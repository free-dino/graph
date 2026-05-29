from __future__ import annotations

import random
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from env import (
    ALPHA,
    DeliveryEnv,
    Order,
    Shipper,
    TIME_UNIT_PER_DAY,
    delivery_reward,
    move_cost,
    r_base,
)
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
NodeId = Tuple[str, int]  # ("start"|"P"|"D", id)
PlanStep = Tuple[int, int, str, int]  # (target_r, target_c, op_type, order_id)

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


class ACOSolver(Solver):
    method_name = "ACO"

    # ----- ACO hyperparameters -----
    ALPHA_TAU: float = 1.0   # trọng số pheromone
    BETA_ETA: float = 3.5    # trọng số heuristic — đặt cao để bám tham lam tốt
    RHO: float = 0.12        # tốc độ bay hơi
    Q: float = 1.0           # hệ số bồi đắp
    TAU_INIT: float = 1.0
    TAU_MIN: float = 0.05
    TAU_MAX: float = 8.0
    ELITE_BONUS: float = 2.0      # bonus cho best-so-far solution
    TOP_RANK: int = 3             # số ant deposit theo rank trong mỗi vòng
    NO_IMPROVE_LIMIT: int = 10    # cắt sớm nếu không cải thiện liên tiếp

    # Giới hạn số đơn unpicked đưa vào một lần solve.
    MAX_UNPICKED_FOR_SOLVE: int = 100
    # Cap candidates per (shipper, step) trong sampling ant để tăng tốc.
    CANDIDATE_CAP: int = 24

    # ----- Replan triggers (adaptive base values; overridden bằng V) -----
    REPLAN_PERIOD: int = 30
    NEW_ORDER_COOLDOWN: int = 4
    STUCK_LIMIT: int = 2
    LOCK_BONUS: float = 1.0  # disabled — đã thử và hurts responsiveness

    def __init__(self, env: DeliveryEnv):
        self.env = env
        if hasattr(env, "public_cfg"):
            self.cfg = env.public_cfg
        elif hasattr(env, "cfg"):
            self.cfg = env.cfg
        else:
            self.cfg = {
                "name": getattr(env, "config_name", "unknown"),
                "N": env.N,
                "C": env.C,
                "G": env.G,
                "T": env.T,
            }
        self.grid = env.grid
        self.orders: List[Order] = []
        self.T: int = int(self.env.T)
        self.C: int = int(self.cfg["C"])
        self.rows: int = len(self.grid)
        self.cols: int = len(self.grid[0]) if self.rows else 0

        # BFS all-pairs
        self._dist: Dict[Position, Dict[Position, int]] = {}
        self._step: Dict[Position, Dict[Position, Move]] = {}
        self._precompute_shortest_paths()

        # Pheromone toàn cục — giữ qua các lần replan để học liên tục.
        self._pheromone: Dict[Tuple[NodeId, NodeId], float] = {}
        self._rng = random.Random(20260520)

        self.plans: Dict[int, List[PlanStep]] = {i: [] for i in range(self.C)}
        self._last_replan_t: int = -(10**9)
        self._stuck_counter: Dict[int, int] = {i: 0 for i in range(self.C)}
        # Theo dõi trạng thái empty của shipper tại lần replan trước.
        self._empty_at_last_replan: Dict[int, bool] = {
            i: True for i in range(self.C)
        }

        # Adaptive triggers theo V (số shipper).
        # Sau khi fix empty-plan trigger, có thể dùng cooldown nhỏ (vì replan
        # không còn fire mỗi tick). Period giữ vừa phải để cho plan thời gian
        # chạy nhưng vẫn refresh định kỳ.
        V = self.C
        if V <= 2:
            self.REPLAN_PERIOD = 40
            self.NEW_ORDER_COOLDOWN = 5
        elif V <= 3:
            self.REPLAN_PERIOD = 25
            self.NEW_ORDER_COOLDOWN = 3
        else:
            self.REPLAN_PERIOD = 25
            self.NEW_ORDER_COOLDOWN = 3

    # ------------------------------------------------------------------
    # BFS precompute.
    # ------------------------------------------------------------------
    def _precompute_shortest_paths(self) -> None:
        free_cells: List[Position] = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] == 0
        ]
        for start in free_cells:
            dist_map: Dict[Position, int] = {start: 0}
            step_map: Dict[Position, Move] = {start: "S"}
            parent: Dict[Position, Tuple[Position, Move]] = {}
            queue: deque[Position] = deque([start])
            while queue:
                cur = queue.popleft()
                cur_dist = dist_map[cur]
                for mv in MOVES:
                    dr, dc = DIRS[mv]
                    nr, nc = cur[0] + dr, cur[1] + dc
                    if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                        continue
                    if self.grid[nr][nc] != 0:
                        continue
                    nxt = (nr, nc)
                    if nxt in dist_map:
                        continue
                    dist_map[nxt] = cur_dist + 1
                    parent[nxt] = (cur, mv)
                    queue.append(nxt)
            for target in dist_map:
                if target == start:
                    continue
                cur = target
                first_move = "S"
                while True:
                    prev, mv = parent[cur]
                    if prev == start:
                        first_move = mv
                        break
                    cur = prev
                step_map[target] = first_move
            self._dist[start] = dist_map
            self._step[start] = step_map

    def _distance(self, a: Position, b: Position) -> int:
        if a == b:
            return 0
        return self._dist.get(a, {}).get(b, INF)

    def _next_move(self, a: Position, b: Position) -> Move:
        if a == b:
            return "S"
        return self._step.get(a, {}).get(b, "S")

    # ------------------------------------------------------------------
    # Heuristic helpers.
    # ------------------------------------------------------------------
    @staticmethod
    def _bag_weight(shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _exp_reward_at(self, order: Order, t_arrival: int) -> float:
        t = min(max(t_arrival, 0), self.T - 1)
        return delivery_reward(order, t, self.T)

    def _move_cost_estimate(self, dist: int, weight_carried: float, w_max: float) -> float:
        if dist <= 0:
            return 0.0
        return -move_cost(weight_carried, w_max) * dist

    # ------------------------------------------------------------------
    # Pheromone helpers.
    # ------------------------------------------------------------------
    def _tau(self, frm: NodeId, to: NodeId) -> float:
        return self._pheromone.get((frm, to), self.TAU_INIT)

    def _clip_tau(self, v: float) -> float:
        return max(self.TAU_MIN, min(self.TAU_MAX, v))

    def _evaporate(self) -> None:
        keep = 1.0 - self.RHO
        for k in list(self._pheromone.keys()):
            new_v = self._pheromone[k] * keep
            if new_v < self.TAU_MIN:
                new_v = self.TAU_MIN
            self._pheromone[k] = new_v

    def _deposit(self, edges: List[Tuple[NodeId, NodeId]], amount: float) -> None:
        if amount <= 0 or not edges:
            return
        for e in edges:
            cur = self._pheromone.get(e, self.TAU_INIT)
            self._pheromone[e] = self._clip_tau(cur + amount)

    # ------------------------------------------------------------------
    # Hungarian-iterative construction ant (deterministic, strong baseline).
    # ------------------------------------------------------------------
    def _build_hungarian_ant(
        self,
        obs: dict,
        unpicked: List[Order],
        bag_orders: Dict[int, List[Order]],
        locked_first: Optional[Dict[int, Tuple[str, int]]] = None,
    ) -> Tuple[Dict[int, List[PlanStep]], float, List[Tuple[NodeId, NodeId]]]:
        """Build solution bằng **round-Hungarian**: mỗi round, mỗi shipper
        nhận TỐI ĐA 1 action (chọn theo score giảm dần, không đụng nhau).
        Phân phối công bằng giữa các shipper — trái với one-shot iterative
        Hungarian (sẽ "ăn dồn" cho 1 shipper duy nhất).

        `locked_first` (optional): dict[shipper_id] -> (op_type, oid) — gợi
        ý first action cho shipper. Nếu khớp candidate, boost LOCK_BONUS.
        """
        shippers_list: List[Shipper] = obs["shippers"]
        orders_map: Dict[int, Order] = obs["orders"]
        t_now: int = int(obs.get("t", 0))
        locked_first = locked_first or {}

        routes: Dict[int, List[PlanStep]] = {s.id: [] for s in shippers_list}
        edges: List[Tuple[NodeId, NodeId]] = []
        score = 0.0

        pos: Dict[int, Position] = {s.id: s.position for s in shippers_list}
        bag_set: Dict[int, set] = {s.id: set(s.bag) for s in shippers_list}
        bag_w: Dict[int, float] = {
            s.id: self._bag_weight(s, orders_map) for s in shippers_list
        }
        bag_n: Dict[int, int] = {s.id: len(s.bag) for s in shippers_list}
        cur_time: Dict[int, int] = {s.id: t_now for s in shippers_list}
        last_node: Dict[int, NodeId] = {s.id: ("start", s.id) for s in shippers_list}
        shipper_by_id: Dict[int, Shipper] = {s.id: s for s in shippers_list}

        available: Dict[int, Order] = {o.id: o for o in unpicked}
        handled_first: set = set()  # shipper đã chọn first action

        n_total = (
            sum(s.K_max for s in shippers_list)
            + len(unpicked)
            + sum(len(b) for b in bag_set.values())
        )
        max_rounds = max(4, (n_total // max(1, len(shippers_list))) + 4)

        for _ in range(max_rounds):
            # Build candidates (sid, op, oid, score, d, rwd) cho TẤT CẢ shippers.
            cands: List[Tuple[float, int, str, int, int, float]] = []

            for s in shippers_list:
                sid = s.id
                is_first = (sid in locked_first) and (sid not in handled_first)
                lock_op = locked_first.get(sid, (None, None))[0]
                lock_oid = locked_first.get(sid, (None, None))[1]
                # Dùng t_now cho first action (matching Greedy optimism); dùng
                # cur_time cho subsequent actions trong plan (accuracy).
                t_eval = t_now if len(routes[sid]) == 0 else cur_time[sid]

                # Delivery ứng viên.
                for oid in bag_set[sid]:
                    o = orders_map.get(oid)
                    if o is None:
                        continue
                    d = self._distance(pos[sid], (o.ex, o.ey))
                    if d >= INF:
                        continue
                    t_arrive = t_eval + d + 1
                    rwd = self._exp_reward_at(o, t_arrive)
                    cost = self._move_cost_estimate(d, bag_w[sid], s.W_max)
                    net = rwd - cost
                    if net <= 0.0:
                        net = max(net, 0.05)
                    sc = net / (d + 1.0)
                    if is_first and lock_op == "D" and oid == lock_oid:
                        sc *= self.LOCK_BONUS
                    cands.append((sc, sid, "D", oid, d, rwd))

                # Pickup ứng viên.
                if bag_n[sid] < s.K_max:
                    for oid, o in available.items():
                        if bag_w[sid] + o.w > s.W_max:
                            continue
                        dp = self._distance(pos[sid], (o.sx, o.sy))
                        if dp >= INF:
                            continue
                        dd = self._distance((o.sx, o.sy), (o.ex, o.ey))
                        if dd >= INF:
                            continue
                        t_arrive = t_eval + dp + 1 + dd + 1
                        rwd = self._exp_reward_at(o, t_arrive)
                        cost = self._move_cost_estimate(
                            dp, bag_w[sid], s.W_max
                        ) + self._move_cost_estimate(
                            dd, bag_w[sid] + o.w, s.W_max
                        )
                        net = rwd - cost
                        if net <= 0:
                            continue
                        sc = net / (dp + dd + 2.0)
                        if is_first and lock_op == "P" and oid == lock_oid:
                            sc *= self.LOCK_BONUS
                        cands.append((sc, sid, "P", oid, dp, rwd))

            if not cands:
                break

            cands.sort(key=lambda x: -x[0])
            assigned_this_round: set = set()
            reserved_pickups: set = set()
            applied_any = False

            for sc, sid, op_t, oid, d, rwd in cands:
                if sid in assigned_this_round:
                    continue
                if op_t == "P" and oid in reserved_pickups:
                    continue
                s = shipper_by_id[sid]
                o = orders_map[oid]
                # Validate (state có thể đã đổi sau các round trước).
                if op_t == "D":
                    if oid not in bag_set[sid]:
                        continue
                else:  # P
                    if oid not in available:
                        continue
                    if bag_n[sid] >= s.K_max:
                        continue
                    if bag_w[sid] + o.w > s.W_max:
                        continue

                # Apply.
                if op_t == "P":
                    routes[sid].append((o.sx, o.sy, "P", oid))
                    edges.append((last_node[sid], ("P", oid)))
                    cur_time[sid] += d + 1
                    pos[sid] = (o.sx, o.sy)
                    bag_set[sid].add(oid)
                    bag_w[sid] += o.w
                    bag_n[sid] += 1
                    available.pop(oid, None)
                    reserved_pickups.add(oid)
                    score -= self._move_cost_estimate(
                        d, bag_w[sid] - o.w, s.W_max
                    )
                    last_node[sid] = ("P", oid)
                else:
                    routes[sid].append((o.ex, o.ey, "D", oid))
                    edges.append((last_node[sid], ("D", oid)))
                    cur_time[sid] += d + 1
                    pos[sid] = (o.ex, o.ey)
                    bag_set[sid].discard(oid)
                    bag_w[sid] -= o.w
                    bag_n[sid] -= 1
                    score += rwd - self._move_cost_estimate(
                        d, bag_w[sid] + o.w, s.W_max
                    )
                    last_node[sid] = ("D", oid)

                assigned_this_round.add(sid)
                handled_first.add(sid)
                applied_any = True

            if not applied_any:
                break

        return routes, score, edges

    # ------------------------------------------------------------------
    # Stochastic ant — round-based sampling (mỗi round mỗi shipper ≤ 1 ops)
    # Sử dụng softmax theo τ^α · η^β + pickup-conflict resolution.
    # ------------------------------------------------------------------
    def _build_sampling_ant(
        self,
        obs: dict,
        unpicked: List[Order],
        bag_orders: Dict[int, List[Order]],
        rng: random.Random,
        locked_first: Optional[Dict[int, Tuple[str, int]]] = None,
    ) -> Tuple[Dict[int, List[PlanStep]], float, List[Tuple[NodeId, NodeId]]]:
        orders_map: Dict[int, Order] = obs["orders"]
        shippers_list: List[Shipper] = obs["shippers"]
        t_now: int = int(obs.get("t", 0))
        locked_first = locked_first or {}

        routes: Dict[int, List[PlanStep]] = {s.id: [] for s in shippers_list}
        edges: List[Tuple[NodeId, NodeId]] = []
        score = 0.0

        pos: Dict[int, Position] = {s.id: s.position for s in shippers_list}
        bag_set: Dict[int, set] = {s.id: set(s.bag) for s in shippers_list}
        bag_w: Dict[int, float] = {
            s.id: self._bag_weight(s, orders_map) for s in shippers_list
        }
        bag_n: Dict[int, int] = {s.id: len(s.bag) for s in shippers_list}
        cur_time: Dict[int, int] = {s.id: t_now for s in shippers_list}
        last_node: Dict[int, NodeId] = {
            s.id: ("start", s.id) for s in shippers_list
        }
        shipper_by_id: Dict[int, Shipper] = {s.id: s for s in shippers_list}
        available: Dict[int, Order] = {o.id: o for o in unpicked}
        first_step_done: Dict[int, bool] = {s.id: False for s in shippers_list}

        n_total = (
            sum(s.K_max for s in shippers_list)
            + len(unpicked)
            + sum(len(b) for b in bag_set.values())
        )
        max_rounds = max(4, (n_total // max(1, len(shippers_list))) + 4)

        for _ in range(max_rounds):
            # Sinh candidates cho mọi shipper trong round này.
            cands: List[Tuple[float, int, str, int, int, float]] = []
            # (weight_aco, sid, op_type, oid, dist, exp_reward)

            for s in shippers_list:
                sid = s.id
                lock_op = locked_first.get(sid, (None, None))[0]
                lock_oid = locked_first.get(sid, (None, None))[1]
                want_lock = not first_step_done[sid] and lock_op is not None
                t_eval = t_now if len(routes[sid]) == 0 else cur_time[sid]

                # Delivery candidates.
                for oid in bag_set[sid]:
                    o = orders_map.get(oid)
                    if o is None:
                        continue
                    d = self._distance(pos[sid], (o.ex, o.ey))
                    if d >= INF:
                        continue
                    t_arrive = t_eval + d + 1
                    rwd = self._exp_reward_at(o, t_arrive)
                    cost = self._move_cost_estimate(d, bag_w[sid], s.W_max)
                    net = rwd - cost
                    if net <= 0.0:
                        net = max(net, 0.05)
                    eta = net / (d + 1.0)
                    tau = self._tau(last_node[sid], ("D", oid))
                    w_aco = (max(tau, 1e-9) ** self.ALPHA_TAU) * (
                        max(eta, 1e-9) ** self.BETA_ETA
                    )
                    if want_lock and lock_op == "D" and oid == lock_oid:
                        w_aco *= self.LOCK_BONUS
                    cands.append((w_aco, sid, "D", oid, d, rwd))

                # Pickup candidates.
                if bag_n[sid] < s.K_max:
                    for oid, o in available.items():
                        if bag_w[sid] + o.w > s.W_max:
                            continue
                        dp = self._distance(pos[sid], (o.sx, o.sy))
                        if dp >= INF:
                            continue
                        dd = self._distance((o.sx, o.sy), (o.ex, o.ey))
                        if dd >= INF:
                            continue
                        t_arrive = t_eval + dp + 1 + dd + 1
                        rwd = self._exp_reward_at(o, t_arrive)
                        if rwd <= 0:
                            continue
                        cost = self._move_cost_estimate(
                            dp, bag_w[sid], s.W_max
                        ) + self._move_cost_estimate(
                            dd, bag_w[sid] + o.w, s.W_max
                        )
                        net = rwd - cost
                        if net <= 0:
                            continue
                        eta = net / (dp + dd + 2.0)
                        urgency = 1.0 + max(
                            0.0,
                            (TIME_UNIT_PER_DAY / max(o.et - t_eval, 1))
                            * 0.05,
                        )
                        eta *= urgency
                        tau = self._tau(last_node[sid], ("P", oid))
                        w_aco = (max(tau, 1e-9) ** self.ALPHA_TAU) * (
                            max(eta, 1e-9) ** self.BETA_ETA
                        )
                        if want_lock and lock_op == "P" and oid == lock_oid:
                            w_aco *= self.LOCK_BONUS
                        cands.append((w_aco, sid, "P", oid, dp, rwd))

            if not cands:
                break

            # Cap (tổng candidates) cho tốc độ.
            if len(cands) > self.CANDIDATE_CAP * len(shippers_list):
                cands.sort(key=lambda x: -x[0])
                cands = cands[: self.CANDIDATE_CAP * len(shippers_list)]

            assigned_this_round: set = set()
            reserved_pickups: set = set()
            applied_any = False

            # Lặp: mỗi vòng inner-loop, sample ONE (sid, action), apply,
            # cho tới khi không còn ứng viên hợp lệ trong round.
            while True:
                # Lọc còn hợp lệ trong round.
                valid: List[Tuple[float, int, str, int, int, float]] = []
                total_w = 0.0
                for c in cands:
                    sc, sid, op_t, oid, d, rwd = c
                    if sid in assigned_this_round:
                        continue
                    if op_t == "P" and oid in reserved_pickups:
                        continue
                    valid.append(c)
                    total_w += sc
                if not valid:
                    break

                if total_w <= 0:
                    chosen = max(valid, key=lambda x: x[0])
                else:
                    r = rng.random() * total_w
                    cum = 0.0
                    chosen = valid[-1]
                    for c in valid:
                        cum += c[0]
                        if cum >= r:
                            chosen = c
                            break

                _w, sid, op_t, oid, d, rwd = chosen
                s = shipper_by_id[sid]
                o = orders_map[oid]

                # Validate (định kỳ).
                if op_t == "D" and oid not in bag_set[sid]:
                    assigned_this_round.add(sid)
                    continue
                if op_t == "P":
                    if oid not in available:
                        continue
                    if bag_n[sid] >= s.K_max:
                        assigned_this_round.add(sid)
                        continue
                    if bag_w[sid] + o.w > s.W_max:
                        assigned_this_round.add(sid)
                        continue

                # Apply.
                if op_t == "P":
                    routes[sid].append((o.sx, o.sy, "P", oid))
                    edges.append((last_node[sid], ("P", oid)))
                    cur_time[sid] += d + 1
                    pos[sid] = (o.sx, o.sy)
                    bag_set[sid].add(oid)
                    bag_w[sid] += o.w
                    bag_n[sid] += 1
                    available.pop(oid, None)
                    reserved_pickups.add(oid)
                    score -= self._move_cost_estimate(
                        d, bag_w[sid] - o.w, s.W_max
                    )
                    last_node[sid] = ("P", oid)
                else:
                    routes[sid].append((o.ex, o.ey, "D", oid))
                    edges.append((last_node[sid], ("D", oid)))
                    cur_time[sid] += d + 1
                    pos[sid] = (o.ex, o.ey)
                    bag_set[sid].discard(oid)
                    bag_w[sid] -= o.w
                    bag_n[sid] -= 1
                    score += rwd - self._move_cost_estimate(
                        d, bag_w[sid] + o.w, s.W_max
                    )
                    last_node[sid] = ("D", oid)

                assigned_this_round.add(sid)
                first_step_done[sid] = True
                applied_any = True

            if not applied_any:
                break

        return routes, score, edges

    # ------------------------------------------------------------------
    # Local search trên solution (cải thiện best ant).
    # Đơn giản & nhanh: chỉ 2-opt within-route, single pass, có time cap.
    # ------------------------------------------------------------------
    def _local_search(
        self,
        obs: dict,
        routes: Dict[int, List[PlanStep]],
        deadline: float,
    ) -> Tuple[Dict[int, List[PlanStep]], float, List[Tuple[NodeId, NodeId]]]:
        """2-opt within-route, single pass với best-improvement.

        Tham số `deadline` là wall-clock cutoff (time.time() unit).
        """
        shippers_list: List[Shipper] = obs["shippers"]
        orders_map: Dict[int, Order] = obs["orders"]
        t_now: int = int(obs.get("t", 0))

        cur_routes: Dict[int, List[PlanStep]] = {
            sid: list(r) for sid, r in routes.items()
        }
        cur_score, cur_edges = self._score_routes(
            cur_routes, shippers_list, orders_map, t_now
        )

        for sid in list(cur_routes.keys()):
            if time.time() > deadline:
                break
            route = cur_routes[sid]
            if len(route) < 4:
                continue
            L = len(route)
            improved_this_route = True
            # Tối đa 2 vòng cải thiện liên tiếp trên 1 route.
            sweeps = 0
            while improved_this_route and sweeps < 2 and time.time() < deadline:
                improved_this_route = False
                sweeps += 1
                best_local_score = cur_score
                best_local_route: Optional[List[PlanStep]] = None
                for i in range(L - 1):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, L):
                        candidate = (
                            route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                        )
                        if not self._is_valid_route(candidate):
                            continue
                        trial_routes = dict(cur_routes)
                        trial_routes[sid] = candidate
                        sc, _ = self._score_routes(
                            trial_routes, shippers_list, orders_map, t_now
                        )
                        if sc > best_local_score + 1e-6:
                            best_local_score = sc
                            best_local_route = candidate
                if best_local_route is not None:
                    cur_routes[sid] = best_local_route
                    cur_score = best_local_score
                    route = best_local_route
                    improved_this_route = True

        # Tính lại edges của routes cuối.
        _, cur_edges = self._score_routes(cur_routes, shippers_list, orders_map, t_now)
        return cur_routes, cur_score, cur_edges

    @staticmethod
    def _is_valid_route(route: List[PlanStep]) -> bool:
        """Validate: nếu cặp (P, D) cùng oid đều xuất hiện trong route thì
        P phải đứng trước D. Đơn nằm trong bag (chỉ có D, không có P) hợp lệ.
        """
        pickup_idx: Dict[int, int] = {}
        for idx, step in enumerate(route):
            _r, _c, op_t, oid = step
            if op_t == "P":
                if oid in pickup_idx:
                    return False  # pickup trùng
                pickup_idx[oid] = idx
        for idx, step in enumerate(route):
            _r, _c, op_t, oid = step
            if op_t == "D" and oid in pickup_idx:
                if pickup_idx[oid] > idx:
                    return False
        return True

    def _score_routes(
        self,
        routes: Dict[int, List[PlanStep]],
        shippers_list: List[Shipper],
        orders_map: Dict[int, Order],
        t_now: int,
    ) -> Tuple[float, List[Tuple[NodeId, NodeId]]]:
        """Tính score (sum net reward) + edges (cho deposit pheromone)."""
        edges: List[Tuple[NodeId, NodeId]] = []
        total = 0.0
        for s in shippers_list:
            sid = s.id
            route = routes.get(sid, [])
            pos = s.position
            bag_w = self._bag_weight(s, orders_map)
            cur_time = t_now
            last_node: NodeId = ("start", sid)

            for step in route:
                r, c, op_t, oid = step
                target = (r, c)
                d = self._distance(pos, target)
                if d >= INF:
                    return -float("inf"), edges
                o = orders_map.get(oid)
                if o is None:
                    return -float("inf"), edges

                if op_t == "P":
                    total -= self._move_cost_estimate(d, bag_w, s.W_max)
                    bag_w += o.w
                    edges.append((last_node, ("P", oid)))
                    last_node = ("P", oid)
                else:
                    total -= self._move_cost_estimate(d, bag_w, s.W_max)
                    cur_time_arrive = cur_time + d + 1
                    rwd = self._exp_reward_at(o, cur_time_arrive)
                    total += rwd
                    bag_w -= o.w
                    edges.append((last_node, ("D", oid)))
                    last_node = ("D", oid)
                cur_time += d + 1
                pos = target
        return total, edges

    # ------------------------------------------------------------------
    # ACO main loop.
    # ------------------------------------------------------------------
    def _adaptive_budget(self, n_orders: int) -> Tuple[float, int, int]:
        """Return (time_budget_s, n_ants, n_iterations) tuỳ kích thước bài toán."""
        if n_orders <= 8:
            return 0.5, 10, 20
        if n_orders <= 20:
            return 1.0, 14, 25
        if n_orders <= 40:
            return 1.6, 14, 25
        if n_orders <= 70:
            return 2.2, 12, 22
        return 2.8, 10, 18

    def _aco_search(
        self,
        obs: dict,
        unpicked: List[Order],
        bag_orders: Dict[int, List[Order]],
        locked_first: Optional[Dict[int, Tuple[str, int]]] = None,
    ) -> Dict[int, List[PlanStep]]:
        start = time.time()

        n_orders_total = len(unpicked) + sum(len(b) for b in bag_orders.values())
        time_budget, n_ants, n_iter = self._adaptive_budget(n_orders_total)
        deadline = start + time_budget

        best_routes: Optional[Dict[int, List[PlanStep]]] = None
        best_score = -float("inf")
        best_edges: List[Tuple[NodeId, NodeId]] = []

        # ---- Seed 1: Hungarian-iterative (đảm bảo ≥ Greedy BFS) ----
        h_routes, h_score, h_edges = self._build_hungarian_ant(
            obs, unpicked, bag_orders, locked_first
        )
        if h_score > best_score:
            best_score = h_score
            best_routes = h_routes
            best_edges = h_edges

        # ---- Seed 2: 1 shuffled-greedy ant để diversity ----
        try:
            g_routes, g_score, g_edges = self._build_sampling_ant(
                obs, unpicked, bag_orders, self._rng, locked_first
            )
            if g_score > best_score:
                best_score = g_score
                best_routes = g_routes
                best_edges = g_edges
        except Exception:
            pass

        # ---- ACO iterations (chỉ construction + pheromone update) ----
        no_improve = 0
        for it in range(n_iter):
            if time.time() > deadline:
                break
            if no_improve >= self.NO_IMPROVE_LIMIT:
                break

            iter_solutions: List[
                Tuple[float, List[Tuple[NodeId, NodeId]]]
            ] = []
            iter_best_sc = -float("inf")
            iter_best_routes: Optional[Dict[int, List[PlanStep]]] = None
            iter_best_edges: List[Tuple[NodeId, NodeId]] = []

            for _ in range(n_ants):
                if time.time() > deadline:
                    break
                routes, sc, eds = self._build_sampling_ant(
                    obs, unpicked, bag_orders, self._rng, locked_first
                )
                iter_solutions.append((sc, eds))
                if sc > iter_best_sc:
                    iter_best_sc = sc
                    iter_best_routes = routes
                    iter_best_edges = eds
                if sc > best_score:
                    best_score = sc
                    best_routes = routes
                    best_edges = eds
                    no_improve = -1

            # ---- Pheromone update ----
            self._evaporate()

            # Rank-based deposit (top-R ant trong iteration).
            iter_solutions.sort(key=lambda x: -x[0])
            top = iter_solutions[: self.TOP_RANK]
            for rank, (sc, eds) in enumerate(top):
                if sc <= 0 or not eds:
                    continue
                norm = max(1.0, abs(best_score) if best_score > 0 else sc)
                amt = self.Q * (sc / norm) * ((self.TOP_RANK - rank) / self.TOP_RANK)
                self._deposit(eds, amt)

            # Elite (best-so-far) bonus.
            if best_routes is not None and best_score > 0 and best_edges:
                norm = max(1.0, abs(best_score))
                amt = self.Q * (best_score / norm) * self.ELITE_BONUS
                self._deposit(best_edges, amt)

            no_improve += 1

        # ---- Local search ở cuối, một lần, trên global best ----
        # Cấp 25% budget còn dư cho LS (tối thiểu 0.2s).
        if best_routes is not None:
            ls_deadline = max(
                time.time() + 0.2,
                deadline + (time_budget * 0.25),
            )
            ls_routes, ls_score, _ = self._local_search(
                obs, best_routes, ls_deadline
            )
            if ls_score > best_score:
                best_score = ls_score
                best_routes = ls_routes

        return (
            best_routes
            if best_routes is not None
            else {s.id: [] for s in obs["shippers"]}
        )

    # ------------------------------------------------------------------
    # Replan entrypoint.
    # ------------------------------------------------------------------
    def _replan(self, obs: dict) -> None:
        orders_map: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        C = len(shippers)

        unpicked: List[Order] = []
        bag_orders: Dict[int, List[Order]] = {s.id: [] for s in shippers}
        for o in orders_map.values():
            if o.delivered:
                continue
            if o.picked and 0 <= o.carrier < C:
                bag_orders[o.carrier].append(o)
            elif not o.picked:
                unpicked.append(o)

        if not unpicked and not any(bag_orders.values()):
            for s in shippers:
                self.plans[s.id] = []
            return

        unpicked.sort(key=lambda o: (-o.p, o.et, o.id))
        if len(unpicked) > self.MAX_UNPICKED_FOR_SOLVE:
            unpicked = unpicked[: self.MAX_UNPICKED_FOR_SOLVE]

        # Xây dựng locked_first từ plan cũ — giảm thrashing.
        locked_first: Dict[int, Tuple[str, int]] = {}
        for s in shippers:
            plan = self.plans.get(s.id, [])
            if not plan:
                continue
            _tr, _tc, op_t, oid = plan[0]
            o = orders_map.get(oid)
            if o is None or o.delivered:
                continue
            if op_t == "P":
                if not o.picked:
                    # Pickup vẫn khả thi cho shipper s nếu còn slot/trọng tải
                    if len(s.bag) < s.K_max and (
                        sum(orders_map[b].w for b in s.bag if b in orders_map)
                        + o.w
                        <= s.W_max
                    ):
                        locked_first[s.id] = (op_t, oid)
            else:  # "D"
                if oid in s.bag:
                    locked_first[s.id] = (op_t, oid)

        routes = self._aco_search(obs, unpicked, bag_orders, locked_first)
        for s in shippers:
            self.plans[s.id] = routes.get(s.id, [])
        # Cập nhật snapshot trạng thái empty SAU khi replan đã đặt plan.
        self._empty_at_last_replan = {
            s.id: len(self.plans[s.id]) == 0 for s in shippers
        }

    # ------------------------------------------------------------------
    # Quản lý plan.
    # ------------------------------------------------------------------
    def _advance_plans(self, obs: dict) -> bool:
        orders_map: Dict[int, Order] = obs["orders"]
        invalid = False
        for s in obs["shippers"]:
            plan = self.plans[s.id]
            while plan:
                _tr, _tc, op_t, oid = plan[0]
                o = orders_map.get(oid)
                if op_t == "P":
                    if oid in s.bag:
                        plan.pop(0)
                        continue
                    if o is None or o.delivered or (o.picked and o.carrier != s.id):
                        plan.pop(0)
                        invalid = True
                        continue
                    break
                else:
                    if o is None or o.delivered:
                        plan.pop(0)
                        continue
                    if oid not in s.bag:
                        plan.pop(0)
                        invalid = True
                        continue
                    break
        return invalid

    def _needs_replan(self, obs: dict, invalid_after_advance: bool) -> bool:
        t_now: int = int(obs.get("t", 0))
        orders_map: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        has_unpicked = any(
            (not o.delivered) and (not o.picked) for o in orders_map.values()
        )
        has_bag = any(s.bag for s in shippers)
        if not has_unpicked and not has_bag:
            return False

        if invalid_after_advance:
            return True

        # Empty-plan trigger: shipper hoàn thành plan và có công việc tiềm năng.
        # Quan trọng: chỉ trigger nếu shipper VỪA trở nên rỗng (was non-empty
        # before this tick) HOẶC nếu đã đủ EMPTY_REPLAN_COOLDOWN từ lần
        # replan trước (tránh re-replan mỗi tick khi không có việc khả thi
        # cho shipper).
        empty_cooldown = max(self.NEW_ORDER_COOLDOWN, 3)
        for s in shippers:
            if not self.plans[s.id]:
                if s.bag or has_unpicked:
                    if not self._empty_at_last_replan.get(s.id, False):
                        return True
                    if t_now - self._last_replan_t >= empty_cooldown:
                        return True

        if obs.get("new_order_ids"):
            if t_now - self._last_replan_t >= self.NEW_ORDER_COOLDOWN:
                return True

        for s in shippers:
            if (
                self._stuck_counter.get(s.id, 0) >= self.STUCK_LIMIT
                and self.plans[s.id]
            ):
                return True

        if t_now - self._last_replan_t >= self.REPLAN_PERIOD:
            return True

        return False

    def _action_for(self, shipper: Shipper) -> Tuple[Move, int]:
        plan = self.plans[shipper.id]
        if not plan:
            return ("S", 2 if shipper.bag else 0)
        tr, tc, op_t, _oid = plan[0]
        target = (tr, tc)
        if shipper.position == target:
            return ("S", 1 if op_t == "P" else 2)
        move = self._next_move(shipper.position, target)
        if op_t == "D":
            return (move, 2)
        return (move, 2 if shipper.bag else 0)

    # ------------------------------------------------------------------
    # Main loop.
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            invalid = self._advance_plans(obs)

            if self._needs_replan(obs, invalid):
                self._replan(obs)
                self._last_replan_t = int(obs["t"])
                self._stuck_counter = {s.id: 0 for s in obs["shippers"]}

            actions: Dict[int, Tuple[Move, int]] = {}
            for s in obs["shippers"]:
                actions[s.id] = self._action_for(s)

            prev_positions = {s.id: s.position for s in obs["shippers"]}
            obs, _, done, _ = self.env.step(actions)

            for s in obs["shippers"]:
                prev = prev_positions.get(s.id)
                if prev is None:
                    continue
                if s.position == prev:
                    plan = self.plans[s.id]
                    if plan:
                        tr, tc, _op, _oid = plan[0]
                        if (tr, tc) != s.position:
                            self._stuck_counter[s.id] = (
                                self._stuck_counter.get(s.id, 0) + 1
                            )
                        else:
                            self._stuck_counter[s.id] = 0
                    else:
                        self._stuck_counter[s.id] = 0
                else:
                    self._stuck_counter[s.id] = 0

            if done:
                break

        return self.env.result(
            self.method_name, elapsed_sec=time.time() - start_time
        )
