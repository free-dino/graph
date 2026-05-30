from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from env import (
    ALPHA, DeliveryEnv, Order, Shipper,
    delivery_reward, is_valid_cell, valid_next_pos,
)
from solvers.solver import Solver

INF = 10 ** 9
MOVES = ("U", "D", "L", "R")
Position = Tuple[int, int]
Move = str


class VRPOrToolsSolver(Solver):

    REPLAN_INTERVAL = 3    # re-solve VRP every N steps
    VRP_TIME_LIMIT  = 0.5  # seconds per VRP call

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._dist_cache: Dict[Tuple[Position, Position], int] = {}
        self._move_cache: Dict[Tuple[Position, Position], str] = {}
        # shipper_id -> ordered list of order_ids to pick up
        self._pending_pickups: Dict[int, List[int]] = {}

    # ------------------------------------------------------------------ BFS

    def _bfs_parents(
        self, start: Position, goal: Position
    ) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}
        queue: deque[Position] = deque([start])
        while queue:
            cur = queue.popleft()
            if cur == goal:
                return parent
            for move in MOVES:
                nxt = valid_next_pos(cur, move, self.grid)
                if nxt != cur and nxt not in parent:
                    parent[nxt] = (cur, move)
                    queue.append(nxt)
        return None

    def _distance(self, a: Position, b: Position) -> int:
        if a == b:
            return 0
        key = (a, b)
        if key in self._dist_cache:
            return self._dist_cache[key]
        parent = self._bfs_parents(a, b)
        if parent is None or b not in parent:
            self._dist_cache[key] = INF
            return INF
        dist, cur = 0, b
        while cur != a:
            prev = parent[cur][0]
            if prev is None:
                self._dist_cache[key] = INF
                return INF
            cur = prev
            dist += 1
        self._dist_cache[key] = dist
        return dist

    def _next_move(self, a: Position, b: Position) -> Move:
        if a == b:
            return "S"
        key = (a, b)
        if key in self._move_cache:
            return self._move_cache[key]
        parent = self._bfs_parents(a, b)
        if parent is None or b not in parent:
            self._move_cache[key] = "S"
            return "S"
        cur = b
        while parent[cur][0] != a:
            prev = parent[cur][0]
            if prev is None:
                self._move_cache[key] = "S"
                return "S"
            cur = prev
        self._move_cache[key] = parent[cur][1]
        return parent[cur][1]

    # ------------------------------------------------------------------ Helpers

    def _is_worth_picking(self, order: Order, shipper: Shipper, t: int, T: int) -> bool:
        """False if this order cannot possibly yield a positive reward."""
        dist_p = self._distance(shipper.position, (order.sx, order.sy))
        dist_d = self._distance((order.sx, order.sy), (order.ex, order.ey))
        t_arrive = t + dist_p + dist_d
        return t_arrive < T and delivery_reward(order, t_arrive, T) > 0.0

    def _delivery_slack(self, order: Order, pos: Position, t: int) -> float:
        dist = self._distance(pos, (order.ex, order.ey))
        return ((order.et - t) - dist) / ALPHA[order.p]

    def _pickup_slack(self, order: Order, pos: Position, t: int) -> float:
        dist_p = self._distance(pos, (order.sx, order.sy))
        dist_d = self._distance((order.sx, order.sy), (order.ex, order.ey))
        return ((order.et - t) - dist_p - dist_d) / ALPHA[order.p]

    # ------------------------------------------------------------------ VRP

    def _solve_vrp(self, obs: dict) -> Dict[int, List[int]]:
        shippers: List[Shipper] = obs["shippers"]
        all_orders: Dict[int, Order] = obs["orders"]
        t: int = obs["t"]
        T: int = obs["T"]

        unassigned: List[Order] = [
            o for o in all_orders.values() if not o.picked and not o.delivered
        ]
        result: Dict[int, List[int]] = {s.id: [] for s in shippers}
        if not unassigned or not shippers:
            return result

        shipper_by_id = {s.id: s for s in shippers}

        # Remaining weight and slot capacity per shipper (excluding already-held orders)
        w_cap: Dict[int, float] = {
            s.id: s.W_max - sum(all_orders[oid].w for oid in s.bag if oid in all_orders)
            for s in shippers
        }
        k_cap: Dict[int, int] = {s.id: s.K_max - len(s.bag) for s in shippers}

        # End position of each shipper's planned route (updated as orders are appended)
        route_end: Dict[int, Position] = {s.id: (s.r, s.c) for s in shippers}

        def feasible(sid: int, o: Order) -> bool:
            return (
                w_cap[sid] >= o.w
                and k_cap[sid] > 0
                and self._is_worth_picking(o, shipper_by_id[sid], t, T)
            )

        def append_cost(sid: int, o: Order) -> int:
            """Distance cost of tacking order o onto the end of shipper sid's route."""
            end = route_end[sid]
            return (
                self._distance(end, (o.sx, o.sy))
                + self._distance((o.sx, o.sy), (o.ex, o.ey))
            )

        # ---- Phase 1: regret-based greedy insertion --------------------------
        # Filter to orders that at least one shipper can serve, sort by urgency.
        remaining: List[Order] = [
            o for o in sorted(unassigned, key=lambda x: (-x.p, x.et))
            if any(feasible(s.id, o) for s in shippers)
        ]

        t_phase1_end = time.time() + self.VRP_TIME_LIMIT * 0.55
        while remaining and time.time() < t_phase1_end:
            best_idx   = -1
            best_sid   = None
            best_regret = -INF
            best_cost   = INF

            for idx, o in enumerate(remaining):
                # Costs of assigning o to each feasible shipper, cheapest first
                costs = sorted(
                    (append_cost(s.id, o), s.id)
                    for s in shippers if feasible(s.id, o)
                )
                if not costs:
                    continue
                c1, sid1 = costs[0]
                # Regret = gap between best and second-best option
                regret = costs[1][0] - c1 if len(costs) > 1 else c1
                if regret > best_regret or (regret == best_regret and c1 < best_cost):
                    best_regret, best_cost = regret, c1
                    best_idx, best_sid = idx, sid1

            if best_sid is None:
                break  # no order can be assigned any more

            o = remaining.pop(best_idx)
            result[best_sid].append(o.id)
            route_end[best_sid] = (o.ex, o.ey)
            w_cap[best_sid] -= o.w
            k_cap[best_sid] -= 1

        # ---- Phase 2: Or-opt-1 inter-route relocation ------------------------
        # Helper: position just before index `pos` in a route.
        def prev_pos(route: List[int], pos: int, start: Position) -> Position:
            if pos == 0:
                return start
            prev_o = all_orders[route[pos - 1]]
            return (prev_o.ex, prev_o.ey)

        # Cost change from *removing* the order at src_pos from route.
        # Negative value = we save travel distance by removing it.
        def removal_delta(
            route: List[int], pos: int, start: Position, o: Order
        ) -> int:
            p, d = (o.sx, o.sy), (o.ex, o.ey)
            before_pos = prev_pos(route, pos, start)

            cost_before = self._distance(before_pos, p) + self._distance(p, d)
            if pos + 1 < len(route):
                nxt_pick = (all_orders[route[pos + 1]].sx, all_orders[route[pos + 1]].sy)
                cost_before += self._distance(d, nxt_pick)
                cost_after   = self._distance(before_pos, nxt_pick)
            else:
                cost_after = 0

            return cost_after - cost_before  # negative = savings

        # Cost change from *inserting* order o at dst_pos in route.
        # Positive value = we pay more travel distance to include it.
        def insertion_delta(
            route: List[int], pos: int, start: Position, o: Order
        ) -> int:
            p, d = (o.sx, o.sy), (o.ex, o.ey)
            before_pos = prev_pos(route, pos, start)

            added = self._distance(before_pos, p) + self._distance(p, d)
            if pos < len(route):
                nxt_pick = (all_orders[route[pos]].sx, all_orders[route[pos]].sy)
                added += self._distance(d, nxt_pick) - self._distance(before_pos, nxt_pick)

            return added

        t_phase2_end = time.time() + self.VRP_TIME_LIMIT * 0.45
        improved = True
        while improved and time.time() < t_phase2_end:
            improved = False
            for src_s in shippers:
                src_start = (src_s.r, src_s.c)
                src_pos = 0
                while src_pos < len(result[src_s.id]):
                    oid = result[src_s.id][src_pos]
                    o   = all_orders[oid]
                    rem = removal_delta(result[src_s.id], src_pos, src_start, o)

                    moved = False
                    for dst_s in shippers:
                        if dst_s.id == src_s.id:
                            continue
                        if w_cap[dst_s.id] < o.w or k_cap[dst_s.id] <= 0:
                            continue

                        dst_start = (dst_s.r, dst_s.c)
                        dst_route = result[dst_s.id]
                        for dst_pos in range(len(dst_route) + 1):
                            if rem + insertion_delta(dst_route, dst_pos, dst_start, o) < -1:
                                result[src_s.id].pop(src_pos)
                                result[dst_s.id].insert(dst_pos, oid)
                                w_cap[src_s.id] += o.w
                                w_cap[dst_s.id] -= o.w
                                k_cap[src_s.id] += 1
                                k_cap[dst_s.id] -= 1
                                improved = True
                                moved = True
                                break
                        if moved:
                            break

                    if not moved:
                        src_pos += 1
                    if improved:
                        break
                if improved:
                    break

        return result

    # ------------------------------------------------------------------ Policy

    def _update_assignments(self, obs: dict) -> None:
        """Re-solve VRP and merge new assignments into _pending_pickups.
        Falls back to greedy for any order the VRP did not cover."""
        try:
            vrp_result = self._solve_vrp(obs)
        except Exception:
            vrp_result = {s.id: [] for s in obs["shippers"]}

        for sid, order_ids in vrp_result.items():
            existing     = self._pending_pickups.get(sid, [])
            existing_set = set(existing)
            new_orders   = [oid for oid in order_ids if oid not in existing_set]
            self._pending_pickups[sid] = existing + new_orders

        # Greedy fallback: assign any order not covered by VRP to the best shipper.
        all_orders: Dict[int, Order] = obs["orders"]
        shippers:   List[Shipper]    = obs["shippers"]
        t: int = obs["t"]
        T: int = obs["T"]

        already_assigned = {
            oid for pending in self._pending_pickups.values() for oid in pending
        }
        unrouted = [
            o for o in all_orders.values()
            if not o.picked and not o.delivered and o.id not in already_assigned
        ]
        for o in sorted(unrouted, key=lambda o: (-o.p, o.et)):
            best_sid, best_slack = None, float("inf")
            for s in shippers:
                if not s.can_carry(o, all_orders):
                    continue
                if not self._is_worth_picking(o, s, t, T):
                    continue
                slack = self._pickup_slack(o, s.position, t)
                if slack < best_slack:
                    best_slack, best_sid = slack, s.id
            if best_sid is not None:
                self._pending_pickups.setdefault(best_sid, []).append(o.id)

    def _shipper_action(
        self, s: Shipper, orders: Dict[int, Order], t: int, T: int
    ) -> Tuple[Move, int]:
        pos: Position = (s.r, s.c)

        # Priority 1: deliver in-bag orders — most urgent first
        deliverable = [
            orders[oid] for oid in s.bag if oid in orders and not orders[oid].delivered
        ]
        if deliverable:
            target = min(deliverable, key=lambda o: self._delivery_slack(o, pos, t))
            goal: Position = (target.ex, target.ey)
            move = self._next_move(pos, goal)
            nxt  = valid_next_pos(pos, move, self.grid)
            return (move, 2) if nxt == goal else (move, 0)

        # Priority 2: pick up the most urgent assigned order that is still profitable
        pending = self._pending_pickups.get(s.id, [])
        pending_valid = [
            oid for oid in pending
            if oid in orders
            and not orders[oid].picked
            and not orders[oid].delivered
            and s.can_carry(orders[oid], orders)
            and self._is_worth_picking(orders[oid], s, t, T)
        ]
        self._pending_pickups[s.id] = pending_valid

        if pending_valid:
            best_oid = min(
                pending_valid,
                key=lambda oid: self._pickup_slack(orders[oid], pos, t),
            )
            o    = orders[best_oid]
            goal = (o.sx, o.sy)
            move = self._next_move(pos, goal)
            nxt  = valid_next_pos(pos, move, self.grid)
            return (move, 1) if nxt == goal else (move, 0)

        return "S", 0

    def run(self) -> dict:
        start_time   = time.time()
        obs          = self.env.reset()
        last_replan_t = -1

        while not obs.get("done", False):
            t: int = obs["t"]
            T: int = obs["T"]
            new_orders_arrived = bool(obs.get("new_order_ids"))
            all_queues_empty   = all(
                not self._pending_pickups.get(s.id) for s in obs["shippers"]
            )

            if (
                new_orders_arrived
                or all_queues_empty
                or (t - last_replan_t) >= self.REPLAN_INTERVAL
            ):
                self._update_assignments(obs)
                last_replan_t = t

            actions = {
                s.id: self._shipper_action(s, obs["orders"], t, T)
                for s in obs["shippers"]
            }
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result("VRPOrToolsSolver", time.time() - start_time)
