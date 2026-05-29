from __future__ import annotations

import time
from collections import OrderedDict, deque
from heapq import nsmallest
from typing import Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
MOVE_BY_DELTA: Dict[Tuple[int, int], Move] = {
    (-1, 0): "U",
    (1, 0): "D",
    (0, -1): "L",
    (0, 1): "R",
}

ALPHA = {1: 1.0, 2: 2.0, 3: 3.0}
BETA = {1: 0.1, 2: 0.3, 3: 0.5}

COMPLETION_BONUS = 1.0
COMPLETION_MISS_PENALTY = -4.0
OPPORTUNISTIC_MAX_DETOUR = 1
ENABLE_OPPORTUNISTIC = True


class Task:
    """Mô tả tác vụ nhỏ dùng cho gán việc và lập kế hoạch ngắn hạn."""

    __slots__ = ("kind", "goal", "order_id", "priority", "deadline", "score")

    def __init__(
        self,
        kind: str,
        goal: Position,
        order_id: Optional[int] = None,
        priority: int = 0,
        deadline: int = INF,
        score: float = 0.0,
    ) -> None:
        self.kind = kind
        self.goal = goal
        self.order_id = order_id
        self.priority = priority
        self.deadline = deadline
        self.score = score


class MAPDCBSSolver(Solver):
    """
    Solver MAPD-CBS-lite theo cửa sổ lăn.

    Solver gán một mục tiêu ngắn hạn cho mỗi shipper, lập tiền tố đường đi
    ngắn, đặt chỗ cho các tiền tố ưu tiên cao hơn, rồi chỉ thực thi một bước.
    """

    method_name = "MAPD-CBS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._goal_distance_cache: OrderedDict[Position, List[int]] = OrderedDict()
        self._max_goal_cache = 384
        self._pickup_targets: Dict[int, int] = {}
        self._tasks: Dict[int, Task] = {}
        self._last_assignment_t = -INF
        self._replan_interval = 2
        self._last_expected_pos: Dict[int, Position] = {}
        self._last_move: Dict[int, Move] = {}
        self._last_conflict_losers: Set[int] = set()
        self._start_time = 0.0

    # ------------------------------------------------------------------
    # Grid and BFS helpers
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        """Sinh các bước đi hợp lệ tới 4 ô kề từ một vị trí trên lưới."""
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _is_adjacent_or_same(a: Position, b: Position) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) <= 1

    @staticmethod
    def _base_reward(weight: float) -> float:
        """Mô phỏng các mức thưởng cơ bản của môi trường để chấm điểm cục bộ."""
        if weight <= 0.2:
            return 4.0
        if weight <= 3:
            return 10.0
        if weight <= 10:
            return 15.0
        if weight <= 30:
            return 20.0
        return 30.0

    def _estimated_delivery_reward(self, order: Order, delivery_t: int, horizon_t: int) -> float:
        """Ước lượng reward giao hàng từ trường công khai của order và thời điểm giao dự kiến."""
        base = self._base_reward(order.w)
        priority = int(order.p)
        if delivery_t <= order.et:
            bonus = max(0.0, (order.et - delivery_t) / max(order.et, 1))
            return ALPHA.get(priority, 1.0) * base * (1.0 + bonus)
        factor = max(0.0, 1.0 - (delivery_t - order.et) / max(horizon_t, 1))
        return BETA.get(priority, 0.1) * base * factor

    def _flat_index(self, pos: Position) -> int:
        return pos[0] * len(self.grid[0]) + pos[1]

    def _goal_cache_limit(self) -> int:
        """Giới hạn số distance map theo quy mô lưới."""
        area = len(self.grid) * len(self.grid[0]) if self.grid else 0
        if area >= 10000:
            return 384
        if area >= 3600:
            return 256
        return self._max_goal_cache

    def _trim_goal_cache(self) -> None:
        """Xóa các distance map ít dùng gần đây nhất thay vì clear toàn bộ cache."""
        limit = self._goal_cache_limit()
        while len(self._goal_distance_cache) > limit:
            self._goal_distance_cache.popitem(last=False)

    def _goal_distance_map(self, goal: Position) -> List[int]:
        """Chạy một BFS từ goal ra toàn bản đồ và cache distance map dạng phẳng."""
        cached = self._goal_distance_cache.get(goal)
        if cached is not None:
            self._goal_distance_cache.move_to_end(goal)
            return cached

        rows = len(self.grid)
        cols = len(self.grid[0]) if rows else 0
        dist = [INF] * (rows * cols)
        if not rows or not cols or not is_valid_cell(goal, self.grid):
            self._goal_distance_cache[goal] = dist
            self._trim_goal_cache()
            return dist

        goal_idx = self._flat_index(goal)
        dist[goal_idx] = 0
        queue: deque[Position] = deque([goal])

        while queue:
            r, c = queue.popleft()
            next_dist = dist[r * cols + c] + 1
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if self.grid[nr][nc] != 0:
                    continue
                idx = nr * cols + nc
                if dist[idx] != INF:
                    continue
                dist[idx] = next_dist
                queue.append((nr, nc))

        self._goal_distance_cache[goal] = dist
        self._trim_goal_cache()
        return dist

    def _distance(self, start: Position, goal: Position) -> int:
        """Trả về khoảng cách ngắn nhất bằng cách đọc distance map đã cache theo goal."""
        if start == goal:
            return 0
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return INF
        return self._goal_distance_map(goal)[self._flat_index(start)]

    def _next_position_towards(self, start: Position, goal: Position) -> Position:
        """Chọn ô kề hợp lệ làm giảm khoảng cách đã cache tới goal."""
        if start == goal:
            return start
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return start

        dist_map = self._goal_distance_map(goal)
        current_dist = dist_map[self._flat_index(start)]
        if current_dist >= INF:
            return start

        best_pos = start
        best_dist = current_dist
        for _, nxt in self._neighbors(start):
            d = dist_map[self._flat_index(nxt)]
            if d < best_dist:
                best_dist = d
                best_pos = nxt
        return best_pos

    def _move_between(self, start: Position, nxt: Position) -> Move:
        if start == nxt:
            return "S"
        return MOVE_BY_DELTA.get((nxt[0] - start[0], nxt[1] - start[1]), "S")

    # ------------------------------------------------------------------
    # Task scoring and assignment
    # ------------------------------------------------------------------
    def _carried_orders(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        """Trả về các order chưa giao đang được shipper mang theo."""
        return [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]

    def _select_delivery_task(self, shipper: Shipper, orders: Dict[int, Order], t: int, horizon_t: int) -> Optional[Task]:
        """Chọn order đang mang tốt nhất để giao, ưu tiên reward, độ gấp và điểm giao gần."""
        best: Optional[Task] = None
        for order in self._carried_orders(shipper, orders):
            goal = (order.ex, order.ey)
            dist = self._distance(shipper.position, goal)
            if dist >= INF:
                continue
            delivery_t = t + max(1, dist) - 1
            reward = self._estimated_delivery_reward(order, delivery_t, horizon_t)
            lateness = max(0, delivery_t - order.et)
            near_delivery_bonus = 5.0 + 1.5 * order.p if dist <= 3 else 0.0
            score = reward + 5.0 * order.p + near_delivery_bonus - 0.02 * dist - 0.25 * lateness
            task = Task(
                kind="deliver",
                goal=goal,
                order_id=order.id,
                priority=order.p,
                deadline=order.et,
                score=score,
            )
            if best is None or self._task_better(task, best):
                best = task
        return best

    def _task_better(self, left: Task, right: Task) -> bool:
        return (
            left.score,
            left.priority,
            -left.deadline,
            -(left.order_id or INF),
        ) > (
            right.score,
            right.priority,
            -right.deadline,
            -(right.order_id or INF),
        )

    def _pickup_candidate_score(
        self,
        shipper: Shipper,
        order: Order,
        t: int,
        horizon_t: int,
    ) -> Tuple[float, int, int]:
        """Chấm điểm một pickup đã lộ theo reward, chi phí di chuyển, deadline và khả năng hoàn thành."""
        pickup = (order.sx, order.sy)
        delivery = (order.ex, order.ey)
        d_pick = self._distance(shipper.position, pickup)
        if d_pick >= INF:
            return -INF, INF, INF
        d_deliver = self._distance(pickup, delivery)
        if d_deliver >= INF:
            return -INF, INF, INF

        delivery_t = t + max(1, d_pick) + d_deliver - 1
        reward = self._estimated_delivery_reward(order, delivery_t, horizon_t)
        total_dist = d_pick + d_deliver
        lateness = max(0, delivery_t - order.et)
        slack = max(0, order.et - delivery_t)
        urgency_bonus = min(12.0, 24.0 / max(slack + 1, 1)) if order.p >= 2 else 0.0
        completion_bonus = COMPLETION_BONUS if delivery_t <= horizon_t else COMPLETION_MISS_PENALTY
        score = (
            reward
            + 4.0 * order.p
            + urgency_bonus
            + completion_bonus
            - 0.032 * total_dist
            - 0.45 * lateness
        )
        return score, d_pick, d_deliver

    def _rough_pickup_candidates(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        active_orders: List[Order],
        reserved_order_ids: Set[int],
        t: int,
        limit: int,
    ) -> List[Order]:
        """Lọc và xếp hạng pickup candidate chi phí thấp trước khi tính khoảng cách chính xác."""
        rough: List[Tuple[float, int, Order]] = []
        for order in active_orders:
            if order.delivered or order.picked:
                continue
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue
            pickup = (order.sx, order.sy)
            delivery = (order.ex, order.ey)
            man_pick = self._manhattan(shipper.position, pickup)
            man_total = man_pick + self._manhattan(pickup, delivery)
            rough_delivery_t = t + max(1, man_pick) + self._manhattan(pickup, delivery) - 1
            rough_reward = self._estimated_delivery_reward(order, rough_delivery_t, max(t + 1, order.et + 1))
            rough_score = rough_reward + 4.0 * order.p - 0.03 * man_total - 0.25 * max(0, rough_delivery_t - order.et)
            rough.append((-rough_score, order.id, order))

        if len(rough) > limit:
            rough = nsmallest(limit, rough)
        else:
            rough.sort()
        return [order for _, _, order in rough]

    def _select_pickup_task(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        active_orders: List[Order],
        reserved_order_ids: Set[int],
        t: int,
        horizon_t: int,
        candidate_limit: int,
    ) -> Optional[Task]:
        """Chọn tác vụ pickup và giữ target cũ còn hợp lệ khi có thể."""
        previous_id = self._pickup_targets.get(shipper.id)
        if previous_id is not None and previous_id not in reserved_order_ids:
            order = orders.get(previous_id)
            if (
                order is not None
                and not order.picked
                and not order.delivered
                and shipper.can_carry(order, orders)
            ):
                score, _, _ = self._pickup_candidate_score(shipper, order, t, horizon_t)
                if score > -INF:
                    return Task(
                        kind="pickup",
                        goal=(order.sx, order.sy),
                        order_id=order.id,
                        priority=order.p,
                        deadline=order.et,
                        score=score - 0.5,
                    )

        rough_limit = max(candidate_limit * 3, candidate_limit)
        candidates = self._rough_pickup_candidates(shipper, orders, active_orders, reserved_order_ids, t, rough_limit)
        best: Optional[Task] = None
        for order in candidates[:candidate_limit]:
            score, _, _ = self._pickup_candidate_score(shipper, order, t, horizon_t)
            if score <= -INF:
                continue
            task = Task(
                kind="pickup",
                goal=(order.sx, order.sy),
                order_id=order.id,
                priority=order.p,
                deadline=order.et,
                score=score,
            )
            if best is None or self._task_better(task, best):
                best = task
        return best

    def _select_opportunistic_pickup(
        self,
        shipper: Shipper,
        delivery_task: Task,
        orders: Dict[int, Order],
        active_orders: List[Order],
        reserved_order_ids: Set[int],
        t: int,
        horizon_t: int,
        candidate_limit: int,
    ) -> Optional[Task]:
        """Thử pickup chen ngang cực chặt khi shipper đang trên đường đi giao."""
        if delivery_task.kind != "deliver":
            return None

        direct_dist = self._distance(shipper.position, delivery_task.goal)
        if direct_dist <= 0 or direct_dist >= INF:
            return None

        direct_delivery_t = t + max(1, direct_dist) - 1
        carried_slack = delivery_task.deadline - direct_delivery_t
        if carried_slack < 3:
            return None

        rough = self._rough_pickup_candidates(
            shipper,
            orders,
            active_orders,
            reserved_order_ids,
            t,
            max(4, min(candidate_limit, 10)),
        )

        best: Optional[Task] = None
        for order in rough:
            if order.p < 2:
                continue
            if order.id in reserved_order_ids or not shipper.can_carry(order, orders):
                continue
            pickup = (order.sx, order.sy)
            d_pick = self._distance(shipper.position, pickup)
            if d_pick >= INF:
                continue
            if d_pick > 1:
                continue
            d_pick_to_delivery = self._distance(pickup, delivery_task.goal)
            if d_pick_to_delivery >= INF:
                continue

            detour = d_pick + d_pick_to_delivery - direct_dist
            if detour < 0:
                detour = 0
            if detour > OPPORTUNISTIC_MAX_DETOUR:
                continue

            delayed_delivery_t = t + max(1, d_pick) + d_pick_to_delivery - 1
            delay = delayed_delivery_t - direct_delivery_t
            if delayed_delivery_t > delivery_task.deadline:
                continue
            if delay > min(2, max(0, carried_slack)):
                continue

            pickup_score, _, d_extra_deliver = self._pickup_candidate_score(shipper, order, t, horizon_t)
            if pickup_score <= -INF:
                continue
            route_alignment = max(0.0, 4.0 - detour)
            score = pickup_score + route_alignment + 1.5 * order.p - 2.0 * detour - 0.5 * max(0, d_extra_deliver - direct_dist)
            task = Task(
                kind="pickup",
                goal=pickup,
                order_id=order.id,
                priority=order.p,
                deadline=order.et,
                score=score,
            )
            if best is None or self._task_better(task, best):
                best = task
        return best

    def _task_is_valid(
        self,
        shipper: Shipper,
        task: Optional[Task],
        orders: Dict[int, Order],
        reserved_order_ids: Set[int],
    ) -> bool:
        """Kiểm tra tác vụ đã cache còn khớp với trạng thái quan sát công khai hiện tại hay không."""
        if task is None:
            return False
        if task.kind == "deliver":
            if task.order_id is None or task.order_id not in shipper.bag:
                return False
            order = orders.get(task.order_id)
            return order is not None and not order.delivered and (order.ex, order.ey) == task.goal
        if task.kind == "pickup":
            if task.order_id is None or task.order_id in reserved_order_ids:
                return False
            order = orders.get(task.order_id)
            return (
                order is not None
                and not order.picked
                and not order.delivered
                and shipper.can_carry(order, orders)
                and (order.sx, order.sy) == task.goal
            )
        return not self._carried_orders(shipper, orders)

    def _active_orders(self, orders: Dict[int, Order]) -> List[Order]:
        return [order for order in orders.values() if not order.delivered]

    def _detect_stuck_shippers(self, obs: dict) -> Set[int]:
        """Phát hiện shipper không tới được ô dự kiến ở tick trước."""
        stuck: Set[int] = set(self._last_conflict_losers)
        for shipper in obs["shippers"]:
            expected = self._last_expected_pos.get(shipper.id)
            previous_move = self._last_move.get(shipper.id, "S")
            if expected is not None and previous_move != "S" and shipper.position != expected:
                stuck.add(shipper.id)
        return stuck

    def _assign_tasks(
        self,
        obs: dict,
        horizon: int,
        candidate_limit: int,
        active_orders: List[Order],
        stuck_shipper_ids: Set[int],
    ) -> Dict[int, Task]:
        """Gán một tác vụ ngắn hạn cho mỗi shipper, chỉ dùng các order đã lộ trong obs."""
        t = int(obs["t"])
        horizon_t = max(int(obs.get("T", t + horizon + 1)), t + horizon + 1)
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        active_order_ids = {order.id for order in active_orders}
        self._pickup_targets = {
            sid: oid for sid, oid in self._pickup_targets.items() if oid in active_order_ids
        }

        tasks: Dict[int, Task] = {}
        reserved_pickups: Set[int] = set()
        replan_interval = 1 if len(shippers) <= 5 else self._replan_interval
        replan_all = not self._tasks or (t - self._last_assignment_t) >= replan_interval

        for shipper in sorted(shippers, key=lambda s: s.id):
            previous_task = self._tasks.get(shipper.id)
            if (
                not replan_all
                and shipper.id not in stuck_shipper_ids
                and self._task_is_valid(shipper, previous_task, orders, reserved_pickups)
            ):
                tasks[shipper.id] = previous_task  # type: ignore[assignment]
                if previous_task is not None and previous_task.kind == "pickup" and previous_task.order_id is not None:
                    reserved_pickups.add(previous_task.order_id)
                    self._pickup_targets[shipper.id] = previous_task.order_id
                continue

            delivery_task = self._select_delivery_task(shipper, orders, t, horizon_t)
            if delivery_task is not None:
                opportunistic_task = None
                if ENABLE_OPPORTUNISTIC and len(shippers) >= 10:
                    opportunistic_task = self._select_opportunistic_pickup(
                        shipper,
                        delivery_task,
                        orders,
                        active_orders,
                        reserved_pickups,
                        t,
                        horizon_t,
                        candidate_limit,
                    )
                if opportunistic_task is not None and opportunistic_task.order_id is not None:
                    reserved_pickups.add(opportunistic_task.order_id)
                    self._pickup_targets[shipper.id] = opportunistic_task.order_id
                    tasks[shipper.id] = opportunistic_task
                    continue
                tasks[shipper.id] = delivery_task
                self._pickup_targets.pop(shipper.id, None)
                continue

            pickup_task = self._select_pickup_task(
                shipper,
                orders,
                active_orders,
                reserved_pickups,
                t,
                horizon_t,
                candidate_limit,
            )
            if pickup_task is not None and pickup_task.order_id is not None:
                reserved_pickups.add(pickup_task.order_id)
                self._pickup_targets[shipper.id] = pickup_task.order_id
                tasks[shipper.id] = pickup_task
                continue

            self._pickup_targets.pop(shipper.id, None)
            tasks[shipper.id] = Task(kind="wait", goal=shipper.position)

        self._tasks = tasks
        if replan_all:
            self._last_assignment_t = t
        return tasks

    # ------------------------------------------------------------------
    # CBS-lite path prefix planning
    # ------------------------------------------------------------------
    def _priority_key(
        self,
        shipper: Shipper,
        task: Task,
        orders: Dict[int, Order],
    ) -> Tuple[int, int, int, int, int]:
        """Khóa ưu tiên cho reservation planning: việc đang mang và gấp được đi trước."""
        carried = self._carried_orders(shipper, orders)
        carried_priority = max((order.p for order in carried), default=0)
        earliest_deadline = min((order.et for order in carried), default=task.deadline)
        carrying = 1 if carried else 0
        dist = self._distance(shipper.position, task.goal)
        if dist >= INF:
            dist = 10**6
        return (
            carried_priority if carried_priority else task.priority,
            -earliest_deadline,
            carrying,
            -dist,
            -shipper.id,
        )

    def _base_prefix(self, start: Position, goal: Position, horizon: int) -> List[Position]:
        """Dựng tiền tố đường đi tham lam theo shortest path từ distance cache."""
        prefix = [start]
        current = start
        for _ in range(horizon):
            if current != goal:
                current = self._next_position_towards(current, goal)
            prefix.append(current)
        while len(prefix) <= horizon:
            prefix.append(prefix[-1])
        return prefix

    def _conflicts_with_reservations(
        self,
        prev: Position,
        nxt: Position,
        step: int,
        reserved_vertices: Dict[int, Set[Position]],
        reserved_edges: Dict[int, Set[Tuple[Position, Position]]],
    ) -> bool:
        """Kiểm tra conflict cùng ô và đổi cạnh với các reservation ưu tiên cao hơn."""
        if nxt in reserved_vertices.get(step, set()):
            return True
        if (nxt, prev) in reserved_edges.get(step, set()):
            return True
        return False

    def _repair_prefix_with_reservations(
        self,
        start: Position,
        goal: Position,
        horizon: int,
        reserved_vertices: Dict[int, Set[Position]],
        reserved_edges: Dict[int, Set[Tuple[Position, Position]]],
    ) -> List[Position]:
        """Sửa tiền tố đường đi bằng cách chờ hoặc chọn ô kề an toàn khi gặp reservation."""
        base = self._base_prefix(start, goal, horizon)
        repaired: List[Position] = [start]
        base_index = 1

        for step in range(1, horizon + 1):
            current = repaired[-1]
            desired = base[base_index] if base_index < len(base) else base[-1]
            if desired == current and current != goal and base_index + 1 < len(base):
                desired = base[base_index + 1]
            if not self._is_adjacent_or_same(current, desired) or not is_valid_cell(desired, self.grid):
                desired = self._next_position_towards(current, goal)
            if not self._is_adjacent_or_same(current, desired):
                desired = current

            if not self._conflicts_with_reservations(current, desired, step, reserved_vertices, reserved_edges):
                repaired.append(desired)
                if desired == base[base_index] and base_index < len(base) - 1:
                    base_index += 1
                continue

            wait_ok = not self._conflicts_with_reservations(current, current, step, reserved_vertices, reserved_edges)
            if wait_ok:
                repaired.append(current)
                continue

            alternatives = sorted(
                (nxt for _, nxt in self._neighbors(current) if nxt != current),
                key=lambda pos: (self._distance(pos, goal), self._manhattan(pos, goal), pos[0], pos[1]),
            )
            chosen = current
            for alt in alternatives:
                if not self._conflicts_with_reservations(current, alt, step, reserved_vertices, reserved_edges):
                    chosen = alt
                    break
            repaired.append(chosen)

        return repaired

    def _plan_prefixes(
        self,
        obs: dict,
        tasks: Dict[int, Task],
        horizon: int,
    ) -> Dict[int, List[Position]]:
        """Lập tiền tố ngắn theo thứ tự ưu tiên và đặt reservation cho các shipper ưu tiên thấp hơn."""
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        shipper_by_id = {shipper.id: shipper for shipper in shippers}
        ordered_ids = sorted(
            shipper_by_id,
            key=lambda sid: self._priority_key(shipper_by_id[sid], tasks[sid], orders),
            reverse=True,
        )

        prefixes: Dict[int, List[Position]] = {}
        reserved_vertices: Dict[int, Set[Position]] = {0: {shipper.position for shipper in shippers}}
        reserved_edges: Dict[int, Set[Tuple[Position, Position]]] = {}

        for sid in ordered_ids:
            shipper = shipper_by_id[sid]
            task = tasks.get(sid, Task(kind="wait", goal=shipper.position))
            prefix = self._repair_prefix_with_reservations(
                shipper.position,
                task.goal,
                horizon,
                reserved_vertices,
                reserved_edges,
            )
            prefixes[sid] = prefix
            for step in range(1, min(horizon, len(prefix) - 1) + 1):
                reserved_vertices.setdefault(step, set()).add(prefix[step])
                reserved_edges.setdefault(step, set()).add((prefix[step - 1], prefix[step]))

        return prefixes

    # ------------------------------------------------------------------
    # Action generation and conflict fallback
    # ------------------------------------------------------------------
    def _cargo_op_for_next_pos(
        self,
        shipper: Shipper,
        task: Task,
        orders: Dict[int, Order],
        next_pos: Position,
    ) -> int:
        """Chọn thao tác pickup, delivery hoặc no-op tại ô sau khi di chuyển."""
        for order in self._carried_orders(shipper, orders):
            if (order.ex, order.ey) == next_pos:
                return 2

        if task.kind == "pickup" and task.order_id is not None and next_pos == task.goal:
            order = orders.get(task.order_id)
            if order is not None and shipper.can_carry(order, orders):
                return 1
        return 0

    def _actions_from_prefixes(
        self,
        obs: dict,
        tasks: Dict[int, Task],
        prefixes: Dict[int, List[Position]],
    ) -> Dict[int, Action]:
        """Chuyển các tiền tố đã lập thành action một bước cho môi trường."""
        orders: Dict[int, Order] = obs["orders"]
        actions: Dict[int, Action] = {}
        for shipper in obs["shippers"]:
            prefix = prefixes.get(shipper.id, [shipper.position, shipper.position])
            next_pos = prefix[1] if len(prefix) > 1 else shipper.position
            next_pos = valid_next_pos(shipper.position, self._move_between(shipper.position, next_pos), self.grid)
            move = self._move_between(shipper.position, next_pos)
            task = tasks.get(shipper.id, Task(kind="wait", goal=shipper.position))
            actions[shipper.id] = (move, self._cargo_op_for_next_pos(shipper, task, orders, next_pos))
        return actions

    def _wait_action(self, shipper: Shipper, task: Task, orders: Dict[int, Order]) -> Action:
        return ("S", self._cargo_op_for_next_pos(shipper, task, orders, shipper.position))

    def _repair_first_step_conflicts(
        self,
        obs: dict,
        tasks: Dict[int, Task],
        actions: Dict[int, Action],
    ) -> Dict[int, Action]:
        """Xử lý conflict cùng ô và đổi cạnh ngay trước khi gọi env.step()."""
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        shipper_by_id = {shipper.id: shipper for shipper in shippers}

        priorities = {
            sid: self._priority_key(shipper_by_id[sid], tasks.get(sid, Task("wait", shipper_by_id[sid].position)), orders)
            for sid in shipper_by_id
        }
        old_positions = {sid: shipper.position for sid, shipper in shipper_by_id.items()}
        old_position_owner = {shipper.position: sid for sid, shipper in shipper_by_id.items()}
        all_losers: Set[int] = set()

        for _ in range(max(1, len(shippers))):
            desired = {
                sid: valid_next_pos(shipper.position, actions.get(sid, ("S", 0))[0], self.grid)
                for sid, shipper in shipper_by_id.items()
            }
            losers: Set[int] = set()

            by_target: Dict[Position, List[int]] = {}
            for sid, pos in desired.items():
                by_target.setdefault(pos, []).append(sid)

            for pos, sids in by_target.items():
                if len(sids) <= 1:
                    continue
                occupant = old_position_owner.get(pos)
                if occupant in sids and desired.get(occupant) == pos:
                    winner = occupant
                else:
                    winner = max(sids, key=lambda item: priorities[item])
                losers.update(sid for sid in sids if sid != winner)

            edge_owner: Dict[Tuple[Position, Position], int] = {}
            for sid in sorted(shipper_by_id):
                old = old_positions[sid]
                nxt = desired[sid]
                if old == nxt:
                    continue
                other_sid = edge_owner.get((nxt, old))
                if other_sid is not None:
                    loser = sid if priorities[sid] < priorities[other_sid] else other_sid
                    losers.add(loser)
                edge_owner[(old, nxt)] = sid

            if not losers:
                break

            changed = False
            for sid in losers:
                all_losers.add(sid)
                shipper = shipper_by_id[sid]
                wait = self._wait_action(shipper, tasks.get(sid, Task("wait", shipper.position)), orders)
                if actions.get(sid) != wait:
                    actions[sid] = wait
                    changed = True
            if not changed:
                break

        self._last_conflict_losers = all_losers
        return actions

    def _record_expected_positions(self, obs: dict, actions: Dict[int, Action]) -> None:
        """Ghi lại ô dự kiến đi tới để tick sau phát hiện shipper bị kẹt."""
        self._last_expected_pos = {}
        self._last_move = {}
        for shipper in obs["shippers"]:
            move = actions.get(shipper.id, ("S", 0))[0]
            self._last_move[shipper.id] = move
            self._last_expected_pos[shipper.id] = valid_next_pos(shipper.position, move, self.grid)

    def _safe_actions(self, obs: dict, tasks: Optional[Dict[int, Task]] = None) -> Dict[int, Action]:
        """Fallback bằng cách chờ an toàn, vẫn giao hoặc pickup nếu đang đứng đúng ô hợp lệ."""
        orders: Dict[int, Order] = obs["orders"]
        actions: Dict[int, Action] = {}
        for shipper in obs["shippers"]:
            task = tasks.get(shipper.id, Task("wait", shipper.position)) if tasks else Task("wait", shipper.position)
            actions[shipper.id] = self._wait_action(shipper, task, orders)
        return actions

    def _adaptive_limits(self, obs: dict) -> Tuple[int, int]:
        """Chọn rolling horizon và số candidate theo kích thước đội shipper."""
        c = max(1, int(obs.get("C", len(obs.get("shippers", [])))))
        if c >= 20:
            return 6, 14
        if c >= 10:
            return 8, 18
        return 10, 28

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        """Một bước quyết định online: gán việc, lập prefix, sửa conflict và xuất action."""
        self.grid = obs["grid"]
        horizon, candidate_limit = self._adaptive_limits(obs)

        try:
            orders: Dict[int, Order] = obs["orders"]
            active_orders = self._active_orders(orders)
            stuck_shipper_ids = self._detect_stuck_shippers(obs)
            tasks = self._assign_tasks(obs, horizon, candidate_limit, active_orders, stuck_shipper_ids)
            prefixes = self._plan_prefixes(obs, tasks, horizon)
            actions = self._actions_from_prefixes(obs, tasks, prefixes)
            actions = self._repair_first_step_conflicts(obs, tasks, actions)
        except Exception:
            actions = self._safe_actions(obs)

        self._record_expected_positions(obs, actions)
        return actions

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Chạy vòng lặp mô phỏng online và trả về kết quả chính thức của môi trường."""
        self._start_time = time.time()
        obs = self.env.reset()
        self.grid = obs["grid"]
        self._goal_distance_cache.clear()
        self._tasks = {}
        self._pickup_targets = {}
        self._last_expected_pos = {}
        self._last_move = {}
        self._last_conflict_losers = set()
        self._last_assignment_t = -INF

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - self._start_time,
        )
