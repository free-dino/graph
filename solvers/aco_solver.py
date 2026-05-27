from __future__ import annotations
import hashlib
import random
import time
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9

MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


class ACOSolver(Solver):
    """Sinh viên cài đặt Ant Colony Optimization tại đây."""
    method_name = "ACO"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}
        self._free_cells = [
            (r, c)
            for r, row in enumerate(self.grid)
            for c, val in enumerate(row)
            if val == 0
        ]
        self._pheromone = [
            [0.0 for _ in range(len(self.grid[0]))]
            for _ in range(len(self.grid))
        ]
        for r, c in self._free_cells:
            self._pheromone[r][c] = 1.0

        self._alpha = 1.2
        self._beta = 2.0
        self._evaporation = 0.05
        self._pheromone_min = 0.01
        self._pheromone_max = 5.0
        self._deposit_base = 0.6
        self._rng = random.Random(self._stable_seed(self.env.config_name))
        
    @staticmethod
    def _stable_seed(name: str) -> int:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def _evaporate(self) -> None:
        factor = 1.0 - self._evaporation
        for r, c in self._free_cells:
            self._pheromone[r][c] = max(self._pheromone_min, self._pheromone[r][c] * factor)

    def _deposit(self, pos: Position, order: Order, scale: float = 1.0) -> None:
        r, c = pos
        if not is_valid_cell(pos, self.grid):
            return
        add = self._deposit_base * scale * (1.0 + order.p)
        self._pheromone[r][c] = min(self._pheromone_max, self._pheromone[r][c] + add)

    def _pending_scale(self, order: Order, t: int) -> float:
        slack = max(order.et - t, 0)
        urgency = 1.0 + 1.0 / (1.0 + slack)
        base = 0.07 if order.picked else 0.05
        return base * urgency

    def _update_pheromone(
        self,
        prev_orders: Dict[int, Order],
        new_orders: Dict[int, Order],
        t: int,
    ) -> None:
        self._evaporate()
        delivered = set(prev_orders) - set(new_orders)
        for oid in delivered:
            order = prev_orders[oid]
            self._deposit((order.sx, order.sy), order, scale=0.6)
            self._deposit((order.ex, order.ey), order, scale=1.0)
        appeared = set(new_orders) - set(prev_orders)
        for oid in appeared:
            order = new_orders[oid]
            self._deposit((order.sx, order.sy), order, scale=0.2)
        for order in new_orders.values():
            scale = self._pending_scale(order, t)
            if order.picked:
                self._deposit((order.ex, order.ey), order, scale=scale)
            else:
                self._deposit((order.sx, order.sy), order, scale=scale)

    # ------------------------------------------------------------------
    # BFS utilities
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_parents(
        self,
        start: Position,
        goal: Position,
    ) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None

        queue: Deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}

        while queue:
            current = queue.popleft()
            if current == goal:
                return parent
            for move, nxt in self._neighbors(current):
                if nxt in parent:
                    continue
                parent[nxt] = (current, move)
                queue.append(nxt)
        return None

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0

        key = (start, goal)
        if key in self._distance_cache:
            return self._distance_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._distance_cache[key] = INF
            return INF

        distance = 0
        current = goal
        while current != start:
            previous, _ = parent[current]
            if previous is None:
                self._distance_cache[key] = INF
                return INF
            current = previous
            distance += 1

        self._distance_cache[key] = distance
        return distance

    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"

        key = (start, goal)
        if key in self._next_move_cache:
            return self._next_move_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._next_move_cache[key] = "S"
            return "S"

        current = goal
        while True:
            previous, move = parent[current]
            if previous is None:
                self._next_move_cache[key] = "S"
                return "S"
            if previous == start:
                self._next_move_cache[key] = move
                return move
            current = previous

    # ------------------------------------------------------------------
    # ACO decision utilities
    # ------------------------------------------------------------------
    def _roulette_choice(self, scored: List[Tuple[float, Order]]) -> Optional[Order]:
        if not scored:
            return None
        total = sum(score for score, _ in scored)
        if total <= 0:
            min_score = min(score for score, _ in scored)
            adjusted = [(score - min_score + 1e-6, order) for score, order in scored]
            total = sum(score for score, _ in adjusted)
            threshold = self._rng.random() * total
            acc = 0.0
            for score, order in adjusted:
                acc += score
                if acc >= threshold:
                    return order
            return adjusted[-1][1]
        threshold = self._rng.random() * total
        acc = 0.0
        for score, order in scored:
            acc += score
            if acc >= threshold:
                return order
        return scored[-1][1]

    def _heuristic(self, order: Order, distance: int, slack: int, T: int) -> float:
        urgency = 1.0 + 1.0 / (1.0 + slack)
        priority = 1.0 + order.p
        return (priority * urgency) / (distance + 1.0)

    def _delivery_heuristic(self, order: Order, distance: int, t: int, T: int) -> float:
        slack = max(order.et - t, 0)
        return self._heuristic(order, distance, slack, T)

    def _pickup_heuristic(self, order: Order, total_dist: int, t: int, T: int) -> float:
        slack = max(order.et - t - total_dist, 0)
        return self._heuristic(order, total_dist, slack, T)

    def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order], t: int, T: int) -> Optional[Order]:
        carried_orders = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if not carried_orders:
            return None

        scored: List[Tuple[float, Order]] = []
        for order in carried_orders:
            target = (order.ex, order.ey)
            dist = self._distance(shipper.position, target)
            if dist >= INF:
                continue
            pheromone = self._pheromone[target[0]][target[1]]
            heuristic = self._delivery_heuristic(order, dist, t, T)
            score = (pheromone ** self._alpha) * (heuristic ** self._beta)
            scored.append((score, order))
        if not scored:
            return None
        return self._roulette_choice(scored)

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: Set[int],
        t: int,
        T: int,
    ) -> Optional[Order]:
        scored: List[Tuple[float, Order]] = []
        for order in orders.values():
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue
            pickup = (order.sx, order.sy)
            dist_pick = self._distance(shipper.position, pickup)
            if dist_pick >= INF:
                continue
            dist_drop = self._distance(pickup, (order.ex, order.ey))
            if dist_drop >= INF:
                continue
            total_dist = dist_pick + dist_drop
            pheromone = self._pheromone[pickup[0]][pickup[1]]
            heuristic = self._pickup_heuristic(order, total_dist, t, T)
            score = (pheromone ** self._alpha) * (heuristic ** self._beta)
            scored.append((score, order))
        if not scored:
            return None
        return self._roulette_choice(scored)

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------
    def _move_towards(self, shipper: Shipper, goal: Position) -> Tuple[Move, Position]:
        move = self._next_move(shipper.position, goal)
        next_position = valid_next_pos(shipper.position, move, self.grid)
        return move, next_position

    def _delivery_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.ex, order.ey)
        move, next_position = self._move_towards(shipper, goal)
        return (move, 2) if next_position == goal else (move, 0)

    def _pickup_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.sx, order.sy)
        move, next_position = self._move_towards(shipper, goal)
        return (move, 1) if next_position == goal else (move, 0)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs["t"])
        T = int(obs["T"])

        actions: Dict[int, Action] = {}
        reserved_pickups: Set[int] = set()

        for shipper in sorted(shippers, key=lambda s: s.id):
            delivery_order = self._select_delivery(shipper, orders, t, T)
            if delivery_order is not None:
                actions[shipper.id] = self._delivery_action(shipper, delivery_order)
                continue

            pickup_order = self._select_pickup(shipper, orders, reserved_pickups, t, T)
            if pickup_order is not None:
                reserved_pickups.add(pickup_order.id)
                actions[shipper.id] = self._pickup_action(shipper, pickup_order)
                continue

            actions[shipper.id] = ("S", 0)

        return actions

    def run(self) -> dict:
        # TODO: xây dựng pheromone/heuristic trên đồ thị, mô phỏng và trả về dict kết quả.
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            prev_orders = obs["orders"]
            obs, _, done, _ = self.env.step(actions)
            self._update_pheromone(prev_orders, obs["orders"], int(obs["t"]))
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
