from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from env import (
    ALPHA, BETA, DeliveryEnv, Order, Shipper,
    delivery_reward, is_valid_cell, r_base, valid_next_pos,
)
from solvers.solver import Solver

from ortools.constraint_solver import routing_enums_pb2, pywrapcp

INF = 10 ** 9
MOVES = ("U", "D", "L", "R")
Position = Tuple[int, int]
Move = str

REWARD_SCALE = 100  # scale float rewards → int costs for OR-Tools


class VRPOrToolsSolver(Solver):

    REPLAN_INTERVAL = 3   # re-solve VRP every N steps
    VRP_TIME_LIMIT = 0.5  # seconds per OR-Tools call

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
        dist = 0
        cur: Position = b
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
        cur: Position = b
        while parent[cur][0] != a:
            prev = parent[cur][0]
            if prev is None:
                self._move_cache[key] = "S"
                return "S"
            cur = prev
        self._move_cache[key] = parent[cur][1]
        return parent[cur][1]

    # ================================================================ Reward helpers

    def _max_reward(self, order: Order) -> float:
        """Upper bound on reward: on-time with maximum early bonus."""
        return ALPHA[order.p] * r_base(order.w) * 2.0

    def _late_penalty_rate(self, order: Order) -> int:
        """
        OR-Tools integer cost per time unit delivery exceeds the deadline.
        Reflects the actual per-unit reward gap between on-time and late delivery.
        High-priority orders have a larger gap, so they get a larger penalty.
        """
        gap = (ALPHA[order.p] - BETA[order.p]) * r_base(order.w)
        return max(1, int(gap * REWARD_SCALE))

    def _is_worth_picking(self, order: Order, shipper: Shipper, t: int, T: int) -> bool:
        """False if this order cannot possibly yield a positive reward."""
        dist_p = self._distance(shipper.position, (order.sx, order.sy))
        dist_d = self._distance((order.sx, order.sy), (order.ex, order.ey))
        t_arrive = t + dist_p + dist_d
        return t_arrive < T and delivery_reward(order, t_arrive, T) > 0.0

    def _delivery_slack(self, order: Order, pos: Position, t: int) -> float:
        """
        Remaining time after reaching delivery minus deadline, scaled by priority.
        Negative = already behind. Lower → handle sooner.
        Dividing by ALPHA[p] makes high-priority orders comparatively more urgent.
        """
        dist = self._distance(pos, (order.ex, order.ey))
        return ((order.et - t) - dist) / ALPHA[order.p]

    def _pickup_slack(self, order: Order, pos: Position, t: int) -> float:
        """Like _delivery_slack but also accounts for travel to the pickup point."""
        dist_p = self._distance(pos, (order.sx, order.sy))
        dist_d = self._distance((order.sx, order.sy), (order.ex, order.ey))
        return ((order.et - t) - dist_p - dist_d) / ALPHA[order.p]

    # ------------------------------------------------------------------ VRP

    def _solve_vrp(self, obs: dict) -> Dict[int, List[int]]:
        """
        Capacitated VRP assignment using OR-Tools.

        Model: one node per unassigned order.
          - Arc cost shipper_v  → order_i  = dist(shipper_v_pos, order_i_pickup)
          - Arc cost order_i    → order_j  = dist(order_i_delivery, order_j_pickup)
          - Arc cost order_i    → end_v    = 0   (no return-to-depot cost)
        This naturally encodes that after delivering order i the shipper
        travels to the next order's pickup, without needing PDP constraints.

        Capacity: weight and slot dimensions enforce W_max / K_max.
        Each order is optional via AddDisjunction with a priority-weighted penalty.

        Returns {shipper_id: [order_id, ...]} — assignment per shipper.
        """
        shippers: List[Shipper] = obs["shippers"]
        all_orders: Dict[int, Order] = obs["orders"]

        unassigned: List[Order] = [
            o for o in all_orders.values() if not o.picked and not o.delivered
        ]
        result: Dict[int, List[int]] = {s.id: [] for s in shippers}
        if not unassigned or not shippers:
            return result

        C = len(shippers)
        P = len(unassigned)
        num_nodes = C + P  # shipper nodes + one node per order

        # Positions: shipper current pos; order pickup pos (route SOURCE);
        # order delivery pos (route DESTINATION after visiting order node).
        shipper_pos = [(s.r, s.c) for s in shippers]
        order_pickup = [(o.sx, o.sy) for o in unassigned]
        order_delivery = [(o.ex, o.ey) for o in unassigned]

        def arc_cost(from_node: int, to_node: int) -> int:
            if from_node == to_node:
                return 0
            # Source: shipper start or previous order's delivery point
            src = shipper_pos[from_node] if from_node < C else order_delivery[from_node - C]
            # Destination: next order's pickup point (shipper-end nodes cost 0)
            if to_node < C:
                return 0
            return self._distance(src, order_pickup[to_node - C])

        dist_matrix = [
            [arc_cost(i, j) for j in range(num_nodes)]
            for i in range(num_nodes)
        ]
        finite = [d for row in dist_matrix for d in row if 0 < d < INF]
        big = (max(finite) * 10 + 1) if finite else 10_000
        dist_matrix = [[min(d, big) for d in row] for row in dist_matrix]

        manager = pywrapcp.RoutingIndexManager(
            num_nodes, C, list(range(C)), list(range(C))
        )
        routing = pywrapcp.RoutingModel(manager)

        def transit_cb(fi: int, ti: int) -> int:
            return dist_matrix[manager.IndexToNode(fi)][manager.IndexToNode(ti)]

        transit_id = routing.RegisterTransitCallback(transit_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_id)

        # Weight capacity dimension (cumulative per shipper).
        # This is conservative for sequential execution (delivers weight never
        # decrements) but acts as a useful workload-balancing heuristic: it prevents
        # OR-Tools from over-assigning heavy orders to one shipper.
        def weight_demand(idx: int) -> int:
            node = manager.IndexToNode(idx)
            return int(unassigned[node - C].w * 10) if node >= C else 0

        weight_id = routing.RegisterUnaryTransitCallback(weight_demand)
        weight_caps = [
            max(1, int(
                (s.W_max - sum(all_orders[oid].w for oid in s.bag if oid in all_orders)) * 10
            ))
            for s in shippers
        ]
        routing.AddDimensionWithVehicleCapacity(weight_id, 0, weight_caps, True, "Weight")

        # Slot capacity dimension (number of orders per shipper).
        def slot_demand(idx: int) -> int:
            return 1 if manager.IndexToNode(idx) >= C else 0

        slot_id = routing.RegisterUnaryTransitCallback(slot_demand)
        slot_caps = [max(1, s.K_max - len(s.bag)) for s in shippers]
        routing.AddDimensionWithVehicleCapacity(slot_id, 0, slot_caps, True, "Slots")

        # Each order is optional: not serving it costs max_reward (priority-scaled)
        for i, o in enumerate(unassigned):
            penalty = max(1, int(self._max_reward(o) * REWARD_SCALE))
            routing.AddDisjunction([manager.NodeToIndex(C + i)], penalty)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        params.time_limit.FromMilliseconds(max(200, int(self.VRP_TIME_LIMIT * 1000)))

        solution = routing.SolveWithParameters(params)
        if not solution:
            return result

        for v, s in enumerate(shippers):
            route: List[int] = []
            idx = routing.Start(v)
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node >= C:
                    route.append(unassigned[node - C].id)
                idx = solution.Value(routing.NextVar(idx))
            result[s.id] = route

        return result

    # ------------------------------------------------------------------ Policy

    def _update_assignments(self, obs: dict) -> None:
        """Re-solve VRP and merge new assignments into _pending_pickups.
        Falls back to greedy assignment for any order the VRP did not cover."""
        try:
            vrp_result = self._solve_vrp(obs)
        except Exception:
            vrp_result = {s.id: [] for s in obs["shippers"]}

        for sid, order_ids in vrp_result.items():
            existing = self._pending_pickups.get(sid, [])
            existing_set = set(existing)
            new_orders = [oid for oid in order_ids if oid not in existing_set]
            self._pending_pickups[sid] = existing + new_orders

        # Greedy fallback: assign any order not covered by VRP to the best shipper.
        # This handles VRP capacity drops, infeasible sub-problems, or timeout failures.
        all_orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
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

        # Priority 1: deliver in-bag orders — most urgent (lowest deadline slack) first.
        # Urgency is scaled by ALPHA[p] so high-priority orders are treated as more pressing.
        deliverable = [
            orders[oid] for oid in s.bag if oid in orders and not orders[oid].delivered
        ]
        if deliverable:
            target = min(deliverable, key=lambda o: self._delivery_slack(o, pos, t))
            goal: Position = (target.ex, target.ey)
            move = self._next_move(pos, goal)
            nxt = valid_next_pos(pos, move, self.grid)
            return (move, 2) if nxt == goal else (move, 0)

        # Priority 2: pick up the most urgent assigned order that is still profitable.
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
            o = orders[best_oid]
            goal = (o.sx, o.sy)
            move = self._next_move(pos, goal)
            nxt = valid_next_pos(pos, move, self.grid)
            return (move, 1) if nxt == goal else (move, 0)

        return "S", 0

    # ------------------------------------------------------------------ Run

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()
        last_replan_t = -1

        while not obs.get("done", False):
            t: int = obs["t"]
            T: int = obs["T"]
            new_orders_arrived = bool(obs.get("new_order_ids"))
            all_queues_empty = all(
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
