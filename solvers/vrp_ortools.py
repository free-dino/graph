from __future__ import annotations

import random
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from env import ALPHA, BETA, DeliveryEnv, Order, Shipper, r_base
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
# (target_r, target_c, op_type ∈ {"P","D"}, order_id)
PlanStep = Tuple[int, int, str, int]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

REWARD_SCALE = 100
WEIGHT_SCALE = 100

Index = int
Node = int
TransitCallback = Callable[[Index, Index], int]
UnaryCallback = Callable[[Index], int]


# VRP routing engine
class RoutingIndexManager:
    def __init__(
        self,
        num_nodes: int,
        num_vehicles: int,
        starts: Sequence[int],
        ends: Sequence[int],
    ) -> None:
        self.num_nodes = num_nodes
        self.num_vehicles = num_vehicles
        self.starts = list(starts)
        self.ends = list(ends)

    def IndexToNode(self, index: Index) -> Node:
        return index

    def NodeToIndex(self, node: Node) -> Index:
        return node


class RoutingSearchParameters:
    def __init__(self) -> None:
        self.time_limit_ms: int = 1000
        self.first_solution_strategy: str = "parallel_cheapest_insertion"
        self.local_search_metaheuristic: str = "guided_local_search"


def default_routing_search_parameters() -> RoutingSearchParameters:
    return RoutingSearchParameters()


class DimensionSpec:
    def __init__(
        self,
        name: str,
        transit_cb: Optional[TransitCallback] = None,
        unary_cb: Optional[UnaryCallback] = None,
        slack_max: int = 0,
        capacity: int = 0,
        vehicle_capacity: Optional[List[int]] = None,
        fix_start_cumul_to_zero: bool = False,
    ) -> None:
        self.name = name
        self.transit_cb = transit_cb
        self.unary_cb = unary_cb
        self.slack_max = slack_max
        self.capacity = capacity
        self.vehicle_capacity = vehicle_capacity
        self.fix_start_cumul_to_zero = fix_start_cumul_to_zero
        self.start_cumul: Dict[int, int] = {}
        self.soft_upper_bounds: List[Tuple[Index, int, int]] = []


class PickupDeliveryPair:
    def __init__(self, pickup_index: Index, delivery_index: Index) -> None:
        self.pickup_index = pickup_index
        self.delivery_index = delivery_index


class DisjunctionSpec:
    def __init__(self, indices: Tuple[Index, ...], penalty: int) -> None:
        self.indices = indices
        self.penalty = penalty


class RoutingSolution:
    def __init__(self, next_map: Dict[Index, Index]) -> None:
        self._next = next_map

    def Value(self, index: Index) -> Index:
        return self._next[index]


class RoutingModel:
    PARALLEL_CHEAPEST_INSERTION = "parallel_cheapest_insertion"
    GUIDED_LOCAL_SEARCH = "guided_local_search"

    def __init__(self, manager: RoutingIndexManager) -> None:
        self._manager = manager
        self._transit_callbacks: List[TransitCallback] = []
        self._unary_callbacks: List[UnaryCallback] = []
        self._arc_cost_cb: Optional[TransitCallback] = None
        self._dimensions: Dict[str, DimensionSpec] = {}
        self._pickup_delivery: List[PickupDeliveryPair] = []
        self._disjunctions: List[DisjunctionSpec] = []
        self._fixed_vehicle: Dict[Index, List[int]] = {}
        self._rng = random.Random(42)

    def RegisterTransitCallback(self, cb: TransitCallback) -> int:
        idx = len(self._transit_callbacks)
        self._transit_callbacks.append(cb)
        return idx

    def RegisterUnaryTransitCallback(self, cb: UnaryCallback) -> int:
        idx = len(self._unary_callbacks)
        self._unary_callbacks.append(cb)
        return idx

    def SetArcCostEvaluatorOfAllVehicles(self, callback_index: int) -> None:
        self._arc_cost_cb = self._transit_callbacks[callback_index]

    def AddDimension(
        self,
        callback_index: int,
        slack_max: int,
        capacity: int,
        fix_start_cumul_to_zero: bool,
        name: str,
    ) -> None:
        self._dimensions[name] = DimensionSpec(
            name=name,
            transit_cb=self._transit_callbacks[callback_index],
            slack_max=slack_max,
            capacity=capacity,
            fix_start_cumul_to_zero=fix_start_cumul_to_zero,
        )

    def AddDimensionWithVehicleCapacity(
        self,
        callback_index: int,
        slack_max: int,
        vehicle_capacity: List[int],
        fix_start_cumul_to_zero: bool,
        name: str,
    ) -> None:
        self._dimensions[name] = DimensionSpec(
            name=name,
            unary_cb=self._unary_callbacks[callback_index],
            slack_max=slack_max,
            vehicle_capacity=list(vehicle_capacity),
            fix_start_cumul_to_zero=fix_start_cumul_to_zero,
        )

    def GetDimensionOrDie(self, name: str) -> "RoutingDimension":
        return RoutingDimension(self, name)

    def AddPickupAndDelivery(self, pickup_index: Index, delivery_index: Index) -> None:
        self._pickup_delivery.append(
            PickupDeliveryPair(pickup_index, delivery_index)
        )

    def AddDisjunction(self, indices: Sequence[Index], penalty: int) -> None:
        self._disjunctions.append(DisjunctionSpec(tuple(indices), penalty))

    def VehicleVar(self, index: Index) -> "VehicleVarRef":
        return VehicleVarRef(self, index)

    def Start(self, vehicle: int) -> Index:
        return self._manager.starts[vehicle]

    def End(self, vehicle: int) -> Index:
        return self._manager.ends[vehicle]

    def IsEnd(self, index: Index) -> bool:
        return index in self._manager.ends

    def NextVar(self, index: Index) -> Index:
        return index

    def SolveWithParameters(
        self, params: RoutingSearchParameters
    ) -> Optional[RoutingSolution]:
        return _VRPSolver(self).solve(params)


class VehicleVarRef:
    def __init__(self, model: RoutingModel, index: Index) -> None:
        self._model = model
        self._index = index

    def SetValues(self, vehicles: Sequence[int]) -> None:
        self._model._fixed_vehicle[self._index] = list(vehicles)


class RoutingDimension:
    def __init__(self, model: RoutingModel, name: str) -> None:
        self._model = model
        self._name = name

    def _spec(self) -> DimensionSpec:
        return self._model._dimensions[self._name]

    def CumulVar(self, index: Index) -> "CumulVarRef":
        return CumulVarRef(self._model, self._name, index)

    def SetCumulVarSoftUpperBound(
        self, index: Index, upper_bound: int, coefficient: int
    ) -> None:
        self._spec().soft_upper_bounds.append((index, upper_bound, coefficient))


class CumulVarRef:
    def __init__(self, model: RoutingModel, dim_name: str, index: Index) -> None:
        self._model = model
        self._dim_name = dim_name
        self._index = index

    def SetValue(self, value: int) -> None:
        self._model._dimensions[self._dim_name].start_cumul[self._index] = value


class _RouteState:
    def __init__(self, routes: List[List[Node]], skipped: Set[Node]) -> None:
        self.routes = routes
        self.skipped = skipped


class _VRPSolver:
    def __init__(self, model: RoutingModel) -> None:
        self._m = model
        self._mgr = model._manager
        self._v = self._mgr.num_vehicles
        self._arc = model._arc_cost_cb
        assert self._arc is not None

    def solve(self, params: RoutingSearchParameters) -> Optional[RoutingSolution]:
        state = self._initial_state()
        if state is None:
            return None

        deadline = time.perf_counter() + params.time_limit_ms / 1000.0

        if params.first_solution_strategy == RoutingModel.PARALLEL_CHEAPEST_INSERTION:
            pci = self._parallel_cheapest_insertion(state, deadline)
            if pci is not None:
                state = pci

        best_state = _RouteState(
            routes=[list(r) for r in state.routes],
            skipped=set(state.skipped),
        )
        best_cost = self._total_cost(best_state)

        if params.local_search_metaheuristic == RoutingModel.GUIDED_LOCAL_SEARCH:
            gls_state, gls_cost = self._guided_local_search(
                best_state, best_cost, deadline
            )
            if gls_cost < best_cost:
                best_state, best_cost = gls_state, gls_cost

        if not self._is_feasible(best_state):
            return None
        return self._build_solution(best_state)

    def _initial_state(self) -> Optional[_RouteState]:
        routes: List[List[Node]] = [[] for _ in range(self._v)]
        skipped: Set[Node] = set()
        for node, vehicles in self._m._fixed_vehicle.items():
            if len(vehicles) != 1:
                continue
            routes[vehicles[0]].append(node)
        for v in range(self._v):
            if not self._route_feasible(v, routes[v], skipped):
                return None
        return _RouteState(routes=routes, skipped=skipped)

    def _optional_nodes(self) -> Set[Node]:
        opt: Set[Node] = set()
        for d in self._m._disjunctions:
            opt.update(d.indices)
        return opt

    def _skip_penalty(self, node: Node) -> int:
        for d in self._m._disjunctions:
            if node in d.indices:
                return d.penalty
        return 0

    def _is_mandatory(self, node: Node) -> bool:
        if node in self._m._fixed_vehicle:
            return True
        return node not in self._optional_nodes()

    def _parallel_cheapest_insertion(
        self, state: _RouteState, deadline: float
    ) -> Optional[_RouteState]:
        pairs = list(self._m._pickup_delivery)
        if not pairs:
            return state

        base_cost = self._total_cost(state)
        for pair in pairs:
            if time.perf_counter() >= deadline:
                break
            p, d = pair.pickup_index, pair.delivery_index
            if p in state.skipped or self._node_in_routes(state, p):
                continue

            best_delta = self._skip_penalty(p) + self._skip_penalty(d)
            best_skip = True
            best_v = -1
            best_pos_p = -1
            best_pos_d = -1

            for v in range(self._v):
                if self._m._fixed_vehicle.get(p) and v not in self._m._fixed_vehicle[p]:
                    continue
                if self._m._fixed_vehicle.get(d) and v not in self._m._fixed_vehicle[d]:
                    continue
                route = state.routes[v]
                n = len(route)
                for pos_p in range(n + 1):
                    for pos_d in range(pos_p + 1, n + 2):
                        new_route = (
                            route[:pos_p] + [p] + route[pos_p:pos_d] + [d] + route[pos_d:]
                        )
                        if not self._route_feasible(v, new_route, state.skipped):
                            continue
                        trial_routes = [list(r) for r in state.routes]
                        trial_routes[v] = new_route
                        trial = _RouteState(
                            routes=trial_routes, skipped=set(state.skipped)
                        )
                        if not self._is_feasible(trial):
                            continue
                        delta = self._total_cost(trial) - base_cost
                        if delta < best_delta:
                            best_delta = delta
                            best_skip = False
                            best_v = v
                            best_pos_p = pos_p
                            best_pos_d = pos_d

            if best_skip:
                state.skipped.add(p)
                state.skipped.add(d)
            else:
                route = state.routes[best_v]
                state.routes[best_v] = (
                    route[:best_pos_p]
                    + [p]
                    + route[best_pos_p:best_pos_d]
                    + [d]
                    + route[best_pos_d:]
                )
            base_cost += best_delta

        return state

    def _guided_local_search(
        self, state: _RouteState, init_cost: int, deadline: float
    ) -> Tuple[_RouteState, int]:
        best = _RouteState(
            routes=[list(r) for r in state.routes], skipped=set(state.skipped)
        )
        best_cost = init_cost
        current = _RouteState(
            routes=[list(r) for r in state.routes], skipped=set(state.skipped)
        )

        guide: Dict[Tuple[Node, Node], int] = {}
        stagnation = 0
        max_iters = 200

        iters = 0
        while time.perf_counter() < deadline and iters < max_iters:
            iters += 1
            improved = False
            for _ in range(16):
                neighbor = self._random_neighbor(current)
                if neighbor is None or not self._is_feasible(neighbor):
                    continue
                if self._guided_cost(neighbor, guide) < self._guided_cost(current, guide):
                    current = neighbor
                    improved = True
                    real = self._total_cost(current)
                    if real < best_cost:
                        best = _RouteState(
                            routes=[list(r) for r in current.routes],
                            skipped=set(current.skipped),
                        )
                        best_cost = real
                        stagnation = 0

            if not improved:
                stagnation += 1
                if stagnation >= 40:
                    self._update_penalties(guide, current)
                    stagnation = 0

        return best, best_cost

    def _update_penalties(
        self, guide: Dict[Tuple[Node, Node], int], state: _RouteState
    ) -> None:
        for v in range(self._v):
            path = self._full_path(v, state.routes[v])
            for i in range(len(path) - 1):
                key = (path[i], path[i + 1])
                guide[key] = guide.get(key, 0) + 1

    def _guided_cost(
        self, state: _RouteState, guide: Dict[Tuple[Node, Node], int]
    ) -> int:
        base = self._total_cost(state)
        extra = 0
        for v in range(self._v):
            path = self._full_path(v, state.routes[v])
            for i in range(len(path) - 1):
                extra += guide.get((path[i], path[i + 1]), 0) * 4
        return base + extra

    def _random_neighbor(self, state: _RouteState) -> Optional[_RouteState]:
        op = self._m._rng.randint(0, 5)
        if op == 0:
            return self._relocate_node(state)
        if op == 1:
            return self._swap_nodes(state)
        if op == 2:
            return self._two_opt(state)
        if op == 3:
            return self._toggle_skip_pair(state)
        if op == 4:
            return self._relocate_pair(state)
        return self._relocate_pair(state)

    def _relocate_node(self, state: _RouteState) -> Optional[_RouteState]:
        for pair in self._m._pickup_delivery:
            if self._node_in_routes(state, pair.pickup_index):
                return self._relocate_pair(state)
        return None

    def _relocate_pair(self, state: _RouteState) -> Optional[_RouteState]:
        if not self._m._pickup_delivery:
            return None
        pair = self._m._rng.choice(self._m._pickup_delivery)
        p, d = pair.pickup_index, pair.delivery_index
        if p in state.skipped:
            return None
        trial = _RouteState(
            routes=[list(r) for r in state.routes], skipped=set(state.skipped)
        )
        for route in trial.routes:
            if p in route:
                route.remove(p)
            if d in route:
                route.remove(d)
        v_to = self._m._rng.randrange(self._v)
        route = trial.routes[v_to]
        pos_p = self._m._rng.randrange(len(route) + 1)
        route.insert(pos_p, p)
        pos_d = self._m._rng.randrange(pos_p + 1, len(route) + 1)
        route.insert(pos_d, d)
        return trial

    def _swap_nodes(self, state: _RouteState) -> Optional[_RouteState]:
        trial = _RouteState(
            routes=[list(r) for r in state.routes], skipped=set(state.skipped)
        )
        movable: List[Tuple[int, int]] = []
        for v, route in enumerate(trial.routes):
            for i, n in enumerate(route):
                if not self._is_mandatory(n):
                    movable.append((v, i))
        if len(movable) < 2:
            return None
        (v1, i1), (v2, i2) = self._m._rng.sample(movable, 2)
        trial.routes[v1][i1], trial.routes[v2][i2] = (
            trial.routes[v2][i2],
            trial.routes[v1][i1],
        )
        return trial

    def _two_opt(self, state: _RouteState) -> Optional[_RouteState]:
        v = self._m._rng.randrange(self._v)
        route = state.routes[v]
        if len(route) < 2:
            return None
        i, j = sorted(self._m._rng.sample(range(len(route)), 2))
        trial = _RouteState(
            routes=[list(r) for r in state.routes], skipped=set(state.skipped)
        )
        trial.routes[v] = route[:i] + list(reversed(route[i : j + 1])) + route[j + 1 :]
        return trial

    def _toggle_skip_pair(self, state: _RouteState) -> Optional[_RouteState]:
        if not self._m._pickup_delivery:
            return None
        pair = self._m._rng.choice(self._m._pickup_delivery)
        p, d = pair.pickup_index, pair.delivery_index
        trial = _RouteState(
            routes=[list(r) for r in state.routes], skipped=set(state.skipped)
        )
        if p in state.skipped:
            trial.skipped.discard(p)
            trial.skipped.discard(d)
            v = self._m._rng.randrange(self._v)
            trial.routes[v].extend([p, d])
        else:
            for route in trial.routes:
                if p in route:
                    route.remove(p)
                if d in route:
                    route.remove(d)
            trial.skipped.add(p)
            trial.skipped.add(d)
        return trial

    def _node_in_routes(self, state: _RouteState, node: Node) -> bool:
        return any(node in r for r in state.routes)

    def _full_path(self, vehicle: int, intermediates: List[Node]) -> List[Node]:
        return [self._mgr.starts[vehicle]] + intermediates + [self._mgr.ends[vehicle]]

    def _dimension_transit(self, spec: DimensionSpec, a: Node, b: Node) -> int:
        if spec.transit_cb is None:
            return 0
        return spec.transit_cb(a, b)

    def _dimension_unary(self, spec: DimensionSpec, node: Node) -> int:
        if spec.unary_cb is None:
            return 0
        return spec.unary_cb(node)

    def _evaluate_route(
        self, vehicle: int, intermediates: List[Node]
    ) -> Optional[Dict[str, List[int]]]:
        path = self._full_path(vehicle, intermediates)
        dim_cumuls: Dict[str, List[int]] = {}
        for name, spec in self._m._dimensions.items():
            start_idx = self._mgr.starts[vehicle]
            start_val = spec.start_cumul.get(start_idx, 0)
            cumuls = [start_val]
            cur = start_val
            for i in range(1, len(path)):
                cur += self._dimension_transit(spec, path[i - 1], path[i])
                cur += self._dimension_unary(spec, path[i])
                cumuls.append(cur)
            cap = (
                spec.vehicle_capacity[vehicle]
                if spec.vehicle_capacity is not None
                else spec.capacity
            )
            if cap > 0:
                for c in cumuls:
                    if c < 0 or c > cap:
                        return None
            dim_cumuls[name] = cumuls
        return dim_cumuls

    def _route_feasible(
        self, vehicle: int, intermediates: List[Node], skipped: Set[Node]
    ) -> bool:
        for n in intermediates:
            if n in self._m._fixed_vehicle:
                if vehicle not in self._m._fixed_vehicle[n]:
                    return False
        return self._evaluate_route(vehicle, intermediates) is not None

    def _is_feasible(self, state: _RouteState) -> bool:
        visited: Dict[Node, int] = {}
        for v, route in enumerate(state.routes):
            if not self._route_feasible(v, route, state.skipped):
                return False
            for n in route:
                if n in visited:
                    return False
                visited[n] = v
        for n in state.skipped:
            if self._is_mandatory(n):
                return False
        for pair in self._m._pickup_delivery:
            p, d = pair.pickup_index, pair.delivery_index
            p_vis = p in visited
            d_vis = d in visited
            if p in state.skipped or d in state.skipped:
                if p_vis or d_vis:
                    return False
                continue
            if p_vis != d_vis:
                return False
            if p_vis:
                if visited[p] != visited[d]:
                    return False
                rv = visited[p]
                r = state.routes[rv]
                if r.index(p) >= r.index(d):
                    return False
                if "Time" in self._m._dimensions:
                    ev = self._evaluate_route(rv, r)
                    if ev is None:
                        return False
                    path = self._full_path(rv, r)
                    if ev["Time"][path.index(p)] > ev["Time"][path.index(d)]:
                        return False
        for n, vehicles in self._m._fixed_vehicle.items():
            if n in visited and visited[n] not in vehicles:
                return False
            if n not in visited and n not in state.skipped:
                return False
        return True

    def _soft_penalty(self, state: _RouteState) -> int:
        penalty = 0
        for v, route in enumerate(state.routes):
            path = self._full_path(v, route)
            for name, spec in self._m._dimensions.items():
                ev = self._evaluate_route(v, route)
                if ev is None:
                    continue
                cumuls = ev[name]
                for idx, ub, coeff in spec.soft_upper_bounds:
                    if idx not in path:
                        continue
                    penalty += coeff * max(0, cumuls[path.index(idx)] - ub)
        return penalty

    def _skip_penalties_total(self, state: _RouteState) -> int:
        total = 0
        counted: Set[Node] = set()
        for node in state.skipped:
            if node in counted:
                continue
            total += self._skip_penalty(node)
            counted.add(node)
        return total

    def _arc_cost_total(self, state: _RouteState) -> int:
        total = 0
        for v, route in enumerate(state.routes):
            path = self._full_path(v, route)
            for i in range(len(path) - 1):
                total += self._arc(path[i], path[i + 1])
        return total

    def _total_cost(self, state: _RouteState) -> int:
        return (
            self._arc_cost_total(state)
            + self._soft_penalty(state)
            + self._skip_penalties_total(state)
        )

    def _build_solution(self, state: _RouteState) -> RoutingSolution:
        next_map: Dict[Index, Index] = {}
        for v in range(self._v):
            path = self._full_path(v, state.routes[v])
            for i in range(len(path) - 1):
                next_map[path[i]] = path[i + 1]
        return RoutingSolution(next_map)


# Delivery solver

class VRPOrToolsSolver(Solver):

    method_name = "VRPOrTools"

    REPLAN_PERIOD = 40
    NEW_ORDER_REPLAN_COOLDOWN = 6
    STUCK_LIMIT = 3
    MAX_UNPICKED_FOR_SOLVE = 80

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

        self._dist: Dict[Position, Dict[Position, int]] = {}
        self._step: Dict[Position, Dict[Position, Move]] = {}
        self._precompute_shortest_paths()

        self.plans: Dict[int, List[PlanStep]] = {i: [] for i in range(self.C)}
        self._last_replan_t: int = -(10**9)
        self._last_position: Dict[int, Position] = {}
        self._stuck_counter: Dict[int, int] = {i: 0 for i in range(self.C)}

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

    @staticmethod
    def _bag_weight(shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    @staticmethod
    def _potential_reward(order: Order) -> float:
        return ALPHA[order.p] * r_base(order.w) * 2.0

    @staticmethod
    def _late_floor_reward(order: Order) -> float:
        return BETA[order.p] * r_base(order.w)

    def _greedy_replan(
        self,
        obs: dict,
        unpicked: List[Order],
        bag_orders: Dict[int, List[Order]],
    ) -> None:
        shippers: List[Shipper] = obs["shippers"]
        orders: Dict[int, Order] = obs["orders"]
        reserved: set = set()

        for s in shippers:
            self.plans[s.id] = []
            in_bag = sorted(bag_orders.get(s.id, []), key=lambda o: (o.et, -o.p, o.id))
            for o in in_bag:
                self.plans[s.id].append((o.ex, o.ey, "D", o.id))

        if not unpicked:
            return

        for s in shippers:
            current_pos = s.position
            current_bag_count = len(s.bag)
            current_bag_weight = self._bag_weight(s, orders)

            for _ in range(s.K_max - current_bag_count):
                best: Optional[Order] = None
                best_score = -1.0
                for o in unpicked:
                    if o.id in reserved:
                        continue
                    if current_bag_count >= s.K_max:
                        break
                    if current_bag_weight + o.w > s.W_max:
                        continue
                    d_pickup = self._distance(current_pos, (o.sx, o.sy))
                    if d_pickup >= INF:
                        continue
                    d_total = d_pickup + self._distance((o.sx, o.sy), (o.ex, o.ey))
                    if d_total >= INF:
                        continue
                    score = self._potential_reward(o) / (d_total + 1.0)
                    if score > best_score:
                        best_score = score
                        best = o
                if best is None:
                    break
                reserved.add(best.id)
                self.plans[s.id].append((best.sx, best.sy, "P", best.id))
                self.plans[s.id].append((best.ex, best.ey, "D", best.id))
                current_pos = (best.ex, best.ey)
                current_bag_count += 1
                current_bag_weight += best.w

    def _replan(self, obs: dict) -> None:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t_now: int = int(obs.get("t", 0))
        C = len(shippers)

        unpicked: List[Order] = []
        bag_orders: Dict[int, List[Order]] = {s.id: [] for s in shippers}
        for o in orders.values():
            if o.delivered:
                continue
            if o.picked and 0 <= o.carrier < C:
                bag_orders[o.carrier].append(o)
            elif not o.picked:
                unpicked.append(o)

        has_bag = any(bag_orders[s.id] for s in shippers)
        if not unpicked and not has_bag:
            for s in shippers:
                self.plans[s.id] = []
            return

        unpicked.sort(key=lambda o: (-o.p, o.et, o.id))
        if len(unpicked) > self.MAX_UNPICKED_FOR_SOLVE:
            unpicked = unpicked[: self.MAX_UNPICKED_FOR_SOLVE]

        locations: List[Position] = []
        for s in shippers:
            locations.append(s.position)
        for s in shippers:
            locations.append(s.position)

        unpicked_start = 2 * C
        for o in unpicked:
            locations.append((o.sx, o.sy))
            locations.append((o.ex, o.ey))

        bag_start = len(locations)
        bag_entries: List[Tuple[int, Order]] = []
        for sid in sorted(bag_orders.keys()):
            for o in sorted(bag_orders[sid], key=lambda x: (x.et, -x.p, x.id)):
                bag_entries.append((sid, o))
                locations.append((o.ex, o.ey))

        n_nodes = len(locations)
        starts = list(range(C))
        ends = list(range(C, 2 * C))

        manager = RoutingIndexManager(n_nodes, C, starts, ends)
        routing = RoutingModel(manager)

        def transit_cb(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            if C <= to_node < 2 * C:
                return 0
            return self._distance(locations[from_node], locations[to_node])

        transit_idx = routing.RegisterTransitCallback(transit_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        horizon = max(self.T + 50, t_now + 200)

        def time_cb(from_index: int, to_index: int) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            if C <= to_node < 2 * C:
                return 0
            d = self._distance(locations[from_node], locations[to_node])
            return d + (1 if to_node >= 2 * C else 0)

        time_cb_idx = routing.RegisterTransitCallback(time_cb)
        routing.AddDimension(time_cb_idx, 0, horizon, False, "Time")
        time_dim = routing.GetDimensionOrDie("Time")
        for v in range(C):
            time_dim.CumulVar(routing.Start(v)).SetValue(t_now)

        def _deadline_coeff(o: Order) -> int:
            base = ALPHA[o.p] * r_base(o.w)
            return max(1, int(base * REWARD_SCALE / max(self.T, 1)) + 2)

        for i, o in enumerate(unpicked):
            d_node = unpicked_start + 2 * i + 1
            d_idx = manager.NodeToIndex(d_node)
            ub = max(t_now, min(o.et, horizon - 1))
            time_dim.SetCumulVarSoftUpperBound(d_idx, ub, _deadline_coeff(o))

        for offset, (_sid, o) in enumerate(bag_entries):
            d_node = bag_start + offset
            d_idx = manager.NodeToIndex(d_node)
            ub = max(t_now, min(o.et, horizon - 1))
            time_dim.SetCumulVarSoftUpperBound(d_idx, ub, _deadline_coeff(o))

        def count_cb(from_index: int) -> int:
            n = manager.IndexToNode(from_index)
            if n < 2 * C:
                return 0
            if n < bag_start:
                rel = n - unpicked_start
                return 1 if rel % 2 == 0 else -1
            return -1

        count_cb_idx = routing.RegisterUnaryTransitCallback(count_cb)
        routing.AddDimensionWithVehicleCapacity(
            count_cb_idx,
            0,
            [int(s.K_max) for s in shippers],
            False,
            "Count",
        )
        count_dim = routing.GetDimensionOrDie("Count")
        for v, s in enumerate(shippers):
            count_dim.CumulVar(routing.Start(v)).SetValue(len(s.bag))

        def weight_cb(from_index: int) -> int:
            n = manager.IndexToNode(from_index)
            if n < 2 * C:
                return 0
            if n < bag_start:
                rel = n - unpicked_start
                o = unpicked[rel // 2]
                w = int(round(o.w * WEIGHT_SCALE))
                return w if rel % 2 == 0 else -w
            offset = n - bag_start
            _, o = bag_entries[offset]
            return -int(round(o.w * WEIGHT_SCALE))

        weight_cb_idx = routing.RegisterUnaryTransitCallback(weight_cb)
        routing.AddDimensionWithVehicleCapacity(
            weight_cb_idx,
            0,
            [int(round(s.W_max * WEIGHT_SCALE)) for s in shippers],
            False,
            "Weight",
        )
        weight_dim = routing.GetDimensionOrDie("Weight")
        for v, s in enumerate(shippers):
            w_bag_int = int(round(self._bag_weight(s, orders) * WEIGHT_SCALE))
            weight_dim.CumulVar(routing.Start(v)).SetValue(w_bag_int)

        for i, o in enumerate(unpicked):
            p_node = unpicked_start + 2 * i
            d_node = p_node + 1
            p_idx = manager.NodeToIndex(p_node)
            d_idx = manager.NodeToIndex(d_node)
            routing.AddPickupAndDelivery(p_idx, d_idx)
            potential = int(round(self._potential_reward(o) * REWARD_SCALE))
            routing.AddDisjunction([p_idx], max(potential, 1))
            routing.AddDisjunction([d_idx], max(potential, 1))

        for offset, (sid, o) in enumerate(bag_entries):
            d_node = bag_start + offset
            d_idx = manager.NodeToIndex(d_node)
            routing.VehicleVar(d_idx).SetValues([sid])
            potential = int(round(self._potential_reward(o) * REWARD_SCALE)) * 10
            routing.AddDisjunction([d_idx], max(potential, 1))

        params = default_routing_search_parameters()
        params.first_solution_strategy = RoutingModel.PARALLEL_CHEAPEST_INSERTION
        params.local_search_metaheuristic = RoutingModel.GUIDED_LOCAL_SEARCH
        n_orders = len(unpicked) + len(bag_entries)
        if n_orders <= 4:
            time_limit_ms = 150
        elif n_orders <= 10:
            time_limit_ms = 400
        elif n_orders <= 20:
            time_limit_ms = 800
        elif n_orders <= 40:
            time_limit_ms = 1300
        elif n_orders <= 70:
            time_limit_ms = 1800
        else:
            time_limit_ms = 2300
        params.time_limit_ms = time_limit_ms

        solution = routing.SolveWithParameters(params)
        if solution is None:
            self._greedy_replan(obs, unpicked, bag_orders)
            return

        for v in range(C):
            new_plan: List[PlanStep] = []
            idx = routing.Start(v)
            while not routing.IsEnd(idx):
                idx = solution.Value(routing.NextVar(idx))
                node = manager.IndexToNode(idx)
                if C <= node < 2 * C:
                    break
                if node < bag_start:
                    rel = node - unpicked_start
                    o = unpicked[rel // 2]
                    if rel % 2 == 0:
                        new_plan.append((o.sx, o.sy, "P", o.id))
                    else:
                        new_plan.append((o.ex, o.ey, "D", o.id))
                else:
                    offset = node - bag_start
                    _, o = bag_entries[offset]
                    new_plan.append((o.ex, o.ey, "D", o.id))
            self.plans[v] = new_plan

    def _advance_plans(self, obs: dict) -> bool:
        orders: Dict[int, Order] = obs["orders"]
        invalid = False

        for s in obs["shippers"]:
            plan = self.plans[s.id]
            while plan:
                _tr, _tc, op_t, oid = plan[0]
                o = orders.get(oid)
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
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        has_unpicked = any(
            (not o.delivered) and (not o.picked) for o in orders.values()
        )
        has_bag = any(s.bag for s in shippers)
        if not has_unpicked and not has_bag:
            return False

        if invalid_after_advance:
            return True

        for s in shippers:
            if not self.plans[s.id]:
                if s.bag:
                    return True
                if has_unpicked:
                    return True

        if obs.get("new_order_ids"):
            if t_now - self._last_replan_t >= self.NEW_ORDER_REPLAN_COOLDOWN:
                return True

        for s in shippers:
            if self._stuck_counter.get(s.id, 0) >= self.STUCK_LIMIT and self.plans[s.id]:
                return True

        if t_now - self._last_replan_t >= self.REPLAN_PERIOD:
            return True

        return False

    def _action_for(self, shipper: Shipper) -> Tuple[Move, int]:
        plan = self.plans[shipper.id]
        if not plan:
            return ("S", 2 if shipper.bag else 0)

        target_r, target_c, op_t, _oid = plan[0]
        target: Position = (target_r, target_c)

        if shipper.position == target:
            return ("S", 1 if op_t == "P" else 2)

        move = self._next_move(shipper.position, target)
        if op_t == "D":
            return (move, 2)

        return (move, 0)

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
