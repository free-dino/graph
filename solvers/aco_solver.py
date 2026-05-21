from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from env import (
    ALPHA,
    GAMMA,
    DeliveryEnv,
    Order,
    Shipper,
    delivery_reward,
    is_valid_cell,
    r_base,
    valid_next_pos,
)
from solvers.solver import Solver, default_result

# Type aliases
Move = str
Position = Tuple[int, int]
Action = Tuple[Move, Any]

# ACO parameters (tune if needed)
NUM_ANTS = 10
NUM_ITERATIONS = 30
ALPHA_ACO = 1.0      # pheromone weight
BETA_ACO = 2.0       # heuristic weight
EVAP_RATE = 0.2      # evaporation rate
Q = 100.0            # deposit factor
TAU_MIN = 1e-3
TAU_MAX = 10.0
BASE_REWARD = 100.0  # shift for pheromone deposit when net reward may be negative

# Movement directions (same as env)
MOVES = ("U", "D", "L", "R")


# ---------------------------------------------------------------------------
# BFS All‑Pairs Shortest Path (cached for the whole grid)
# ---------------------------------------------------------------------------
class GridPathCache:
    """Pre‑compute BFS distances and first moves between all free cells."""

    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.free_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if grid[r][c] == 0
        ]
        self.cell_to_id = {cell: i for i, cell in enumerate(self.free_cells)}
        n = len(self.free_cells)
        # distance matrix (inf if unreachable)
        self.dist = [[float("inf")] * n for _ in range(n)]
        # next_move matrix: first direction from cell i to reach cell j
        self.next_move: List[List[Optional[Move]]] = [
            [None] * n for _ in range(n)
        ]
        self._compute_all()

    def _compute_all(self):
        """BFS from every free cell."""
        for idx, start in enumerate(self.free_cells):
            self.dist[idx][idx] = 0
            self.next_move[idx][idx] = "S"
            q = deque()
            q.append(start)
            parent: Dict[Position, Tuple[Optional[Position], Move]] = {
                start: (None, "S")
            }
            visited = {start}
            while q:
                cur = q.popleft()
                for move in MOVES:
                    nxt = valid_next_pos(cur, move, self.grid)
                    if nxt != cur and nxt not in visited and is_valid_cell(nxt, self.grid):
                        visited.add(nxt)
                        parent[nxt] = (cur, move)
                        q.append(nxt)
            # Fill distances and first moves from start to all reachable cells
            for target, _ in parent.items():
                if target == start:
                    continue
                # reconstruct path to get distance and first move
                steps = 0
                node = target
                while node != start:
                    prev, move = parent[node]
                    if prev is None:
                        break
                    if prev == start:
                        first_move = move
                    node = prev
                    steps += 1
                j = self.cell_to_id[target]
                self.dist[idx][j] = steps
                self.next_move[idx][j] = first_move if steps > 0 else "S"

    def distance(self, pos1: Position, pos2: Position) -> float:
        if pos1 == pos2:
            return 0
        i = self.cell_to_id.get(pos1)
        j = self.cell_to_id.get(pos2)
        if i is None or j is None:
            return float("inf")
        return self.dist[i][j]

    def first_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"
        i = self.cell_to_id.get(start)
        j = self.cell_to_id.get(goal)
        if i is None or j is None:
            return "S"
        move = self.next_move[i][j]
        return move if move is not None else "S"


# ---------------------------------------------------------------------------
# Node types for the ACO task graph
# ---------------------------------------------------------------------------
class Node:
    """Abstract node in the ACO construction graph."""
    def __init__(self, idx: int, loc: Position):
        self.idx = idx
        self.loc = loc

    def __repr__(self):
        return f"{self.__class__.__name__}({self.idx}, loc={self.loc})"


class StartNode(Node):
    """Dummy node representing a shipper's starting position."""
    def __init__(self, idx: int, shipper_id: int, loc: Position):
        super().__init__(idx, loc)
        self.shipper_id = shipper_id


class PickupNode(Node):
    """Pickup task for an order."""
    def __init__(self, idx: int, order: Order):
        super().__init__(idx, (order.sx, order.sy))
        self.order = order


class DeliveryNode(Node):
    """Delivery task for an order."""
    def __init__(self, idx: int, order: Order):
        super().__init__(idx, (order.ex, order.ey))
        self.order = order


# ---------------------------------------------------------------------------
# ACO Solver
# ---------------------------------------------------------------------------
class ACOSolver(Solver):
    method_name = "ACO"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.path_cache = GridPathCache(self.env.grid)
        # Parameters
        self.alpha = ALPHA_ACO
        self.beta = BETA_ACO
        self.rho = EVAP_RATE
        self.Q = Q
        self.tau_min = TAU_MIN
        self.tau_max = TAU_MAX
        self.base_reward = BASE_REWARD
        self.num_ants = NUM_ANTS
        self.num_iters = NUM_ITERATIONS

    # ------------------------------------------------------------------
    # Heuristic calculation
    # ------------------------------------------------------------------
    def _heuristic(
        self,
        node: Node,
        shipper_state: dict,
        t_now: int,
        T: int,
    ) -> float:
        """Attractiveness of a task given the shipper's current state."""
        pos = shipper_state["pos"]
        t = shipper_state["time"]
        # Distance from current position to the task's location
        d = self.path_cache.distance(pos, node.loc)
        if d == float("inf"):
            return 1e-12

        if isinstance(node, DeliveryNode):
            order = node.order
            arrival = t + int(d)
            # estimated reward if delivered at arrival
            reward = delivery_reward(order, arrival, T)
            # Ensure positive
            return max(reward, 1e-6) / (d + 1)
        elif isinstance(node, PickupNode):
            order = node.order
            # Potential maximum reward (if delivered on time)
            pot = r_base(order.w) * ALPHA[order.p]
            # Urgency: less slack → more attractive
            slack = max(0, order.et - t - d)
            urgency = 1.0 + max(0.0, 1.0 - slack / max(T, 1))
            return pot * urgency / (d + 1)
        else:
            return 1e-12

    # ------------------------------------------------------------------
    # Solution construction (one ant)
    # ------------------------------------------------------------------
    def _construct_solution(
        self,
        shippers: List[Shipper],
        nodes: List[Node],
        start_nodes: List[StartNode],
        pickups: Dict[int, PickupNode],
        deliveries: Dict[int, DeliveryNode],
        tau: List[List[float]],
        t_now: int,
        T: int,
        rng: random.Random,
    ) -> Tuple[List[int], float]:
        """
        Build a sequence of node indices (including start nodes) for all shippers.
        Returns the sequence and its total reward (fitness). Invalid solutions
        receive reward = -inf.
        """
        C = len(shippers)
        n_nodes = len(nodes)
        seq = []
        total_reward = 0.0

        # Global set of unassigned orders (not yet picked up)
        assigned_orders: Set[int] = set()
        # Copy shipper states (we'll modify them during construction)
        shipper_states = []
        for s in shippers:
            load_w = sum(
                self.env.orders[oid].w for oid in s.bag if oid in self.env.orders
            )
            state = {
                "pos": s.position,
                "time": t_now,
                "bag": list(s.bag),          # list of order ids
                "load_w": load_w,
                "load_k": len(s.bag),
                "W_max": s.W_max,
                "K_max": s.K_max,
            }
            shipper_states.append(state)

        # Process shippers in order 0..C-1
        for s_id in range(C):
            state = shipper_states[s_id]
            # Add start node for this shipper
            start_node = start_nodes[s_id]
            seq.append(start_node.idx)
            current_node_idx = start_node.idx

            # Feasible tasks for this shipper
            feasible = set()
            # Delivery tasks for orders already in bag
            for oid in state["bag"]:
                if oid in deliveries:
                    feasible.add(deliveries[oid].idx)
            # Pickup tasks for orders not yet assigned globally and that fit
            for oid, pnode in pickups.items():
                if oid not in assigned_orders and state["load_w"] + pnode.order.w <= state["W_max"] and state["load_k"] + 1 <= state["K_max"]:
                    feasible.add(pnode.idx)

            # Build the shipper's route until no more feasible tasks or bag empty
            while feasible:
                # Compute probabilities
                probs = []
                total_prob = 0.0
                tau_current = tau[current_node_idx]
                for j in feasible:
                    node_j = nodes[j]
                    eta = self._heuristic(node_j, state, t_now, T)
                    phero = tau_current[j] ** self.alpha
                    prob = phero * (eta ** self.beta)
                    probs.append((j, prob))
                    total_prob += prob
                if total_prob <= 0:
                    # Fall back to uniform
                    probs = [(j, 1.0) for j in feasible]
                    total_prob = len(probs)

                # Roulette wheel selection
                choice = rng.random() * total_prob
                cum = 0.0
                chosen_j = None
                for j, prob in probs:
                    cum += prob
                    if choice <= cum:
                        chosen_j = j
                        break
                if chosen_j is None:
                    chosen_j = probs[-1][0]

                # Execute chosen task
                node = nodes[chosen_j]
                d = self.path_cache.distance(state["pos"], node.loc)
                if d == float("inf"):
                    # Unreachable – abort this ant (invalid)
                    return [], -float("inf")
                d = int(d)

                # Move cost: based on current load
                load_ratio = state["load_w"] / max(state["W_max"], 1.0)
                move_cost_per_step = -0.01 * (1.0 + GAMMA * load_ratio)
                total_reward += d * move_cost_per_step

                # Update position and time
                state["pos"] = node.loc
                state["time"] += d

                if isinstance(node, PickupNode):
                    order = node.order
                    # Pickup
                    state["bag"].append(order.id)
                    state["load_w"] += order.w
                    state["load_k"] += 1
                    assigned_orders.add(order.id)
                    # Now the delivery of this order becomes available for this shipper
                    if order.id in deliveries:
                        # add delivery node to feasible set (but it won't be selected
                        # until we move away and come back? Actually, after the move
                        # we are already at the pickup location. In the real env you can
                        # do pickup and then at the same cell do other things later.
                        # The delivery node is now a future task. We add it now.
                        feasible.add(deliveries[order.id].idx)
                elif isinstance(node, DeliveryNode):
                    order = node.order
                    # Delivery reward
                    reward = delivery_reward(order, state["time"], T)
                    total_reward += reward
                    # Remove from bag
                    try:
                        state["bag"].remove(order.id)
                    except ValueError:
                        pass
                    state["load_w"] -= order.w
                    state["load_k"] -= 1
                    # No longer available
                    feasible.discard(chosen_j)
                else:
                    pass  # should not happen

                # Remove the just executed task from feasible set
                feasible.discard(chosen_j)

                # Update feasible set: after pickup/delivery, some pickups may become
                # infeasible due to capacity.
                # Remove pickups that no longer fit
                to_remove = []
                for j in feasible:
                    node_j = nodes[j]
                    if isinstance(node_j, PickupNode):
                        o = node_j.order
                        if state["load_w"] + o.w > state["W_max"] or state["load_k"] + 1 > state["K_max"]:
                            to_remove.append(j)
                for j in to_remove:
                    feasible.remove(j)

                # Set current node index for next pheromone lookup
                current_node_idx = chosen_j

            # After while, check if all bag items are delivered
            if state["bag"]:
                # This ant could not deliver all mandatory orders → invalid
                return [], -float("inf")

        # The sequence now contains all start nodes and chosen tasks.
        # Any remaining unassigned orders are simply ignored (not delivered).
        return seq, total_reward

    # ------------------------------------------------------------------
    # Pheromone update
    # ------------------------------------------------------------------
    def _update_pheromone(
        self,
        tau: List[List[float]],
        best_seq: List[int],
        best_reward: float,
        worst_reward: float,
    ):
        """Evaporate and deposit on the edges of the best sequence."""
        n = len(tau)
        # Evaporation
        for i in range(n):
            for j in range(n):
                tau[i][j] *= (1.0 - self.rho)
                tau[i][j] = max(tau[i][j], self.tau_min)

        # Deposit on edges of the best sequence
        if not best_seq or best_reward == -float("inf"):
            return
        # Transform reward to positive deposit amount
        deposit = self.Q * (self.base_reward + best_reward) / (self.base_reward + abs(best_reward) + 1e-6)
        for idx in range(len(best_seq) - 1):
            i = best_seq[idx]
            j = best_seq[idx + 1]
            tau[i][j] += deposit
            if tau[i][j] > self.tau_max:
                tau[i][j] = self.tau_max

    # ------------------------------------------------------------------
    # Planning step: run ACO to decide actions
    # ------------------------------------------------------------------
    def plan(self, obs: dict) -> Dict[int, Action]:
        t_now = obs["t"]
        T = self.env.T
        orders_dict: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]

        # Build node sets
        nodes_list: List[Node] = []
        start_nodes: List[StartNode] = []
        pickups: Dict[int, PickupNode] = {}
        deliveries: Dict[int, DeliveryNode] = {}

        # Start nodes for each shipper
        for s in shippers:
            node = StartNode(len(nodes_list), s.id, s.position)
            nodes_list.append(node)
            start_nodes.append(node)

        # Task nodes from current orders
        for oid, order in orders_dict.items():
            if order.picked:
                # It's in some shipper's bag (must be)
                dnode = DeliveryNode(len(nodes_list), order)
                nodes_list.append(dnode)
                deliveries[oid] = dnode
            else:
                pnode = PickupNode(len(nodes_list), order)
                nodes_list.append(pnode)
                pickups[oid] = pnode

        n_nodes = len(nodes_list)
        if n_nodes == 0:
            # No tasks, all shippers stay
            return {s.id: ("S", 0) for s in shippers}

        # Initialize pheromone matrix (symmetric not needed)
        tau0 = 1.0 / max(n_nodes, 1)
        tau = [[tau0] * n_nodes for _ in range(n_nodes)]

        rng = random.Random(self.env.config_name)  # deterministic per config, but can be mixed with step
        best_seq_global = []
        best_reward_global = -float("inf")

        # ACO iterations
        for _ in range(self.num_iters):
            best_seq_iter = []
            best_reward_iter = -float("inf")
            worst_reward_iter = float("inf")

            for _ in range(self.num_ants):
                seq, reward = self._construct_solution(
                    shippers, nodes_list, start_nodes,
                    pickups, deliveries, tau, t_now, T, rng
                )
                if reward > best_reward_iter:
                    best_reward_iter = reward
                    best_seq_iter = seq
                if reward < worst_reward_iter:
                    worst_reward_iter = reward

            if best_reward_iter > best_reward_global:
                best_reward_global = best_reward_iter
                best_seq_global = best_seq_iter

            # Update pheromone using iteration best
            self._update_pheromone(tau, best_seq_iter, best_reward_iter, worst_reward_iter)

        # Decode the best sequence into actions for each shipper
        actions: Dict[int, Action] = {}
        # Find the start of each shipper's segment in the sequence
        # The global sequence is: S0, tasks..., S1, tasks..., etc.
        # We can parse it easily by scanning for start nodes.
        seq = best_seq_global
        if not seq:
            # Fallback: all stay
            for s in shippers:
                actions[s.id] = ("S", 0)
            return actions

        # For each shipper, find its first task after its start node.
        for sid, s in enumerate(shippers):
            start_node = start_nodes[sid]
            # Find index of start node in seq
            try:
                idx = seq.index(start_node.idx)
            except ValueError:
                actions[s.id] = ("S", 0)
                continue
            # The next node (if any) is the first task for this shipper
            if idx + 1 < len(seq):
                next_node_idx = seq[idx + 1]
                # Check if next node is a start node (shipper had no tasks)
                if isinstance(nodes_list[next_node_idx], StartNode):
                    actions[s.id] = ("S", 0)
                else:
                    next_node = nodes_list[next_node_idx]
                    goal = next_node.loc
                    move = self.path_cache.first_move(s.position, goal)
                    next_pos = valid_next_pos(s.position, move, self.env.grid)
                    if isinstance(next_node, DeliveryNode):
                        op = 2 if next_pos == goal else 0
                        actions[s.id] = (move, op)
                    elif isinstance(next_node, PickupNode):
                        op = 1 if next_pos == goal else 0
                        actions[s.id] = (move, op)
                    else:
                        actions[s.id] = (move, 0)
            else:
                actions[s.id] = ("S", 0)

        return actions

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            actions = self.plan(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )