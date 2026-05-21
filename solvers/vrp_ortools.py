from __future__ import annotations
from env import DeliveryEnv, Order
from solvers.solver import Solver, default_result


class VRPOrToolsSolver(Solver):
    """Sinh viên cài đặt VRP + OR-Tools tại đây."""

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)

    def run(self) -> dict:
        # TODO: mô hình hóa các đơn đã quan sát thành bài toán VRP động và trả về dict kết quả.
        return default_result("VRP-OrTools", self.env.config_name, self.env.G, self.orders)
