from typing import Dict, List

from domain.node import Node


class Solution:
    def __init__(self):
        self.routes: Dict[int, List[Node]] = {}
        # total travel cost: 总成本
        self.cost: float = 0
        # reduced cost: 检验数（只有这一次迭代有用）
        self.reduced_cost: float = 0
        # flow in each edge: 各边上的流量
        self.flow: Dict[int, float] = {}
