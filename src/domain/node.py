# -- coding: utf-8 --
from typing import Dict, Union


class Node:
    """
    Node in the graph
    """
    def __init__(self, node_id: int) -> None:
        self.id = node_id
        self.successors: Dict[int, int] = {}  # successors in the graph
        self.predecessors: Dict[int, int] = {}  # predecessors in the graph
        self.label = float('inf')  # for Dijkstra algorithm
        self.spPred: Union[Node, None] = None  # also for Dijkstra algorithm

    def __str__(self) -> str:
        return str(self.id)

    def __repr__(self) -> str:
        return str(self.id)

    def __lt__(self, other):
        return self.id < other.id

    def add_successor(self, edge_id: int, node_id: int):
        self.successors[node_id] = edge_id

    def add_predecessor(self, edge_id: int, node_id: int):
        self.predecessors[node_id] = edge_id
