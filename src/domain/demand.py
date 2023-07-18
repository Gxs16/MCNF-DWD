# -- coding: utf-8 --
from typing import List, Dict

from src.domain.node import Node


def hash_route(route: List[Node]) -> str:
    """
    Hash the route to a string
    """
    s = ""
    for node in route:
        s += str(node.id) + "->"
    return s


class Demand:
    """
    Demand in the graph
    """
    def __init__(self, demand_id: int, o: Node, d: Node, quantity: float) -> None:
        self.id = demand_id
        self.o = o
        self.d = d
        self.quantity = quantity  # the quantity of the flow
        self.routes: Dict[str, float] = {}  # store the resulting routes of the MCNF problem

    def update_route(self, route: list, ratio: float):
        """
        Update the dict of the routes, where route is a list of Nodes and ratio is the ratio of flow on the route
        """
        self.routes[hash_route(route)] = self.routes.setdefault(hash_route(route), 0) + ratio
