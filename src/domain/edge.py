# -- coding: utf-8 --

from src.domain.node import Node


class Edge:
    """
    Edge in the graph
    """

    def __init__(self, edge_id: int, u: Node, v: Node, weight: float, capacity: float = float('inf')) -> None:
        self.id = edge_id
        # The edge is from Node u to Node v
        self.u = u
        self.u.add_successor(edge_id, v.id)
        self.v = v
        self.v.add_predecessor(edge_id, u.id)

        self.capacity = capacity  # capacity
        self.weight = weight  # weight, may change in the future
        self.initial_weight = weight  # record the initial weight
        self.capacity_used = 0  # capacity that has been used by the current flow
        self.capacity_left = capacity  # capacity not used

    def __str__(self) -> str:
        return "{:.0f}: {:.0f}->{:.0f} ({:.2f}, {:.2f})".format(self.id, self.u.id, self.v.id, self.initial_weight,
                                                                self.capacity)

    def __repr__(self) -> str:
        return "{:.0f}: {:.0f}->{:.0f} ({:.2f}, {:.2f})".format(self.id, self.u.id, self.v.id, self.initial_weight,
                                                                self.capacity)

    # Update capacity by using a given quantity
    def use_capacity(self, capacity_used: float):
        self.capacity_used += capacity_used
        self.capacity_left = self.capacity_left - capacity_used
