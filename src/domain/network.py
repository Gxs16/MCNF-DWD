# -- coding: utf-8 --
from typing import Dict, Tuple

import pandas as pd

from domain.demand import Demand
from domain.edge import Edge
from domain.node import Node


class Network:
    """
    The network
    """

    def __init__(self) -> None:
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[int, Edge] = {}
        self.demands: Dict[int, Demand] = {}
        self.edge_dict: Dict[Tuple[int, int], int] = {}  # map the pair of nodes' id to the edge's id
        self.bounded_edges = {}  # 有容量上界的边
        self.solutions = []  # The list of extreme solutions: 极点的列表（对应DW算法中的列）
        self.obj_model = None

    def add_node(self, node_id: int):
        """
        Add a node
        """
        self.nodes[node_id] = Node(node_id)

    def add_edge(self, edge_id: int, uid: int, vid: int, weight: float, capacity: float = float('inf')):
        """
        Add an edge, if the end nodes of the edge do not exit, it will add them
        """
        if uid not in self.nodes:
            self.add_node(uid)
        if vid not in self.nodes:
            self.add_node(vid)
        u = self.nodes[uid]
        v = self.nodes[vid]
        edge = Edge(edge_id, u, v, weight, capacity)
        self.edges[edge_id] = edge
        self.edge_dict[(uid, vid)] = edge_id

    def add_demand(self, demand_id: int, oid: int, did: int, quantity: float = 0):
        """
        Add a demand
        """
        o = self.nodes[oid]
        d = self.nodes[did]
        demand = Demand(demand_id, o, d, quantity)
        self.demands[demand_id] = demand

    def load_network(self, network_file: str = "./network/SiouxFalls_net.csv",
                     demand_file: str = "./network/SiouxFalls_trips.csv"):
        """
        Load network and demand from a file, example file shown in ./network/*
        """
        network = pd.read_csv(network_file, sep='\t')
        networkDf = network.to_dict("index")
        for edgeId, edgeInfo in networkDf.items():
            self.add_edge(edgeId, edgeInfo['init_node'], edgeInfo['term_node'], edgeInfo['length'],
                          edgeInfo['capacity'] * 2)
        demand = pd.read_csv(demand_file, sep='\t')
        demandDf = demand.to_dict("index")
        for demandId, demandInfo in demandDf.items():
            if demandInfo["demand"] > 1e-10:
                self.add_demand(demandId, demandInfo["init_node"], demandInfo["term_node"], demandInfo["demand"])

    def reset_node_label(self):
        """
        Reset the labels and spPred of all nodes (for new executions of shortest path algorithm)
        """
        for node in self.nodes.values():
            node.label = float('inf')
            node.spPred = None

    def reset_edge_capacity(self):
        """
        Reset the capacity of all edges to the initial state
        """
        for edge in self.edges.values():
            edge.capacity_used = 0
            edge.capacity_left = edge.capacity

    def reset_edge_weight(self):
        """
        Reset the weight of all edges to the initial value
        """
        for edge in self.edges.values():
            edge.weight = edge.initial_weight




