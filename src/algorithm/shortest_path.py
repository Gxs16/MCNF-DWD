# -- coding: utf-8 --
import heapq

from domain.network import Network
from domain.node import Node


def dijkstra(network: Network, u: Node, v: Node = None):
    """
    Dijkstra algorithm, return the list of Nodes from u to v and the shortest distance
    """
    network.reset_node_label()
    network.nodes[u.id].label = 0
    network.nodes[u.id].shortest_path_predecessor = None
    min_heap = [(u.label, u)]
    while min_heap:
        current_node = heapq.heappop(min_heap)[1]
        current_label = current_node.label
        if v is not None:
            if v.id == current_node.id:
                path = [v]
                sp_predecessor = v.shortest_path_predecessor
                while sp_predecessor is not None:
                    path.append(sp_predecessor)
                    sp_predecessor = sp_predecessor.shortest_path_predecessor
                path.reverse()
                return path, v.label
        for nodeId, edgeId in current_node.successors.items():
            success_node = network.nodes[nodeId]
            success_edge = network.edges[edgeId]
            if current_label + success_edge.weight < success_node.label:
                success_node.label = current_label + success_edge.weight
                success_node.shortest_path_predecessor = current_node
                heapq.heappush(min_heap, (success_node.label, success_node))
