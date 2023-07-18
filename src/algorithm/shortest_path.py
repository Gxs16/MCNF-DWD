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
    network.nodes[u.id].spPred = None
    minHeap = [(u.label, u)]
    while minHeap:
        currentNode = heapq.heappop(minHeap)[1]
        currentLabel = currentNode.label
        if v is not None:
            if v.id == currentNode.id:
                path = [v]
                spPred = v.spPred
                while spPred is not None:
                    path.append(spPred)
                    spPred = spPred.spPred
                path.reverse()
                return path, v.label
        for nodeId, edgeId in currentNode.successors.items():
            success_node = network.nodes[nodeId]
            success_edge = network.edges[edgeId]
            if currentLabel + success_edge.weight < success_node.label:
                success_node.label = currentLabel + success_edge.weight
                success_node.spPred = currentNode
                heapq.heappush(minHeap, (success_node.label, success_node))
