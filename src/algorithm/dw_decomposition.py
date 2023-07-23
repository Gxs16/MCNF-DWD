# -- coding: utf-8 --
import logging
import time

import cplex
from cplex import SparsePair

from algorithm.shortest_path import dijkstra
from domain.network import Network
from domain.node import Node
from domain.solution import Solution

logger = logging.getLogger(__name__)


def solve(network: Network, M=1e6, epsilon=1e-6):
    # Find edges with upper bounded capacity: 找到有容量上界的边
    for edgeId, edge in network.edges.items():
        if edge.capacity < float('inf'):
            network.bounded_edges[edgeId] = edge
    # Initialization: 初始化模型
    master_problem = cplex.Cplex()
    # Add variables: 添加变量，包括lambda、松弛变量和人工变量
    num_edges = len(network.bounded_edges)
    a_index_range = master_problem.variables.add(obj=[M] * num_edges, lb=[0] * num_edges, names=['a'] * num_edges)
    coefficients_a = [-1] * num_edges + [0]

    capacity_index_range = master_problem.linear_constraints.add(lin_expr=[SparsePair()] * num_edges,
                                                                 senses=['L'] * num_edges,
                                                                 rhs=[network.bounded_edges[k].capacity for k in network.bounded_edges.keys()])
    master_problem.linear_constraints.set_coefficients(list(zip(capacity_index_range, a_index_range, coefficients_a)))

    usage_index_range = master_problem.linear_constraints.add(lin_expr=[SparsePair()],
                                                              senses=['E'],
                                                              rhs=[1])

    master_problem.objective.set_sense(master_problem.objective.sense.minimize)

    # Start! 记录开始时间
    start_time = time.time()
    # Solve the Restricted Master Problem: 求解受限主问题RMP

    # Get the dual: 得到对偶变量的值
    dual_vars = [0.0 for _ in capacity_index_range] + [1e9]
    # Start the iteration: 迭代开始，迭代次数设置为0
    iter_num = 0
    # Solve the Sub problem according to the dual, and find new column: 根据对偶变量的值求解子问题（SP），得到检验数，同时新的极点会添加到极点列表里
    solution = sub_problem(network, dual_vars)
    while solution.reduced_cost < -epsilon and iter_num < 2000:
        network.solutions.append(solution)
        # Calculate the coefficient of the new column: 计算该极点对应的新列的系数
        lamb_index_range = master_problem.variables.add(obj=[solution.cost], lb=[0], ub=[1],
                                                        names=[f'lamb_{len(network.solutions) - 1}'])
        coefficients = [solution.flow[k] for k in network.bounded_edges.keys()]
        for lamb_index in lamb_index_range:
            master_problem.linear_constraints.set_coefficients(list(zip(capacity_index_range, [lamb_index] * num_edges, coefficients)))
        for lamb_index in lamb_index_range:
            master_problem.linear_constraints.set_coefficients(list(zip(usage_index_range, [lamb_index], [1])))
        # Solve the Restricted Master Problem: 求解受限主问题RMP
        master_problem.solve()
        # Get the dual: 得到对偶变量的值
        dual_vars = [d for d in master_problem.solution.get_dual_values(list(capacity_index_range))] + [d for d in master_problem.solution.get_dual_values(list(usage_index_range))]
        # Solve the Sub problem according to the dual, and find new column: 根据对偶变量的值求解子问题（SP），得到检验数，同时新的极点会添加到极点列表里
        solution = sub_problem(network, dual_vars)
        # Output: 输出
        iter_num += 1
        logger.info(f"Iter: {iter_num} | Master Problem: {master_problem.solution.get_objective_value()} | Sub Problem: {solution.reduced_cost}")
    # End! 记录结束时间
    end_time = time.time()
    # Save the model: 模型保存和输出
    network.obj_model = master_problem
    logger.info("Iteration time: {:.2f}s. Objective: {:.2f}.".format(end_time - start_time, master_problem.solution.get_objective_value()))
    # Process the final solution: 得到各OD的路径和流量信息
    display_sol(network)


def sub_problem(network: Network, dual_vars: list):
    """
    Solve the sub problem
    """
    # Set the weight of all edges according to the dual: 调整路网上各边的权重
    i = 0
    for edgeId, edge in network.bounded_edges.items():
        edge.weight = edge.initial_weight - dual_vars[i]
        i += 1
    # Reset the capacity of all edges: 重置图上各边的流量
    network.reset_edge_capacity()
    # Obtain a new extreme solution as well as its information: 得到一个新极点（对应一列）
    solution = generate_path_for_demand(network)
    # Calculate the reduced cost (add dual associated with sum(lambda)=1): 计算检验数
    solution.reduced_cost -= dual_vars[i]
    return solution


def generate_path_for_demand(network: Network) -> Solution:
    """
    For each demand, assign all the flow to the current shortest path to generate an extreme solution (i.e., a column in DW formulation)
    """
    # the information of the extreme solution: 储存该极点的信息
    solution = Solution()
    # For each demand, assign all the flow to the current shortest path: 对于每一个OD对，计算在调整权重后的路网上的最短路径，并分配流量
    for demandId, demand in network.demands.items():
        # calculate the shortest path: 得到最短路径
        sp, _ = dijkstra(network, demand.o, demand.d)
        # the information of the extreme solution -- route: 极点信息——路径
        solution.routes[demandId] = sp
        # add the total travel cost and the reduced cost according to the path: 根据路径计算总成本和检验数
        last_node = demand.o
        for node in sp[1:]:
            # find the edge according to the id of end nodes: 根据路径列表中的点找到对应的边
            edge_id = network.edge_dict[(last_node.id, node.id)]
            edge = network.edges[edge_id]
            # update the capacity of the edge: 改变边上的流量
            edge.use_capacity(demand.quantity)
            # total travel cost: 总成本
            solution.cost += demand.quantity * edge.initial_weight
            # reduced cost: 检验数
            solution.reduced_cost += demand.quantity * edge.weight
            last_node = node
    # Calculate flow in each edge: 计算各边上的流量
    for edge_id, edge in network.edges.items():
        solution.flow[edge_id] = edge.capacity_used
    # Return the extreme solution as well as the information: 返回新极点（及其信息）
    return solution


def display_sol(network: Network):
    """
    Process the final solution of the total model
    """
    num_vars = network.obj_model.variables.get_num()
    for i in range(num_vars):
        var_name = network.obj_model.variables.get_names(i)
        var_value = network.obj_model.solution.get_values(i)
        if var_value > 0:
            logger.info(var_name + '\t' + str(var_value))
    # Reset capacity and weight of all edges
    network.reset_edge_capacity()
    network.reset_edge_weight()
    # Obtain the lambda
    lams = {}

    num_vars = network.obj_model.variables.get_num()
    for i in range(num_vars):
        var_name = network.obj_model.variables.get_names(i)
        var_value = network.obj_model.solution.get_values(i)
        if len(var_name) > 3 and var_name[:3] == "lam":
            lam_idx = int(var_name[5:])
            lams[lam_idx] = var_value
    # For each demand, obtain its route(s) and the ratio of flow on the route according to lambda
    for solId, ratio in lams.items():
        solution = network.solutions[solId]
        routes_dict = solution.routes
        for demandId, route in routes_dict.items():
            demand = network.demands[demandId]
            demand.update_route(route, ratio)
            last_node = demand.o
            for node in route[1:]:
                edge_id = network.edge_dict[(last_node.id, node.id)]
                edge = network.edges[edge_id]
                edge.use_capacity(demand.quantity * ratio)
                last_node = node
    # Output
    for demand in network.demands.values():
        for route, ratio in demand.routes.items():
            if ratio > 0:
                logger.info("{:.0f}\t{:.6f}\t".format(demand.id, ratio) + route)
    for edge in network.edges.values():
        logger.info("{:.0f}\t{:.1f}/{:.1f}".format(edge.id, edge.capacity_used, edge.capacity))
