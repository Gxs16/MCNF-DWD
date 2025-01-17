# -- coding: utf-8 --
import logging
import time

from docplex.mp.model import Model

from algorithm.shortest_path import dijkstra
from domain.network import Network
from domain.solution import Solution
from logger_config import LoggerStream

logger = logging.getLogger(__name__)


def solve(network: Network, big_m=1e8, epsilon=1e-6):
    # Find edges with upper bounded capacity: 找到有容量上界的边
    for edgeId, edge in network.edges.items():
        if edge.capacity < float('inf'):
            network.bounded_edges[edgeId] = edge
    # Initialization: 初始化模型
    master_problem = Model()
    # Add variables: 添加变量，包括lambda、松弛变量和人工变量
    slack = master_problem.continuous_var(lb=0, name='s')
    capacity_cts = [master_problem.sum([]) <= network.bounded_edges[k].capacity for k in network.bounded_edges.keys()]
    master_problem.add_constraints(capacity_cts)
    usage_ct = slack == 1
    master_problem.add_constraint(usage_ct)
    obj = big_m * slack
    master_problem.minimize(obj)

    start_time = time.time()
    iter_num = 0
    while True:
        master_problem.solve(log_output=LoggerStream())

        dual_vars = master_problem.dual_values(capacity_cts) + master_problem.dual_values([usage_ct])

        solution = sub_problem(network, dual_vars)

        logger.info(f"Iter: {iter_num} | Master Problem: {master_problem.solution.get_objective_value()} | Sub Problem: {solution.reduced_cost}")

        if solution.reduced_cost < -epsilon and iter_num < 2000:
            network.solutions.append(solution)
            lamb = master_problem.continuous_var(lb=0, name=f'lamb_{len(network.solutions) - 1}')
            for i, k in enumerate(network.bounded_edges.keys()):
                capacity_cts[i].lhs += solution.flow[k] * lamb
            usage_ct.lhs += lamb
            obj += solution.cost * lamb
            master_problem.minimize(obj)
            iter_num += 1
        else:
            break
    end_time = time.time()
    network.obj_model = master_problem
    logger.info("Iteration time: {:.2f}s. Objective: {:.2f}.".format(end_time - start_time, master_problem.objective_value))
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

    # Obtain the lambda
    lams = {}
    for i in network.obj_model.iter_continuous_vars():
        var_name = i.name
        var_value = i.solution_value
        if var_name.startswith("lamb"):
            lam_idx = int(var_name[5:])
            lams[lam_idx] = var_value
        if var_value > 0:
            logger.info(var_name + '\t' + str(var_value))
    # Reset capacity and weight of all edges
    network.reset_edge_capacity()
    network.reset_edge_weight()

    # For each demand, obtain its route(s) and the ratio of flow on the route according to lambda
    for sol_id, ratio in lams.items():
        solution = network.solutions[sol_id]
        routes_dict = solution.routes
        for demand_id, route in routes_dict.items():
            demand = network.demands[demand_id]
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
