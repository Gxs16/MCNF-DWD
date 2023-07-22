# -- coding: utf-8 --
import time

import cplex
from cplex import SparsePair

from algorithm.shortest_path import dijkstra
from domain.demand import Demand
from domain.edge import Edge
from domain.network import Network
from domain.node import Node
from domain.solution import Solution


def solve(network: Network, M=1e6, epsilon=1e-6, output=True, output_folder="./output/"):
    # Find an initial extreme solution by assigning all the flow to the current shortest path: 使用最短路算法找到一个极点（对应一列）
    solution = generate_path_for_demand(network)
    network.solutions.append(solution)
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
    s_index_range = master_problem.variables.add(obj=[0] * num_edges, lb=[0] * num_edges, names=['s'] * num_edges)
    coefficients_s = [1] * num_edges + [0]

    lamb_index_range = master_problem.variables.add(obj=[network.solutions[0].cost], lb=[0], ub=[1], names=['lamb_0'])
    coefficients_lamb = [network.solutions[0].flow[k] for k in network.bounded_edges.keys()]

    capacity_index_range = master_problem.linear_constraints.add(lin_expr=[SparsePair()] * num_edges,
                                                                 senses=['E'] * num_edges,
                                                                 rhs=[network.bounded_edges[k].capacity for k in network.bounded_edges.keys()])
    master_problem.linear_constraints.set_coefficients(list(zip(capacity_index_range, a_index_range, coefficients_a)))
    master_problem.linear_constraints.set_coefficients(list(zip(capacity_index_range, s_index_range, coefficients_s)))
    for lamb_index in lamb_index_range:
        master_problem.linear_constraints.set_coefficients(list(zip(capacity_index_range, [lamb_index]*num_edges, coefficients_lamb)))

    usage_index_range = master_problem.linear_constraints.add(lin_expr=[SparsePair()],
                                                              senses=['E'],
                                                              rhs=[1])
    for lamb_index in lamb_index_range:
        master_problem.linear_constraints.set_coefficients(list(zip(usage_index_range, [lamb_index], [1])))

    # Start! 记录开始时间
    startTime = time.time()
    # Solve the Restricted Master Problem: 求解受限主问题RMP
    master_problem.objective.set_sense(master_problem.objective.sense.minimize)
    master_problem.solve()

    # Get the dual: 得到对偶变量的值
    dual_vars = [d for d in master_problem.solution.get_dual_values(list(capacity_index_range))] + [d for d in master_problem.solution.get_dual_values(list(usage_index_range))]
    # Start the iteration: 迭代开始，迭代次数设置为0
    iterNum = 0
    # Solve the Subproblem according to the dual, and find new column: 根据对偶变量的值求解子问题（SP），得到检验数，同时新的极点会添加到极点列表里
    reducedCost = sub_problem(network, dual_vars)
    # Output: 输出
    iterationFile = open(output_folder + "iterations.txt", 'w')
    iterationFile.write("iter_num\tobj\treduced_cost\n")
    if output:
        print(
            "{:.0f}\t\t\t{:.2f}\t\t\t{:.2f}".format(iterNum, master_problem.solution.get_objective_value(), reducedCost))
    else:
        iterationFile.write(
            "{:.0f}\t{:.2f}\t{:.2f}\n".format(iterNum, master_problem.solution.get_objective_value(), reducedCost))
    while reducedCost > epsilon and iterNum < 2000:
        # Get the latest column: 取最新添加的极点
        solution = network.solutions[-1]
        # Calculate the coefficient of the new column: 计算该极点对应的新列的系数
        lamb_index_range = master_problem.variables.add(obj=[solution.cost], lb=[0], ub=[1], names=[f'lamb_{len(network.solutions)-1}'])
        coefficients = [solution.flow[k] for k in network.bounded_edges.keys()]
        for lamb_index in lamb_index_range:
            master_problem.linear_constraints.set_coefficients(list(zip(capacity_index_range, [lamb_index]*num_edges, coefficients)))
        for lamb_index in lamb_index_range:
            master_problem.linear_constraints.set_coefficients(list(zip(usage_index_range, [lamb_index], [1])))
        # Solve the Restricted Master Problem: 求解受限主问题RMP
        master_problem.solve()
        # Get the dual: 得到对偶变量的值
        dual_vars = [d for d in master_problem.solution.get_dual_values(list(capacity_index_range))] + [d for d in master_problem.solution.get_dual_values(list(usage_index_range))]
        # Solve the Subproblem according to the dual, and find new column: 根据对偶变量的值求解子问题（SP），得到检验数，同时新的极点会添加到极点列表里
        reducedCost = sub_problem(network, dual_vars)
        # Output: 输出
        iterNum += 1
        if output:
            print("{:.0f}\t\t\t{:.2f}\t\t\t{:.2f}".format(iterNum, master_problem.solution.get_objective_value(),
                                                          reducedCost))
        else:
            iterationFile.write(
                "{:.0f}\t{:.2f}\t{:.2f}\n".format(iterNum, master_problem.solution.get_objective_value(), reducedCost))
    # End! 记录结束时间
    endTime = time.time()
    # Save the model: 模型保存和输出
    network.obj_model = master_problem
    iterationFile.close()
    print("Iteration time: {:.2f}s. Objective: {:.2f}.".format(endTime - startTime,
                                                               master_problem.solution.get_objective_value()))
    varFile = open(output_folder + "variables.txt", 'w')
    varFile.write("var_name\tvalue\n")
    num_vars = network.obj_model.variables.get_num()
    for i in range(num_vars):
        varName = network.obj_model.variables.get_names(i)
        varValue = network.obj_model.solution.get_values(i)
        if output:
            print(varName + '\t' + str(varValue))
        else:
            varFile.write("{:}\t{:.6f}".format(varName, varValue) + '\n')
    varFile.close()
    # Process the final solution: 得到各OD的路径和流量信息
    retrieve_sol(network, output, output_folder)


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
    reducedCost = -solution.reduced_cost + dual_vars[i]
    # If the reduced cost is positive, add the solution to the list of columns: 如果检验数为正，就将该极点添加到极点列表中
    if reducedCost > 0:
        network.solutions.append(solution)
    # Return the reduced cost: 返回检验数
    return reducedCost


def generate_path_for_demand(network: Network) -> Solution:
    """
    For each demand, assign all the flow to the current shortest path to generate an extreme solution (i.e., a column in DW formulation)
    """
    # the information of the extreme solution: 储存该极点的信息
    solution = Solution()
    # total travel cost of the flow of the solution: 总成本（通行成本）
    totalCost = 0
    # reduced cost of the column: 检验数
    reducedCost = 0
    # For each demand, assign all the flow to the current shortest path: 对于每一个OD对，计算在调整权重后的路网上的最短路径，并分配流量
    for demandId, demand in network.demands.items():
        # calculate the shortest path: 得到最短路径
        sp, _ = dijkstra(network, demand.o, demand.d)
        # the information of the extreme solution -- route: 极点信息——路径
        solution.routes[demandId] = sp
        # add the total travel cost and the reduced cost according to the path: 根据路径计算总成本和检验数
        lastNode = demand.o
        for node in sp[1:]:
            node: Node
            # find the edge according to the id of end nodes: 根据路径列表中的点找到对应的边
            edgeId = network.edge_dict[(lastNode.id, node.id)]
            edge: Edge = network.edges[edgeId]
            # update the capacity of the edge: 改变边上的流量
            edge.use_capacity(demand.quantity)
            # total travel cost: 总成本
            totalCost += demand.quantity * edge.initial_weight
            # reduced cost: 检验数
            reducedCost += demand.quantity * edge.weight
            lastNode = node
    # the information of the extreme solution: 极点信息
    solution.cost = totalCost
    solution.reduced_cost = reducedCost
    # Calculate flow in each edge: 计算各边上的流量
    for edgeId, edge in network.edges.items():
        solution.flow[edgeId] = edge.capacity_used
    # Return the extreme solution as well as the information: 返回新极点（及其信息）
    return solution


def retrieve_sol(network: Network, output=0, output_file_folder="./output/"):
    """
    Process the final solution of the total model
    """
    # Reset capacity and weight of all edges
    network.reset_edge_capacity()
    network.reset_edge_weight()
    # Obtain the lambda
    lams = {}

    num_vars = network.obj_model.variables.get_num()
    for i in range(num_vars):
        varName = network.obj_model.variables.get_names(i)
        varValue = network.obj_model.solution.get_values(i)
        if len(varName) > 3 and varName[:3] == "lam":
            lamIdx = int(varName[5:])
            lams[lamIdx] = varValue
    # For each demand, obtain its route(s) and the ratio of flow on the route according to lambda
    for solId, ratio in lams.items():
        solution = network.solutions[solId]
        routesDict = solution.routes
        for demandId, route in routesDict.items():
            demand = network.demands[demandId]
            demand.update_route(route, ratio)
            lastNode = demand.o
            for node in route[1:]:
                edgeId = network.edge_dict[(lastNode.id, node.id)]
                edge = network.edges[edgeId]
                edge.use_capacity(demand.quantity * ratio)
                lastNode = node
    # Output
    routeFile = open(output_file_folder + "routes.txt", 'w')
    routeFile.write("id\tratio\troute\n")
    edgeFile = open(output_file_folder + "flow.txt", 'w')
    edgeFile.write("id\tuid\tvid\tflow\tcapacity\n")
    for demand in network.demands.values():
        demand: Demand
        for route, ratio in demand.routes.items():
            if output:
                print("{:.0f}\t{:.6f}\t".format(demand.id, ratio) + route)
            else:
                routeFile.write("{:.0f}\t{:.6f}\t".format(demand.id, ratio) + route + '\n')
    for edge in network.edges.values():
        edge: Edge
        if output:
            print("{:.0f}\t{:.1f}/{:.1f}".format(edge.id, edge.capacity_used, edge.capacity))
        else:
            edgeFile.write(
                "{:.0f}\t{:.0f}\t{:.0f}\t{:.1f}\t{:.1f}\n".format(edge.id, edge.u.id, edge.v.id, edge.capacity_used,
                                                                  edge.capacity))
    routeFile.close()
    edgeFile.close()
