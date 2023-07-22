# -- coding: utf-8 --
from algorithm.dw_decomposition import solve
from domain.network import Network

if __name__ == "__main__":
    network = Network()
    network.load_network()
    solve(network)
    sensitivityFile = open("./output/sensitivity.txt", 'w')
    sensitivityFile.write("m\tobj\n")
    for m in range(15):
        network = Network()
        network.load_network()
        solve(network, M=m)
        sensitivityFile.write("{:.0f}\t{:.6f}\n".format(m, network.obj_model.solution.get_objective_value()))
    sensitivityFile.close()