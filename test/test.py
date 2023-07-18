# -- coding: utf-8 --
from domain.network import Network

if __name__ == "__main__":
    network = Network()
    network.load_network()
    network.dwDecomposition()
    sensitivityFile = open("./output/sensitivity.txt", 'w')
    sensitivityFile.write("m\tobj\n")
    for m in range(15):
        network = Network()
        network.load_network()
        network.dwDecomposition(m)
        sensitivityFile.write("{:.0f}\t{:.6f}\n".format(m, network.obj_model.getObjective().getValue()))
    sensitivityFile.close()