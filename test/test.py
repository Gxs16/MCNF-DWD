# -- coding: utf-8 --
import logging.config

from algorithm.dw_decomposition import solve
from domain.network import Network
from logger_config import log_config_dict

logging.config.dictConfig(log_config_dict)

if __name__ == "__main__":
    network = Network()
    network.load_network()
    solve(network)