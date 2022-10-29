import os
import pandas as pd
import matplotlib.pyplot as plt
from hubpersistence.read_data_from_csv import read_graph_structure_from_csv
from hubpersistence.weighted_graph import WeightedGraph

class TimeVaryingHubs(object):
    """Computes hubs for a graph time series and plots the resulting dynamical
    hubs ranking

    Attributes
    ----------
    graphs : list
        List of WeightedGraph instances
    """
    def __init__(self, graphs):
        self.graphs = graphs
        self.compute_filtrations()

    def compute_filtrations(self):
        [self.compute_filtration(graph) for graph in self.graphs]

    def compute_persistence(self, persistence_function = "steady"):
        self.persistence_function = persistence_function
        [graph.steady_persistence() if persistence_function == "steady" else
         graph.ranging_persistence() for graph in self.graphs]

    @staticmethod
    def compute_filtration(graph):
        graph.build_graph()
        graph.build_filtered_subgraphs()
        graph.get_temporary_hubs_along_filtration()
        return graph


if __name__ == "__main__":
    path_to_folder = 'data/literature/GOT/'
    csv_files = ['1', '2', '3', '45']
    graphs = []

    for file_name in csv_files:
        path_to_csv = os.path.join(path_to_folder, file_name + '.csv')
        graph_structure = read_graph_structure_from_csv(path_to_csv)
        graphs.append(WeightedGraph(graph_structure))

    tvg = TimeVaryingHubs(graphs)
    tvg.compute_persistence(persistence_function = "ranging")
