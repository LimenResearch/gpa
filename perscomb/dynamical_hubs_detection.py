from __future__ import absolute_import, division, print_function
import os
from sys import platform
if platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")
import pandas as pd
import matplotlib.pyplot as plt
from perscomb.read_data_from_csv import read_graph_structure_from_csv
from perscomb.weighted_graph import WeightedGraph

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
        [graph.steady_hubs_persistence() if persistence_function == "steady" else
         graph.ranging_hubs_persistence() for graph in self.graphs]

    @staticmethod
    def compute_filtration(graph):
        graph.build_graph()
        graph.build_filtered_subgraphs()
        graph.get_temporary_hubs_along_filtration()
        return graph

    def export_csv_for_plot(self, save_as = None):
        info_dicts = [self.get_structure(graph, i) for i, graph in
                      enumerate(self.graphs)]
        #get union of the vertices
        vertices_in_time = set().union(*[d["vertex"] for d in info_dicts])

        for j, d in enumerate(info_dicts):
            V = set(d["vertex"])
            for v in vertices_in_time:
                if v not in V:
                    d["vertex"].append(v)
                    d["observation"].append(j)
                    d["persistence"].append(0)
        info_dict = info_dicts[0]

        for d in info_dicts[1:]:
            for k in info_dict:
                info_dict[k].extend(d[k])

        data_frame = pd.DataFrame(info_dict)
        if save_as is None:
            save_as = './test.csv'

        data_frame.to_csv(save_as, index = False)

    def get_structure(self, graph, index):
        if self.persistence_function == "steady":
            nodes_color_map,\
            nodelist,\
            node_size, cornerpoints = graph.plot_steady_persistence_diagram(return_attr = True)
        elif self.persistence_function == "ranging":
            nodes_color_map,\
            nodelist,\
            node_size, cornerpoints = graph.plot_ranging_persistence_diagram(return_attr = True)
        return graph.export_graph_and_hubs_as_dict(nodes_color_map, nodelist,
                                                   node_size, cornerpoints,
                                                   observation = index)

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
    tvg.export_csv_for_plot(save_as = './got_dynamical_ranging.csv')
