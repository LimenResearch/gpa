from __future__ import absolute_import, division, print_function
from sys import platform
import sys
sys.path.append('../../')
if platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pprint import pprint
from hubpersistence.read_data_from_csv import read_csv_distance_matrix
from hubpersistence.weighted_graph import WeightedGraph


def opp(array):
    return -array

def max_(array):
    return(max(array) - array)

def reproduce_thesis_figure():
    above_max_diagonal_gap = True
    path_to_csv = '../../data/languages/SSWL5_norm.csv'
    graph_structure = read_csv_distance_matrix(path_to_csv)
    graph = WeightedGraph(graph_structure)
    graph.build_graph()
    graph.build_filtered_subgraphs(weight_transform = max_ , sublevel= True)
    graph.get_temporary_hubs_along_filtration()
    # Steady
    graph.steady_hubs_persistence(above_max_diagonal_gap = above_max_diagonal_gap,
                                   gap_number = 2)
    graph.plot_steady_persistence_diagram(show = True)
    fig, ax = plt.subplots()
    graph.steady_pd.plot_gudhi(ax,
            persistence_to_plot = graph.steady_pd.persistence_to_plot)
    if hasattr(graph.steady_pd, 'proper_cornerpoints_above_gap'):
        graph.steady_pd.plot_nth_widest_gap(ax_handle =ax, n = graph.gap_number)
        graph.steady_pd.mark_points_above_diagonal_gaps(ax)
    plt.savefig('steady_ling_gap2' + '.pdf', dpi= 300)
    print('Steady hubs:', [c.vertex for c in graph.steady_cornerpoints])
    print ('steady hubs above gap:',  graph.steady_pd.proper_cornerpoints_above_gap)
    pprint ([(c.birth, c.death, c.vertex) for c in graph.steady_pd.proper_cornerpoints_above_gap])
    # Ranging
    graph.ranging_hubs_persistence(above_max_diagonal_gap = above_max_diagonal_gap,
                                   gap_number = 2)
    graph.plot_ranging_persistence_diagram(show = True)
    fig, ax = plt.subplots()
    graph.ranging_pd.plot_gudhi(ax,
            persistence_to_plot = graph.ranging_pd.persistence_to_plot)
    if hasattr(graph.ranging_pd, 'proper_cornerpoints_above_gap'):
        graph.ranging_pd.plot_nth_widest_gap(ax_handle =ax, n = graph.gap_number)
        graph.ranging_pd.mark_points_above_diagonal_gaps(ax)
    plt.savefig('ranging_ling_gap2' + '.pdf', dpi= 300)
    print('Ranging hubs:', [c.vertex for c in graph.ranging_cornerpoints])
    print ('ranging hubs above gap:', graph.ranging_pd.proper_cornerpoints_above_gap)
    pprint ([(c.birth, c.death, c.vertex) for c in graph.ranging_pd.proper_cornerpoints_above_gap])

if __name__ == "__main__":
    reproduce_thesis_figure()
