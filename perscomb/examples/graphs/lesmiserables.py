from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../../')
from sys import platform
if platform == "darwin":
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pprint import pprint
from perscomb.read_data_from_csv import read_graph_structure_from_csv
from perscomb.weighted_graph import WeightedGraph

def export_json_3d_graph():
    path_to_csv = '../../data/literature/lesmiserables.csv'
    graph_structure = read_graph_structure_from_csv(path_to_csv)
    graph = WeightedGraph(graph_structure)
    graph.build_graph()
    graph.build_filtered_subgraphs()
    graph.get_temporary_hubs_along_filtration()

    graph.steady_hubs_persistence()
    graph.ranging_hubs_persistence()
    graph.plot_steady_persistence_diagram(export_json = True)
    graph.plot_ranging_persistence_diagram(export_json = True)
    plt.show()

def reproduce_thesis_figure():
    above_max_diagonal_gap = True
    path_to_csv = '../../data/literature/lesmiserables.csv'
    graph_structure = read_graph_structure_from_csv(path_to_csv)
    graph = WeightedGraph(graph_structure)
    graph.build_graph()
    graph.build_filtered_subgraphs()
    graph.get_temporary_hubs_along_filtration()
    # Steady
    graph.steady_hubs_persistence(above_max_diagonal_gap = above_max_diagonal_gap , gap_number = 1)
    graph.plot_steady_persistence_diagram()
    fig, ax = plt.subplots()
    graph.steady_pd.plot_gudhi(ax,
            persistence_to_plot = graph.steady_pd.persistence_to_plot)
    if hasattr(graph.steady_pd, 'proper_cornerpoints_above_gap'):
        graph.steady_pd.plot_nth_widest_gap(ax_handle =ax, n = graph.gap_number)
        graph.steady_pd.mark_points_above_diagonal_gaps(ax)
    plt.savefig('gap2_steady'+ '.pdf', dpi=300)
    print ('steady hubs above gap:',  graph.steady_pd.proper_cornerpoints_above_gap)
    pprint ([(c.birth, c.death, c.vertex) for c in graph.steady_pd.proper_cornerpoints_above_gap])
    # Ranging
    graph.ranging_hubs_persistence(above_max_diagonal_gap = above_max_diagonal_gap , gap_number = 1)
    graph.plot_ranging_persistence_diagram()
    fig, ax = plt.subplots()
    graph.ranging_pd.plot_gudhi(ax,
            persistence_to_plot = graph.ranging_pd.persistence_to_plot)
    if hasattr(graph.ranging_pd, 'proper_cornerpoints_above_gap'):
        graph.ranging_pd.plot_nth_widest_gap(ax_handle =ax, n = graph.gap_number)
        graph.ranging_pd.mark_points_above_diagonal_gaps(ax)
    plt.savefig('gap2_ranging'+ '.pdf', dpi=300)
    print ('ranging hubs above gap:',  graph.ranging_pd.proper_cornerpoints_above_gap)
    pprint ([(c.birth, c.death, c.vertex) for c in graph.ranging_pd.proper_cornerpoints_above_gap])

if __name__ == "__main__":
    reproduce_thesis_figure()
