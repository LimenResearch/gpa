#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import pandas as pd
import itertools
import matplotlib.pyplot as plt

def read_csv_distance_matrix(path_to_csv, skip_character = -1, show = False):
    """Matrices are normally used for complete graphs. If it is not the case
    edges without weights should correspond to a specific matrix entry value.
    Our default is -1 since negative weights are not used in our examples.
    """
    d = pd.read_csv(path_to_csv, sep = ",", index_col = 0)
    dd = pd.DataFrame(data = d.values , columns = d.columns, index = d.columns)
    dd = dd.fillna(skip_character)
    if show:
        plot_values(dd)
    graph_structure = []

    for pair in itertools.combinations(dd.columns, r = 2):
        if dd[pair[1]][pair[0]] != skip_character:
            graph_structure.append(pair + tuple([dd[pair[1]][pair[0]]]))

    return graph_structure

def read_graph_structure_from_csv(path_to_csv, sep = ","):
    d = pd.read_csv(path_to_csv, sep = sep)
    graph_structure = []

    for index, row in d.iterrows():
        graph_structure.append(tuple([row[0], row[1], row[2] + 10]))

    return graph_structure

def read_csv_frequencies_matrix(path_to_csv, skip_character = -1, show = False):
    d = pd.read_csv(path_to_csv, sep = ",", index_col = 0)
    dd = pd.DataFrame(data = d.values , columns = d.columns, index = d.columns)
    dd = dd.fillna(skip_character)
    if show:
        plot_values(dd)
    graph_structure = []

    for pair in itertools.combinations(dd.columns, r = 2):
        sum = 0
        if dd[pair[0]][pair[1]] != skip_character:
            sum += dd[pair[0]][pair[1]]
        if dd[pair[1]][pair[0]] != skip_character:
            sum += dd[pair[1]][pair[0]]
        # non avrò mai la somma =0 perchè non sommerò mai +1 e -1 giusto?
        if sum > 0:
            graph_structure.append(pair + tuple([sum]))

    return graph_structure


def plot_values(dataframe):
    fig, ax = plt.subplots(figsize=(18, 18))
    im = ax.imshow(dataframe.values)
    labels = [s.title() for s in dataframe.index]
    ax.set_xticks(range(len(dataframe.index)))
    ax.set_xticklabels(labels, rotation="vertical")
    ax.set_yticks(range(len(dataframe.index)))
    ax.set_yticklabels(labels)
    fig.colorbar(im)
    plt.savefig()
