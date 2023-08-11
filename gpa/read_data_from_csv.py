import pandas as pd
import itertools

def read_csv_distance_matrix(path_to_csv, skip_character = -1):
    """Matrices are normally used for complete graphs. If it is not the case
    edges without weights should correspond to a specific matrix entry value.
    Our default is -1 since negative weights are not used in our examples.
    """
    d = pd.read_csv(path_to_csv, sep = ";")
    dd = pd.DataFrame(data = d.values , columns = d.columns, index = d.columns)
    dd = dd.fillna(skip_character)

    graph_structure = []
    for pair in itertools.combinations(dd.columns, r = 2):
        if dd[pair[0]][pair[1]] != skip_character:
            graph_structure.append(pair + tuple([dd[pair[0]][pair[1]]]))

    return graph_structure


def read_graph_as_df(path_to_csv, sep=",", **kwargs):
    return pd.read_csv(path_to_csv, sep = sep, **kwargs)


def read_graph_structure_from_csv(path_to_csv, sep = ";"):
    d = read_graph_as_df(path_to_csv, sep = sep)

    graph_structure = []
    for _, row in d.iterrows():
        graph_structure.append(tuple([row[0], row[1], row[2]]))

    return graph_structure