import os
from itertools import chain, combinations
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import cm
from collections import Counter


def grouped(iterable, n):
    """Groups iterable by considering n consecutive elements
    """
    return zip(*[iter(iterable)]*n)


def get_powerset_(l):
    """Returns the powerset built on the elements of l, excluding the empty set
    """
    if not isinstance(l, list):
        l = list(l)
    return chain.from_iterable(combinations(l, r) for r in range(2,len(l)+1))

def get_supersets(l):
    values = set(map(lambda x:len(x), l))
    sets_dict_by_cardinality = {x:[y for y in l if len(y)==x] for x in values}
    supersets = []

    for i in values:
        superset = [key for key in sets_dict_by_cardinality[i]
                    if not any(
                        [val.intersection(key)==key
                         for j in values if j !=i
                         for val in sets_dict_by_cardinality[j]])]
        if len(superset) > 0:
            supersets.extend(superset)

    return supersets

"""Local Gudhi plot
"""
def __min_birth_max_death(persistence, band_boot=0.):
    """This function returns (min_birth, max_death) from the persistence.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param band_boot: bootstrap band
    :type band_boot: float.
    :returns: (float, float) -- (min_birth, max_death).
    """
    # Look for minimum birth date and maximum death date for plot optimisation
    max_death = 0
    min_birth = persistence[0][1][0]
    for interval in reversed(persistence):
        if float(interval[1][1]) != float('inf'):
            if float(interval[1][1]) > max_death:
                max_death = float(interval[1][1])
        if float(interval[1][0]) > max_death:
            max_death = float(interval[1][0])
        if float(interval[1][0]) < min_birth:
            min_birth = float(interval[1][0])
    if band_boot > 0.:
        max_death += band_boot
    return (min_birth, max_death)

"""
Only 13 colors for the palette
"""
palette = ['#ff0000', '#00ff00', '#0000ff', '#00ffff', '#ff00ff', '#ffff00',
           '#000000', '#880000', '#008800', '#000088', '#888800', '#880088',
           '#008888']

def show_palette_values(alpha=0.6):
    """This function shows palette color values in function of the dimension.

    :param alpha: alpha value in [0.0, 1.0] for horizontal bars (default is 0.6).
    :type alpha: float.
    :returns: plot the dimension palette values.
    """
    colors = []
    for color in palette:
        colors.append(color)

    y_pos = np.arange(len(palette))

    plt.barh(y_pos, y_pos + 1, align='center', alpha=alpha, color=colors)
    plt.ylabel('Dimension')
    plt.title('Dimension palette values')
    return plt

def plot_persistence_diagram(persistence=[], persistence_file='', alpha=0.6,
                             band_boot=0., max_plots=0, cornerpoints = None,
                             coloring = None):
    """This function plots the persistence diagram with an optional confidence band.

    :param persistence: The persistence to plot.
    :type persistence: list of tuples(dimension, tuple(birth, death)).
    :param persistence_file: A persistence file style name (reset persistence if both are set).
    :type persistence_file: string
    :param alpha: alpha value in [0.0, 1.0] for points and horizontal infinity line (default is 0.6).
    :type alpha: float.
    :param band_boot: bootstrap band (not displayed if :math:`\leq` 0.)
    :type band_boot: float.
    :param max_plots: number of maximal plots to be displayed
    :type max_plots: int.
    :returns: plot -- A diagram plot of persistence.
    """
    if persistence_file is not '':
        if os.path.isfile(persistence_file):
            # Reset persistence
            persistence = []
            diag = read_persistence_intervals_grouped_by_dimension(persistence_file=persistence_file)
            for key in diag.keys():
                for persistence_interval in diag[key]:
                    persistence.append((key, persistence_interval))
        else:
            print("file " + persistence_file + " not found.")
            return None

    if max_plots > 0 and max_plots < len(persistence):
        # Sort by life time, then takes only the max_plots elements
        persistence = sorted(persistence, key=lambda life_time:
                             life_time[1][1]-life_time[1][0], reverse=True)[:max_plots]

    (min_birth, max_death) = __min_birth_max_death(persistence, band_boot)
    ind = 0
    delta = ((max_death - min_birth) / 10.0)
    # Replace infinity values with max_death + delta for diagram to be more
    # readable
    infinity = max_death + delta
    axis_start = min_birth - delta
    plt.plot([axis_start,infinity],[axis_start, infinity], color='k',
             alpha=alpha)
    plt.axhline(infinity,linewidth=1.0, color='k', alpha=alpha)
    plt.text(axis_start, infinity, r'$\infty$', color='k', alpha=alpha)
    # bootstrap band
    if band_boot > 0.:
        plt.fill_between(x, x, x+band_boot, alpha=alpha, facecolor='red')
    if cornerpoints is not None:
        reversed_cornerpoints = reversed(cornerpoints)
    else:
        reversed_cornerpoints = [None] * len(persistence)

    # Draw points in loop
    for interval, cp in zip(reversed(persistence), reversed_cornerpoints):
        print("plotting ", interval)
        if coloring is None or cp is None:
            color = palette[interval[0]]
            label = None
        else:
            color, label = get_color_from_cornerpoint(cp, coloring)
        print("color: ", color)
        print("label: ", label)
        if float(interval[1][1]) != float('inf'):
            print("finite death")
            print("interval: ", interval[1][0], interval[1][1])
            plt.plot(interval[1][0], interval[1][1], alpha=alpha,
                        color = color, label=label, marker='o')
            plt.plot([interval[1][0],interval[1][0]],[interval[1][1], interval[1][0]],
                     color = color, alpha = alpha/2,
                     linestyle="dashed")
            plt.plot([interval[1][0],interval[1][1]],[interval[1][1], interval[1][1]],
                     color = color, alpha = alpha/2,
                     linestyle="dashed")
        else:
            print("infinte death")
            print("interval: ", interval[1][0], infinity)
            plt.plot(interval[1][0], infinity, alpha=alpha,
                        color = color, label=label, marker='o')
            plt.plot([interval[1][0],interval[1][0]],[interval[1][0], infinity],
                     color = color, alpha = alpha)
        ind = ind + 1

    plt.title('Persistence diagram')
    plt.xlabel('Birth')
    plt.ylabel('Death')
    if coloring is not None:
        plt.legend()
    # Ends plot on infinity value and starts a little bit before min_birth
    plt.axis([axis_start, infinity, axis_start, infinity + delta])
    return plt


def get_color_from_cornerpoint(cp, coloring):
    try:
        room_count = Counter([v[0] for v in cp.vertex])
    except:
        room_count = Counter([v for v in cp.vertex])
    print("room count ", room_count)
    color = np.sum([np.asarray(coloring[int(room)]) * room_count[room] / len(cp.vertex)
                   for room in room_count], axis=0)
    label = str(list(room_count.keys()))
    return color, label



if __name__ == "__main__":
    ### get_powerset_
    a = [1,2,3]
    b = get_powerset_(a)
    ### get_supersets
    a = [set([10]), set([1]), set([2]), set([1,3]), set([1,2]) , set([5,6,7])]
    b = get_supersets(a)
    print(b)
