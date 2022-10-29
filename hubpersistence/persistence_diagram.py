import numpy as np
from math import sqrt
from collections import Counter
import colorsys
from hubpersistence.utils import plot_persistence_diagram


class CornerPoint:
    """A point of a persistence diagram

    Attributes
    ----------

    k : int
        degree (used normally in homological contexts)
    birth : float
        birth of the homological class represented by the cornerpoint
    death : float
        death of the homological class represented by the cornerpoint
    vertex : string
        for hubs persistence the vertex of class of vertices represented by the
        cornerpoint
    """
    def __init__(self, k, birth, death, vertex = None, color=None):
        self.k = k
        self.birth = birth if birth <= death else death
        self.death = death if death >= birth else birth
        self.persistence = abs(death - birth)
        self.vertex = vertex
        self.color = color
        self.above_the_gap = False

    @property
    def is_cornerline(self):
        """True if self is a cornerline
        """
        return self.persistence == np.inf

    @property
    def is_proper(self):
        """True if self is a proper cornerpoint (not a cornerline)
        """
        return self.persistence != np.inf

    def __eq__(self, other):
        """True if self and other are the same cornerpoint
        """

        return self.birth == other.birth and self.death == other.death

    def __hash__(self):
        return hash((self.k, self.birth, self.death, self.persistence))

    def __repr__(self):
        return "Birth: {}\nDeath: {}\nVertices: {}".format(self.birth,
                                                           self.death,
                                                           self.vertex)


class PersistenceDiagram(object):
    """A persistence diagram is a multiset of 2-dimensional points called
    cornerpoints. The class allows to create a persistence diagram in two ways:

    * Giving a filtered complex, in this case it is possible to compute the
    persistence and hence the cornerpoints by using methods of this class

    * Giving a list of cornerpoints (with repetitions)

    Attributes
    ----------
    filtered_complex : <gudhi.SimplexTree>
        A SimplexTree instance
    cornerpoints : list
        List of tuples of the form (k, (b, d))
    """
    def __init__(self, cornerpoints = None):
        self.cornerpoints = cornerpoints
        self.get_cornerpoints_multiset()
        self.get_persistence_from_cornerpoints()

    def get_cornerpoints(self, proper = True):
        if proper:
            self.get_proper_cornerpoints()
            cps = self.proper_cornerpoints
        else:
            cps = self.cornerpoints
        return np.asarray([[c.birth, c.death] for c in cps])

    def get_persistence_from_cornerpoints(self):
        """Gets persistence in gudhi format from self.cornerpoints
        """
        self.persistence_to_plot = [(c.k, (c.birth, c.death))
                                    for c in self.cornerpoints]

    def plot_gudhi(self, ax_handle, cornerpoints, persistence_to_plot = None,
                   coloring=None):
        """plots the persistence diagram in ax_handle

        coloring : list
            if not None, is the list of colors to be used on the
            persistence diagram's points.
        """
        if persistence_to_plot is None:
            persistence_to_plot = self.persistence
        ax_handle = plot_persistence_diagram(persistence_to_plot,
                                             cornerpoints = cornerpoints,
                                             coloring = coloring)
        return ax_handle

    def get_cornerpoint_objects(self):
        """Creates a list of CornerPoint instances
        """
        self.cornerpoints = [CornerPoint(k, b, d) for (k, (b, d)) in self.persistence]
        self.cornerpoints.sort(key=lambda x: x.persistence)
        self.get_cornerpoints_multiset()

    def get_cornerpoints_multiset(self):
        """Organises cornerpoints as a multiset in the form
        cornerpoint : multiplicity and generate as many colors as cornerpoints
        to eventually colorcode them.

        """
        self.cornerpoints_multiset = Counter(self.cornerpoints)
        self.colors = generate_n_distinct_colors(len(self.cornerpoints_multiset))

    def get_proper_cornerpoints(self):
        """Gets the list of proper cornerpoints (the ones with persistence
        smaller than infinity).
        """
        self.proper_cornerpoints = [c for c in self.cornerpoints_multiset
                                            if c.is_proper and not np.isnan(c.persistence)]
        self.proper_cornerpoints.sort(key=lambda x: x.persistence)

    def get_nth_widest_gap(self, n = 0):
        """Computes the widest gap according to the definition originally given
        in [2]_

        Parameters
        ----------

        n : int
            nth gap to consider. 1 is maximal, 2 the second maximal gap
            et cetera

        .. [2] Kurlin, Vitaliy. "A fast persistence-based segmentation of noisy
              2D clouds with provable guarantees." Pattern recognition letters
              83 (2016): 3-12.
        """
        if not hasattr(self, "proper_cornerpoints"):
            self.get_proper_cornerpoints()
        self.gap_number = n
        diagonal_gaps = np.diff([p.persistence for p in self.proper_cornerpoints])
        dg_index = np.argmax(diagonal_gaps)
        dg_index = np.argsort(diagonal_gaps)[::-1][self.gap_number]
        self.proper_cornerpoints_above_gap = self.proper_cornerpoints[dg_index + 1 :]
        [setattr(c, 'above_the_gap', True) for c in self.proper_cornerpoints_above_gap]
        return self.proper_cornerpoints[dg_index], self.proper_cornerpoints[dg_index + 1]

    def get_n_most_persistent_cornerpoints(self, n):
        """Get the first n cornerpoints according to their poersistence
        """
        if not hasattr(self, "proper_cornerpoints"):
            self.get_proper_cornerpoints()
        self.proper_cornerpoints_reversed = self.proper_cornerpoints[::-1]
        return self.proper_cornerpoints[: n + 1]

    def plot_nth_widest_gap(self, ax_handle = None, n = 0):
        """Plots the widest gap on the persistence diagram already plotted in
        ax_handle
        """
        if ax_handle is None:
            fig, ax_handle = plt.subplots()
        if hasattr(self, 'gap_number'):
            n = self.gap_number
        l, u = self.get_nth_widest_gap(n = n)
        x = np.asarray(ax_handle.get_xlim())
        y_l =  x + l.persistence
        y_u = x + u.persistence
        ax_handle.plot(x, y_l, ls="--", c=".3")
        ax_handle.plot(x, y_u, ls="--", c=".3")
        ax_handle.fill_between(x, y_l, y_u, facecolor='yellow', alpha=0.7)
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])
        ax_handle.set_title("Visualizing the {} widest gap".format(ordinal(n)))

    def mark_points_above_diagonal_gaps(self, ax_handle):
        """Marks the points above the widest gap by circling them in red
        """
        for c in self.proper_cornerpoints_above_gap:
            ax_handle.plot(c.birth, c.death, 'o', ms=14, markerfacecolor="None",
             markeredgecolor='red', markeredgewidth=5)


def generate_n_distinct_colors(n):
    """Generates n distinct colors
    """
    hsv_tuples = [(i * 1.0 / n, 0.5, 0.5) for i in range(n)]
    return map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
