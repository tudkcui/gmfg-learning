import itertools
import string

import networkx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib as mpl
from cycler import cycler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import meshgrid

from experiments import args_parser


def plot():
    plt.rcParams.update({
        "figure.dpi": 600,
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 22,
        "font.sans-serif": ["Helvetica"]})
    cmap = pl.cm.Greys

    i = 1

    plt.subplot(1,2,i)
    plt.gca().text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=plt.gca().transAxes,
                   size=22, weight='bold')
    i += 1

    N = 5
    G = networkx.Graph()
    G.add_nodes_from(range(N))
    G.add_edge(0, 1)
    G.add_edge(0, 3)
    G.add_edge(0, 4)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(2, 4)
    networkx.relabel_nodes(G, lambda x: x + 1, copy=False)
    networkx.draw_networkx(G, with_labels=True)
    networkx.relabel_nodes(G, lambda x: x - 1, copy=False)

    plt.subplot(1,2,i)
    plt.gca().text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=plt.gca().transAxes,
                   size=22, weight='bold')
    i += 1

    center_points = np.array([i / N + 1 / 2 / N for i in range(N)])
    def step_graphon(x, y):
        x_bin = (np.abs(np.expand_dims(x, 2) - np.expand_dims(center_points, [0,1]))).argmin(axis=2)
        y_bin = (np.abs(np.expand_dims(y, 2) - np.expand_dims(center_points, [0,1]))).argmin(axis=2)
        return np.vectorize(lambda x, y: G.has_edge(x, y))(x_bin, y_bin)

    x = np.linspace(0, 1, 250)
    y = np.linspace(0, 1, 250)
    X, Y = meshgrid(x, y)
    Z = step_graphon(X, Y).tolist()

    # plt.title(r'$W(x,y)$')
    plt.rcParams['axes.grid'] = False
    plt.pcolor(x, y, Z, cmap=cmap, vmin=0, vmax=1)

    plt.xlabel('$x$')
    plt.ylabel('$y$')

    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
    plt.gcf().add_axes(ax_cb)
    cb1.set_label(r'$W(x,y)$')
    # lbpos = get(cb1, 'title')


    plt.gcf().set_size_inches(8, 4)
    plt.tight_layout()
    plt.savefig('./figures/step-graphons.png', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()
