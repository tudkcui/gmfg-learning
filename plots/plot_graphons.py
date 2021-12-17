import itertools
import pickle
import string

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib as mpl
from cycler import cycler
from gym.spaces import Discrete
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import meshgrid

from experiments import args_parser
from games.graphons import uniform_attachment_graphon, ranked_attachment_graphon, er_graphon


def plot():
    plt.rcParams.update({
        "figure.dpi": 600,
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 22,
        "font.sans-serif": ["Helvetica"]})
    cmap = pl.cm.Greys

    i = 1
    for graphon in ['unif-att', 'rank-att', 'er']:
        clist = itertools.cycle(cycler(color='rbgcmyk'))
        plt.subplot(1,3,i)
        plt.gca().text(-0.01, 1.06, '(' + string.ascii_lowercase[i-1] + ')', transform=plt.gca().transAxes,
            size=22, weight='bold')
        i += 1

        graphon_label = r'$W_\mathrm{unif}$' if graphon == 'unif-att' else \
                        r'$W_\mathrm{rank}$' if graphon == 'rank-att' else \
                        r'$W_\mathrm{er}$'
        color_bar_label = r'$W_\mathrm{unif}(x,y)$' if graphon == 'unif-att' else \
                          r'$W_\mathrm{rank}(x,y)$' if graphon == 'rank-att' else \
                          r'$W_\mathrm{er}(x,y)$'
        graphon = uniform_attachment_graphon if graphon == 'unif-att' \
        else ranked_attachment_graphon if graphon == 'rank-att' else er_graphon

        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = meshgrid(x, y)
        Z = graphon(X, Y).tolist()

        # plt.title(color_bar_label)
        plt.rcParams['axes.grid'] = False
        plt.pcolor(x, y, Z, cmap=cmap, vmin=0, vmax=1)

        plt.xlabel('$x$')
        plt.ylabel('$y$')

        plt.xlim([0, 1])
        plt.ylim([0, 1])

        divider = make_axes_locatable(plt.gca())
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
        plt.gcf().add_axes(ax_cb)
        cb1.set_label(color_bar_label)

        # plt.title(color_bar_label)

    # plt.subplot(1,3,i)
    # plt.gca().text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=plt.gca().transAxes,
    #                size=22, weight='bold')
    # i += 1
    #
    # N = 5
    # G = networkx.erdos_renyi_graph(N, 0.7)
    # networkx.relabel_nodes(G, lambda x: x + 1, copy=False)
    # networkx.draw(G, with_labels=True)
    # networkx.relabel_nodes(G, lambda x: x - 1, copy=False)
    #
    # plt.subplot(1,3,i)
    # plt.gca().text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=plt.gca().transAxes,
    #                size=22, weight='bold')
    # i += 1
    #
    # center_points = np.array([i / N + 1 / 2 / N for i in range(N)])
    # def step_graphon(x, y):
    #     x_bin = (np.abs(np.expand_dims(x, 2) - np.expand_dims(center_points, [0,1]))).argmin(axis=2)
    #     y_bin = (np.abs(np.expand_dims(y, 2) - np.expand_dims(center_points, [0,1]))).argmin(axis=2)
    #     return np.vectorize(lambda x, y: G.has_edge(x, y))(x_bin, y_bin)
    #
    # x = np.arange(0, 1, 0.01)
    # y = np.arange(0, 1, 0.01)
    # X, Y = meshgrid(x, y)
    # Z = step_graphon(X, Y).tolist()
    #
    # plt.title(color_bar_label)
    # plt.rcParams['axes.grid'] = False
    # plt.pcolor(x, y, Z, cmap=cmap, vmin=0, vmax=1)
    #
    # divider = make_axes_locatable(plt.gca())
    # ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    # cb1 = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, orientation='vertical')
    # plt.gcf().add_axes(ax_cb)
    #
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')



    plt.gcf().set_size_inches(13.5, 4)
    plt.tight_layout(w_pad=-0.05)
    plt.savefig('./figures/graphons.png', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot()
