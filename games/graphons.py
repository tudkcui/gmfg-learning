import numpy as np


def uniform_attachment_graphon(x, y):
    return 1 - np.maximum(x, y)


def ranked_attachment_graphon(x, y):
    return 1 - x * y


def er_graphon(x, y, p=0.5):
    return np.ones_like(x) * p
