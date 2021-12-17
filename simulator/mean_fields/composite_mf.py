from simulator.mean_fields.base import MeanField


class CompositeMeanField(MeanField):
    """
    Implements a mean field from a partial mean field for every time step.
    """

    def __init__(self, state_space, partials):
        super().__init__(state_space)
        self.partials = partials

    def evaluate_integral(self, t, f):
        return self.partials[t].evaluate_integral(t, f)
