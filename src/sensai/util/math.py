import scipy.stats

from .string import objectRepr


class NormalDistribution:
    def __init__(self, mean=0, std=1, unitMax=False):
        """
        :param mean: the mean
        :param std: the standard deviation
        :param unitMax: if True, scales the distribution's pdf such that its maximum value becomes 1
        """
        self.unitMax = unitMax
        self.mean = mean
        self.std = std
        self.norm = scipy.stats.norm(loc=mean, scale=std)

    def pdf(self, x):
        v = self.norm.pdf(x)
        if self.unitMax:
            v /= self.norm.pdf(self.mean)
        return v

    def __str__(self):
        return objectRepr(self, ["mean", "std", "unitMax"])
