from abc import ABC

import torch
from torch import nn
from torch.nn import functional as F


class MCDropoutCapableNNModule(nn.Module, ABC):
    """
    Base class for NN modules that are to support MC-Dropout.
    Support can be added by applying the _dropout function in the module's forward method.
    Then, to apply inference that samples results, call inferMCDropout rather than just using __call__.
    """

    def __init__(self):
        super().__init__()
        self._applyMCDropout = False
        self._pMCDropoutOverride = None

    def __setstate__(self, d):
        if "_applyMCDropout" not in d:
            d["_applyMCDropout"] = False
        if "_pMCDropoutOverride" not in d:
            d["_pMCDropoutOverride"] = None
        super().__setstate__(d)

    def _dropout(self, x, pTraining=None, pInference=None):
        """
        This method is to to applied within the module's forward method to apply dropouts during training and/or inference.

        :param x: the model input tensor
        :param pTraining: the probability with which to apply dropouts during training; if None, apply no dropout
        :param pInference:  the probability with which to apply dropouts during MC-Dropout-based inference (via inferMCDropout,
            which may override the probability via its optional argument);
            if None, a dropout is not to be applied
        :return: a potentially modified version of x with some elements dropped out, depending on application context and dropout probabilities
        """
        if self.training and pTraining is not None:
            return F.dropout(x, pTraining)
        elif not self.training and self._applyMCDropout and pInference is not None:
            return F.dropout(x, pInference if self._pMCDropoutOverride is None else self._pMCDropoutOverride)
        else:
            return x

    def _enableMCDropout(self, enabled=True, pMCDropoutOverride=None):
        self._applyMCDropout = enabled
        self._pMCDropoutOverride = pMCDropoutOverride

    def inferMCDropout(self, x, numSamples, p=None):
        """
        Applies inference using MC-Dropout, drawing the given number of samples.

        :param x: the model input
        :param numSamples: the number of samples to draw with MC-Dropout
        :param p: the dropout probability to apply, overriding the probability specified by the model's forward method; if None, use model's default
        :return: a pair (y, sd) where y the mean output tensor and sd is a tensor of the same dimension containing standard deviations
        """
        results = []
        self._enableMCDropout(True, pMCDropoutOverride=p)
        try:
            for i in range(numSamples):
                y = self(x)
                results.append(y)
        finally:
            self._enableMCDropout(False)
        results = torch.stack(results)
        mean = torch.mean(results, 0)
        stddev = torch.std(results, 0, unbiased=False)
        return mean, stddev


class MultiLayerPerceptronModule(MCDropoutCapableNNModule):
    def __init__(self, inputDim, outputDim, hiddenDims, hidActivationFn=torch.sigmoid, outputActivationFn=torch.sigmoid,
            pDropout=None):
        super().__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.hiddenDims = hiddenDims
        self.hidActivationFn = hidActivationFn
        self.outputActivationFn = outputActivationFn
        self.pDropout = pDropout
        self.layers = nn.ModuleList()
        if pDropout is not None:
            self.dropout = nn.Dropout(p=pDropout)
        else:
            self.dropout = None
        prevDim = inputDim
        for dim in [*hiddenDims, outputDim]:
            self.layers.append(nn.Linear(prevDim, dim))
            prevDim = dim

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            isLast = i+1 == len(self.layers)
            x = layer(x)
            if not isLast and self.dropout is not None:
                x = self.dropout(x)
            activation = self.hidActivationFn if not isLast else self.outputActivationFn
            if activation is not None:
                x = activation(x)
        return x
