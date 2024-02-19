from enum import Enum
from typing import Union, Callable

import torch
from torch import nn
from torch.nn import functional as F

from sensai.util.pickle import setstate
from ...torch_base import MCDropoutCapableNNModule
from ...torch_enums import ActivationFunction


class LSTNetwork(MCDropoutCapableNNModule):
    """
    Network for (auto-regressive) time-series prediction with long- and short-term dependencies as proposed by G. Lai et al.
    It applies two parallel paths to a time series of size (numInputTimeSlices, inputDimPerTimeSlice):
    
        * Complex path with the following stages:
        
            * Convolutions on the time series input data (CNNs):
              For a CNN with numCnnTimeSlices (= kernel size), it produces an output series of size numInputTimeSlices-numCnnTimeSlices+1.
              If the number of parallel convolutions is numConvolutions, the total output size of this stage is thus
              numConvolutions*(numInputTimeSlices-numCnnTimeSlices+1)
            * Two RNN components which process the CNN output in parallel:
            
                * RNN (GRU)
                  The output dimension of this stage is the hidden state of the GRU after seeing the entire
                  input data from the previous stage, i.e. if has size hidRNN.
                * Skip-RNN (GRU), which processes time series elements that are 'skip' time slices apart.
                  It does this by grouping the input such that 'skip' GRUs are applied in parallel, which all use the same parameters.
                  If the hidden state dimension of each GRU is hidSkip, then the output size of this stage is skip*hidSkip.
                  
            * Dense layer
            
        * Direct regression dense layer (so-called "highway" path) which uses the features of the last hwWindow time slices to
          directly make a prediction
        
    The model ultimately combines the outputs of these two paths via a combination function.
    Many parts of the model are optional and can be completely disabled.
    The model can produce one or more (potentially multi-dimensional) outputs, where each output typically corresponds
    to a time slice for which a prediction is made.

    The model expects as input a tensor of size (batchSize, numInputTimeSlices, inputDimPerTimeSlice).
    As output, the model will produce a tensor of size (batchSize, numOutputTimeSlices, outputDimPerTimeSlice)
    if mode==REGRESSION and a tensor of size (batchSize, outputDimPerTimeSlice=numClasses, numOutputTimeSlices)
    if mode==CLASSIFICATION; the latter shape matches what is required by the multi-dimensional case of loss function
    CrossEntropyLoss, for example, and therefore is suitable for classification use cases.
    For mode==ENCODER, the model will produce a tensor of size (batch_size, hidRNN + skip * hidSkip).
    """

    class Mode(Enum):
        REGRESSION = "regression"
        CLASSIFICATION = "classification"
        ENCODER = "encoder"

    def __init__(self,
            num_input_time_slices: int,
            input_dim_per_time_slice: int,
            num_output_time_slices: int = 1,
            output_dim_per_time_slice: int = 1,
            num_convolutions: int = 100,
            num_cnn_time_slices: int = 6,
            hid_rnn: int = 100,
            skip: int = 0,
            hid_skip: int = 5,
            hw_window: int = 0,
            hw_combine: str = "plus",
            dropout=0.2,
            output_activation: Union[str, ActivationFunction, Callable] = "sigmoid",
            mode: Mode = Mode.REGRESSION):
        """
        :param num_input_time_slices: the number of input time slices
        :param input_dim_per_time_slice: the dimension of the input data per time slice
        :param num_output_time_slices: the number of time slices predicted by the model
        :param output_dim_per_time_slice: the number of dimensions per output time slice. While this is the number of
            target variables per time slice for regression problems, this must be the number of classes for classification problems.
        :param num_cnn_time_slices: the number of time slices considered by each convolution (i.e. it is one of the dimensions of the matrix used for
            convolutions, the other dimension being inputDimPerTimeSlice), a.k.a. "Ck"
        :param num_convolutions: the number of separate convolutions to apply, i.e. the number of independent convolution matrices, a.k.a "hidC";
            if it is 0, then the entire complex processing path is not applied.
        :param hid_rnn: the number of hidden output dimensions for the RNN stage
        :param skip: the number of time slices to skip for the skip-RNN. If it is 0, then the skip-RNN is not used.
        :param hid_skip: the number of output dimensions of each of the skip parallel RNNs
        :param hw_window: the number of time slices from the end of the input time series to consider as input for the highway component.
            If it is 0, the highway component is not used.
        :param hw_combine: {"plus", "product", "bilinear"} the function with which the highway component's output is combined with the complex path's output
        :param dropout: the dropout probability to use during training (dropouts are applied after every major step in the evaluation path)
        :param output_activation: the output activation function
        :param mode: the prediction mode. For `CLASSIFICATION`, the output tensor dimension ordering is adapted to suit loss functions such
            as CrossEntropyLoss. When set to `ENCODER`, will output the latent representation prior to the dense layer in the complex path
            of the network (see class docstring).
        """
        if num_convolutions == 0 and hw_window == 0:
            raise ValueError("No processing paths remain")
        if num_input_time_slices < num_cnn_time_slices or (hw_window != 0 and hw_window < num_input_time_slices):
            raise Exception("Inconsistent numbers of times slices provided")

        super().__init__()
        self.inputDimPerTimeSlice = input_dim_per_time_slice
        self.timeSeriesDimPerTimeSlice = output_dim_per_time_slice
        self.totalOutputDim = self.timeSeriesDimPerTimeSlice * num_output_time_slices
        self.numOutputTimeSlices = num_output_time_slices
        self.window = num_input_time_slices
        self.hidRNN = hid_rnn
        self.numConv = num_convolutions
        self.hidSkip = hid_skip
        self.Ck = num_cnn_time_slices  # the "height" of the CNN filter/kernel; the "width" being inputDimPerTimeSlice
        self.convSeqLength = self.window - self.Ck + 1  # the length of the output sequence produced by the CNN for each kernel matrix
        self.skip = skip
        self.hw = hw_window
        self.pDropout = dropout
        self.mode = mode

        # configure CNN-RNN path
        if self.numConv > 0:
            self.conv1 = nn.Conv2d(1, self.numConv, kernel_size=(self.Ck, self.inputDimPerTimeSlice))  # produce numConv sequences using numConv kernel matrices of size (height=Ck, width=inputDimPerTimeSlice)
            self.GRU1 = nn.GRU(self.numConv, self.hidRNN)
            if self.skip > 0:
                self.skipRnnSeqLength = self.convSeqLength // self.skip  # we divide by skip to obtain the sequence length, because, in order to support skipping via a regrouping of the tensor, the Skip-RNN processes skip entries of the series in parallel to produce skip hidden output vectors
                if self.skipRnnSeqLength == 0:
                    raise Exception("Window size %d is not large enough for skip length %d; would result in Skip-RNN sequence length of 0!" % (self.window, self.skip))
                self.GRUskip = nn.GRU(self.numConv, self.hidSkip)
                self.linear1 = nn.Linear(self.hidRNN + self.skip * self.hidSkip, self.totalOutputDim)
            else:
                self.linear1 = nn.Linear(self.hidRNN, self.totalOutputDim)

        # configure highway component
        if self.hw > 0:
            # direct mapping from all inputs to all outputs
            self.highway = nn.Linear(self.hw * self.inputDimPerTimeSlice, self.totalOutputDim)
            if hw_combine == 'plus':
                self.highwayCombine = self._plus
            elif hw_combine == 'product':
                self.highwayCombine = self._product
            elif hw_combine == 'bilinear':
                self.highwayCombine = nn.Bilinear(self.totalOutputDim, self.totalOutputDim, self.totalOutputDim)
            else:
                raise ValueError("Unknown highway combination function '%s'" % hw_combine)

        self.output = ActivationFunction.torch_function_from_any(output_activation)

    def __setstate__(self, state):
        if "isClassification" in state:
            state["mode"] = self.Mode.CLASSIFICATION if state["isClassification"] else self.Mode.REGRESSION
        setstate(LSTNetwork, self, state, removed_properties=["isClassification"])

    @staticmethod
    def compute_encoder_dim(hid_rnn: int, skip: int, hid_skip: int) -> int:
        return hid_rnn + skip * hid_skip

    def get_encoder_dim(self):
        """
        :return: the vector dimension that is output for the case where mode=ENCODER
        """
        return self.compute_encoder_dim(self.hidRNN, self.skip, self.hidSkip)

    def forward(self, x):
        batch_size = x.size(0)
        # x has size (batch_size, window=numInputTimeSlices, inputDimPerTimeSlice)

        dropout = lambda x: self._dropout(x, p_training=self.pDropout, p_inference=self.pDropout)

        res = None

        if self.numConv > 0:
            # CNN
            # convSeqLength = self.window - self.Ck + 1
            # convolution produces, via numConv kernel matrices of dimension (height=Ck, width=inputDimPerTimeSlice), from an original input sequence of length window, numConv output sequences of length convSeqLength
            c = x.view(batch_size, 1, self.window, self.inputDimPerTimeSlice)  # insert one dim of size 1 (one channel): (batch_size, 1, height=window, width=inputDimPerTimeSlice)
            c = F.relu(self.conv1(c))  # (batch_size, channels=numConv, convSeqLength, 1)
            c = dropout(c)
            c = torch.squeeze(c, 3)  # drops last dimension, i.e. new size (batch_size, numConv, convSeqLength)

            # RNN
            # It processes the numConv sequences of length convSeqLength obtained through convolution and keep the hidden state at the end, which is comprised of hidR entries
            # Specifically, it squashes the numConv sequences of length convSeqLength to a vector of size hidS (by iterating through the sequences and applying the same model in each step, processing all batches in parallel)
            r = c.permute(2, 0, 1).contiguous()  # (convSeqLength, batch_size, numConv)
            self.GRU1.flatten_parameters()
            _, r = self.GRU1(r)  # maps (seq_len=convSeqLength, batch=batch_size, input_size=numConv) -> hidden state (num_layers=1, batch=batch_size, hidden_size=hidR)
            r = torch.squeeze(r, 0)  # (batch_size, hidR)
            r = dropout(r)

            # Skip-RNN
            if self.skip > 0:
                s = c[:, :, -(self.skipRnnSeqLength * self.skip):].contiguous()  # (batch_size, numConv, convSeqLength) -> (batch_size, numConv, skipRnnSeqLength * skip)
                s = s.view(batch_size, self.numConv, self.skipRnnSeqLength, self.skip)  # (batch_size, numConv, skipRnnSeqLength, skip)
                s = s.permute(2, 0, 3, 1).contiguous()  # (skipRnnSeqLength, batch_size, skip, numConv)
                s = s.view(self.skipRnnSeqLength, batch_size * self.skip, self.numConv)  # (skipRnnSeqLength, batch_size * skip, numConv)
                # Why the above view makes sense:
                # skipRnnSeqLength is the sequence length considered for the RNN, i.e. the number of steps that is taken for each sequence.
                # The batch_size*skip elements of the second dimension are all processed in parallel, i.e. there are batch_size*skip RNNs being applied in parallel.
                # By scaling the actual batch size with 'skip', we process 'skip' RNNs of each batch in parallel, such that each RNN consecutively processes entries that are 'skip' steps apart
                self.GRUskip.flatten_parameters()
                _, s = self.GRUskip(s)  # maps (seq_len=skipRnnSeqLength, batch=batch_size * skip, input_size=numConv) -> hidden state (num_layers=1, batch=batch_size * skip, hidden_size=hidS)
                # Because of the way the data is grouped, we obtain not one vector of size hidS but skip vectors of size hidS
                s = s.view(batch_size, self.skip * self.hidSkip)  # regroup by batch -> (batch_size, skip * hidS)
                s = dropout(s)
                r = torch.cat((r, s), 1)  # (batch_size, hidR + skip * hidS)

            if self.mode == self.Mode.ENCODER:
                return r

            res = self.linear1(r)  # (batch_size, totalOutputDim)

        # auto-regressive highway model
        if self.hw > 0:
            resHW = x[:, -self.hw:, :]  # keep only the last hw entries for each input: (batch_size, hw, inputDimPerTimeSlice)
            resHW = resHW.view(-1, self.hw * self.inputDimPerTimeSlice)  # (batch_size, hw * inputDimPerTimeSlice)
            resHW = self.highway(resHW)  # (batch_size, totalOutputDim)
            if res is None:
                res = resHW
            else:
                res = self.highwayCombine(res, resHW)  # (batch_size, totalOutputDim)

        if self.output:
            res = self.output(res)

        res = res.view(batch_size, self.numOutputTimeSlices, self.timeSeriesDimPerTimeSlice)
        if self.mode == self.Mode.CLASSIFICATION:
            res = res.permute(0, 2, 1)
        return res

    @staticmethod
    def _plus(x, y):
        return x + y

    @staticmethod
    def _product(x, y):
        return x * y
