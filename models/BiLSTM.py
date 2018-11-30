# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6 or above
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Initializes a multi layer bidirectional LSTM.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2018"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : BiLSTM

__variables__   :

__methods__     :
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# import unittest
# from torchsummary import summary
from logger.logger import logger


class BiLSTM(nn.Module):
    def __init__(self, input_size, hid_size, num_layers=1, dropout=0.0, dropout_extrenal=True,
                 bidirectional=True, bias=True, use_cuda=True):
        """
        Initializes a multi layer bidirectional LSTM based on parameter values.

        :param input_size: Number of features in input.
        :param hid_size: Number of neurons in the hidden layer.
        :param num_layers: Number of recurrent layers.
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first: If True, then the input and text_lstm tensors are provided as (batch, seq, feature)
        :param dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer.
        :param dropout_extrenal: Flag if dropout should be used externally as Pytorch uses dropout only on last layer
               and if there is only one layer, dropout will not be used.
        :param bidirectional: If True, becomes a bidirectional RNN. Default: False
        """
        super(BiLSTM, self).__init__()
        self.use_cuda = use_cuda
        if self.use_cuda and torch.cuda.is_available():
            self.lstm = nn.LSTM(input_size=input_size,
                                num_layers=num_layers,
                                hidden_size=hid_size,
                                bias=bias,
                                dropout=dropout,
                                batch_first=True,  # If True, then the input and lstm tensors are provided as (batch, seq, feature)
                                bidirectional=bidirectional).cuda()
        else:
            self.lstm = nn.LSTM(input_size=input_size,
                                num_layers=num_layers,
                                hidden_size=hid_size,
                                bias=bias,
                                dropout=dropout,
                                batch_first=True,
                                bidirectional=bidirectional)

        if dropout_extrenal and dropout > 0.0:  # Need to use dropout externally as Pytorch LSTM uses dropout only on
            # last layer and if there is only one layer, dropout will not be used.
            self.lstm_out = F.dropout(self.lstm, p=dropout, training=False, inplace=False)

    def forward(self, inputs, batch_size=64, requires_grad=False):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.

        :param requires_grad:
        :param batch_size: Size of a batch per forward call.
        :param inputs: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        self.batch_size = batch_size
        if self.use_cuda and torch.cuda.is_available():
            c0 = Variable(torch.rand(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),
                          requires_grad=requires_grad).cuda()
            h0 = Variable(torch.rand(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),
                          requires_grad=requires_grad).cuda()
        else:
            logger.debug(self.batch_size)
            c0 = Variable(torch.rand(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),
                          requires_grad=requires_grad)
            h0 = Variable(torch.rand(self.lstm.num_layers * 2, self.batch_size, self.lstm.hidden_size),
                          requires_grad=requires_grad)
        logger.debug(self.lstm)
        logger.debug(inputs.shape)
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        logger.debug(output)
        logger.debug(output.shape)
        # assert input.shape == text_lstm.shape, "Input {} and Output {} shape should match.".format(input.shape, text_lstm.shape)

        return output, hn, cn


# class BidirectionalLSTMTest(unittest.TestCase):
#     def setUp(self):
#         pass
#
#     def tearDown(self):
#         pass
#
#     def test_forward(self):
#         pass


if __name__ == '__main__':
    # unittest.main()
    test_blstm = BiLSTM(input_size=3, hid_size=2, num_layers=1)  # (4, 2, 1, 1) batch_size, input_size, hid_size, num_layers -> (4, 1, 2)
    # hid_size-> 1: 442, 2: 444, 3: 446, 4:448
    # input = torch.rand(1, 4, 2)  # (1, 4, 2): 2 cols, 4 rows = input:  tensor([[[0.9409, 0.6746], [0.8177, 0.3506], [0.9565, 0.5537], [0.3677, 0.8891]]])
    input = torch.rand(4, 5, 3)  # (4, 1, 2) batch_size, <don't matter> ,input_size
    logger.debug("input: {}".format(input))
    # logger.debug("y: ", y)
    result = test_blstm.forward(input,batch_size=4)
    # logger.debug("result: {}".format(result))
    logger.debug("result: {}".format(result[0].shape))
    logger.debug("result: {}".format(result[1].shape))
    logger.debug("result: {}".format(result[2].shape))
