# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Initializes a multi layer bidirectional LSTM.

__description__ : Initializes a multi layer bidirectional LSTM.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : BiLSTM

__variables__   :

__methods__     :
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.Weight_Init import weight_init
from logger.logger import logger
from config import configuration as config


class BiLSTM(nn.Module):
    """
        Class for Bidirectional LSTM operations.
    """

    def __init__(self,batch_size: int = config["sampling"]["batch_size"],
                 use_cuda: bool = config["model"]["use_cuda"],
                 bidirectional: bool = config["lstm_params"]["bidirectional"],
                 input_size: int = config["prep_vecs"]["input_size"],
                 hid_size: int = config["lstm_params"]["hid_size"],
                 dropout: float = config["model"]["dropout"],
                 bias: bool = config["lstm_params"]["bias"],
                 num_layers: int = config["lstm_params"]["num_layers"],
                 batch_first: bool = config["lstm_params"]["batch_first"]) -> None:
        """
        Initializes a multi layer bidirectional LSTM based on parameter values.

        :param input_size: Number of features in input.
        :param hid_size: Number of neurons in the hidden layer.
        :param num_layers: Number of recurrent layers.
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first: If True, then the input and text_lstm tensors are provided as (batch, seq, feature)
               and if there is only one layer, dropout will not be used.
        :param bidirectional: If True, becomes a bidirectional RNN. Default: False
        """
        super(BiLSTM,self).__init__()
        self.use_cuda = use_cuda
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        if self.use_cuda and torch.cuda.is_available():
            self.lstm = nn.LSTM(input_size=input_size,
                                hidden_size=hid_size,
                                dropout=dropout,
                                bias=bias,
                                num_layers=num_layers,
                                batch_first=batch_first,  ## If True, then the input and lstm tensors are provided as
                                ## (batch, # seq, feature) rather than (seq, batch, feature)
                                bidirectional=bidirectional).cuda()
        else:
            self.lstm = nn.LSTM(input_size=input_size,
                                hidden_size=hid_size,
                                dropout=dropout,
                                bias=bias,
                                num_layers=num_layers,
                                batch_first=batch_first,  ## If True, then the input and lstm tensors are provided as
                                ## (batch, # seq, feature) rather than (seq, batch, feature)
                                bidirectional=bidirectional)
        # The linear layer that maps from hidden state space to tag space
        # self.hidden2tag = nn.Linear(hid_size, tagset_size)
        # self.hidden = self.init_hid(self.batch_size)
        # logger.debug(self.lstm.weight_ih_l0)
        # logger.debug(self.lstm.weight_hh_l0)
        # logger.debug(self.lstm.bias_ih_l0)
        # logger.debug(self.lstm.bias_hh_l0)
        # # logger.debug(self.lstm.weight_ih_l0_reverse)
        # logger.debug(self.lstm._all_weights)
        weight_init(self.lstm)
        # logger.debug(self.lstm.weight_ih_l0)
        # logger.debug(self.lstm.weight_hh_l0)
        # logger.debug(self.lstm.bias_ih_l0)
        # logger.debug(self.lstm.bias_hh_l0)
        # logger.debug(self.lstm.weight_ih_l0_reverse)
        # logger.debug(self.lstm._all_weights)

    def init_hid(self,batch_size: int = config["sampling"]["batch_size"],requires_grad: bool = True,r1: int = -1,
                 r2: int = 1,num_directions: int = 2) -> [torch.Tensor,torch.Tensor]:
        """
        Generates h0 and c0 for LSTM initialization with values range from r1 to r2.

        :param num_directions: For bidirectional, num_layers should be multiplied by 2.
        :param r2: Max value of the range. To generate numbers in range(r1, r2)
        :param r1: Min value of the range
        :param batch_size:
        :param requires_grad:
        :return:
        """
        if not self.bidirectional:
            num_directions = 1
        # r1, r2 = -1, 1  ## To generate numbers in range(-1,1)
        if self.use_cuda and torch.cuda.is_available():
            cell_init = Variable((r2 - r1) * torch.rand(self.lstm.num_layers * num_directions,batch_size,
                                                        self.lstm.hidden_size) - r2,requires_grad=requires_grad).cuda()
            hid_init = Variable((r2 - r1) * torch.rand(self.lstm.num_layers * num_directions,batch_size,
                                                       self.lstm.hidden_size) - r2,requires_grad=requires_grad).cuda()
        else:
            cell_init = Variable((r2 - r1) * torch.rand(self.lstm.num_layers * num_directions,batch_size,
                                                        self.lstm.hidden_size) - r2,requires_grad=requires_grad)
            hid_init = Variable((r2 - r1) * torch.rand(self.lstm.num_layers * num_directions,batch_size,
                                                       self.lstm.hidden_size) - r2,requires_grad=requires_grad)
        return hid_init,cell_init

    def forward(self,inputs: torch.Tensor,training: bool = True,requires_grad: bool = True,
                dropout: float = config["model"]["dropout"],
                dropout_external: float = config["model"]["dropout_external"]) -> [torch.Tensor,torch.Tensor]:
        """
        Runs the bidirectional LSTM, produces outputs and hidden states.

        https://pytorch.org/docs/master/nn.html#torch.nn.LSTM

        input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence. The input
        can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() or
        torch.nn.utils.rnn.pack_sequence() for details.

        h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state
        for each element in the batch. If the LSTM is bidirectional, num_directions should be 2, else it should be 1.

        c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for
        each element in the batch.

        If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.

        :param requires_grad:
        :param training: Signifies if network is training, during inference dropout is disabled.
        :param dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer.
        :param dropout_external: Flag if dropout should be used externally as Pytorch uses dropout only on last layer.
        :param inputs: The inputs should be a list of shape  (seq_len, batch, input_size).
        :return: Returns the LSTM outputs: (seq_len, batch, num_directions * hidden_size), as well as the cell state (cn: (num_layers * num_directions, batch, hidden_size)) and final hidden representations (hn: (num_layers * num_directions, batch, hidden_size)).
        """
        (h0,c0) = self.init_hid(self.batch_size,requires_grad=requires_grad)
        # logger.debug("self.hidden 1: {}".format(self.hidden))
        outputs,(hn,cn) = self.lstm(inputs,(h0,c0))

        if dropout_external and dropout > 0.0:  ## Use dropout externally as Pytorch LSTM applies dropout only on
            ## last layer and if there is only one layer, dropout will not be applied.
            logger.info("Applying dropout externally.")
            outputs = F.dropout(outputs,p=dropout,training=training,inplace=False)
        # assert input.shape == text_lstm.shape, "Input {} and Output {} shape should match.".format(input.shape, text_lstm.shape)
        return outputs,hn


if __name__ == '__main__':
    test_blstm = BiLSTM(input_size=3,hid_size=2,batch_size=2,num_layers=1,dropout=0.2,use_cuda=False,
                        bidirectional=True)
    print(test_blstm)
    input = torch.rand(2,1,3)  ## (batch_size, seq_size, input_size)
    logger.debug("input: {}".format(input))
    logger.debug("input Shape: {}".format(input.shape))
    result = test_blstm.forward(input,
                                dropout_external=True)  ## output.shape = (batch_size, seq_size, 2 * hid_size); hn.shape = (2 * num_layers, batch_size, hid_size) = cn.shape
    # logger.debug("result: {}".format(result))
    logger.debug("result: {}".format(result[0]))
    logger.debug("result: {}".format(result[0].shape))
    logger.debug("result: {}".format(result[1][0].shape))
    logger.debug("result: {}".format(result[1][1].shape))
    result = test_blstm.forward(input,dropout_external=True)
    logger.debug("result: {}".format(result[0]))
    logger.debug("result: {}".format(result[0].shape))
