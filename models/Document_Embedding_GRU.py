# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Encodes a wiki document with it's labels using GRU.

__description__ :
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "23-05-2019"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : BiGRU

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


class Document_Embedding_GRU(nn.Module):
    """ Class for Bidirectional GRU operations. """

    def __init__(self,batch_size=config["sampling"]["batch_size"],
                 use_cuda=config["model"]["use_cuda"],
                 bidirectional=config["lstm_params"]["bidirectional"],
                 input_size=config["prep_vecs"]["input_size"],
                 hid_size=config["lstm_params"]["hid_size"],
                 dropout=config["model"]["dropout"],
                 bias=config["lstm_params"]["bias"],
                 num_layers=config["lstm_params"]["num_layers"],
                 batch_first=config["lstm_params"]["batch_first"],
                 lr: float = config["model"]["optimizer"]["learning_rate"]):
        """
        Initializes a multi layer bidirectional GRU based on parameter values.

        :param input_size: Number of features in input.
        :param hid_size: Number of neurons in the hidden layer.
        :param num_layers: Number of recurrent layers.
        :param bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first: If True, then the input and text_lstm tensors are provided as (batch, seq, feature)
               and if there is only one layer, dropout will not be used.
        :param bidirectional: If True, becomes a bidirectional RNN. Default: False
        """
        super(Document_Embedding_GRU,self).__init__()
        self.use_cuda = use_cuda
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.lr = lr
        if self.use_cuda and torch.cuda.is_available():
            self.doc_gru = nn.GRU(input_size=input_size,
                                  hidden_size=hid_size,
                                  num_layers=num_layers,
                                  bias=bias,
                                  batch_first=batch_first,  ## If True, then the input and gru tensors are provided as
                                  ## (batch, # seq, feature) rather than (seq, batch, feature)
                                  dropout=dropout,
                                  bidirectional=bidirectional).cuda()
        else:
            self.doc_gru = nn.GRU(input_size=input_size,
                                  hidden_size=hid_size,
                                  num_layers=num_layers,
                                  bias=bias,
                                  batch_first=batch_first,  ## If True, then the input and gru tensors are provided as
                                  ## (batch, # seq, feature) rather than (seq, batch, feature)
                                  dropout=dropout,
                                  bidirectional=bidirectional)
        weight_init(self.doc_gru)

    def init_hidden(self,batch_size=config["sampling"]["batch_size"],requires_grad=True,r1=-1,r2=1,num_directions=2):
        """
        Generates h0 and c0 for GRU initialization with values range from r1 to r2.

        :param num_directions: For bidirectional, num_layers should be multiplied by 2.
        :param r2: Max value of the range. To generate numbers in range(r1, r2)
        :param r1: Min value of the range
        :param batch_size:
        :param requires_grad:
        :return:
        """
        if not self.bidirectional:
            num_directions = 1
        if self.use_cuda and torch.cuda.is_available():
            cell_init = Variable((r2 - r1) * torch.rand(self.doc_gru.num_layers * num_directions,batch_size,
                                                        self.doc_gru.hidden_size) - r2,
                                 requires_grad=requires_grad).cuda()
            hid_init = Variable((r2 - r1) * torch.rand(self.doc_gru.num_layers * num_directions,batch_size,
                                                       self.doc_gru.hidden_size) - r2,
                                requires_grad=requires_grad).cuda()
        else:
            cell_init = Variable((r2 - r1) * torch.rand(self.doc_gru.num_layers * num_directions,batch_size,
                                                        self.doc_gru.hidden_size) - r2,requires_grad=requires_grad)
            hid_init = Variable((r2 - r1) * torch.rand(self.doc_gru.num_layers * num_directions,batch_size,
                                                       self.doc_gru.hidden_size) - r2,requires_grad=requires_grad)
        return hid_init,cell_init

    def forward(self,inputs,label_enb,training=True,requires_grad=True,dropout=config["model"]["dropout"],
                dropout_external=config["model"]["dropout_external"]):
        """
        Runs the bidirectional GRU, produces outputs and hidden states.

        https://pytorch.org/docs/master/nn.html#torch.nn.GRU

        input of shape (seq_len, batch, input_size): tensor containing the features of the input sequence. The input
        can also be a packed variable length sequence. See torch.nn.utils.rnn.pack_padded_sequence() or
        torch.nn.utils.rnn.pack_sequence() for details.

        h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state
        for each element in the batch. If the GRU is bidirectional, num_directions should be 2, else it should be 1.

        c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for
        each element in the batch.

        If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.

        :param label_enb:
        :param requires_grad:
        :param training: Signifies if network is training, during inference dropout is disabled.
        :param dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer.
        :param dropout_external: Flag if dropout should be used externally as Pytorch uses dropout only on last layer.
        :param inputs: The inputs should be a list of shape  (seq_len, batch, input_size).
        :return: Returns the GRU outputs: (seq_len, batch, num_directions * hidden_size), as well as the cell state (cn: (num_layers * num_directions, batch, hidden_size)) and final hidden representations (hn: (num_layers * num_directions, batch, hidden_size)).
        """
        optimizer = self.__create_optimizer(self.doc_gru, self.lr)  # Creating the optimizer
        (h0,c0) = self.init_hidden(self.batch_size,requires_grad=requires_grad)
        outputs,hn = self.doc_gru(inputs,h0)
        ## take hidden avg
        loss = F.cosine_embedding_loss(hn, label_enb)
        optimizer.zero_grad()

        ## Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        optimizer.step()

        if dropout_external and dropout > 0.0:  ## Need to use dropout externally as Pytorch GRU applies dropout only
            ## on last layer and if there is only one layer, dropout will not be applied.
            logger.info("Applying dropout externally.")
            outputs = F.dropout(outputs,p=dropout,training=training)
        return outputs,hn

    def __create_optimizer(self, model, new_lr, optimizer_type=config["model"]["optimizer"]["optimizer_type"],
                           weight_decay=config["model"]["optimizer"]["weight_decay"],
                           rho=config["model"]["optimizer"]["rho"],
                           momentum=config["model"]["optimizer"]["momentum"],
                           dampening=config["model"]["optimizer"]["dampening"],
                           alpha=config["model"]["optimizer"]["alpha"],
                           centered=config["model"]["optimizer"]["centered"]):
        """Setup optimizer_type"""
        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=new_lr,
                                        momentum=momentum,
                                        dampening=dampening,
                                        weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=new_lr,
                                         weight_decay=weight_decay)
        elif optimizer_type == 'adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(),
                                             lr=new_lr,
                                             rho=rho,
                                             weight_decay=weight_decay)
        elif optimizer_type == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(),
                                            lr=new_lr,
                                            lr_decay=self.lr_decay,
                                            weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(),
                                            lr=new_lr,
                                            alpha=alpha,
                                            momentum=0.9,
                                            centered=centered,
                                            weight_decay=weight_decay)
        else:
            raise Exception('Optimizer not supported: [{0}]'.format(optimizer_type))
        return optimizer


if __name__ == '__main__':
    test_doc_gru = Document_Embedding_GRU(input_size=3,hid_size=2,batch_size=2,num_layers=1,dropout=0.2,use_cuda=False,
                                          bidirectional=True)
    print(test_doc_gru)
    input = torch.rand(2,1,3)  ## (batch_size, seq_size, input_size)
    logger.debug("input: {}".format(input))
    logger.debug("input Shape: {}".format(input.shape))
    result = test_doc_gru.forward(input, dropout_external=True)  ## output.shape = (batch_size, seq_size, 2 * hid_size);
    ## hn.shape = (2 * num_layers, batch_size, hid_size) = cn.shape
    # logger.debug("result: {}".format(result))
    logger.debug("result: {}".format(result[0]))
    logger.debug("result: {}".format(result[0].shape))
    logger.debug("result: {}".format(result[1][0].shape))
    logger.debug("result: {}".format(result[1][1].shape))
    result = test_doc_gru.forward(input,dropout_external=True)
    logger.debug("result: {}".format(result[0]))
    logger.debug("result: {}".format(result[0].shape))
