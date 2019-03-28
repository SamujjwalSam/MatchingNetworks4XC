# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Builds embeddings for pre-trained sentences using either LSTM or CNNText.

__description__ : Builds embeddings for pre-trained sentences using either LSTM or CNNText.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : EmbedText

__variables__   :

__methods__     :
"""

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

from logger.logger import logger
from models.Weight_Init import weight_init
from config import configuration as config


class EmbedText(nn.Module):
    """Builds context sensitive embeddings for pre-trained sentences using either LSTM or CNN."""

    def __init__(self, num_channels, layer_size, classify_count=config["model"]["classify_count"], requires_grad=True,
                 hid_size=config["lstm_params"]["hid_size"], g_encoder=config["model"]["g_encoder"], use_cuda=config["model"]["use_cuda"],
                 input_size=config["prep_vecs"]["input_size"], bidirectional=config["lstm_params"]["bidirectional"]):
        """
        Builds embeddings for pre-trained sentences using either LSTM or CNN.

        :param layer_size: Number of layers in the LSTM.
        :param g_encoder: Wheather to use LSTM or CNN to encode the inputs.
        :param classify_count: Number of categories. If num_categories>0, we want a FC layer at the end with num_categories size.
        :param num_channels: Number of channels of samples
        :param useDroput: use Dropout with p=0.1 in each Conv block
        """
        super(EmbedText, self).__init__()
        self.g_encoder = g_encoder
        self.use_cuda = use_cuda

        if self.g_encoder == "lstm":
            self.bidirectional = bidirectional
            self.lstm = self.LSTM_layer()
            self.hidden = self.init_hid(requires_grad=requires_grad)
            # logger.debug((self.hidden))
            self.output_size = hid_size * 2  ## 2 because bidirectional.
        elif self.g_encoder == "cnn":
            logger.debug((num_channels, layer_size))
            self.conv1 = self.CNN_layer(num_channels, layer_size)
            self.conv2 = self.CNN_layer(layer_size, layer_size)
            self.conv3 = self.CNN_layer(layer_size, layer_size)
            self.conv4 = self.CNN_layer(layer_size, layer_size)

            finalSize = int(math.floor(input_size / (2 * 2 * 2 * 2)))  # (2 * 2 * 2 * 2) for 4 CNN layers.
            self.output_size = finalSize * finalSize * layer_size
        else:
            raise Exception("Unknown 'g_encoder': [{}]. \n"
                            "Supported types are: ['lstm', 'cnn'].".format(self.g_encoder))

        if classify_count > 0:  ## We want a linear layer as last layer of CNN network.
            self.use_linear_last = True
            self.last_linear_layer = nn.Linear(self.output_size, classify_count)
            self.output_size = classify_count
        else:
            self.use_linear_last = False

            if self.g_encoder == "lstm":
                weight_init(self.lstm)
            elif self.g_encoder == "cnn":
                # weight_init(self.conv1)
                # weight_init(self.conv2)
                # weight_init(self.conv3)
                # weight_init(self.conv4)
                self.weights_init(self.conv1)
                self.weights_init(self.conv2)
                self.weights_init(self.conv3)
                self.weights_init(self.conv4)

    def forward(self, inputs, dropout_external=config["model"]["dropout_external"]):
        """
        Runs the CNNText producing the embeddings and the gradients.

        :param requires_grad:
        :param dropout_external:
        :param inputs: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        if self.g_encoder == "lstm":
            output, _ = self.lstm(inputs, self.hidden)
        elif self.g_encoder == "cnn":
            output = self.conv1(inputs)
            output = self.conv2(output)
            output = self.conv3(output)
            output = self.conv4(output)
            # output = output.view(output.size(0), -1)
        else:
            raise Exception("Unknown g_encoder: [{}]. \n"
                            "Supported types are: ['lstm', 'cnn'].".format(self.g_encoder))

        if self.use_linear_last:
            output = self.last_linear_layer(output)
        return output

    # def lstm_layer(self):
    #     fw_lstm_cells_encoder = [nn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
    #                              for i in range(len(self.layer_sizes))]
    #     bw_lstm_cells_encoder = [nn.LSTMCell(num_units=self.layer_sizes[i], activation=tf.nn.tanh)
    #                              for i in range(len(self.layer_sizes))]

    def CNN_layer(self, in_planes, out_planes, kernel_size=config["cnn_params"]["kernel_size"],
                  stride=config["cnn_params"]["stride"], padding=config["cnn_params"]["padding"],
                  bias=config["cnn_params"]["bias"], dropout=config["model"]["dropout"]):
        """Convolution with padding to get embeddings for input pre-trained sentences.

        Default: Convolution = 1 x 1 with stride = 2.
        """
        seq = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=kernel_size, stride=stride + 1)
        )
        if dropout > 0.0:  ## Add dropout module
            list_seq = list(seq.modules())[1:]
            list_seq.append(nn.Dropout(dropout))
            seq = nn.Sequential(*list_seq)

        return seq

    def LSTM_layer(self, bidirectional=config["lstm_params"]["bidirectional"], dropout=config["model"]["dropout"],
                   input_size=config["prep_vecs"]["input_size"], batch_first=config["lstm_params"]["batch_first"],
                   num_layers=config["lstm_params"]["num_layers"], hid_size=config["lstm_params"]["hid_size"],
                   bias=config["lstm_params"]["bias"]):
        """Convolution with padding to get embeddings for input pre-trained sentences.

        Default:
        """
        lstm = nn.LSTM(input_size=input_size, hidden_size=hid_size, dropout=dropout, bias=bias, num_layers=num_layers,
                       batch_first=batch_first, bidirectional=bidirectional)

        return lstm

    def init_hid(self, batch_size=config["sampling"]["batch_size"], requires_grad=True, low_val=-1, high_val=1):
        """
        Generates h0 and c0 for LSTM initialization with values range from r1 to r2.

        :param high_val: Max value of the range
        :param low_val: Min value of the range
        :param batch_size:
        :param requires_grad:
        :return:
        """
        num_directions = 2  ## For bidirectional, num_layers should be multiplied by 2.
        if not self.bidirectional:
            num_directions = 1
        # r1, r2 = -1, 1  ## To generate numbers in range(-1,1)
        if self.use_cuda and torch.cuda.is_available():
            cell_init = Variable((high_val - low_val)
                                 * torch.rand(self.lstm.num_layers * num_directions, batch_size, self.lstm.hidden_size)
                                 - high_val, requires_grad=requires_grad).cuda()
            hid_init = Variable((high_val - low_val)
                                * torch.rand(self.lstm.num_layers * num_directions, batch_size, self.lstm.hidden_size)
                                - high_val, requires_grad=requires_grad).cuda()
        else:
            cell_init = Variable((high_val - low_val)
                                 * torch.rand(self.lstm.num_layers * num_directions, batch_size, self.lstm.hidden_size)
                                 - high_val, requires_grad=requires_grad)
            hid_init = Variable((high_val - low_val)
                                * torch.rand(self.lstm.num_layers * num_directions, batch_size, self.lstm.hidden_size)
                                - high_val, requires_grad=requires_grad)
        return hid_init, cell_init

    def weights_init(self, module):
        """
        Initialize weights to all the layers of the Network.
        :param module:
        """
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    import torch

    a = torch.ones(1, 2, 4)  ## batch_size, <don't matter>, input_size
    logger.debug(a)
    cls = EmbedText(input_size=4, layer_size=1, classify_count=2, g_encoder="lstm")
    sim = cls.forward(a, batch_size=1, dropout_external=True)
    logger.debug(sim)
    logger.debug(sim.shape)
