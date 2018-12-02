# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6 or above
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Builds embeddings for pre-trained sentences using either LSTM and/or CNNText.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2018"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : EmbedText

__variables__   :

__methods__     :
"""

import math

import numpy as np
import torch.nn as nn
import torch.nn.init as init

from logger.logger import logger
from models import BiLSTM as BiLSTM


# from torch.autograd import Variable
# import unittest


class EmbedText(nn.Module):
    """Builds context sensitive embeddings for pre-trained sentences using either LSTM or CNN."""

    def __init__(self, num_categories=0, hid_size=2, num_layers=1, model_type="lstm", num_channels=1, dropout=0.2,
                 input_size=28):
        """
        Builds embeddings for pre-trained sentences using either LSTM or CNN.

        :param num_layers: Number of layers in the LSTM.
        :param model_type: Wheather to use LSTM or CNN to encode the inputs.
        :param num_categories: Number of categories. If num_categories>0, we want a FC layer at the end with num_categories size.
        :param num_channels: Number of channels of samples
        :param useDroput: use Dropout with p=0.1 in each Conv block
        """
        super(EmbedText, self).__init__()
        self.model_type = model_type

        if self.model_type == "lstm":
            self.text_lstm = BiLSTM.BiLSTM(input_size, hid_size=hid_size, num_layers=num_layers, dropout=dropout,
                                           bidirectional=True)
            # self.weights_init(self.text_lstm)
            self.output_size = hid_size * 2  # TODO: final text_lstm size of LSTM.
        elif self.model_type == "cnn":  # TODO: Decide on CNN architecture to use.
            self.layer1 = self.CNN_layer(num_channels, num_layers, dropout=dropout)
            self.layer2 = self.CNN_layer(num_layers, num_layers, dropout=dropout)
            self.layer3 = self.CNN_layer(num_layers, num_layers, dropout=dropout)
            self.layer4 = self.CNN_layer(num_layers, num_layers, dropout=dropout)

            finalSize = int(math.floor(input_size / (2 * 2 * 2 * 2)))  # (2 * 2 * 2 * 2) for 4 CNN layers.
            self.output_size = finalSize * finalSize * num_layers
        if num_categories > 0:  # We want a linear layer as last layer of CNN network.
            self.use_linear_last = True
            self.last_linear_layer = nn.Linear(self.output_size, num_categories)
            self.output_size = num_categories
        else:
            self.use_linear_last = False

            # self.weights_init(self.layer1)
            # self.weights_init(self.layer2)
            # self.weights_init(self.layer3)
            # self.weights_init(self.layer4)

    def forward(self, inputs, batch_size=64, dropout_extrenal=False):
        """
        Runs the CNNText producing the embeddings and the gradients.

        :param batch_size:
        :param inputs: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        logger.info(inputs.shape)
        if self.model_type == "lstm":
            output, hn, cn = self.text_lstm(inputs, batch_size, dropout_extrenal=dropout_extrenal)
        elif self.model_type == "cnn":
            output = self.layer1(inputs)
            output = self.layer2(output)
            output = self.layer3(output)
            output = self.layer4(output)
            output = output.view(output.size(0), -1)
        else:
            logger.warn("Unknown model_type: [{}]. Supported types are: ['lstm','cnn'].".format(self.model_type))
            raise NotImplementedError

        if self.use_linear_last:
            output = self.last_linear_layer(output)
        return output

    def CNN_layer(self, in_planes, out_planes, kernel_size=1, stride=1, padding=1, bias=True, dropout=0.1):
        """Convolution with padding to get embeddings for input pre-trained sentences.

        Default: Convolution = 3 x 3.
        """
        seq = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        if dropout > 0.0:  # Add dropout module
            list_seq = list(seq.modules())[1:]
            list_seq.append(nn.Dropout(dropout))
            seq = nn.Sequential(*list_seq)

        return seq

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
            else:
                logger.warn("Unknown module instance: [{}]. Weight initialization failed.".format(m))
                raise NotImplementedError


# class ClassifierTest(unittest.TestCase):
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
    import torch

    a = torch.ones(1, 2, 4)  # batch_size, <don't matter>, input_size
    logger.debug(a)
    cls = EmbedText(input_size=4, num_layers=1, num_categories=2, model_type="lstm")
    sim = cls.forward(a, batch_size=1, dropout_extrenal=True)
    logger.debug(sim)
    logger.debug(sim.shape)
