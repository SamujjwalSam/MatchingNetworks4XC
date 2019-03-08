# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Builds embeddings for pre-trained sentences using either LSTM and/or CNNText.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : EmbedText

__variables__   :

__methods__     :
"""

import math
import torch.nn as nn

from logger.logger import logger
from models import BiLSTM
from models.Weight_Init import weight_init
from config import configuration as config


class EmbedText(nn.Module):
    """Builds context sensitive embeddings for pre-trained sentences using either LSTM or CNN."""

    def __init__(self, layer_size, num_channels, classify_count=config["model"]["classify_count"],
                 hid_size=config["prep_vecs"]["hid_size"], model_type=config["model"]["g_encoder"],
                 input_size=config["prep_vecs"]["input_size"]):
        """
        Builds embeddings for pre-trained sentences using either LSTM or CNN.

        :param layer_size: Number of layers in the LSTM.
        :param model_type: Wheather to use LSTM or CNN to encode the inputs.
        :param classify_count: Number of categories. If num_categories>0, we want a FC layer at the end with num_categories size.
        :param num_channels: Number of channels of samples
        :param useDroput: use Dropout with p=0.1 in each Conv block
        """
        super(EmbedText, self).__init__()
        self.model_type = model_type

        if self.model_type == "lstm":
            self.text_lstm = BiLSTM()
            self.output_size = hid_size * 2  ## 2 because BiLSTM.
        elif self.model_type == "cnn":  # TODO: Decide on CNN architecture to use.
            self.conv1 = self.CNN_layer(num_channels, layer_size)
            self.conv2 = self.CNN_layer(layer_size, layer_size)
            self.conv3 = self.CNN_layer(layer_size, layer_size)
            self.conv4 = self.CNN_layer(layer_size, layer_size)

            finalSize = int(math.floor(input_size / (2 * 2 * 2 * 2)))  # (2 * 2 * 2 * 2) for 4 CNN layers.
            self.output_size = finalSize * finalSize * layer_size
        else:
            raise Exception("Unknown 'g_encoder': [{}]. \n"
                            "Supported types are: ['lstm', 'cnn'].".format(self.model_type))

        if classify_count > 0:  ## We want a linear layer as last layer of CNN network.
            self.use_linear_last = True
            self.last_linear_layer = nn.Linear(self.output_size, classify_count)
            self.output_size = classify_count
        else:
            self.use_linear_last = False

            if self.model_type == "cnn":
                weight_init(self.conv1)
                weight_init(self.conv2)
                weight_init(self.conv3)
                weight_init(self.conv4)

    def forward(self, inputs, batch_size=config["sampling"]["batch_size"], dropout_external=config["model"]["dropout_external"], requires_grad=True):
        """
        Runs the CNNText producing the embeddings and the gradients.

        :param dropout_external:
        :param batch_size:
        :param inputs: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        if self.model_type == "lstm":
            _, output = self.text_lstm(inputs, dropout_external=dropout_external, requires_grad=requires_grad)
        elif self.model_type == "cnn":
            # if inputs.shape[1] != 1: logger.warning("input shape: {}".format(inputs.shape))
            output = self.conv1(inputs)
            output = self.conv2(output)
            output = self.conv3(output)
            output = self.conv4(output)
            # output = output.view(output.size(0), -1)
        else:
            raise Exception("Unknown model_type: [{}]. \n"
                            "Supported types are: ['lstm', 'cnn'].".format(self.model_type))

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

        Default: Convolution = 3 x 3.
        """
        seq = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=kernel_size, stride=stride+1)
        )
        if dropout > 0.0:  ## Add dropout module
            list_seq = list(seq.modules())[1:]
            list_seq.append(nn.Dropout(dropout))
            seq = nn.Sequential(*list_seq)

        return seq


if __name__ == '__main__':
    import torch

    a = torch.ones(1, 2, 4)  ## batch_size, <don't matter>, input_size
    logger.debug(a)
    cls = EmbedText(input_size=4, layer_size=1, classify_count=2, model_type="lstm")
    sim = cls.forward(a, batch_size=1, dropout_external=True)
    logger.debug(sim)
    logger.debug(sim.shape)
