##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# import torch
import torch.nn as nn
import torch.nn.init as init
# from torch.autograd import Variable
import torch.nn.functional as F
# import unittest
import numpy as np
import math

from logger.logger import logger
from models import BiLSTM as BiLSTM
# print(BiLSTM.__file__)


class EmbedText(nn.Module):
    def __init__(self, layer_size, model_type = "lstm",nClasses=0, num_channels=1, dropout=0.2, image_size=28):
        """
        Builds embeddings for pre-trained sentences using either LSTM and/or CNNText.

        :param layer_sizes: A list of length 4 containing the layer sizes
        :param model_type: Wheather to use LSTM or CNN to encode the inputs.
        :param nClasses: If nClasses>0, we want a FC layer at the end with nClasses size.
        :param num_channels: Number of channels of images
        :param useDroput: use Dropout with p=0.1 in each Conv block
        """
        super(EmbedText, self).__init__()
        
        if model_type == "lstm":
            self.output = LSTM_layer(self, input_dim, hid_size, num_layers, batch_size, bidirectional=True, dropout_extrenal=False, dropout=dropout)
            self.weights_init(self.output)
            self.outSize =   # TODO: final output size of LSTM.
        elif model_type == "cnn":  # TODO: Decide on CNN architecture to use.
            self.layer1 = self.CNN_layer(num_channels, layer_size, dropout)
            self.layer2 = self.CNN_layer(layer_size, layer_size, dropout)
            self.layer3 = self.CNN_layer(layer_size, layer_size, dropout)
            self.layer4 = self.CNN_layer(layer_size, layer_size, dropout)

            finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))  # (2 * 2 * 2 * 2) for 4 CNN layers.
            self.outSize = finalSize * finalSize * layer_size
            if nClasses > 0:  # We want a linear layer as last layer.
                self.useClassification = True
                self.layer5 = nn.Linear(self.outSize, nClasses)
                self.outSize = nClasses
            else:
                self.useClassification = False

            self.weights_init(self.layer1)
            self.weights_init(self.layer2)
            self.weights_init(self.layer3)
            self.weights_init(self.layer4)

    def weights_init(self, module):
        """
        Initialize weights to all the layers of the Network.
        :param module:
        """
        for m in module.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.LSTM):
                init.xavier_uniform_(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            else:
                logger.warn("Unknown module! Weight initialization failed.")
                raise NotImplementedError

    def forward(self, inputs):
        """
        Runs the CNNText producing the embeddings and the gradients.

        :param inputs: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        if model_type == "lstm":
            LSTM_layer(self, input_dim, hid_size, num_layers, batch_size, bidirectional=True, dropout_extrenal=False, dropout=0.1)
        elif model_type == "cnn": 
            x = self.layer1(inputs)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = x.view(x.size(0), -1)
            if self.useClassification:
                x = self.layer5(x)
        return x

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

    def LSTM_layer(self, input_dim, hid_size, num_layers, batch_size, bidirectional=True, dropout_extrenal=False, dropout=0.1):
        """LSTM to get embeddings for input pre-trained sentences."""
        lstm_out, hn, cn = BiLSTM.BiLSTM(input_dim, hid_size, num_layers, batch_size, dropout, bidirectional)
        if dropout_extrenal:  # Need to use dropout externally as Pytorch LSTM uses dropout only on last layer. If there is only one layer, dropout will not be used.
            nn.Dropout(dropout)
            lstm_out = F.dropout(lstm_out,p=dropout,training=False,inplace=False)
        # Need to add a linear layer at the end of LSTM?
        return lstm_out


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
    pass
