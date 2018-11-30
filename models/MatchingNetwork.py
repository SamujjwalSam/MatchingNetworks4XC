# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6 or above
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Builds a matching network, the training and evaluation ops as well as data augmentation routines.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2018"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : MatchingNetwork

__variables__   :

__methods__     :
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# import unittest
import numpy as np

from models import Attn as A
from models import BiLSTM as B
from models import CosineDistance as C
from models import EmbedText as E
from logger.logger import logger

"""
Variable naming convention:

    Categories -> Labels / Classes
    Sample -> [Feature, Categories]
    _hot -> multi-hot format
    x_hat -> test sample
"""


class MatchingNetwork(nn.Module):
    """Builds a matching network, the training and evaluation ops as well as data augmentation routines."""
    def __init__(self, dropout, batch_size=100, num_channels=1, learning_rate=0.001, fce=True, num_classes_per_set=5, num_samples_per_class=1, num_categories=0, input_size=28):
        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.

        :param dropout: A placeholder of type torch.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the images  # TODO
        :param is_training: Flag indicating whether we are training or evaluating
        :param rotate_flag: Flag indicating whether to rotate the images  # TODO
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNNText embeddings)
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        :param num_categories: total number of classes. It changes the text_lstm size of the classifier g with a final FC layer.
        :param image_input: size of the input image. It is needed in case we want to create the last FC classification   # TODO
        """
        super(MatchingNetwork, self).__init__()
        self.batch_size = batch_size
        # self.dropout = dropout
        self.num_classes_per_set = num_classes_per_set  # TODO
        self.num_samples_per_class = num_samples_per_class  # TODO
        # self.learning_rate = learning_rate
        self.fce = fce

        self.g = E.EmbedText(layer_size=64, num_channels=num_channels, num_categories=num_categories, input_size=input_size)
        if self.fce:
            self.lstm = B.BiLSTM(batch_size=self.batch_size, hid_size=32, input_dim=self.g.output_size)
        self.cosine_dis = C.CosineDistance()
        self.attn = A.Attn()

    def forward(self, support_set, support_set_hot, x_hat, x_hat_hot):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.

        :param support_set: A tensor containing the support set images [batch_size, sequence_size, n_channels, 28, 28]
        :param support_set_hot: A tensor containing the support set labels [batch_size, sequence_size, n_classes]
        :param x_hat: A tensor containing the target image (image to produce label for) [batch_size, n_channels, 28, 28]  # TODO
        :param x_hat_hot: A tensor containing the target label [batch_size, 1]  # TODO
        :return: 
        """
        logger.debug("Shapes: (support_set {}, support_set_hot {}, x_hat {}, x_hat_hot{})".format(
              support_set.shape, support_set_hot.shape, x_hat.shape, x_hat_hot.shape))
        # produce embeddings for support set images
        encoded_images = []
        for i in np.arange(support_set.size(1)):  # TODO
            gen_encode = self.g(support_set[:, i, :, :, :])
            encoded_images.append(gen_encode)

        # produce embeddings for target images
        for i in np.arange(x_hat.size(1)):  # TODO
            gen_encode = self.g(x_hat[:, i, :, :, :])
            encoded_images.append(gen_encode)
            outputs = torch.stack(encoded_images)

            if self.fce:
                outputs, hn, cn = self.lstm(outputs)

            # get similarity between support set embeddings and target
            similarities = self.cosine_dis(support_set=outputs[:-1], input_image=outputs[-1])
            similarities = similarities.t()

            # produce predictions for target probabilities
            preds = self.attn(similarities, support_set_y=support_set_hot)

            # calculate accuracy and crossentropy loss
            values, indices = preds.max(1)
            if i == 0:  # TODO
                accuracy = torch.mean((indices.squeeze() == x_hat_hot[:, i]).float())
                crossentropy_loss = F.cross_entropy(preds, x_hat_hot[:, i].long())
            else:
                accuracy = accuracy + torch.mean((indices.squeeze() == x_hat_hot[:, i]).float())
                crossentropy_loss = crossentropy_loss + F.cross_entropy(preds, x_hat_hot[:, i].long())

            # delete the last target image encoding of encoded_images
            encoded_images.pop()

        return accuracy / x_hat.size(1), crossentropy_loss / x_hat.size(1)


# class MatchingNetworkTest(unittest.TestCase):
#     def setUp(self):
#         pass
#
#     def tearDown(self):
#         pass
#
#     def test_accuracy(self):
#         pass


if __name__ == '__main__':
    # unittest.main()
    pass
