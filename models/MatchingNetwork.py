# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Builds a matching network, the training and evaluation ops as well as data_loader augmentation routines.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : MatchingNetwork

__variables__   :

__methods__     :
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from logger.logger import logger
from models import Attn
from models import BiLSTM
from models import PairCosineSim
from models import EmbedText


class MatchingNetwork(nn.Module):
    """Builds a matching network, the training and evaluation ops as well as data_loader augmentation routines."""

    def __init__(self, input_size=28, hid_size=1, fce=True, use_cuda=True, num_categories=0, dropout=0.2):
        """
        Builds a matching network, the training and evaluation ops as well as data_loader augmentation routines.

        :param dropout: A placeholder of type torch.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the samples.
        :param is_training: Flag indicating whether we are training or evaluating
        :param rotate_flag: Flag indicating whether to rotate the samples.
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNNText embeddings)
        # :param num_classes_per_set: Integer indicating the number of classes per set
        # :param num_samples_per_class: Integer indicating the number of samples per class
        :param num_categories: total number of classes. It changes the text_lstm size of the classifier g with a final FC layer.
        :param sample_input: size of the input sample. It is needed in case we want to create the last FC classification.
        """
        super(MatchingNetwork, self).__init__()
        self.fce = fce

        self.attn = Attn()
        self.cosine_dis = PairCosineSim.PairCosineSim()
        self.g = EmbedText(num_layers=1,
                                     model_type="lstm",
                                     num_categories=num_categories,
                                     input_size=input_size,
                                     hid_size=hid_size,
                                     dropout=dropout,
                                     use_cuda=use_cuda)
        if self.fce:
            self.lstm = BiLSTM(hid_size=hid_size,
                                      input_size=self.g.output_size,
                                      dropout=dropout,
                                      use_cuda=use_cuda)

    def forward(self, supports_x, supports_hots, hats_x, hats_hots, batch_size=64):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.

        :param batch_size:
        :param supports_x: A tensor containing the support set samples.
            [batch_size, sequence_size, n_channels, 28]
            torch.Size([32, 25, 1, 28])
        :param supports_hots: A tensor containing the support set labels.
            [batch_size, sequence_size, n_classes]
            torch.Size([32, 25, 5])
        :param hats_x: A tensor containing the target sample (sample to produce label for).
            [batch_size, n_channels, 28]
            torch.Size([32, 5, 1, 28])
        :param hats_hots: A tensor containing the target label.
            [batch_size, 1]
            torch.Size([32, 5])
        :return:
        """
        encoded_supports = self.g(supports_x, batch_size=batch_size)

        # produce embeddings for target samples
        encoded_x_hat = self.g(hats_x, batch_size=batch_size)
        if self.fce:
            encoded_supports, hn, cn = self.lstm(encoded_supports, batch_size=batch_size)
            encoded_x_hat, hn, cn = self.lstm(encoded_x_hat, batch_size=batch_size)

        # get similarity between support set embeddings and target
        similarities = self.cosine_dis(support_sets=encoded_supports, X_hats=encoded_x_hat)
        # similarities = similarities.t()  # TODO: need to transpose?

        # produce predictions for target probabilities
        hats_preds = self.attn(similarities,
                               support_set_y=supports_hots)  # batch_size x # classes = hats_hots.shape

        assert hats_preds.shape == hats_hots.shape, "hats_preds.shape ({}) == ({}) hats_hots.shape" \
            .format(hats_preds.shape, hats_hots.shape)

        # calculate accuracy and crossentropy loss
        # values, indices = hats_preds.max(1)
        # accuracy = torch.mean((indices.squeeze() == hats_hots).float())

        # Need to calculate loss for each sample but for whole batch.
        crossentropy_loss_x_hats = 0
        for j in np.arange(hats_preds.size(1)):
            crossentropy_loss_x_hats += F.binary_cross_entropy_with_logits(hats_hots[:, j, :], hats_preds[:, j, :])
            # crossentropy_loss_x_hats += F.cross_entropy(hats_preds[:, j, :], hats_hots.long()[:, j, :])
        crossentropy_loss = crossentropy_loss_x_hats / hats_x.size(1)

        return crossentropy_loss, hats_preds


if __name__ == '__main__':
    import torch

    support_set = torch.rand(4, 2, 4)  # [batch_size, sequence_size, input_size]
    support_set_hot = torch.zeros(4, 2, 1)  # [batch_size, n_classes]
    x_hat = torch.rand(4, 2, 4)
    x_hat_hot = torch.zeros(4, 2, 1)
    logger.debug(support_set_hot)
    logger.debug(x_hat_hot)

    # support_set = torch.tensor([[[1., 0.4],
    #                              [1., 1.]],
    #                             [[1., 0.4],
    #                              [0., 1.5]],
    #                             [[1., 0.4],
    #                              [1., 1.5]]])
    #
    support_set_hot = torch.tensor([[[1., 0.],
                                     [0., 1.]],

                                    [[1., 0.],
                                     [0., 1.]],

                                    [[1., 0.],
                                     [0., 1.]],

                                    [[1., 0.],
                                     [0., 1.]]])
    #
    # x_hat = torch.tensor([[[1., 0.4],
    #                        [0., 1.5]],
    #                       [[1., 0.4],
    #                        [1., 1.5]]])
    #
    x_hat_hot = torch.tensor([[[1., 0.],
                               [0., 1.]],

                              [[1., 0.],
                               [0., 1.]],

                              [[1., 0.],
                               [0., 1.]],

                              [[1., 0.],
                               [0., 1.]]])

    logger.debug(support_set.shape)
    logger.debug(support_set_hot.shape)
    logger.debug(x_hat.shape)
    logger.debug(x_hat_hot.shape)

    cls = MatchingNetwork(input_size=4, hid_size=4, num_categories=5, use_cuda=False)
    logger.debug(cls)
    sim = cls.forward(support_set, support_set_hot, x_hat, x_hat_hot, batch_size=4)
    logger.debug(sim)
