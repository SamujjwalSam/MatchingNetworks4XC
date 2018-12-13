# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6 or above
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Builds a matching network, the training and evaluation ops as well as data_loader augmentation routines.
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

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from logger.logger import logger
from models import Attn as A
from models import BiLSTM as B
from models import PairCosineSim as C
from models import EmbedText as E


class MatchingNetwork(nn.Module):
    """Builds a matching network, the training and evaluation ops as well as data_loader augmentation routines."""

    def __init__(self, input_size=28, hid_size=1, fce=True,
                 # num_classes_per_set=5,
                 # num_samples_per_class=1,
                 num_categories=0, dropout=0.2):
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
        # self.dropout = dropout
        # self.num_classes_per_set = num_classes_per_set
        # self.num_samples_per_class = num_samples_per_class
        # self.learning_rate = learning_rate
        self.fce = fce

        self.g = E.EmbedText(num_layers=1, model_type="lstm",
                             num_categories=num_categories,
                             input_size=input_size,
                             hid_size=hid_size,
                             dropout=dropout)
        if self.fce:
            self.lstm = B.BiLSTM(hid_size=hid_size, input_size=self.g.output_size, dropout=dropout)
        self.cosine_dis = C.PairCosineSim()
        self.attn = A.Attn()

    def forward(self, support_set, support_set_hot, x_hats, x_hats_hots, batch_size=64):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.

        :param batch_size:
        :param support_set: A tensor containing the support set samples.
            [batch_size, sequence_size, n_channels, 28]
            torch.Size([32, 25, 1, 28])
        :param support_set_hot: A tensor containing the support set labels.
            [batch_size, sequence_size, n_classes]
            torch.Size([32, 25, 5])
        :param x_hats: A tensor containing the target sample (sample to produce label for).
            [batch_size, n_channels, 28]
            torch.Size([32, 5, 1, 28])
        :param x_hats_hots: A tensor containing the target label.
            [batch_size, 1]
            torch.Size([32, 5])
        :return:
        """
        logger.debug("support_set: {}, support_set_hot: {}, x_hats: {}, x_hats_hots: {}".format(
              support_set.shape, support_set_hot.shape, x_hats.shape, x_hats_hots.shape))
        # produce embeddings for support set samples
        logger.debug(support_set.shape)
        encoded_supports = self.g(support_set, batch_size=batch_size)
        logger.debug(encoded_supports.shape)
        # encoded_supports = []
        # for i in np.arange(support_set.size(1)):
        #     logger.debug(support_set[:, i, :].shape)
        #     gen_encode = self.g(support_set[:, i, :],batch_size=32)
        #     encoded_supports.append(gen_encode)
        # encoded_supports = torch.stack(encoded_supports)

        # produce embeddings for target samples
        encoded_x_hat = self.g(x_hats, batch_size=batch_size)
        logger.debug(encoded_x_hat.shape)
        if self.fce:
            logger.debug(encoded_supports.shape)
            encoded_supports, hn, cn = self.lstm(encoded_supports, batch_size=batch_size)
            encoded_x_hat, hn, cn = self.lstm(encoded_x_hat, batch_size=batch_size)

        logger.debug(encoded_supports.shape)
        # get similarity between support set embeddings and target
        similarities = self.cosine_dis(support_set=encoded_supports, X_hats=encoded_x_hat)
        logger.debug(similarities.shape)
        # similarities = similarities.t()  # TODO: need to transpose?

        # produce predictions for target probabilities
        x_hats_preds = self.attn(similarities, support_set_y=support_set_hot)  # batch_size x Multi-hot vector size == x_hats_hots.shape
        logger.debug(x_hats_preds)
        logger.debug(x_hats_preds.shape)

        # assert x_hats_preds.shape == x_hats_hots.shape, "x_hats_preds.shape ({}) == ({}) x_hats_hots.shape".format(x_hats_preds.shape, x_hats_hots.shape)

        # calculate accuracy and crossentropy loss
        values, indices = x_hats_preds.max(1)
        # logger.debug(values)
        # logger.debug(values.shape)
        # logger.debug(indices)
        # logger.debug(indices.shape)
        # logger.debug(indices.squeeze())
        # logger.debug(indices.squeeze().shape)

        # accuracy = torch.mean((indices.squeeze() == x_hats_hots).float())
        # logger.info("x_hats_preds.shape ({}) == ({}) x_hats_hots.shape".format(x_hats_preds, x_hats_hots))
        logger.info("x_hats_preds.shape ({}) == ({}) x_hats_hots.shape".format(x_hats_preds.shape, x_hats_hots.shape))

        # Need to calculate loss for each sample but for whole batch.
        crossentropy_loss_x_hats = 0
        for j in np.arange(x_hats_preds.size(1)):
            crossentropy_loss_x_hats += F.multilabel_margin_loss(x_hats_preds[:,j,:], x_hats_hots.long()[:, j, :])
        crossentropy_loss = crossentropy_loss_x_hats / x_hats.size(1)

        return crossentropy_loss


if __name__ == '__main__':
    import torch

    support_set = torch.rand(4, 5, 4)  # [batch_size, sequence_size, input_size]
    support_set_hot = torch.ones(4, 5)  # [batch_size, n_classes]
    x_hat = torch.rand(4, 5, 4)
    x_hat_hot = torch.ones(4, 5)
    # logger.debug(support_set)
    # logger.debug(support_set_hot)
    # logger.debug(x_hat)
    # logger.debug(x_hat_hot)
    cls = MatchingNetwork(input_size=4, hid_size=4, num_categories=5)
    logger.debug(cls)
    sim = cls.forward(support_set, support_set_hot, x_hat, x_hat_hot, batch_size=4)
    logger.debug(sim)
    # logger.debug(sim.shape)
