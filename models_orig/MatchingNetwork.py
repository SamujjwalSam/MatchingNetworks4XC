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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models_orig import Attn
from models_orig import PairCosineSim
from logger.logger import logger
from models import BiLSTM
from models import EmbedText
from config import configuration as config


class MatchingNetwork(nn.Module):
    """Builds a matching network, the training and evaluation ops as well as data_loader augmentation routines."""
    def __init__(self, num_channels, layer_size, fce=config["model"]["fce"]):
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
        :param classify_count: total number of classes. It changes the text_lstm size of the classifier g with a final FC layer.
        :param sample_input: size of the input sample. It is needed in case we want to create the last FC classification.
        """
        super(MatchingNetwork, self).__init__()
        self.fce = fce

        self.attn = Attn()
        self.cosine_dis = PairCosineSim.PairCosineSim()
        self.g = EmbedText(num_channels, layer_size)
        if self.fce:
            self.lstm = BiLSTM(input_size=self.g.output_size)

    def forward(self, supports_x, supports_hots, hats_x, hats_hots, target_cat_indices, batch_size=config["sampling"]["batch_size"]):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.

        :param target_cat_indices:
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
        # encoded_supports = self.g(supports_x, batch_size=batch_size)
        encoded_supports = []
        for i in np.arange(supports_x.size(1)):
            encoded_support = self.g(supports_x[:, i, :].unsqueeze(1))
            encoded_supports.append(encoded_support.squeeze())
        encoded_supports = torch.stack(encoded_supports)
        # logger.debug("encoded_supports: {}".format(encoded_supports))

        accuracy = 0
        crossentropy_loss = 0
        hats_preds = []
        ## Produce embeddings for target samples
        for i in np.arange(hats_x.size(1)):
            encoded_x_hat = self.g(hats_x[:, i, :].unsqueeze(1), batch_size=batch_size)
            # logger.debug("encoded_x_hat output: {}".format(encoded_x_hat))

            if self.fce:
                encoded_supports, _ = self.lstm(encoded_supports, batch_size=batch_size)
                encoded_x_hat, _ = self.lstm(encoded_x_hat, batch_size=batch_size)
                # logger.debug("FCE encoded_supports: {}".format(encoded_supports))
                # logger.debug("FCE encoded_x_hat: {}".format(encoded_x_hat))

            ## Get similarity between support set embeddings and target
            similarities = self.cosine_dis.forward_orig(supports=encoded_supports, target=encoded_x_hat.squeeze())
            similarities = similarities.t()
            similarities_squeezed = similarities.squeeze(1)
            # logger.debug("similarities: {}".format(similarities))
            # logger.debug("similarities shape: {}".format(similarities.shape))

            ## Produce predictions for target probabilities
            hats_pred = self.attn.forward_orig(similarities_squeezed,support_set_y=supports_hots)  # batch_size x # classes = hats_hots.shape
            # logger.debug("target_cat_indices: {}".format(target_cat_indices))
            # logger.debug("hats_pred output: {}".format(hats_pred))

            ## Calculate accuracy and crossentropy loss
            values, indices = hats_pred.max(1)
            if i == 0:
                accuracy = torch.mean((indices.unsqueeze(1) == hats_hots[:, i].long()).float())
                crossentropy_loss = F.cross_entropy(hats_pred, target_cat_indices[:, i].squeeze().long())
            else:
                accuracy = accuracy + torch.mean((indices.unsqueeze(1) == hats_hots[:, i].long()).float())
                crossentropy_loss = crossentropy_loss + F.cross_entropy(hats_pred, target_cat_indices[:, i].long())
            crossentropy_loss = crossentropy_loss / hats_x.size(1)
            hats_preds.append(hats_pred)
        hats_preds = torch.stack(hats_preds,dim=1)
        logger.info("Accuracy: [{}]".format(accuracy / hats_hots.size(1)))
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

    cls = MatchingNetwork(input_size=4, hid_size=4, classify_count=5, use_cuda=False)
    logger.debug(cls)
    sim = cls.forward(support_set, support_set_hot, x_hat, x_hat_hot, batch_size=4)
    logger.debug(sim)
