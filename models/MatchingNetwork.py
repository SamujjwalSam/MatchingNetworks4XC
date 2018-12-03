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

# import unittest
import torch.nn as nn
import torch.nn.functional as F

from logger.logger import logger
from models import Attn as A
from models import BiLSTM as B
from models import PairCosineSim as C
from models import EmbedText as E

"""
Variable naming:
    Name used   ->  Meaning
-----------------------------------------
    Categories  ->  Labels / Classes
    Sample      ->  [Feature, Categories]
    *_hot       ->  multi-hot format
    x_hat       ->  test sample
"""


class MatchingNetwork(nn.Module):
    """Builds a matching network, the training and evaluation ops as well as data augmentation routines."""

    def __init__(self, input_size=28, num_channels=1, fce=True, num_classes_per_set=5, num_samples_per_class=1,
                 num_categories=0, dropout=0.2):
        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.

        :param dropout: A placeholder of type torch.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the samples.
        :param is_training: Flag indicating whether we are training or evaluating
        :param rotate_flag: Flag indicating whether to rotate the samples.
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNNText embeddings)
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        :param num_categories: total number of classes. It changes the text_lstm size of the classifier g with a final FC layer.
        :param sample_input: size of the input sample. It is needed in case we want to create the last FC classification.
        """
        super(MatchingNetwork, self).__init__()
        # self.dropout = dropout
        # self.num_classes_per_set = num_classes_per_set
        # self.num_samples_per_class = num_samples_per_class
        # self.learning_rate = learning_rate
        self.fce = fce

        self.g = E.EmbedText(num_layers=1, model_type="lstm", num_channels=num_channels, num_categories=num_categories,
                             input_size=input_size, dropout=dropout)
        if self.fce:
            self.lstm = B.BiLSTM(hid_size=10, input_size=self.g.output_size, dropout=dropout)
        self.cosine_dis = C.PairCosineSim()
        self.attn = A.Attn()

    def forward(self, support_set, support_set_hot, x_hat, x_hat_hot):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.

        :param support_set: A tensor containing the support set samples.
            [batch_size, sequence_size, n_channels, 28]
            torch.Size([32, 25, 1, 28])
        :param support_set_hot: A tensor containing the support set labels.
            [batch_size, sequence_size, n_classes]
            torch.Size([32, 25, 5])
        :param x_hat: A tensor containing the target sample (sample to produce label for).
            [batch_size, n_channels, 28]
            torch.Size([32, 5, 1, 28])
        :param x_hat_hot: A tensor containing the target label.
            [batch_size, 1]
            torch.Size([32, 5])
        :return:
        """
        logger.debug("support_set: {}, support_set_hot: {}, x_hat: {}, x_hat_hot: {}".format(
              support_set.shape, support_set_hot.shape, x_hat.shape, x_hat_hot.shape))
        # produce embeddings for support set samples
        logger.debug(support_set.shape)
        encoded_supports = self.g(support_set, batch_size=32)
        logger.debug(encoded_supports.shape)
        # encoded_supports = []
        # for i in np.arange(support_set.size(1)):
        #     logger.debug(support_set[:, i, :].shape)
        #     gen_encode = self.g(support_set[:, i, :],batch_size=32)
        #     encoded_supports.append(gen_encode)
        # encoded_supports = torch.stack(encoded_supports)

        # produce embeddings for target samples
        encoded_x_hat = self.g(x_hat, batch_size=32)
        logger.debug(encoded_x_hat.shape)
        if self.fce:
            logger.debug(encoded_supports.shape)
            encoded_supports, hn, cn = self.lstm(encoded_supports, batch_size=32)
            encoded_x_hat, hn, cn = self.lstm(encoded_x_hat, batch_size=32)

        logger.debug(encoded_supports.shape)
        # get similarity between support set embeddings and target
        similarities = self.cosine_dis(support_set=encoded_supports, X_hat=encoded_x_hat)
        logger.debug(similarities.shape)
        # similarities = similarities.t()  # TODO: need to transpose?

        # produce predictions for target probabilities
        preds = self.attn(similarities, support_set_y=support_set_hot)

        # calculate accuracy and crossentropy loss
        values, indices = preds.max(1)

        accuracy = torch.mean((indices.squeeze() == x_hat_hot).float())
        crossentropy_loss = F.cross_entropy(preds, x_hat_hot.long())

        # produce embeddings for target samples
        # for i in np.arange(x_hat.size(0)):  # Running loop on batch_size.
        # x_hat = torch.unsqueeze(x_hat,dim=2)
        # encoded_x_hat = self.g(x_hat[i, :, :],batch_size=32)

        # if self.fce:
        #     logger.debug(encoded_supports.shape)
        #     encoded_supports, hn, cn = self.lstm(encoded_supports,batch_size=32)
        #     encoded_x_hat, hn, cn = self.lstm(encoded_x_hat,batch_size=32)
        # logger.debug(encoded_x_hat.shape)
        # encoded_x_hat = torch.squeeze(encoded_x_hat,dim=1)
        # logger.debug(encoded_x_hat.shape)

            # get similarity between support set embeddings and target
        # similarities = self.cosine_dis(support_set=encoded_supports, X_hat=encoded_x_hat)
        # similarities = similarities.t()

            # produce predictions for target probabilities
        # preds = self.attn(similarities, support_set_y=support_set_hot)

            # calculate accuracy and crossentropy loss
        # values, indices = preds.max(1)
        # if i == 0:
        #     accuracy = torch.mean((indices.squeeze() == x_hat_hot[:, i]).float())
        #     crossentropy_loss = F.cross_entropy(preds, x_hat_hot[:, i].long())
        # else:
        #     accuracy = accuracy + torch.mean((indices.squeeze() == x_hat_hot[:, i]).float())
        #     crossentropy_loss = crossentropy_loss + F.cross_entropy(preds, x_hat_hot[:, i].long())

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
    import torch

    support_set = torch.rand(32, 5, 28)  # [batch_size, sequence_size, n_channels, 28]
    support_set_hot = torch.ones(32, 5, 5)  # [batch_size, sequence_size, n_classes]
    x_hat = torch.rand(32, 5, 28)
    x_hat_hot = torch.ones(32, 5)
    # logger.debug(support_set)
    # logger.debug(support_set_hot)
    # logger.debug(x_hat)
    # logger.debug(x_hat_hot)
    cls = MatchingNetwork(input_size=28, num_categories=0)
    logger.debug(cls)
    sim = cls.forward(support_set, support_set_hot, x_hat, x_hat_hot)
    logger.debug(sim)
    # logger.debug(sim.shape)
