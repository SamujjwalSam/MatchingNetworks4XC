# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Builds a matching network, the training and evaluation ops as well as data_loader augmentation routines.

__description__ : Builds a matching network, the training and evaluation ops as well as data_loader augmentation routines.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : MatchingNetwork

__variables__   :

__methods__     :
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import Attn
from models import PairCosineSim
from logger.logger import logger
from models import BiLSTM
from models import EmbedText

from config import configuration as config


class MatchingNetwork(nn.Module):
    """Builds a matching network, the training and evaluation ops as well as data_loader augmentation routines."""

    def __init__(self,num_channels: int,layer_size: int,fce: bool = config["model"]["fce"]) -> None:
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
        super(MatchingNetwork,self).__init__()
        self.fce = fce

        self.attn = Attn()
        self.cosine_dis = PairCosineSim.PairCosineSim()
        self.g = EmbedText(num_channels,layer_size)
        if self.fce:
            self.lstm = BiLSTM(input_size=self.g.output_size + 1)

    def forward(self,supports_x: torch.Tensor,supports_hots: torch.Tensor,targets_x: torch.Tensor,
                targets_hots: torch.Tensor,target_cat_indices: torch.Tensor,requires_grad: bool = True,
                batch_size: int = config["sampling"]["batch_size"],
                dropout_external: float = config["model"]["dropout_external"]) -> [torch.Tensor,torch.Tensor]:
        """
        Builds graph for Matching Networks, produces losses and summary statistics.

        :param target_cat_indices:
        :param requires_grad:
        :param dropout_external:
        :param batch_size:
        :param supports_x: A tensor containing the support set samples.
            [batch_size, sequence_size, n_channels, 28]
            torch.Size([32, 25, 1, 28])
        :param supports_hots: A tensor containing the support set labels.
            [batch_size, sequence_size, n_classes]
            torch.Size([32, 25, 5])
        :param targets_x: A tensor containing the target sample (sample to produce label for).
            [batch_size, n_channels, 28]
            torch.Size([32, 5, 1, 28])
        :param targets_hots: A tensor containing the target label.
            [batch_size, 1]
            torch.Size([32, 5])
        :return:
        """
        ## Convert target indices to Pytorch multi-label loss format.
        # target_y_mlml = self.create_mlml_data(target_cat_indices, output_shape=targets_hots.shape)

        logger.debug("targets: \n{}".format(targets_x))
        logger.debug("supports: \n{}".format(supports_x))
        logger.debug(
            "targets X targets: \n{}".format(self.cosine_dis(supports=targets_x,targets=targets_x,normalize=False)))
        logger.debug(
            "supports X supports: \n{}".format(self.cosine_dis(supports=supports_x,targets=supports_x,normalize=False)))
        logger.debug(
            "supports X targets: \n{}".format(self.cosine_dis(supports=supports_x,targets=targets_x,normalize=False)))
        ## Encode supports
        supports_x = self.g(supports_x,dropout_external=dropout_external)
        # logger.debug("supports_x [{}] output: {}".format(supports_x.shape,supports_x))

        ## Encode targets
        # targets_x = self.g(targets_x, dropout_external=dropout_external)
        # logger.debug("targets_x[{}] output: {}".format(targets_x.shape,targets_x))
        # logger.debug("emb_targets: \n{}".format(targets_x))
        logger.debug("emb_supports: \n{}".format(supports_x))

        # logger.debug("emb_targets X emb_targets: \n{}".format(self.cosine_dis(supports=targets_x, targets=targets_x, normalize=False)))
        logger.debug("emb_supports X emb_supports: \n{}".format(
            self.cosine_dis(supports=supports_x,targets=supports_x,normalize=False)))
        logger.debug("emb_supports X emb_targets: \n{}".format(
            self.cosine_dis(supports=supports_x,targets=targets_x,normalize=False)))
        if self.fce:
            # supports_x, _ = self.lstm(supports_x, requires_grad=requires_grad)
            targets_x,_ = self.lstm(targets_x,requires_grad=requires_grad)
            # logger.debug("FCE supports_x: \n{}".format(supports_x))
            logger.debug("FCE targets_x: \n{}".format(targets_x))
            logger.debug("FCE_targets X FCE_targets: \n{}".format(
                self.cosine_dis(supports=targets_x,targets=targets_x,normalize=False)))
            # logger.debug("FCE supports_x X FCE supports_x: \n{}".format(self.cosine_dis(supports=supports_x, targets=supports_x, normalize=False)))

        ## Calculate similarity between encoded supports and targets
        similarities = self.cosine_dis(supports=supports_x,targets=targets_x,normalize=True)

        logger.debug("FCE supports_x X FCE targets_x: \n{}".format(similarities))

        ## Produce predictions for target probabilities. targets_preds.shape = batch_size x # classes
        targets_preds = self.attn(similarities,supports_hots=supports_hots.float())
        # logger.debug("target_cat_indices: {}".format(target_cat_indices))
        # logger.debug("targets_preds output: {}".format(targets_preds))

        loss = 0
        # multilabel_batch_loss = F.multilabel_margin_loss(targets_preds, target_y_mlml.long())

        ## Calculate loss, need to calculate loss for each sample but for whole batch.
        for j in np.arange(targets_preds.size(1)):
            # logger.debug((targets_preds[:, j, :].shape, target_y_mlml.long()[:, j, :].shape))
            # logger.debug(targets_preds[:, j, :])
            # logger.debug(target_y_mlml.long()[:, j, :])
            # logger.debug(targets_preds[:, j, :][0])
            # logger.debug(target_y_mlml.long()[:, j, :][0])
            # loss += F.multilabel_margin_loss(targets_preds[:, j, :], target_y_mlml.long()[:, j, :])
            loss += F.cross_entropy(targets_preds[:,j,:],target_cat_indices[:,j].long())
        multilabel_batch_loss = loss / targets_x.size(1)

        return multilabel_batch_loss,targets_preds

    @staticmethod
    def create_mlml_data(target_cat_indices: list,output_shape: tuple) -> torch.Tensor:
        """
        Generates true labels in proper format for Pytorch Multilabel_Margin_Loss.

        Converts target indices to Pytorch 'multilabel_margin_loss' format. Takes class indices at the beginning and
        rest should be filled with -1.
        Link 1: https://gist.github.com/bartolsthoorn/36c813a4becec1b260392f5353c8b7cc#gistcomment-2742606
        Link 2: https://gist.github.com/bartolsthoorn/36c813a4becec1b260392f5353c8b7cc#gistcomment-2840045

        :param output_shape: Shape of the output = batch_size, target count, # labels.
        :param target_cat_indices: List of categories.
        """
        target_y_mlml = torch.full(output_shape,-1)  ## Createing a tensor filled with -1.
        ## Replacing -1 with label indices at the begining of the tensor row.
        for i in np.arange(target_y_mlml.size(0)):
            # logger.debug(target_cat_indices[i])
            for j in np.arange(target_y_mlml.size(1)):
                # logger.debug(type(target_cat_indices[i][j]))
                # logger.debug((i,j,target_cat_indices[i][j]))
                for k in range(len(target_cat_indices[i][j])):
                    # logger.debug(target_cat_indices[i][j][k])
                    target_y_mlml[i,j,k] = torch.tensor(target_cat_indices[i][j][k],dtype=torch.int32)

        return target_y_mlml.long()


if __name__ == '__main__':
    import torch

    support_set = torch.rand(4,2,4)  # [batch_size, sequence_size, input_size]
    support_set_hot = torch.zeros(4,2,1)  # [batch_size, n_classes]
    x_hat = torch.rand(4,2,4)
    x_hat_hot = torch.zeros(4,2,1)
    logger.debug(support_set_hot)
    logger.debug(x_hat_hot)

    # support_set = torch.tensor([[[1., 0.4],
    #                              [1., 1.]],
    #                             [[1., 0.4],
    #                              [0., 1.5]],
    #                             [[1., 0.4],
    #                              [1., 1.5]]])
    #
    support_set_hot = torch.tensor([[[1.,0.],
                                     [0.,1.]],

                                    [[1.,0.],
                                     [0.,1.]],

                                    [[1.,0.],
                                     [0.,1.]],

                                    [[1.,0.],
                                     [0.,1.]]])
    #
    # x_hat = torch.tensor([[[1., 0.4],
    #                        [0., 1.5]],
    #                       [[1., 0.4],
    #                        [1., 1.5]]])
    #
    x_hat_hot = torch.tensor([[[1.,0.],
                               [0.,1.]],

                              [[1.,0.],
                               [0.,1.]],

                              [[1.,0.],
                               [0.,1.]],

                              [[1.,0.],
                               [0.,1.]]])

    logger.debug(support_set.shape)
    logger.debug(support_set_hot.shape)
    logger.debug(x_hat.shape)
    logger.debug(x_hat_hot.shape)

    cls = MatchingNetwork(input_size=4,hid_size=4,classify_count=5,use_cuda=False)
    logger.debug(cls)
    sim = cls.forward(support_set,support_set_hot,x_hat,x_hat_hot,batch_size=4)
    logger.debug(sim)
