# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Calculates cosine similarity of support sets with target sample.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : PairCosineSim

__variables__   :

__methods__     :
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger.logger import logger


class PairCosineSim(nn.Module):
    def __init__(self):
        super(PairCosineSim, self).__init__()

    def flatten_except_batchdim(self, tensor_data, batch_dim=0):
        """
        Flattens a tensor except on [batch_dim] dimension.

        :param tensor_data: [batch_size, sequence_size, input_size]
        :param batch_dim: Dimension of batch, Default: 0 if batch_first = Ture, else 1 for Pytorch tensors.
        :return: [batch_size, (sequence_size x input_size)]
        """
        if len(tensor_data.shape) == 2:
            logger.debug("Flattening 2D tensor to (1, dim), [batch_dim] not used.")
            tensor_data_flat = tensor_data.contiguous().view(1, tensor_data.numel())
        elif len(tensor_data.shape) == 3:
            logger.debug("Flattening 3D tensor to 2D except dim: [batch_dim={}].".format(batch_dim))
            logger.debug("tensor_data.shape: [{}].".format(tensor_data.shape))
            tensor_data_flat = tensor_data.contiguous().view(tensor_data.shape[batch_dim], -1)
            # tensor_data_flat = tensor_data.view(tensor_data.shape[batch_dim], -1)
        else:
            logger.warn("Tensor shape not supported. Got: [{}].".format(tensor_data.shape))
            raise NotImplementedError
        return tensor_data_flat

    def cosine_sim_2d(self, tensor1, tensor2, dim=1):
        """
        Calculates cosine similarity between two 2D tensors of same shape. [Batch_size, input_size]

        NOTE: for more than 2D tensors, flatten on one dimension.
        :param tensor1:
        :param tensor2:
        :param dim: Axis for norm calculation.
        :return:
        """
        assert tensor1.shape == tensor2.shape, "Shape of all tensors should be same."
        tensor1_norm = tensor1 / tensor1.norm(dim=dim)[:, None]
        tensor2_norm = tensor2 / tensor2.norm(dim=dim)[:, None]
        cosine_sim = torch.mm(tensor1_norm, tensor2_norm.transpose(0, 1))
        logger.debug(cosine_sim.shape)

        return cosine_sim

    def forward(self, support_set, X_hats, normalize=False, dim=0):
        """
        Calculates pairwise cosine similarity of support sets with target sample.

        :param dim:
        :param normalize: Whether to normalize the matrix to range: (0,1) from (-1,+1)
        :param support_set: The embeddings of the support set samples, tensor of shape [batch_size, sequence_length, input_size]
        :param X_hats: The embedding of the target sample, tensor of shape [batch_size, input_size] -> [batch_size, sequence_length, input_size]
        :return: Tensor with cosine similarities of shape [batch_size, target_size, support_size]
        """
        eps = 1e-10
        batch_x_hats_similarities = []
        # logger.debug((support_set.shape,X_hats.shape))
        for i in np.arange(X_hats.size(0)):
            x_hats_similarities = []
            for j in np.arange(X_hats.size(1)):
                # logger.debug((X_hats[i,j,:], support_set[i,:,:]))
                # logger.debug((X_hats[i,j,:].shape, support_set[i,:,:].shape))
                x_hat_similarities = F.cosine_similarity(X_hats[i,j,:].unsqueeze(0), support_set[i,:,:],eps=eps)
                # logger.debug("x_hat_similarities: {}".format(x_hat_similarities))
                # logger.debug("x_hat_similarities.shape: {}".format(x_hat_similarities.shape))
                x_hats_similarities.append(x_hat_similarities)
            batch_x_hat_similarities = torch.stack(x_hats_similarities)
            # logger.debug("batch_x_hat_similarities.shape: {}".format(batch_x_hat_similarities.shape))
            batch_x_hats_similarities.append(batch_x_hat_similarities)
        # logger.debug("batch_x_hats_similarities: {}".format(batch_x_hats_similarities))
        batch_x_hats_similarities = torch.stack(batch_x_hats_similarities)
        # logger.debug("batch_x_hats_similarities.shape: {}".format(batch_x_hats_similarities.shape))
        if normalize:
            batch_x_hats_similarities = torch.add(batch_x_hats_similarities,1)
            batch_x_hats_similarities = torch.mul(batch_x_hats_similarities,0.5)
        # logger.debug(batch_x_hats_similarities)
        # logger.debug(batch_x_hats_similarities.shape)
        return batch_x_hats_similarities


if __name__ == '__main__':
    a = torch.tensor([[[1., 0.4],
                       [1., 1.],
                       [0., 1.5]],
                      [[1., 0.6],
                       [1., 1.],
                       [0., 1.5]]])

    b = torch.tensor([[[1., 0.4],
                       [1., 1.5]],
                      [[1., 0.7],
                       [1., 1.5]]])

    output = torch.tensor([[0.8103, 1.0000, 0.8793],
                           [0.9804, 0.8793, 1.0000]])

    a = torch.rand(5, 8, 7)
    b = torch.rand(5, 2, 7)
    # logger.debug(a)
    logger.debug(a.shape)
    # logger.debug(b)
    logger.debug(b.shape)
    test_DN = PairCosineSim()
    sim = test_DN.forward(a, b)
    logger.debug(sim)
    logger.debug(sim.shape)
