# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Calculates cosine similarity of support sets with target sample.

__description__ : Calculates cosine similarity of support sets with target sample.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : PairCosineSim

__variables__   :

__methods__     :
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from logger.logger import logger


class PairCosineSim(nn.Module):
    def __init__(self):
        super(PairCosineSim,self).__init__()

    def forward(self,supports: torch.Tensor,targets: torch.Tensor,normalize: bool = True,
                test: bool = False) -> torch.Tensor:
        """
        Calculates pairwise cosine similarity of support sets with target sample.

        :param test: Flag to denote if checking with sklearn.cosine_similarity is needed.
        :param normalize: Whether to normalize the matrix to range: (0,1) from (-1,+1)
        :param supports: The embeddings of the support set samples, tensor of shape [batch_size, sequence_length, input_size]
        :param targets: The embedding of the target sample, tensor of shape [batch_size, input_size] -> [batch_size, sequence_length, input_size]

        :return: Tensor with cosine similarities of shape [batch_size, target_size, support_size]
        """
        eps = 1e-10
        batch_targets_similarities = []
        if test:
            targets_detached = targets.clone().detach()  ## Creating clone() and detaching from graph.
            supports_detached = supports.clone().detach()
        for i in np.arange(targets.size(0)):
            targets_similarities = []
            for j in np.arange(targets.size(1)):
                target_similarities = F.cosine_similarity(targets[i,j,:].unsqueeze(0),supports[i,:,:],eps=eps)
                targets_similarities.append(target_similarities)
            batch_x_hat_similarities = torch.stack(targets_similarities)

            if test:
                logger.debug("Computed sim: {}".format(batch_x_hat_similarities))
                sim = cosine_similarity(targets_detached[i,:,:].numpy(),supports_detached[i,:,:].numpy())
                logger.debug("sklearn sim: {}".format(sim))
            batch_targets_similarities.append(batch_x_hat_similarities)
        batch_targets_similarities = torch.stack(batch_targets_similarities)
        if normalize:
            batch_targets_similarities = torch.add(batch_targets_similarities,1)
            batch_targets_similarities = torch.mul(batch_targets_similarities,0.5)
        return batch_targets_similarities

    def flatten_except_batchdim(self,tensor_data,batch_dim=0):
        """
        Flattens a tensor except on [batch_dim] dimension.

        :param tensor_data: [batch_size, sequence_size, input_size]
        :param batch_dim: Dimension of batch, Default: 0 if batch_first = Ture, else 1 for Pytorch tensors.
        :return: [batch_size, (sequence_size x input_size)]
        """
        if len(tensor_data.shape) == 2:
            logger.info("Flattening 2D tensor to (1, dim), [batch_dim] not used.")
            tensor_data_flat = tensor_data.contiguous().view(1,tensor_data.numel())
        elif len(tensor_data.shape) == 3:
            logger.info("Flattening 3D tensor to 2D except dim: [batch_dim={}].".format(batch_dim))
            logger.info("tensor_data.shape: [{}].".format(tensor_data.shape))
            tensor_data_flat = tensor_data.contiguous().view(tensor_data.shape[batch_dim],-1)
        else:
            logger.warn("Tensor shape not supported. Got: [{}].".format(tensor_data.shape))
            raise NotImplementedError
        return tensor_data_flat

    @staticmethod
    def cosine_sim_2d(tensor1,tensor2,dim=1):
        """
        Calculates cosine similarity between two 2D tensors of same shape. [Batch_size, input_size]

        NOTE: for more than 2D tensors, flatten on one dimension.
        :param tensor1:
        :param tensor2:
        :param dim: Axis for norm calculation.
        :return:
        """
        assert tensor1.shape == tensor2.shape,"Shape of all tensors should be same."
        tensor1_norm = tensor1 / tensor1.norm(dim=dim)[:,None]
        tensor2_norm = tensor2 / tensor2.norm(dim=dim)[:,None]
        cosine_sim = torch.mm(tensor1_norm,tensor2_norm.transpose(0,1))
        logger.debug(cosine_sim.shape)

        return cosine_sim


if __name__ == '__main__':
    a1_np = np.array([[[1.,0.4],
                       [1.,1.],
                       [0.,1.5]],
                      [[1.,0.6],
                       [1.,1.],
                       [0.,1.5]]])
    a1_pt = torch.from_numpy(a1_np)
    a2_np = np.array([[[1.,2.],
                       [3.,4.],
                       [5.,6.]],
                      [[1.,7.],
                       [2.,5.],
                       [5.,6.]]])
    a2_pt = torch.from_numpy(a2_np)

    b1_np = np.array([[[1.,0.4],
                       [1.,1.5]],
                      [[1.,0.7],
                       [1.,1.5]]])
    b1_pt = torch.from_numpy(b1_np)
    b2_np = np.array([[[1.,2.],
                       [3.,4.]],
                      [[1.,7.],
                       [5.,6.]]])
    b2_pt = torch.from_numpy(b2_np)

    output = torch.tensor([[0.8103,1.0000,0.8793],
                           [0.9804,0.8793,1.0000]])

    # a = torch.rand(5, 8, 7)
    # b = torch.rand(5, 2, 7)
    # logger.debug(a)
    # logger.debug(a.shape)
    # logger.debug(b)
    # logger.debug(b.shape)
    test_DN = PairCosineSim()
    sim = test_DN.forward(a1_pt,b1_pt,test=True)
    # logger.debug(sim)
    logger.debug(sim.shape)
    sim = test_DN.forward(a2_pt,b2_pt,test=True)
    # logger.debug(sim)
    logger.debug(sim.shape)
