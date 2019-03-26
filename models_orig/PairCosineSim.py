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
        super(PairCosineSim, self).__init__()

    def forward(self, supports, target):
        """
        Calculates pairwise cosine similarity of support sets with target sample.

        :param supports: The embeddings of the support set samples, tensor of shape [batch_size, sequence_length, input_size]
        :param targets: The embedding of the target sample, tensor of shape [batch_size, input_size] -> [batch_size, sequence_length, input_size]

        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """
        eps = 1e-10
        similarities = []
        # for i in np.arange(supports.size(1)):
        #     support_image = supports[:, i, :]
        for support_image in supports:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
            target_unsqueeze = target.unsqueeze(1)
            support_image_unsqueeze = support_image.unsqueeze(2)
            # target = target.squeeze(1)
            dot_product = target_unsqueeze.bmm(support_image_unsqueeze)
            # dot_product = target.bmm(support_image)
            dot_product = dot_product.squeeze()
            cos_sim = dot_product * support_magnitude
            similarities.append(cos_sim)
        similarities = torch.stack(similarities)
        return similarities


if __name__ == '__main__':
    a1_np = np.array([[[1., 0.4],
                       [1., 1.],
                       [0., 1.5]],
                      [[1., 0.6],
                       [1., 1.],
                       [0., 1.5]]])
    a1_pt = torch.from_numpy(a1_np)
    a2_np = np.array([[[1., 2.],
                       [3., 4.],
                       [5., 6.]],
                      [[1., 7.],
                       [2., 5.],
                       [5., 6.]]])
    a2_pt = torch.from_numpy(a2_np)

    b1_np = np.array([[[1., 0.4],
                       [1., 1.5]],
                      [[1., 0.7],
                       [1., 1.5]]])
    b1_pt = torch.from_numpy(b1_np)
    b2_np = np.array([[[1., 2.],
                       [3., 4.]],
                      [[1., 7.],
                       [5., 6.]]])
    b2_pt = torch.from_numpy(b2_np)

    output = torch.tensor([[0.8103, 1.0000, 0.8793],
                           [0.9804, 0.8793, 1.0000]])

    # a = torch.rand(5, 8, 7)
    # b = torch.rand(5, 2, 7)
    # logger.debug(a)
    # logger.debug(a.shape)
    # logger.debug(b)
    # logger.debug(b.shape)
    test_DN = PairCosineSim()
    sim = test_DN.forward(a1_pt, b1_pt, test=True)
    # logger.debug(sim)
    logger.debug(sim.shape)
    sim = test_DN.forward(a2_pt, b2_pt, test=True)
    # logger.debug(sim)
    logger.debug(sim.shape)
