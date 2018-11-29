# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6 or above
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Calculates cosine similarity of support sets with target sample.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2018"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : CosineDistance

__variables__   :

__methods__     :
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# import unittest
from logger.logger import logger


class CosineDistance(nn.Module):
    def __init__(self):
        super(CosineDistance, self).__init__()

    def forward_orig(self, support_set, input_image):
        """
        Calculates cosine similarity of support sets with target sample.

        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_magnitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities

    def forward(self, support_set, X_hat, normalize=True):
        """
        Calculates cosine similarity of support sets with target sample.

        :param normalize: Whether to normalize the matrix to range: (0,1) from (-1,+1)
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param X_hat: The embedding of the target image, tensor of shape [batch_size, 64]
        :return: Softmax pdf. Tensor with cosine similarities of shape [batch_size, sequence_length]
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            cosine_similarity = F.cosine_similarity(support_image, X_hat, eps=eps)
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        # logger.debug(similarities)
        # logger.debug(type(similarities))
        if normalize:
            similarities = torch.add(similarities,1)
            similarities = torch.mul(similarities,0.5)
        # logger.debug(similarities)
        # logger.debug(type(similarities))
        return similarities


# class DistanceNetworkTest(unittest.TestCase):
#     def setUp(self):
#         pass
#
#     def tearDown(self):
#         pass
#
#     def test_forward(self):
#         pass


if __name__ == '__main__':
    a = torch.ones(2,2,3)
    logger.debug(a)
    b = torch.ones(2,3)
    logger.debug(b)
    test_DN = CosineDistance()
    sim = test_DN.forward(a,b)
    logger.debug(sim.shape)
    logger.debug(type(sim))
    logger.debug(sim)
    sim = test_DN.builtin_cosine(a,b)
    logger.debug(sim.shape)
    logger.debug(type(sim))
    logger.debug(sim)
    # unittest.main()
