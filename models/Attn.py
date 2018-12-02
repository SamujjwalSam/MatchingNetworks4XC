# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6 or above
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Produces pdfs over the support set classes for the target set datapoint.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2018"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : Attn

__variables__   :

__methods__     :
"""

import torch.nn as nn

from logger.logger import logger
# import unittest.
from models import CosineDistance as C


class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, similarities, support_set_y):
        """
        Produces pdfs over the support set classes for the target set datapoint.

        :param similarities: A tensor with cosine similarities of size [sequence_length, batch_size]
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image [sequence_length,  batch_size, num_classes]
        :return: Softmax pdf
        """
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        return preds


# class AttentionalClassifyTest(unittest.TestCase):
#     def setUp(self):
#         pass
#
#     def tearDown(self):
#         pass
#
#     def test_forward(self):
#         pass


if __name__ == '__main__':
    # unittest.main()
    import torch
    a = torch.ones(2,2,3)
    logger.debug(a)
    b = torch.ones(2,3)
    logger.debug(b)
    test_DN = C.CosineDistance()
    sim = test_DN.forward_old(a, b)
    logger.debug(sim.shape)
    logger.debug(type(sim))
    sim = test_DN(a,b)
    logger.debug(sim.shape)
    logger.debug(type(sim))

    sim = torch.rand(2, 4)
    logger.debug("sim: {}".format(sim))
    # y = torch.ones(2, 4, 2)
    y = torch.tensor([[[1., 0.],
                       [0., 0.],
                       [0., 1.],
                       [1., 0.]], [[0., 1.],
                                   [0., 0.],
                                   [1., 1.],
                                   [1., 0.]]])
    logger.debug("y: {}".format(y))
    test_attn = Attn()
    result = test_attn(sim, y)
    logger.debug("result: {}".format(result))
