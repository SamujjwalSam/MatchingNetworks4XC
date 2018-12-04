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
from models import PairCosineSim as C


class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, similarities, support_set_y, dim=1):
        """
        Produces pdfs over the support set classes for the target samples.

        :param dim: Dimension along which Softmax will be computed (so every slice along dim will sum to 1).
        :param similarities: A tensor with cosine similarities of size [batch_size, sequence_length]
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image [batch_size, sequence_length,  num_classes]
        :return: Softmax pdf
        """
        softmax = nn.Softmax(dim=dim)
        softmax_similarities = softmax(similarities)
        # logger.debug(softmax_similarities)
        # logger.debug(softmax_similarities.shape)
        # logger.debug(support_set_y.shape)
        preds = softmax_similarities.mm(support_set_y)
        # logger.debug(preds)
        # logger.debug(preds.shape)
        return preds


if __name__ == '__main__':
    import torch

    a = torch.tensor([[[1., 0.4],
                       [1., 1.]],
                      [[1., 0.4],
                       [0., 1.5]],
                      [[1., 0.4],
                       [1., 1.5]]])

    b = torch.tensor([[[1., 0.4],
                       [0., 1.5]],
                      [[1., 0.4],
                       [1., 1.5]]])
    # a = torch.ones(2,2,3)
    logger.debug(a)
    logger.debug(a.shape)
    # b = torch.ones(2,3)
    logger.debug(b)
    logger.debug(b.shape)
    test_DN = C.PairCosineSim()
    sim = test_DN(a, b)
    logger.debug(sim.shape)
    logger.debug("sim: {}".format(sim))

    # y = torch.ones(3, 2)
    y = torch.tensor([[1., 0.],
                      [0., 1.],
                      [1., 0.]])
    logger.debug("y.shape: {}".format(y.shape))
    test_attn = Attn()
    result = test_attn(sim, y)
    logger.info("result: {}".format(result))
