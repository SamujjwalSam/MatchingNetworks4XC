# coding=utf-8
# !/usr/bin/python3.6
"""
__synopsis__    : This file implements the method described in: Vinyals, Oriol, et al. "Matching networks for one shot
                  learning." Advances in Neural Information Processing Systems. 2016.

__description__ : This is a Matching Networks implementation by Facebbok research located at: https://github.com/facebookresearch/low-shot-shrink-hallucinate/blob/master/matching_network.py
__project__     : MNXC_2
__author__      : Facebook Research
__version__     : ":  "
__copyright__   : Copyright 2017-present, Facebook, Inc. All rights reserved.
__license__     : License is located at: https://github.com/facebookresearch/low-shot-shrink-hallucinate/blob/master/LICENSE

__classes__     : FullyContextualEmbedding, MatchingNetwork

__variables__   :

__methods__     :
"""

import os,json,argparse
import h5py
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from logger.logger import logger


def encode_lowshot_trainset(model,base_classes,train_file_handle,novel_idx,lowshotn,num_base=100):
    """

    :param model:
    :param base_classes:
    :param train_file_handle:
    :param novel_idx:
    :param lowshotn:
    :param num_base:
    :return:
    """
    all_labels = train_file_handle['all_labels'][...]
    all_feats = train_file_handle['all_feats']

    feats = []
    Y = []
    # for each base class, randomly pick 100 examples
    for i,k in enumerate(base_classes):
        idx = np.where(all_labels == k)[0]
        idx = np.sort(np.random.choice(idx,num_base,replace=False))
        feats.append(all_feats[list(idx)])
        Y_this = np.zeros((num_base,1000))
        Y_this[:,k] = 1
        Y.append(Y_this)

    # next get the novel classes
    sorted_novel_idx = np.sort(novel_idx.reshape(-1))
    novel_feats = all_feats[list(sorted_novel_idx)]
    novel_labels = all_labels[sorted_novel_idx]
    Y_novel = np.zeros((novel_feats.shape[0],1000))
    Y_novel[np.arange(novel_feats.shape[0]),novel_labels] = 1

    num_repeats = int(np.ceil(float(num_base) / float(lowshotn)))
    novel_feats = np.tile(novel_feats,(num_repeats,1))
    Y_novel = np.tile(Y_novel,(num_repeats,1))

    feats.append(novel_feats)
    Y.append(Y_novel)

    feats = np.concatenate(feats,axis=0)
    Y = np.concatenate(Y,axis=0)

    model = model.cuda()
    feats = Variable(torch.Tensor(feats).cuda())
    Y = Variable(torch.Tensor(Y).cuda())

    G,G_norm = model.encode_training_set(feats)
    logger.debug((novel_feats.shape,len(base_classes)))
    return G,G_norm,Y


def perelement_accuracy(scores,label_ind):
    """

    :param scores:
    :param label_ind:
    :return:
    """
    topk_scores,topk_labels = scores.topk(5,1,True,True)
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)),axis=1)
    return top1_correct.astype(float),top5_correct.astype(float)


def run_test(model,G,G_norm,Y,test_file_handle,base_classes,novel_classes,batchsize=128):
    """

    :param model:
    :param G:
    :param G_norm:
    :param Y:
    :param test_file_handle:
    :param base_classes:
    :param novel_classes:
    :param batchsize:
    :return:
    """
    count = test_file_handle['count'][0]
    all_feats = test_file_handle['all_feats']
    all_labels = test_file_handle['all_labels'][:count]
    top1 = None
    top5 = None
    for i in range(0,count,batchsize):
        stop = min(i + batchsize,count)
        F = all_feats[range(i,stop)]
        F = Variable(torch.Tensor(F).cuda())
        L = all_labels[i:stop]

        scores = model.get_logprobs(F,G,G_norm,Y)
        top1_this,top5_this = perelement_accuracy(scores.data,L)

        top1 = top1_this if top1 is None else np.concatenate((top1,top1_this))
        top5 = top5_this if top5 is None else np.concatenate((top5,top5_this))

    is_novel = np.in1d(all_labels,novel_classes)
    is_base = np.in1d(all_labels,base_classes)
    is_either = is_novel | is_base
    top1_novel = np.mean(top1[is_novel])
    top1_base = np.mean(top1[is_base])
    top1_all = np.mean(top1[is_either])
    top5_novel = np.mean(top5[is_novel])
    top5_base = np.mean(top5[is_base])
    top5_all = np.mean(top5[is_either])
    return np.array([top1_novel,top5_novel,top1_base,top5_base,top1_all,top5_all])


def parse_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',required=True,type=int)
    parser.add_argument('--trainfile',required=True,type=str)
    parser.add_argument('--testfile',type=str)
    parser.add_argument('--lowshotmeta',required=True,type=str)
    parser.add_argument('--experimentpath',type=str)
    parser.add_argument('--experimentid',default=1,type=int)
    parser.add_argument('--lowshotn',default=1,type=int)
    parser.add_argument('--testsetup',default=0,type=int)
    parser.add_argument('--modelfile',required=True,type=str)
    parser.add_argument('--K',default=5,type=int)
    parser.add_argument('--outdir',type=str)
    return parser.parse_args()


def main():
    """

    """
    params = parse_args()
    with open(params.lowshotmeta,'r') as f:
        lowshotmeta = json.load(f)

    if params.test:
        with open(params.experimentpath.format(params.experimentid),'r') as f:
            exp = json.load(f)
        novel_idx = np.array(exp)[:,:params.lowshotn]
        if params.testsetup:
            novel_classes = lowshotmeta['novel_classes_2']
            base_classes = lowshotmeta['base_classes_2']
        else:
            novel_classes = lowshotmeta['novel_classes_1']
            base_classes = lowshotmeta['base_classes_1']

        novel_idx = np.sort(novel_idx[novel_classes,:].reshape(-1))
        train_f = h5py.File(params.trainfile,'r')
        test_f = h5py.File(params.testfile,'r')

        featdim = train_f['all_feats'][0].size
        model = MatchingNetwork(featdim,params.K)
        model = model.cuda()
        tmp = torch.load(params.modelfile)
        model.load_state_dict(tmp)

        G,G_norm,Y = encode_lowshot_trainset(model,base_classes,train_f,novel_idx,params.lowshotn)
        accs = run_test(model,G,G_norm,Y,test_f,base_classes,novel_classes)
        modelrootdir = os.path.basename(os.path.dirname(params.trainfile))
        outpath = os.path.join(params.outdir,'MN_' + modelrootdir + '_expid_{:d}_lowshotn_{:d}.json'.format(
            params.experimentid,params.lowshotn))
        with open(outpath,'w') as f:
            json.dump(dict(expid=params.experimentid,lowshotn=params.lowshotn,accs=accs.tolist()),f)
        train_f.close()
        test_f.close()

    else:
        base_classes = lowshotmeta['base_classes_1']
        base_classes.extend(lowshotmeta['base_classes_2'])

        train_f = h5py.File(params.trainfile,'r')

        featdim = train_f['all_feats'][0].size
        model = MatchingNetwork(featdim,params.K)
        model = model.cuda()

        model = train_matching_network(model,train_f,base_classes)
        torch.save(model.state_dict(),params.modelfile)


class FullyContextualEmbedding(nn.Module):
    """

    """

    def __init__(self,feat_dim: int,K: int) -> None:
        super(FullyContextualEmbedding,self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim * 2,feat_dim)
        self.softmax = nn.Softmax()
        self.c_0 = Variable(torch.zeros(1,feat_dim))
        self.feat_dim = feat_dim
        self.K = K

    def forward(self,f: torch.Tensor,G: torch.Tensor) -> torch.Tensor:
        """

        :param f:
        :param G:
        :return:
        """
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose(0,1)
        for k in range(self.K):
            logit_a = h.mm(G_T)
            a = self.softmax(logit_a)
            r = a.mm(G)
            x = torch.cat((f,r),1)

            h,c = self.lstmcell(x,(h,c))
            h = h + f

        return h

    def cuda(self):
        super(FullyContextualEmbedding,self).cuda()
        self.c_0 = self.c_0.cuda()
        return self


class MatchingNetwork(nn.Module):
    def __init__(self,feat_dim,K):
        super(MatchingNetwork,self).__init__()
        self.FCE = FullyContextualEmbedding(feat_dim,K)
        self.G_encoder = nn.LSTM(feat_dim,feat_dim,num_layers=1,batch_first=True,bidirectional=True)
        self.softmax = nn.Softmax()
        self.feat_dim = feat_dim

    def encode_training_set(self,S):
        """

        :param S:
        :return:
        """
        # out_G = self.G_encoder(S.unsqueeze(0))[0]
        out_G = self.G_encoder(S.unsqueeze(0))[0]  ## Ignoring hidden representation.
        out_G = out_G.squeeze(0)
        g1 = out_G[:,:S.size(1)]
        g2 = out_G[:,S.size(1):]
        G = S + g1 + g2
        # G = S + out_G[:,:S.size(1)] + out_G[:,S.size(1):]
        G_pow = G.pow(2)

        G_sum0 = G_pow.sum(0)
        G_pow20 = G_sum0.pow(0.5)
        G_norm0 = G_pow20.expand_as(G)
        G_normalized0 = G.div(G_norm0 + 0.00001)

        G_sum = G_pow.sum(1)
        G_pow2 = G_sum.pow(0.5)
        G_norm = G_pow2.expand_as(G)

        # G_norm = G.pow(2).sum(1).pow(0.5).expand_as(G)
        G_normalized = G.div(G_norm + 0.00001)
        return G,G_normalized

    def get_logprobs(self,f: torch.Tensor,G: torch.Tensor,G_normalized: torch.Tensor,Y_S: torch.Tensor) -> torch.Tensor:
        """

        :param f:
        :param G:
        :param G_normalized:
        :param Y_S:
        :return:
        """
        F = self.FCE(f,G)
        g_n_t = G_normalized.transpose(0,1)
        scores = F.mm(g_n_t)
        # scores = F.mm(G_normalized.transpose(0,1))
        softmax = self.softmax(scores)
        # logprobs = softmax.mm(Y_S).log()
        probs = softmax.mm(Y_S)
        logprobs = probs.log()
        return logprobs

    def forward(self,f,S,Y_S):
        """

        :param f:
        :param S:
        :param Y_S:
        :return:
        """
        G,G_normalized = self.encode_training_set(S)
        logprobs = self.get_logprobs(f,G,G_normalized,Y_S)
        return logprobs

    def cuda(self):
        """

        :return:
        """
        super(MatchingNetwork,self).cuda()
        self.FCE = self.FCE.cuda()
        return self


def train_matching_network(model,file_handle,base_classes,m=389,n=10,initlr=0.1,momentum=0.9,wd=0.001,step_after=20000,
                           niter=60000):
    """ Trains the matching Network.

    :param model:
    :param file_handle:
    :param base_classes:
    :param m:
    :param n:
    :param initlr:
    :param momentum:
    :param wd:
    :param step_after:
    :param niter:
    :return:
    """
    model = model.cuda()
    lr = initlr
    optimizer = torch.optim.SGD(model.parameters(),lr,momentum=momentum,dampening=momentum,weight_decay=wd)

    loss_fn = nn.NLLLoss()
    all_labels = file_handle['all_labels'][...]

    total_loss = 0.0
    loss_count = 0.0
    for it in range(niter):
        optimizer.zero_grad()

        rand_labels = np.random.choice(base_classes,m,replace=False)
        num = np.random.choice(n,m) + 1
        batchsize = int(np.sum(num))

        train_feats = torch.zeros(batchsize,model.feat_dim)
        train_Y = torch.zeros(batchsize,m)
        test_feats = torch.zeros(m,model.feat_dim)
        test_labels = torch.range(0,m - 1)

        count = 0
        for j in range(m):
            idx = np.where(all_labels == rand_labels[j])[0]
            train_idx = np.sort(np.random.choice(idx,num[j],replace=False))
            test_idx = np.random.choice(idx)

            F_tmp = file_handle['all_feats'][list(train_idx)]
            train_feats[count:count + num[j]] = torch.Tensor(F_tmp)
            train_Y[count:count + num[j],j] = 1
            F_tmp = file_handle['all_feats'][test_idx]
            test_feats[j] = torch.Tensor(F_tmp)
            count = count + num[j]

        train_feats = Variable(train_feats.cuda())
        train_Y = Variable(train_Y.cuda())
        test_feats = Variable(test_feats.cuda())
        test_labels = Variable(test_labels.long().cuda())

        logprob = model(test_feats,train_feats,train_Y)
        loss = loss_fn(logprob,test_labels)
        loss.backward()
        optimizer.step()

        if (it + 1) % step_after == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        total_loss = total_loss + loss.data[0]
        loss_count += 1

        if (it + 1) % 1 == 0:
            logger.debug('{:d}:{:f}'.format(it,total_loss / loss_count))
            total_loss = 0.0
            loss_count = 0.0

    return model


if __name__ == '__main__':
    f = torch.rand(1,4)  # [batch_size, sequence_size, input_size]
    S = torch.rand(1,4)
    Y_S = torch.zeros_like(S)

    logger.debug((f,S,Y_S))

    model = MatchingNetwork(feat_dim=4,K=2)
    result = model.forward(f,S,Y_S)
    logger.debug(result)
