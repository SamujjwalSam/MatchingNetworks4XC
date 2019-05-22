# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Builds the experiment using matching network, the training and evaluation ops as well as data_loader augmentation routines.

__description__ : Builds the experiment using matching network, the training and evaluation ops as well as data_loader augmentation routines.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     : Run_Network

__variables__   :

__methods__     :
"""

from os.path import join
import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from logger.logger import logger
from metrics.metrics import Metrics
from models.MatchingNetwork import MatchingNetwork
from config import configuration as config


# import torch.onnx
# import hiddenlayer as hl
# from torchviz import make_dot


class Run_Network:
    """
    Initializes an Run_Network object. The Run_Network object takes care of setting up our experiment
    and provides helper functions such as training and validating to simplify out training
    and evaluation procedures.
    """

    def __init__(self,data_formatter,use_cuda: bool = config["model"]["use_cuda"],
                 batch_size: int = config["sampling"]["batch_size"],
                 lr_decay: int = config["model"]["optimizer"]["lr_decay"],
                 lr: float = config["model"]["optimizer"]["learning_rate"]) -> None:
        """Builds the Matching Network with all required parameters.

        Provides helper functions such as training() and validating() to simplify out training
        and evaluation procedures.

        :param data_formatter: A data_loader provider class
        :param targets_per_category: An integer indicating the number of targets per categories.
        :param vectorizer: How to generate feature vectors.
        :param categories_per_batch: An integer indicating the number of categories per batch.
        :param supports_per_category:
        :param hid_size: Size of the output of BiLSTM.
        :param input_size: Size of sample vectors.
        :param classify_count: Number of categories. Not Required. Always 0.
        :param weight_decay:Weight decay.
        :param dropout: Value for dropout, range: [0,1]. Default: 0.2
        :param optim: Optimiser to use.
        :param lr_decay: Learning rate decay.
        :param lr: Learning rate.
        :param batch_size: The experiment batch size.
        :param fce: Whether to use full context embeddings or not.

        :return: a matching_network object, along with the losses, the training ops and the init op.
        """
        self.use_cuda = use_cuda
        self.data_formatter = data_formatter
        self.dataset_name = self.data_formatter.dataset_name
        self.dataset_dir = self.data_formatter.dataset_dir
        self.data_formatter.prepare_data(load_type='train')  # Loading appropriate data

        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.total_train_iter = 0
        self.test_metrics = Metrics()
        layer_size = config["sampling"]["categories_per_batch"] * config["sampling"]["supports_per_category"]
        self.match_net = MatchingNetwork(layer_size=layer_size, num_channels=layer_size)
        logger.info("Matching Network Summary: \n{}".format(self.match_net))

        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available and self.use_cuda:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(seed=0)
            self.match_net.cuda()

    def training(self, num_train_epoch, view_grads=config["model"]["view_grads"],
                 view_train_precision=config["model"]["view_train_precision"]):
        """
        Runs one training epoch.

        :param total_epoch:
        :param view_train_precision:
        :param view_grads: Flag to denote if gradients are to be printed.
        :param num_train_epoch: Number of batches to train.
        :return: mean_training_multilabel_margin_loss.
        """
        total_loss = 0.
        # self.data_formatter.prepare_data(load_type='train')  # Loading appropriate data
        optimizer = self.__create_optimizer(self.match_net, self.lr)  # Creating the optimizer

        with tqdm.tqdm(total=num_train_epoch) as pbar:
            for i in range(num_train_epoch):  # 1 train epoch
                logger.info("Total EPOCHS: [{}]".format(self.total_train_iter))
                x_supports, y_support_hots, x_hats, y_hats_hots, target_cat_indices = \
                    self.data_formatter.get_batches()
                x_supports = Variable(torch.from_numpy(x_supports), requires_grad=True).float()
                y_support_hots = Variable(torch.from_numpy(y_support_hots), requires_grad=False).float()
                x_hats = Variable(torch.from_numpy(x_hats), requires_grad=True).float()
                y_hats_hots = Variable(torch.from_numpy(y_hats_hots), requires_grad=False).float()
                target_cat_indices = Variable(torch.from_numpy(target_cat_indices), requires_grad=False).float()

                ## Print Model Summary:
                # make_dot(loss, self.match_net)
                # hl.build_graph(self.match_net, args=(x_supports, y_support_hots, x_hats, y_hats_hots, target_cat_indices))

                if self.cuda_available and self.use_cuda:
                    loss, targets_preds = self.match_net(x_supports.cuda(), y_support_hots.cuda(), x_hats.cuda(),
                                                         y_hats_hots.cuda(), target_cat_indices,
                                                         batch_size=self.batch_size)
                else:
                    loss, targets_preds = self.match_net(x_supports, y_support_hots, x_hats, y_hats_hots,
                                                         target_cat_indices, batch_size=self.batch_size)

                ## Before the backward pass, use the optimizer object to zero all of the gradients for the variables
                ## it will update (which are the learnable weights of the model)
                optimizer.zero_grad()

                ## Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                ## Print Weights and Gradients:
                if view_grads and self.total_train_iter % 5 == 0:
                    logger.debug(self.match_net.named_modules())
                    for name, param in self.match_net.named_parameters():
                        logger.debug((name, param.data.shape, param.grad.shape))
                    # logger.debug(self.match_net.state_dict())

                ## Calling the step function on an Optimizer makes an update to its parameters
                optimizer.step()

                ## Update the optimizer learning rate
                self.__adjust_learning_rate(optimizer)

                if view_train_precision:
                    logger.info("TRAIN Precisions:")
                    precision_1 = self.test_metrics.precision_k_hot(y_hats_hots, targets_preds, k=1)
                    logger.info("Precision @ 1: {}".format(precision_1))
                    precision_3 = self.test_metrics.precision_k_hot(y_hats_hots, targets_preds, k=3)
                    logger.info("Precision @ 3: {}".format(precision_3))
                    precision_5 = self.test_metrics.precision_k_hot(y_hats_hots, targets_preds, k=5)
                    logger.info("Precision @ 5: {}".format(precision_5))

                iter_out = "TRAIN Loss: {}".format(loss.item())
                logger.info(iter_out)

                pbar.set_description(iter_out)
                print('\n')
                pbar.update(1)
                total_loss += loss.item()

                # self.total_train_iter += 1
                # if self.total_train_iter % 2000 == 0:
                #     self.lr /= 2
                #     logger.debug("change learning rate: [{}]".format(self.lr))

        total_loss = total_loss / num_train_epoch
        return total_loss

    def validating(self, epoch_count, num_val_epoch=1):
        """
        Runs one validation epoch.

        :param epoch_count: Number of epoch running now.
        :param num_val_epoch: Number of batches to train on
        :return: mean_validation_categorical_crossentropy_loss
        """
        total_val_loss = 0.
        total_p1 = 0.
        total_p3 = 0.
        total_p5 = 0.
        self.data_formatter.prepare_data(load_type='val')

        with tqdm.tqdm(total=num_val_epoch) as pbar:
            # with torch.no_grad():
            for i in range(num_val_epoch):  # 1 validation epoch
                x_supports, y_support_hots, x_targets, y_target_hots, target_cat_indices = \
                    self.data_formatter.get_batches(val=True)
                logger.info("Shapes: x_supports [{}], y_support_hots [{}], x_targets [{}], y_target_hots [{}]"
                            .format(x_supports.shape, y_support_hots.shape, x_targets.shape, y_target_hots.shape))

                x_supports = Variable(torch.from_numpy(x_supports), requires_grad=False).float()
                y_support_hots = Variable(torch.from_numpy(y_support_hots), requires_grad=False).float()
                x_targets = Variable(torch.from_numpy(x_targets), requires_grad=False).float()
                y_target_hots = Variable(torch.from_numpy(y_target_hots), requires_grad=False).float()
                target_cat_indices = Variable(torch.from_numpy(target_cat_indices), requires_grad=False).float()

                if self.cuda_available and self.use_cuda:
                    loss, targets_preds, encoded_x_hat = self.match_net(x_supports.cuda(), y_support_hots.cuda(),
                                                                        x_targets.cuda(), y_target_hots.cuda(),
                                                                        batch_size=self.batch_size,
                                                                        requires_grad=False, print_accuracy=True)
                else:
                    loss, targets_preds = self.match_net(x_supports, y_support_hots, x_targets,
                                                         y_target_hots, target_cat_indices, requires_grad=False,
                                                         batch_size=self.batch_size)

                logger.debug("Saving encoded_x_hat named: [{}] at: [{}]".format(self.dataset_name
                                                                                + "_val_encoded_x_hat_" + str(
                    epoch_count), join(self.dataset_dir, self.dataset_name)))
                logger.debug("VALIDATION epoch_count: [{}]".format(epoch_count))

                iter_out = "VALIDATION Loss: {}".format(loss.item())
                logger.info(iter_out)

                logger.info("VALIDATION Precisions:")
                precision_1 = self.test_metrics.precision_k_hot(y_target_hots, targets_preds, k=1)
                logger.info("Precision @ 1: {}".format(precision_1))
                precision_3 = self.test_metrics.precision_k_hot(y_target_hots, targets_preds, k=3)
                logger.info("Precision @ 3: {}".format(precision_3))
                precision_5 = self.test_metrics.precision_k_hot(y_target_hots, targets_preds, k=5)
                logger.info("Precision @ 5: {}".format(precision_5))

                pbar.set_description(iter_out)
                print('\n')
                pbar.update(1)

                total_val_loss += loss.item()
                total_p1 += precision_1
                total_p3 += precision_3
                total_p5 += precision_5
            total_val_loss /= num_val_epoch
            total_p1 /= num_val_epoch
            total_p3 /= num_val_epoch
            total_p5 /= num_val_epoch

        return total_val_loss, total_p1, total_p3, total_p5

    def testing(self, total_test_batches=1):
        """
        Runs one testing epoch.

        :param batch_size:
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss
        """
        total_test_loss = 0.
        total_p1 = 0.
        total_p3 = 0.
        total_p5 = 0.
        # self.data_formatter.prepare_data(load_type='test')
        with tqdm.tqdm(total=total_test_batches) as pbar:
            with torch.no_grad():
                for i in range(total_test_batches):  # 1 test epoch
                    x_supports, y_support_hots, x_targets, y_target_hots, target_cat_indices = \
                        self.data_formatter.get_test_data(return_cat_indices=True)
                    x_supports = Variable(torch.from_numpy(x_supports), requires_grad=False).float().unsqueeze(0)
                    y_support_hots = Variable(torch.from_numpy(y_support_hots), requires_grad=False).long().unsqueeze(0)
                    x_targets = Variable(torch.from_numpy(x_targets), requires_grad=False).float().unsqueeze(0)
                    y_target_hots = Variable(torch.from_numpy(y_target_hots), requires_grad=False).long().unsqueeze(0)
                    target_cat_indices = Variable(torch.from_numpy(target_cat_indices), requires_grad=False).float()

                    if self.cuda_available and self.use_cuda:
                        loss, targets_preds = self.match_net(x_supports.cuda(), y_support_hots.cuda(),
                                                             x_targets.cuda(), y_target_hots.cuda(),
                                                             target_cat_indices)
                    else:
                        loss, targets_preds = self.match_net(x_supports, y_support_hots, x_targets, y_target_hots,
                                                             target_cat_indices)
                        # target_cat_indices, testing=True)

                    ## Storing predictions
                    # torch.save(model.state_dict(), PATH)
                    torch.save(targets_preds,
                               join(self.dataset_dir, self.dataset_name, self.dataset_name + '_targets_preds.t'))

                    ## Calculate loss and precisions for this batch
                    iter_out = "TEST Loss: {}".format(loss.item())
                    logger.info(iter_out)
                    logger.info("TEST Precisions:")
                    precision_1 = self.test_metrics.precision_k_hot(y_target_hots, targets_preds, k=1)
                    logger.info("Precision @ 1: {}".format(precision_1))
                    precision_3 = self.test_metrics.precision_k_hot(y_target_hots, targets_preds, k=3)
                    logger.info("Precision @ 3: {}".format(precision_3))
                    precision_5 = self.test_metrics.precision_k_hot(y_target_hots, targets_preds, k=5)
                    logger.info("Precision @ 5: {}".format(precision_5))

                    pbar.set_description(iter_out)
                    print('\n')
                    pbar.update(1)

                    total_test_loss += loss.item()
                    total_p1 += precision_1
                    total_p3 += precision_3
                    total_p5 += precision_5
                ## Calculate loss and precisions for all samples.
                total_test_loss /= total_test_batches
                total_p1 /= total_test_batches
                total_p3 /= total_test_batches
                total_p5 /= total_test_batches
        return total_test_loss, total_p1, total_p3, total_p5

    def __adjust_learning_rate(self, optimizer):
        """
        Updates the learning rate given the learning rate decay.

        The routine has been implemented according to the original Lua SGD optimizer_type
        """
        for group in optimizer.param_groups:
            if 'step' not in group:
                group['step'] = 0
            group['step'] += 1

            group['lr'] = self.lr / (1 + group['step'] * self.lr_decay)

    def __create_optimizer(self, model, new_lr, optimizer_type=config["model"]["optimizer"]["optimizer_type"],
                           weight_decay=config["model"]["optimizer"]["weight_decay"],
                           rho=config["model"]["optimizer"]["rho"],
                           momentum=config["model"]["optimizer"]["momentum"],
                           dampening=config["model"]["optimizer"]["dampening"],
                           alpha=config["model"]["optimizer"]["alpha"],
                           centered=config["model"]["optimizer"]["centered"]):
        """Setup optimizer_type"""
        if optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=new_lr,
                                        momentum=momentum,
                                        dampening=dampening,
                                        weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=new_lr,
                                         weight_decay=weight_decay)
        elif optimizer_type == 'adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(),
                                             lr=new_lr,
                                             rho=rho,
                                             weight_decay=weight_decay)
        elif optimizer_type == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(),
                                            lr=new_lr,
                                            lr_decay=self.lr_decay,
                                            weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(),
                                            lr=new_lr,
                                            alpha=alpha,
                                            momentum=0.9,
                                            centered=centered,
                                            weight_decay=weight_decay)
        else:
            raise Exception('Optimizer not supported: [{0}]'.format(optimizer_type))
        return optimizer


if __name__ == '__main__':
    logger.debug("Building Model...")
    cls = Run_Network(input_size=300, hid_size=100, fce=True, batch_size=32, supports_per_category=15,
                      categories_per_batch=25)
    cls.training(num_train_epoch=10)
