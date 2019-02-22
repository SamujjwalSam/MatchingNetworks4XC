# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Builds the experiment using matching network, the training and evaluation ops as well as data_loader augmentation routines.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

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
from utils import util
from metrics.metrics import Metrics
from models.MatchingNetwork import MatchingNetwork

seed_val = 0


# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed=seed_val)


class Run_Network:
    """
    Initializes an Run_Network object. The Run_Network object takes care of setting up our experiment
    and provides helper functions such as run_training_epoch and run_validation_epoch to simplify out training
    and evaluation procedures.
    """

    def __init__(self, data_formatter, dataset_name, dataset_dir, use_cuda=False, categories_per_batch=15, vectorizer="doc2vec",
                 input_size=300, hid_size=100, lr=1e-03, lr_decay=1e-6, weight_decay=1e-4, optim="adam", dropout=0.2,
                 num_categories=0, fce=True, batch_size=32, supports_per_category=20, targets_per_category=5):
        """Builds the Matching Network with all required parameters.

        Provides helper functions such as run_training_epoch() and run_validation_epoch() to simplify out training
        and evaluation procedures.

        :param data_formatter: A data_loader provider class
        :param targets_per_category: An integer indicating the number of targets per categories.
        :param vectorizer: How to generate feature vectors.
        :param categories_per_batch: An integer indicating the number of categories per batch.
        :param supports_per_category:
        :param hid_size: Size of the output of BiLSTM.
        :param input_size: Size of sample vectors.
        :param num_categories: Number of categories. Not Required. Always 0.
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
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

        self.optimizer = optim
        self.lr = lr
        self.lr_decay = lr_decay
        self.input_size = input_size
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.supports_per_category = supports_per_category
        self.targets_per_category = targets_per_category
        self.categories_per_batch = categories_per_batch
        self.vectorizer = vectorizer
        self.total_train_iter = 0
        self.test_metrics = Metrics()
        self.match_net = MatchingNetwork(dropout=dropout,
                                         fce=fce,
                                         num_categories=num_categories,
                                         input_size=input_size,
                                         hid_size=hid_size,
                                         use_cuda=self.use_cuda)
        logger.info("Matching Network Summary: \n{}".format(self.match_net))

        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available and self.use_cuda:
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(seed=seed_val)
            self.match_net.cuda()

    def run_training_epoch(self, num_train_epoch, print_grads=False):
        """
        Runs one training epoch.

        :param print_grads: Flag to denote if gradients are to be printed.
        :param num_train_epoch: Number of batches to train.
        :return: mean_training_multilabel_margin_loss.
        """
        total_c_loss = 0.
        self.data_formatter.prepare_data(load_type='train')  # Loading appropriate data
        optimizer = self.__create_optimizer(self.match_net, self.lr)  # Creating the optimizer

        with tqdm.tqdm(total=num_train_epoch) as pbar:
            for i in range(num_train_epoch):  # 1 train epoch
                x_supports, y_support_hots, x_hats, y_hats_hots, target_cat_indices = \
                    self.data_formatter.get_batches(
                        batch_size=self.batch_size,
                        categories_per_batch=self.categories_per_batch,
                        supports_per_category=self.supports_per_category,
                        targets_per_category=self.targets_per_category,
                        vectorizer=self.vectorizer,
                        input_size=self.input_size)

                x_supports = Variable(torch.from_numpy(x_supports)).float()
                y_support_hots = Variable(torch.from_numpy(y_support_hots), requires_grad=False).float()
                # support_cat_indices = Variable(torch.from_numpy(support_cat_indices), requires_grad=False).float()
                x_hats = Variable(torch.from_numpy(x_hats)).float()
                y_hats_hots = Variable(torch.from_numpy(y_hats_hots), requires_grad=False).float()
                target_cat_indices = Variable(torch.from_numpy(target_cat_indices), requires_grad=False).float()

                if self.cuda_available and self.use_cuda:
                    cc_loss, hats_preds = self.match_net(x_supports.cuda(), y_support_hots.cuda(),
                                                         x_hats.cuda(), y_hats_hots.cuda(),
                                                         batch_size=self.batch_size)
                else:
                    cc_loss, hats_preds = self.match_net(x_supports, y_support_hots, x_hats, y_hats_hots,
                                                         target_cat_indices, batch_size=self.batch_size)
                    # ,num_categories=self.categories_per_batch)

                # Before the backward pass, use the optimizer object to zero all of the gradients for the variables
                # it will update (which are the learnable weights of the model)
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                cc_loss.backward()

                if print_grads:
                    for f in self.match_net.parameters():
                        logger.debug(f.data)
                        logger.debug(f.grad)

                # Calling the step function on an Optimizer makes an update to its parameters
                optimizer.step()

                # update the optimizer learning rate
                self.__adjust_learning_rate(optimizer)

                precision_1 = self.test_metrics.precision_k_hot(y_hats_hots, hats_preds, k=1)
                precision_3 = self.test_metrics.precision_k_hot(y_hats_hots, hats_preds, k=3)
                precision_5 = self.test_metrics.precision_k_hot(y_hats_hots, hats_preds, k=5)

                logger.info("TRAIN Precisions:")
                logger.info("Precision @ 1: {}".format(precision_1))
                logger.info("Precision @ 3: {}".format(precision_3))
                logger.info("Precision @ 5: {}".format(precision_5))

                iter_out = "TRAIN Loss: {}".format(cc_loss.item())
                logger.debug(iter_out)

                pbar.set_description(iter_out)
                print('\n')
                pbar.update(1)
                total_c_loss += cc_loss.item()

                self.total_train_iter += 1
                if self.total_train_iter % 2000 == 0:
                    self.lr /= 2
                    logger.debug("change learning rate: [{}]".format(self.lr))

        total_c_loss = total_c_loss / num_train_epoch
        return total_c_loss

    def run_validation_epoch(self, num_val_epoch, epoch_count):
        """
        Runs one validation epoch.

        :param batch_size:
        :param num_val_epoch: Number of batches to train on
        :return: mean_validation_categorical_crossentropy_loss
        """
        total_val_c_loss = 0.
        self.data_formatter.prepare_data(load_type='val')

        with tqdm.tqdm(total=num_val_epoch) as pbar:
            with torch.no_grad():
                for i in range(num_val_epoch):  # 1 validation epoch
                    x_supports, y_support_hots, x_targets, y_target_hots, target_cat_indices = \
                        self.data_formatter.get_batches(
                            batch_size=self.batch_size,
                            categories_per_batch=self.categories_per_batch,
                            supports_per_category=self.supports_per_category,
                            targets_per_category=self.targets_per_category,
                            vectorizer=self.vectorizer,
                            input_size=self.input_size,
                            val=True)
                    logger.info("Shapes: x_supports [{}], y_support_hots [{}], x_targets [{}], y_target_hots [{}]"
                                .format(x_supports.shape, y_support_hots.shape, x_targets.shape, y_target_hots.shape))

                    x_supports = Variable(torch.from_numpy(x_supports), requires_grad=False).float()
                    y_support_hots = Variable(torch.from_numpy(y_support_hots), requires_grad=False).float()
                    # support_cat_indices = Variable(torch.from_numpy(support_cat_indices), requires_grad=False).float()
                    x_targets = Variable(torch.from_numpy(x_targets), requires_grad=False).float()
                    y_target_hots = Variable(torch.from_numpy(y_target_hots), requires_grad=False).float()
                    target_cat_indices = Variable(torch.from_numpy(target_cat_indices), requires_grad=False).float()

                    if self.cuda_available and self.use_cuda:
                        cc_loss, hats_preds, encoded_x_hat = self.match_net(x_supports.cuda(), y_support_hots.cuda(),
                                                                            x_targets.cuda(), y_target_hots.cuda(),
                                                                            batch_size=self.batch_size,
                                                                            print_accuracy=True)
                    else:
                        cc_loss, hats_preds, encoded_x_hat = self.match_net(x_supports, y_support_hots, x_targets,
                                                                            y_target_hots, target_cat_indices,
                                                                            batch_size=self.batch_size,
                                                                            print_accuracy=True)

                    logger.debug("Saving encoded_x_hat named: [{}] at: [{}]".format(self.dataset_name
                                + "_val_encoded_x_hat_" +str(epoch_count),join(self.dataset_dir, self.dataset_name)))
                    # util.save_npz(encoded_x_hat, self.dataset_name + "_val_encoded_x_hat_" + str(epoch_count),
                    #               file_path=join(self.dataset_dir, self.dataset_name), overwrite=True)
                    # torch.save(encoded_x_hat, join(self.dataset_dir, self.dataset_name,
                    #                                self.dataset_name + "_val_encoded_x_hat_" + str(
                    #                                    epoch_count) + ".tensor"))
                    util.save_json(encoded_x_hat.tolist(), self.dataset_name + "_val_encoded_x_hat_" + str(epoch_count)
                                   + ".tensor", join(self.dataset_dir, self.dataset_name))
                    logger.debug(
                        "VALIDATION epoch_count: [{}]. \nencoded_x_hats: {}".format(epoch_count, encoded_x_hat))

                    iter_out = "VALIDATION Loss: {}".format(cc_loss.item())
                    logger.info(iter_out)

                    precision_1 = self.test_metrics.precision_k_hot(y_target_hots, hats_preds, k=1)
                    precision_3 = self.test_metrics.precision_k_hot(y_target_hots, hats_preds, k=3)
                    precision_5 = self.test_metrics.precision_k_hot(y_target_hots, hats_preds, k=5)

                    logger.info("VALIDATION Precisions:")
                    logger.info("Precision @ 1: {}".format(precision_1))
                    logger.info("Precision @ 3: {}".format(precision_3))
                    logger.info("Precision @ 5: {}".format(precision_5))

                    pbar.set_description(iter_out)
                    print('\n')
                    pbar.update(1)

                    total_val_c_loss += cc_loss.item()
                total_val_c_loss = total_val_c_loss / num_val_epoch

        return total_val_c_loss

    def run_testing_epoch(self, total_test_batches, epoch_count):
        """
        Runs one testing epoch.

        :param batch_size:
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss
        """
        total_test_c_loss = 0.
        self.data_formatter.prepare_data(load_type='test')

        with tqdm.tqdm(total=total_test_batches) as pbar:
            with torch.no_grad():
                for i in range(total_test_batches):  # 1 test epoch
                    x_supports, y_support_hots, x_targets, y_target_hots = \
                        self.data_formatter.get_batches(
                            batch_size=self.batch_size,
                            categories_per_batch=self.categories_per_batch,
                            supports_per_category=self.supports_per_category,
                            vectorizer=self.vectorizer,
                            input_size=self.input_size)

                    x_supports = Variable(torch.from_numpy(x_supports), requires_grad=False).float()
                    y_support_hots = Variable(torch.from_numpy(y_support_hots), requires_grad=False).float()
                    x_targets = Variable(torch.from_numpy(x_targets), requires_grad=False).float()
                    y_target_hots = Variable(torch.from_numpy(y_target_hots), requires_grad=False).float()

                    if self.cuda_available and self.use_cuda:
                        cc_loss, hats_preds, encoded_x_hat = self.match_net(x_supports.cuda(), y_support_hots.cuda(),
                                                                            x_targets.cuda(), y_target_hots.cuda(),
                                                                            batch_size=self.batch_size,
                                                                            print_accuracy=True)
                    else:
                        cc_loss, hats_preds, encoded_x_hat = self.match_net(x_supports, y_support_hots,
                                                                            x_targets, y_target_hots,
                                                                            batch_size=self.batch_size,
                                                                            print_accuracy=True)

                    util.save_npz(encoded_x_hat, self.dataset_name + "_test_encoded_x_hat_" + str(epoch_count),
                                  file_path=join(self.dataset_dir, self.dataset_name), overwrite=True)
                    logger.debug("TEST epoch_count: [{}]. \nencoded_x_hats: {}".format(epoch_count, encoded_x_hat))

                    iter_out = "TEST Loss: {}".format(cc_loss.item())
                    logger.info(iter_out)

                    logger.info("TEST Precisions:")
                    precision_1 = self.test_metrics.precision_k_hot(y_target_hots, hats_preds, k=1)
                    precision_3 = self.test_metrics.precision_k_hot(y_target_hots, hats_preds, k=3)
                    precision_5 = self.test_metrics.precision_k_hot(y_target_hots, hats_preds, k=5)
                    logger.info("Precision @ 1: {}".format(precision_1))
                    logger.info("Precision @ 3: {}".format(precision_3))
                    logger.info("Precision @ 5: {}".format(precision_5))

                    pbar.set_description(iter_out)
                    print('\n')
                    pbar.update(1)

                    total_test_c_loss += cc_loss.item()
                total_test_c_loss = total_test_c_loss / total_test_batches
        return total_test_c_loss

    def __adjust_learning_rate(self, optimizer):
        """
        Updates the learning rate given the learning rate decay.

        The routine has been implemented according to the original Lua SGD optimizer
        """
        for group in optimizer.param_groups:
            if 'step' not in group:
                group['step'] = 0
            group['step'] += 1

            group['lr'] = self.lr / (1 + group['step'] * self.lr_decay)

    def __create_optimizer(self, model, new_lr):
        """Setup optimizer"""
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=new_lr,
                                        momentum=0.9,
                                        dampening=0.9,
                                        weight_decay=self.weight_decay)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=new_lr,
                                         weight_decay=self.weight_decay)
        elif self.optimizer == 'adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(),
                                             lr=new_lr,
                                             rho=0.9,
                                             weight_decay=self.weight_decay)
        elif self.optimizer == 'dagrad':
            optimizer = torch.optim.Adagrad(model.parameters(),
                                            lr=new_lr,
                                            lr_decay=self.lr_decay,
                                            weight_decay=self.weight_decay)
        elif self.optimizer == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(),
                                            lr=new_lr,
                                            alpha=0.99,
                                            momentum=0.9,
                                            centered=False,
                                            weight_decay=self.weight_decay)
        else:
            raise Exception('Optimizer not supported: [{0}]'.format(self.optimizer))
        return optimizer

    def run_training_epoch_orig(self, num_train_epoch):
        """
        Runs one training epoch.

        :param input_size:
        :param batch_size:
        :param supports_per_category:
        :param categories_per_batch:
        :param num_train_epoch: Number of batches to train.
        :return: mean_training_multilabel_margin_loss.
        """
        total_c_loss = 0.
        self.data_formatter.prepare_data(load_type='train')  # Loading appropriate data
        optimizer = self.__create_optimizer(self.match_net, self.lr)  # Creating the optimizer

        with tqdm.tqdm(total=num_train_epoch) as pbar:
            for i in range(num_train_epoch):  # 1 train epoch
                x_supports, y_support_hots, x_hats, y_hats_hots, target_cat_indices = \
                    self.data_formatter.get_batches(
                        batch_size=self.batch_size,
                        categories_per_batch=self.categories_per_batch,
                        supports_per_category=self.supports_per_category,
                        targets_per_category=self.targets_per_category,
                        vectorizer=self.vectorizer,
                        input_size=self.input_size)

                x_supports = Variable(torch.from_numpy(x_supports)).float()
                y_support_hots = Variable(torch.from_numpy(y_support_hots), requires_grad=False).float()
                # support_cat_indices = Variable(torch.from_numpy(support_cat_indices), requires_grad=False).float()
                x_hats = Variable(torch.from_numpy(x_hats)).float()
                y_hats_hots = Variable(torch.from_numpy(y_hats_hots), requires_grad=False).float()
                target_cat_indices = Variable(torch.from_numpy(target_cat_indices), requires_grad=False).float()

                if self.cuda_available and self.use_cuda:
                    cc_loss, hats_preds = self.match_net.forward2(x_supports.cuda(), y_support_hots.cuda(),
                                                                  x_hats.cuda(), y_hats_hots.cuda(),
                                                                  target_cat_indices.cuda(),
                                                                  batch_size=self.batch_size)
                else:
                    cc_loss, hats_preds = self.match_net.forward2(x_supports, y_support_hots,
                                                                  x_hats, y_hats_hots, target_cat_indices,
                                                                  batch_size=self.batch_size)

                # Before the backward pass, use the optimizer object to zero all of the gradients for the variables
                # it will update (which are the learnable weights of the model)
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                cc_loss.backward()

                # Calling the step function on an Optimizer makes an update to its parameters
                optimizer.step()

                # update the optimizer learning rate
                self.__adjust_learning_rate(optimizer)

                iter_out = "TRAIN Loss: {}".format(cc_loss.item())
                logger.info(iter_out)

                precision_1 = self.test_metrics.precision_k_hot(y_hats_hots, hats_preds, k=1)
                precision_3 = self.test_metrics.precision_k_hot(y_hats_hots, hats_preds, k=3)
                precision_5 = self.test_metrics.precision_k_hot(y_hats_hots, hats_preds, k=5)

                logger.info("TRAIN Precisions:")
                logger.info("Precision @ 1: {}".format(precision_1))
                logger.info("Precision @ 3: {}".format(precision_3))
                logger.info("Precision @ 5: {}".format(precision_5))

                pbar.set_description(iter_out)
                print('\n')
                pbar.update(1)
                total_c_loss += cc_loss.item()

                self.total_train_iter += 1
                if self.total_train_iter % 2000 == 0:
                    self.lr /= 2
                    logger.debug("change learning rate: [{}]".format(self.lr))

        total_c_loss = total_c_loss / num_train_epoch
        return total_c_loss


if __name__ == '__main__':
    logger.debug("Building Model...")
    cls = Run_Network(input_size=300, hid_size=100, fce=True, batch_size=32, supports_per_category=15, categories_per_batch=25)
    cls.run_training_epoch(num_train_epoch=10)
