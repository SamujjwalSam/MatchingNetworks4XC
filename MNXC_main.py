# coding=utf-8

#  coding=utf-8
#  !/usr/bin/python3.6
#
#  """
#  Author : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
#  Version : "0.1"
#  Date : "5/1/19 11:44 AM"
#  Copyright : "Copyright (c) 2019. All rights reserved."
#  Licence : "This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree."
#  Last modified : 5/1/19 11:41 AM.
#  """

# !/usr/bin/python3.6  # Please use python 3.6
"""
__synopsis__    : Main file to run Matching networks for Extreme Classification.
__description__ :
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"

__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     :

__variables__   :

__methods__     :
"""

# import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# TIME_STAMP = datetime.utcnow().isoformat()

from logger.logger import logger
from utils import util
from models.BuildMN import BuildMN
from data_loaders.PrepareData import PrepareData
from data_loaders.common_data_handler import Common_JSON_Handler

"""
TODOs:
-----------------------------------------
    Prepare Delicious-T140.
    Investigate loss and whole code.
    Vectorize code.
    Implement TF-IDF weighted vectors.
=========================================

Variable naming:
-----------------------------------------

    Name used   ->  Meaning
    -------------------------------------
    Categories  ->  Labels / Classes {}
    Sample      ->  [Feature, Categories]
    *_hot       ->  multi-hot format
    x_hat       ->  test sample
    no_cat_ids [list]  -> ids for which no categories were found.
=========================================

Data formats:
-----------------------------------------
    sentences : Texts after parsing and cleaning.
    sentences =  {
                    "id1": "text_1",
                    "id2": "text_2"
                 }
    
    classes   : OrderedDict of id to classes.
    classes =    {     
                    "id1" : [class_id_1,class_id_2],
                    "id2" : [class_id_2,class_id_10]
                 }
    
    categories : Dict of class texts.
    categories = {
                    "text"       : id
                    "Computer Science" : class_id_1,
                    "Machine Learning" : class_id_2
                 }

    samples :    {
                    "sentences":"",
                    "classes":""
                 }
==========================================

To solve MKL problem: Adding <conda-env-root>/Library/bin to the path in the run configuration solves the issue, but adding it to the interpreter paths in the project settings doesn't.
https://stackoverflow.com/questions/35478526/pyinstaller-numpy-intel-mkl-fatal-error-cannot-load-mkl-intel-thread-dll
"""


def main(args):
    config = util.load_json(args.config, ext=False)
    util.print_json(config, "Config")
    plat = util.get_platform()

    use_cuda = False
    if plat == "Linux": use_cuda = True

    data_loader = Common_JSON_Handler(dataset_type=config["xc_datasets"][config["data"]["dataset_name"]],
                                      dataset_name=config["data"]["dataset_name"],
                                      data_dir=config["paths"]["dataset_dir"][plat])

    data_formatter = PrepareData(dataset=data_loader,
                                 dataset_name=config["data"]["dataset_name"],
                                 dataset_dir=config["paths"]["dataset_dir"][plat])

    match_net = BuildMN(data_formatter, use_cuda=use_cuda)

    match_net.prepare_mn(num_categories=0,
                         fce=True,
                         input_size=config["model"]["input_size"],
                         hid_size=config["model"]["hid_size"],
                         lr=config["model"]["learning_rate"],
                         lr_decay=config["model"]["lr_decay"],
                         weight_decay=config["model"]["weight_decay"],
                         optim=config["model"]["optim"],
                         dropout=config["model"]["dropout"],
                         batch_size=config["model"]["batch_size"],
                         samples_per_category=config["model"]["samples_per_category"],
                         num_cat=config["model"]["num_cat"])

    num_epochs = config["model"]["num_epochs"]

    train_epoch_losses = []
    val_epoch_losses = []
    for epoch in range(num_epochs):
        train_epoch_loss = match_net.run_training_epoch(num_train_epoch=config["model"]["num_train_epoch"], )
        train_epoch_losses.append(train_epoch_loss)
        logger.info("Train epoch loss: [{}]".format(train_epoch_loss))
        logger.info("[{}] epochs of training completed. \nStarting Validation...".format(epoch))
        val_epoch_loss = match_net.run_validation_epoch(num_val_epoch=1)
        val_epoch_losses.append(val_epoch_loss)
        logger.info("Validation epoch loss: [{}]".format(val_epoch_loss))

    logger.info("Train losses: [{}]".format(train_epoch_losses))
    logger.info("Validation losses: [{}]".format(val_epoch_losses))


if __name__ == '__main__':
    parser = ArgumentParser(description="Script to setup and call MNXC",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            epilog="Example: python MNXC_input.py --dataset_url /Users/monojitdey/Downloads/ "
                                   "--dataset_name Wiki10-31K --test_file /Wiki10/wiki10_test.txt"
                                   "--pretrain_dir /pretrain/glove6B.txt")
    # Config arguments
    parser.add_argument('-config',
                        help='Config to read details',
                        default='MNXC.config')
    parser.add_argument('--dataset_dir',
                        help='Path to dataset folder.', type=str,
                        default="")
    parser.add_argument('--dataset_name',
                        help='Name of the dataset to use.', type=str,
                        default='all')
    parser.add_argument('--train_path',
                        help='Path to train file (Absolute or Relative to [dataset_url]).', type=str,
                        default='train')
    parser.add_argument('--test_path',
                        help='Path to test file (Absolute or Relative to [dataset_url]).', type=str,
                        default='test')
    parser.add_argument('--solution_path',
                        help='Path to result folder (Absolute or Relative to [dataset_url]).', type=str,
                        default='result')
    parser.add_argument('--pretrain_dir',
                        help='Path to pre-trained embedding file. Default: [dataset_url/pretrain].', type=str,
                        default='pretrain')

    # Training configuration arguments
    parser.add_argument('--device', type=str, default='cpu',
                        help='PyTorch device string <device_name>:<device_id>')
    parser.add_argument('--seed', type=int, default=None,
                        help='Manually set the seed for the experiments for reproducibility.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train.')
    parser.add_argument('--interval', type=int, default=-1,
                        help='Interval between two status updates during training.')

    # Optimizer arguments
    parser.add_argument('--optimizer_cfg', type=str,
                        help='Optimizer configuration in YAML format for model.')

    # Post-training arguments
    parser.add_argument('--save_model', type=str, default=None,
                        choices=['all', 'inputAE', 'outputAE', 'regressor'], nargs='+',
                        help='Options to save the model partially or completely.')

    args = parser.parse_args()
    logger.debug("Arguments: {}".format(args))
    main(args)
