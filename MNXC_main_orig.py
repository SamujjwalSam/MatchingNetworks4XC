# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
"""
__synopsis__    : Main file to run Matching Networks for Extreme Classification.

__description__ :
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"

__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.

__classes__     :

__variables__   :

__methods__     :
"""

import torch
from os.path import join,isfile
from matplotlib import pyplot as plt
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
# TIME_STAMP = datetime.utcnow().isoformat()

from config import Config
from config import platform as plat
from config import username as user
from logger.logger import logger
from models_orig.Run_Network import Run_Network
from data_loaders.PrepareData import PrepareData
from data_loaders.common_data_handler import Common_JSON_Handler

## Turn interactive plotting off
plt.ioff()

seed_val = 0
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed=seed_val)

""" Details
TODOs:
-----------------------------------------
    Prepare Delicious-T140.
    Implement TF-IDF weighted vectors.
=========================================

Variable naming:
-----------------------------------------

    Name used   ->  Meaning
    -------------------------------------
    Categories  ->  Labels / Classes {}
    Sample      ->  [Feature, Categories]
    *_hot       ->  multi-hot vector
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
Config values for testing:
------------------------------------------
        
        "hid_size" : 2,
        "input_size" : 3,

        "num_epochs" : 2,
        "num_train_epoch" : 3,
        "batch_size" : 4,
        "categories_per_batch" : 5,
        "supports_per_category" : 2,
        "targets_per_category" : 1
                      

        "hid_size" : 2,
        "input_size" : 3,

        "num_epochs" : 20,
        "num_train_epoch" : 30,
        "batch_size" : 64,
        "categories_per_batch" : 5,
        "supports_per_category" : 10,
        "targets_per_category" : 1
==========================================

To solve MKL problem: Adding <conda-env-root>/Library/bin to the path in the run configuration solves the issue, but adding it to the interpreter paths in the project settings doesn't.
https://stackoverflow.com/questions/35478526/pyinstaller-numpy-intel-mkl-fatal-error-cannot-load-mkl-intel-thread-dll
"""


def plot_occurance(losses: list,plot_name='val_loss.jpg',clear=True,log=False):
    """
    Plots the validation loss against epochs.

    :param losses:
    :param plot_name:
    :param clear:
    :param log:
    """

    logger.info("Plotting losses...")
    fig = plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch no")
    if log:
        plt.yscale('log')
    plt.ylabel("Val Loss ")
    plt.title("Validation Losses")
    plt.savefig(plot_name)
    if clear:
        plt.cla()
    plt.close(fig)  # Closing the figure so it won't get displayed in console.


def main(args):
    """
    Main function to run Matching Networks for Extreme Classification.

    :param args: Dict of all the arguments.
    """
    config_cls = Config()
    config = config_cls.get_config()

    ## Training Phase
    data_loader = Common_JSON_Handler()
    data_formatter = PrepareData(dataset_loader=data_loader)
    match_net = Run_Network(data_formatter=data_formatter)

    train_losses = []
    val_losses = []
    val_p1s = []
    val_p3s = []
    val_p5s = []
    separator_length = 92
    for epoch in range(config["sampling"]["num_epochs"]):
        train_epoch_loss = match_net.training(num_train_epoch=config["sampling"]["num_train_epoch"])
        train_losses.append(train_epoch_loss)
        logger.info("Train epoch loss: [{}]".format(train_epoch_loss))
        logger.info("[{}] epochs of training completed. \nStarting Validation...".format(epoch))
        logger.info("-" * separator_length)

        # val_epoch_loss, val_p1, val_p3, val_p5 = match_net.testing()
        val_epoch_loss,val_p1,val_p3,val_p5 = match_net.validating(epoch_count=epoch)
        val_losses.append(val_epoch_loss)
        val_p1s.append(val_p1)
        val_p3s.append(val_p3)
        val_p5s.append(val_p5)
        logger.info("Validation epoch loss: [{}]".format(val_epoch_loss))
        logger.info("Validation losses: [{}]".format(val_losses))
        logger.info("=" * separator_length)

    ## Storing trained model
    torch.save(match_net.match_net.state_dict(),
               join(config["paths"]["dataset_dir"][plat],config["data"]["dataset_name"],
                    config["data"]["dataset_name"] + '_' + str(config["sampling"]["num_epochs"])))

    logger.info("#" * separator_length)
    # logger.info("Train losses: [{}]".format(train_losses))
    logger.info("Validation losses: [{}]".format(val_losses))
    # plot_occurance(val_losses)
    logger.info("Validation Precisions 1: [{}]".format(val_p1s))
    logger.info("Validation Precisions 3: [{}]".format(val_p3s))
    logger.info("Validation Precisions 5: [{}]".format(val_p5s))
    # plot_occurance(val_p1s)
    # plot_occurance(val_p3s)
    # plot_occurance(val_p5s)

    ## Inference Phase
    logger.info("=" * separator_length)
    logger.info("\nStarting Inference...")
    test_epoch_loss,test_p1,test_p3,test_p5 = match_net.testing()
    logger.info("Test losses: [{}]".format(test_epoch_loss))
    logger.info("Test Precision 1: [{}]".format(test_p1))
    logger.info("Test Precision 3: [{}]".format(test_p3))
    logger.info("Test Precision 5: [{}]".format(test_p5))
    # plot_occurance(test_epoch_loss)
    # plot_occurance(test_p1)
    # plot_occurance(test_p3)
    # plot_occurance(test_p5)

    ## Loading model
    # model = MatchingNetwork(layer_size=1, num_channels=1)
    # model.load_state_dict(torch.load(join(config["paths"]["dataset_dir"][plat],config["data"]["dataset_name"],
    #                 config["data"]["dataset_name"]+'_'+str(config["sampling"]["num_epochs"]))))
    # model.eval()  ## call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.


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
                        help='Path to dataset folder.',type=str,
                        default="")
    parser.add_argument('--dataset_name',
                        help='Name of the dataset to use.',type=str,
                        default='all')
    parser.add_argument('--train_path',
                        help='Path to train file (Absolute or Relative to [dataset_url]).',type=str,
                        default='train')
    parser.add_argument('--test_path',
                        help='Path to test file (Absolute or Relative to [dataset_url]).',type=str,
                        default='test')
    parser.add_argument('--solution_path',
                        help='Path to result folder (Absolute or Relative to [dataset_url]).',type=str,
                        default='result')
    parser.add_argument('--pretrain_dir',
                        help='Path to pre-trained embedding file. Default: [dataset_url/pretrain].',type=str,
                        default='pretrain')

    # Training configuration arguments
    parser.add_argument('--device',type=str,default='cpu',
                        help='PyTorch device string <device_name>:<device_id>')
    parser.add_argument('--seed',type=int,default=None,
                        help='Manually set the seed for the experiments for reproducibility.')
    parser.add_argument('--batch_size',type=int,default=32,
                        help='Batch size for training.')
    parser.add_argument('--epochs',type=int,default=20,
                        help='Number of epochs to train.')
    parser.add_argument('--interval',type=int,default=-1,
                        help='Interval between two status updates during training.')

    # Optimizer arguments
    parser.add_argument('--optimizer_cfg',type=str,
                        help='Optimizer configuration in YAML format for model.')

    # Post-training arguments
    parser.add_argument('--save_model',type=str,default=None,
                        choices=['all','inputAE','outputAE','regressor'],nargs='+',
                        help='Options to save the model partially or completely.')

    args = parser.parse_args()
    # logger.debug("Arguments: {}".format(args))
    main(args)
