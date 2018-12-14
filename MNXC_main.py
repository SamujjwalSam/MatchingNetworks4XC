# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6 or above
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ :
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2018 sam"
__license__     : "Python"

__classes__     : Class,

__variables__   :

__methods__     :
"""

import os
import pandas as pd
# import ray.dataframe as pd

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from collections import OrderedDict

from logger.logger import logger
from utils import util
from models.BuildMN import BuildMN

# from pretrained.TextEncoder import TextEncoder
# from neighborhood.neighborhood_graph import Neighborhood

"""
Variable naming:

    Name used   ->  Meaning
-----------------------------------------
    Categories  ->  Labels / Classes {}
    Sample      ->  [Feature, Categories]
    *_hot       ->  multi-hot format
    x_hat       ->  test sample
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
                    "label_text"       : class_id_0
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

# Globals-----
PLATFORM = 'win'
TIME_STAMP = datetime.utcnow().isoformat()
dataset_name, dataset_url, dataset_dir, train_path, test_path, solution_path, pretrain_dir = None, None, None, None, None, None, None
seed_val = 42


# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed=seed_val)
# Globals-----


def read_csv_pd(config):
    """
    Loads csv files as pandas dataframe.

    :param config:
    :return:
    """
    tr = pd.read_csv(os.path.join(config["paths"]["dataset_dir"], config["paths"]["dataset_name"], "train.csv"))
    return tr


def read_fastai_csv(dataset_path="D:\\Datasets\\nlp", dataset_name="ag_news_csv", file_name="train.csv", tag="train"):
    """
    Loads csv files as pandas dataframe.

    :param file_name:
    :param dataset_path:
    :param dataset_name:
    :param tag:
    :return: [list]
    """
    import csv
    rows = []
    with open(os.path.join(dataset_path, dataset_name, file_name)) as csvfile:
        csvreader = csv.reader(csvfile, quotechar='"', delimiter=',',
                               quoting=csv.QUOTE_ALL, skipinitialspace=True)
        # header = csvreader.next()
        i = 0
        for row in csvreader:
            d = {"classes": [int(row[0]) - 1], "text": row[1] + ". " + row[2], "sample_id": str(i), "tag": tag}
            rows.append(d)
            i += 1
        # logger.debug("Total no. of rows: %d" % csvreader.line_num)
    return rows


def load_rcv1(sklearn_data='D:\Datasets\Extreme Classification\scikit_learn_data', subset='all', download=True, rand=0,
              shuffle=False, X_y=False):
    """
    Loads RCV1 dataset using sklearn.datasets.fetch_rcv1().

    Dataset Details:
        Version: RCV1-v2, vectors, full sets, topics multilabels.
        Classes	        103
        Samples total	804414
        Dimensionality	47236
        Features	real, between 0 and 1
    :param sklearn_data:
    :param subset:
    :param download:
    :param rand:
    :param shuffle:
    :param X_y:
    :return:
    """
    from sklearn.datasets import fetch_rcv1
    rcv1 = fetch_rcv1(data_home=sklearn_data, subset=subset, download_if_missing=download, random_state=rand,
                      shuffle=shuffle, return_X_y=X_y)
    return rcv1


def download_fastai_dataset(config, dataset='ag_news_csv'):
    """
    Downloads, untars and loads fastai datasets.

    :param config:
    :param dataset:
    :return:
    """
    # logger.debug(config["paths"]["dataset_url"], os.path.join(config["paths"]["dataset_dir"], config["paths"]["dataset_name"]))
    data = untar_data(url=config["paths"]["dataset_url"],
                      fname=os.path.join(config["paths"]["dataset_dir"][PLATFORM],
                                         config["data_loader"]["dataset_name"]),
                      dest=config["paths"]["dataset_dir"][PLATFORM])
    # logger.debug(data_loader)
    return data


def prepare_datasets(config, dataset='ag_news_csv'):
    """
    Downloads, untars and loads dataset.

    :param dataset:
    :param config: Config file dict.
    :return:
    """
    if dataset in config["fastai_datasets"]:
        fastai_path = download_fastai_dataset(config, dataset=dataset)
        tr = read_fastai_csv(dataset_path=fastai_path, dataset_name="ag_news_csv", tag="train")
        ts = read_fastai_csv(dataset_path=fastai_path, dataset_name="ag_news_csv", file_name="test.csv", tag="test")
        X_tr, Y_tr = OrderedDict(), OrderedDict()
        for sample in tr:  # Dividing tr into X_tr and Y_tr.
            X_tr[sample["sample_id"]] = sample["text"]
            Y_tr[sample["sample_id"]] = sample["classes"]

        X_ts, Y_ts = OrderedDict(), OrderedDict()
        for sample in ts:
            X_ts[sample["sample_id"]] = sample["text"]
            Y_ts[sample["sample_id"]] = sample["classes"]

        labels = OrderedDict()
        with open(os.path.join(fastai_path, dataset, "classes.txt")) as cls:
            for i, line in enumerate(cls):
                line = line.rstrip("\n\r")
                labels[i] = line
        # logger.debug(categories)
        return X_tr, Y_tr, X_ts, Y_ts, labels
    else:
        logger.debug("Warning: Dataset not found.")
        return False


def remove_dup_list(seq, case=False):  # Dave Kirby
    """Removes duplicates from a list. Order preserving"""
    seen = set()
    if case: return [x.lower() for x in seq if
                     x.lower() not in seen and not seen.add(x)]
    return [x for x in seq if x not in seen and not seen.add(x)]


def main(args):
    config = util.load_json(args.config, ext=False)
    logger.debug("Config: [{}]".format(config))
    plat = util.get_platform()

    use_cuda = False
    if plat == "Linux": use_cuda = True

    cls = BuildMN(dataset_name=config["data"]["dataset_name"],
                  dataset_dir=config["paths"]["dataset_dir"][plat],
                  use_cuda=use_cuda)

    cls.prepare_mn(num_categories=0,
                   fce=True,
                   input_size=config["model"]["input_size"],
                   hid_size=config["model"]["hid_size"],
                   lr=config["model"]["learning_rate"],
                   lr_decay=config["model"]["lr_decay"],
                   weight_decay=config["model"]["weight_decay"],
                   optim=config["model"]["optim"],
                   dropout=config["model"]["dropout"])

    num_epochs = config["model"]["num_epochs"]

    for epoch in range(num_epochs):
        train_epoch_loss = cls.run_training_epoch(total_train_batches=config["model"]["total_train_batches"],
                                                  batch_size=config["model"]["batch_size"],
                                                  samples_per_category=config["model"]["samples_per_category"],
                                                  num_cat=config["model"]["num_cat"])

        logger.info("Train epoch loss: [{}]".format(train_epoch_loss))
        val_epoch_loss = cls.run_validation_epoch(total_val_batches=1)
        logger.info("Validation epoch loss: [{}]".format(val_epoch_loss))


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
