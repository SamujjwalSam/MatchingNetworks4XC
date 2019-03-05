# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
"""
__synopsis__    : Short summary of the script.
__description__ : 
__project__     : MNXC
__author__      : sam 
__version__     : ":  "
__date__        : "04-03-2019"
__copyright__   : "Copyright (c) 2019, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree."

__classes__     : config
    
__variables__   :
    
__methods__     :

__last_modified__: 
"""

from utils import util


global configuration
configuration = {
    "data": {
        "dataset_name": "Wiki10-31K",
        "val_split": 0.2,
        "test_split": 0.3,
        "show_stat":False
    },

    "xc_datasets": {
        "AmazonCat-14K": "json",
        "Amazon-3M": "json",
        "Delicious-T140": "html",
        "Wiki10-31K": "html",
        "Wiki10-31K_onehot": "html",
        "Wiki10-31K_fixed5": "html",
        "AmazonCat-13K": "txt",
        "Amazon-670K": "txt",
        "EurLex": "uniq"
    },

    "model": {
        "num_folds": 5,
        "max_nb_words": 20000,
        "max_sequence_length": 100,
        "learning_rate": 1,
        "lr_decay": 1e-6,
        "weight_decay": 1e-4,
        "optim": "adam",
        "dropout": 0.2,
        "clipnorm": 3.0,
        "data_slice": 5120,

        "g_encoder": "cnn",
        "vectorizer": "doc2vec",
        "use_cuda": False,
        "sents_chunk_mode": "word_sum",
        "normalize_inputs": False,
        "sample_repeat_mode": "append",
        "tfidf_avg": False,
        "dropout_external": 0.0,
        "kernel_size": 1,
        "stride": 1,
        "padding": 1,
        "min_word_count": 1,
        "context": 10,
        "fce": False,

        "hid_size": 64,
        "input_size": 300,

        "num_epochs": 30,
        "num_train_epoch": 20,
        "batch_size": 32,
        "categories_per_batch": 5,
        "supports_per_category": 4,
        "targets_per_category": 4
    },

    "prep_vecs": {
        "window": 7,
        "negative": 10,
        "num_chunks": 10,
        "sents_chunk_mode": "word_avg"
    },

    "paths": {
        "result_file": "result.txt",
        "log_dir": "/logs",

        "pretrain_dir": {
            "Windows": "D:\\Datasets\\Extreme Classification",
            "Linux": "/raid/ravi/pretrain",
            "OSX": "/home/cs16resch01001/datasets/Extreme Classification"
        },

        "dataset_dir": {
            "Windows": "D:\\Datasets\\Extreme Classification",
            "Linux": "/home/cs16resch01001/datasets/Extreme Classification",
            "OSX": "/home/cs16resch01001/datasets/Extreme Classification"
        }
    }
}


class Config(object):
    """
        Details of the class.
    """

    def __init__(self):
        super(Config, self).__init__()

        self.configuration = configuration

    def get_config(self):
        """

        :return:
        """
        return self.configuration

    def print_config(self):
        """
        Prints the config.
        """
        util.print_json(self.configuration, "System Configuration")


def main():
    """
    Main module to start code
    :param args:
        Type: tuple
        Required
        Read Only
    :return:
    """
    pass


if __name__ == "__main__":
    main()
