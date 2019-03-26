# coding=utf-8
# !/usr/bin/python3.6  # Please use python 3.6
"""
__synopsis__    : Config file for global variables.

__description__ : We can print the configuration on every run and get the hyper-parameter configuration for each run.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "04-03-2019"
__copyright__   : "Copyright (c) 2019, All rights reserved."
__license__     : "This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree."

__classes__     : config
    
__variables__   : configuration, seed, platform
    
__methods__     :

__last_modified__: 
"""

import json
from logger.logger import logger

global seed
seed = 0

global configuration
configuration = {
    "data": {
        "dataset_name": "Wiki10-31K",
        "val_split": 0.1,
        "test_split": 0.3,
        "show_stat": False
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
        "max_sequence_length": 100,
        "max_vec_len": 5000,
        "dropout": 0.2,
        "dropout_external": 0.0,
        "clipnorm": 1.0,
        "data_slice": 5120,

        "g_encoder": "cnn",
        "use_cuda": False,
        "normalize_inputs": False,
        "tfidf_avg": False,
        "kernel_size": 1,
        "stride": 1,
        "padding": 1,
        "context": 10,
        "classify_count": 0,
        "fce": False,
        "optimizer": {
            "optimizer_type": "adam",
            "learning_rate": 1e-4,
            "lr_decay": 1e-6,
            "weight_decay": 1e-4,
            "momentum":0.9,
            "dampening":0.9,
            "alpha":0.99,
            "rho":0.9,
            "centered":False
        }
    },

    "lstm_params": {
        "num_layers": 1,
        "bias": True,
        "batch_first": True,
        "bidirectional": True,
    },

    "cnn_params": {
        "padding": 1,
        "stride": 1,
        "kernel_size": 1,
        "bias": True,
    },

    "sampling": {
        "num_epochs": 30,
        "num_train_epoch": 20,
        "batch_size": 64,
        "categories_per_batch": 5,
        "supports_per_category": 5,
        "targets_per_category": 5
    },

    "prep_vecs": {
        "max_nb_words": 20000,
        "min_word_count": 1,
        "window": 7,
        "min_count": 1,
        "negative": 10,
        "num_chunks": 10,
        "vectorizer": "doc2vec",
        "sample_repeat_mode": "append",
        "hid_size": 64,
        "input_size": 300,
        "tfidf_avg": False
    },

    "text_process": {
        "encoding": 'latin-1',
        "sents_chunk_mode": "word_sum",
        "workers": 5
    },

    "paths": {
        "result_file": "result.txt",
        "log_dir": "/logs",

        "pretrain_dir": {
            "Windows": "D:\\Datasets\\Extreme Classification",
            "Linux": "/raid/ravi/pretrain",  ## Code path: /home/cs14resch11001/codes/MNXC
            "OSX": "/home/cs16resch01001/datasets/Extreme Classification"
        },

        "dataset_dir": {
            "Windows": "D:\\Datasets\\Extreme Classification",
            "Linux": "/raid/ravi/Datasets/Extreme Classification",
            "OSX": "/home/cs16resch01001/datasets/Extreme Classification"
        }
    },

    "html_parser": {
        "alt_text": True,
        "ignore_table": True,
        "decode_errors": "ignore",
        "default_alt": "<IMG>",
        "ignore_link": True,
        "reference_links": True,
        "bypass_tables": True,
        "ignore_emphasis": True,
        "unicode_snob": True,
        "no_automatic_links": True,
        "no_skip_internal_links": True,
        "single_line_break": True,
        "escape_all": True
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

    def print_config(self, indent=4, sort=True):
        """
        Prints the config.
        """
        logger.info("[{}] : {}".format("Configuration", json.dumps(self.configuration, indent=indent, sort_keys=sort)))

    def get_platform(self):
        """
        Returns dataset path based on OS.

        :return: Returns dataset path based on OS.
        """
        import platform

        if platform.system() == 'Windows':
            return platform.system()
        elif platform.system() == 'Linux':
            return platform.system()
        else:  ## OS X returns name 'Darwin'
            return "OSX"


config_cls = Config()
config_cls.print_config()

global platform
platform = config_cls.get_platform()


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
