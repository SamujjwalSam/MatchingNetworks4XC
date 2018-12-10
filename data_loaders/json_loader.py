# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6 or above
"""
__synopsis__    : JSONLoader.
__description__ : Class to process and load json files from a directory.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2018"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : JSONLoader,

__variables__   :

__methods__     :
"""

import os
import torch.utils.data
# from collections import OrderedDict

from utils import util
from logger.logger import logger


class JSONLoader(torch.utils.data.Dataset):
    """
    Class to process and load json files from data directory.

    sentences : Wikipedia english texts after parsing and cleaning.
    sentences = {"id1": "wiki_text_1", "id2": "wiki_text_2"}
    
    classes   : OrderedDict of id to classes.
    classes = {"id1": [class_id_1,class_id_2],"id2": [class_id_2,class_id_10]}
    
    categories : Dict of class texts.
    categories = {"Computer Science":class_id_1, "Machine Learning":class_id_2}

    samples : {"sentences":"",
                  "classes":""
                 }
    """
    def __init__(self, dataset_name="Wiki10-31k", run_mode="train",data_dir: str = "D:\Datasets\Extreme Classification"):
        """
        Initializes the html loader.

        Args:
            data_dir : Path to the file containing the html files.
            dataset_name : Name of the dataset.
        """
        super(JSONLoader, self).__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(self.data_dir,self.dataset_name)
        logger.debug("Dataset name:%s" % self.dataset_name)
        logger.debug("JSON directory:%s" % self.data_dir)
        logger.debug("Check if processed json file already exists at [{}], then load."
                     .format(os.path.join(self.dataset_dir, self.dataset_name + "_sentences.json")))

        if run_mode == "train":
            if os.path.isfile(os.path.join(self.dataset_dir, self.dataset_name + "_sentences_train.json")) \
                    and os.path.isfile(os.path.join(self.dataset_dir, self.dataset_name + "_classes_train.json")):
                self.sentences_train = util.load_json("_sentences_train", file_path=self.dataset_dir)
                self.classes_train = util.load_json("_classes_train", file_path=self.dataset_dir)
            else:
                self.load_full_json()
        elif run_mode == "val":
            if os.path.isfile(os.path.join(self.dataset_dir, self.dataset_name + "_sentences_val.json")) \
                    and os.path.isfile(os.path.join(self.dataset_dir, self.dataset_name + "_classes_val.json")):
                self.sentences_val = util.load_json("_sentences_val", file_path=self.dataset_dir)
                self.classes_train = util.load_json("_classes_val", file_path=self.dataset_dir)
            else:
                self.load_full_json()
        elif run_mode == "test":
            if os.path.isfile(os.path.join(self.dataset_dir, self.dataset_name + "_sentences_test.json")) \
                    and os.path.isfile(os.path.join(self.dataset_dir, self.dataset_name + "_classes_test.json")):
                self.sentences_test = util.load_json("_sentences_test", file_path=self.dataset_dir)
                self.classes_test = util.load_json("_classes_test", file_path=self.dataset_dir)
            else:
                self.load_full_json()
        else:
            raise Exception("Unknown running mode: [{}]. \n Available options: ['train','val','test']".format(run_mode))

    def load_full_json(self):
        """
        Loads dataset.

        :param dataset_name:
        """
        if os.path.isfile(os.path.join(self.dataset_dir, self.dataset_name + "_sentences.json")) \
                and os.path.isfile(os.path.join(self.dataset_dir, self.dataset_name + "_classes.json")) \
                and os.path.isfile(os.path.join(self.dataset_dir, self.dataset_name + "_categories.json")):
            logger.info("Loading pre-processed json files from: [{}]".format(os.path.join(self.dataset_dir, self.dataset_name + "_sentences.json")))
            self.sentences = util.load_json(self.dataset_name + "_sentences", file_path=self.dataset_dir)
            self.classes = util.load_json(self.dataset_name + "_classes", file_path=self.dataset_dir)
            self.categories = util.load_json(self.dataset_name + "_categories", file_path=self.dataset_dir)
            assert len(self.sentences) == len(self.classes),\
                "Count of sentences [{0}] and classes [{1}] should match.".format(len(self.sentences),len(self.classes))
            self.split_data()
        else:
            logger.debug("Load data from JSON files.")
            logger.debug("Create 3 separate dicts of sentences, classes and categories from HTML.")
            self.samples = util.load_json(self.data_dir)
            logger.debug(len(self.samples))
            # logger.debug(self.samples)
            # self.categories_filter_regex = None
            logger.debug("Creating 3 separate dicts of sentences[id->texts], classes[id->class_ids]"
                         "and categories[class_name : class_id] from HTML.")
            sentences, classes, categories, no_cat_ids = self.filter_text(self.samples)
            util.save_json(no_cat_ids, self.dataset_name + "_no_cat_ids", file_path=self.dataset_dir)  # Storing the ids which were not processed.
            self.classes = classes
            # logger.debug("Cleaning categories.")
            categories_cleaned, categories_dup_dict = util.clean_categories(categories)
            # logger.debug(type(categories_dup_dict))
            if categories_dup_dict:
                util.save_json(categories_dup_dict, self.dataset_name + "categories_dup_dict", file_path=self.dataset_dir)  # Storing the duplicate categories for future dedup removal.
                self.classes = util.dedup_data(self.classes, categories_dup_dict)
            sentences_cleaned = util.clean_sentences(sentences)
            self.sentences = sentences_cleaned
            self.categories = categories_cleaned
            self.n_categories = len(categories)
            assert len(self.sentences) == len(self.classes), \
                "Count of sentences [{0}] and classes [{1}] should match.".format(len(self.sentences), len(self.classes))
            util.save_json(self.sentences, self.dataset_name + "_sentences", file_path=self.dataset_dir)
            util.save_json(self.classes, self.dataset_name + "_classes", file_path=self.dataset_dir)
            util.save_json(self.categories, self.dataset_name + "_categories", file_path=self.dataset_dir)
            logger.debug("Saved sentences [{0}], classes [{1}] and categories [{2}] as json files.".format(
                os.path.join(self.dataset_dir + "_sentences.json"),
                os.path.join(self.dataset_dir + "_classes.json"),
                os.path.join(self.dataset_dir + "_categories.json")))
            self.split_data()
        self.num_samples = len(self.sentences)
        # return self.sentences,self.classes,self.categories

    def __len__(self):
        logger.debug("Number of samples: [{}] for dataset: [{}]".format(self.num_samples, self.dataset_name))
        return self.num_samples

    def split_data(self):
        """
        Splits a dict
        :param data:
        :param split_size:
        :return:
        """
        keys = self.sentences.keys()
        keys, selected_keys = util.get_batch_keys(keys, batch_size=int(len(keys) * 0.3))
        self.sentences_train, self.classes_train, self.sentences_test, self.classes_test = util.create_batch(self.sentences, self.classes, selected_keys)
        keys, selected_keys = util.get_batch_keys(keys, batch_size=int(len(keys) * 0.3))
        self.sentences_train, self.classes_train, self.sentences_val, self.classes_val = util.create_batch(self.sentences_train,self.classes_train, selected_keys)

        util.save_json(self.sentences_train, self.dataset_name + "_sentences_train", file_path=self.dataset_dir)
        util.save_json(self.classes_train, self.dataset_name + "_classes_train", file_path=self.dataset_dir)
        util.save_json(self.sentences_val, self.dataset_name + "_sentences_val", file_path=self.dataset_dir)
        util.save_json(self.classes_val, self.dataset_name + "_classes_val", file_path=self.dataset_dir)
        util.save_json(self.sentences_test, self.dataset_name + "_sentences_test", file_path=self.dataset_dir)
        util.save_json(self.classes_test, self.dataset_name + "_classes_test", file_path=self.dataset_dir)
        # return train, test

    def __getitem__(self, idx):
        # TODO: correct this part. -> Probably not required.
        return (torch.from_numpy(self.sentences[idx].todense().reshape(-1)),
                torch.from_numpy(self.classes[idx].todense().reshape(-1)))

    def get_data(self):
        """
        Function to get the entire dataset
        """
        return self.sentences, self.classes, self.categories

    def get_train_data(self):
        """
        Function to get the entire dataset
        """
        return self.sentences_train, self.classes_train

    def get_val_data(self):
        """
        Function to get the entire dataset
        """
        return self.sentences_val, self.classes_val

    def get_test_data(self):
        """
        Function to get the entire dataset
        """
        return self.sentences_test, self.classes_test

    def get_batch(self, batch_size=64):
        """
        :returns: A batch of samples.
        """
        return (self.sentences, self.classes), self.categories

    def get_sentences(self):
        """
        Function to get the entire set of features
        """
        return self.sentences

    def get_classes(self):
        """
        Function to get the entire set of classes.
        """
        return self.classes

    def get_categories(self) -> dict:
        """
        Function to get the entire set of categories
        """
        return self.categories

    def get_categories_count(self):
        """
        Function to get the entire set of categories
        """
        return self.n_categories


def main():
    # config = read_config(args)
    cls = JSONLoader("D:\Datasets\Extreme Classification\html_test")
    data_dict = cls.get_categories_count()
    logger.debug(data_dict)
    return False


if __name__ == '__main__':
    main()
