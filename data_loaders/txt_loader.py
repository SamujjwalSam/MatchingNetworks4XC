# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6 or above
"""
__synopsis__    : TXTLoader.
__description__ : Class to process and load txt files from a directory.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2018"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : TXTLoader,

__variables__   :

__methods__     :
"""

import os
import torch.utils.data
# from collections import OrderedDict

from utils import util
from logger.logger import logger


class TXTLoader(torch.utils.data.Dataset):
    """
    Class to process and load txt files from a directory.
    
    Datasets: Wiki10-31K
    
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
    def __init__(self, data_dir: str = None, dataset_name: str = None):
        """
        Initializes the html loader.

        Args:
            data_dir : Path to the file containing the html files.
            dataset_name : Name of the dataset.
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        logger.debug("Dataset name:%s" % self.dataset_name)
        logger.debug("HTML directory:%s" % self.data_dir)
        logger.debug("Check if processed json file already exists at [{}], then load.".format(os.path.join(self.data_dir, dataset_name + "_sentences.json")))
        if os.path.isfile(os.path.join(self.data_dir, dataset_name + "_sentences.json")) \
                and os.path.isfile(os.path.join(self.data_dir, dataset_name + "_classes.json")) \
                and os.path.isfile(os.path.join(self.data_dir, dataset_name + "_categories.json")):
            logger.info("Loading pre-processed json files from: [{}]".format(os.path.join(self.data_dir, dataset_name + "_sentences.json")))
            self.sentences = util.load_json(dataset_name + "_sentences", file_path=self.data_dir)
            self.classes = util.load_json(dataset_name + "_classes", file_path=self.data_dir)
            self.categories = util.load_json(dataset_name + "_categories", file_path=self.data_dir)
            assert len(self.sentences) == len(self.classes),\
                "Count of sentences [{0}] and classes [{1}] should match.".format(len(self.sentences),len(self.classes))
        else:
            logger.debug("Load data from HTML files.")
            logger.debug("Create 3 separate dicts of sentences, classes and categories from HTML.")
            self.samples = util.load_json(self.data_dir)
            logger.debug(len(self.samples))
            # logger.debug(self.samples)
            # self.categories_filter_regex = None
            logger.debug("Creating 3 separate dicts of sentences[id->texts], classes[id->class_ids]"
                         "and categories[class_name : class_id] from HTML.")
            sentences, classes, categories, no_cat_ids = self.filter_wiki(self.samples)
            util.save_json(no_cat_ids, dataset_name + "_no_cat_ids",
                           file_path=self.data_dir)  # Storing the ids which was not processed.
            self.classes = classes
            # logger.debug("Cleaning categories.")
            categories_cleaned, categories_dup_dict = util.clean_categories(categories)
            # logger.debug(type(categories_dup_dict))
            if categories_dup_dict:
                util.save_json(categories_dup_dict, dataset_name + "categories_dup_dict", file_path=self.data_dir)  # Storing the duplicate categories.
                self.classes = util.dedup_data(self.classes, categories_dup_dict)
            sentences_cleaned = util.clean_sentences(sentences)
            self.sentences = sentences_cleaned
            self.categories = categories_cleaned
            self.n_categories = len(categories)
            assert len(self.sentences) == len(self.classes), \
                "Count of sentences [{0}] and classes [{1}] should match.".format(len(self.sentences), len(self.classes))
            util.save_json(self.sentences, dataset_name + "_sentences", file_path=self.data_dir)
            util.save_json(self.classes, dataset_name + "_classes", file_path=self.data_dir)
            util.save_json(self.categories, dataset_name + "_categories", file_path=self.data_dir)
            logger.debug("Saved sentences [{0}], classes [{1}] and categories [{2}] as json files.".format(
                os.path.join(self.data_dir, dataset_name + "_sentences.json"),
                os.path.join(self.data_dir, dataset_name + "_classes.json"),
                os.path.join(self.data_dir, dataset_name + "_categories.json")))
        self.num_samples = len(self.sentences)
        # return self.sentences,self.classes,self.categories

    def __len__(self):
        logger.debug("Number of samples: [{}] for dataset: [{}]".format(self.num_samples, self.dataset_name))
        return self.num_samples

    def __getitem__(self, idx):
        # TODO: correct this part.
        return (torch.from_numpy(self.sentences[idx].todense().reshape(-1)),
                torch.from_numpy(self.classes[idx].todense().reshape(-1)))

    def read_txt(self, txt_dir):
        return None

    def get_data(self):
        """
        Function to get the entire dataset
        """
        return (self.sentences, self.classes), self.categories

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
    cls = TXTLoader()
    data_dict = cls.read_txt("D:\Datasets\Extreme Classification\html_test")
    logger.debug(data_dict)
    return False


if __name__ == '__main__':
    main()
