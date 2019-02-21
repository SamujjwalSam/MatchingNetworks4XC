# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : WIKIHTMLLoader.
__description__ : Class to process and load html files from a directory.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : HTMLLoader,

__variables__   :

__methods__     :
"""

from os.path import join, exists, split, isfile
import torch.utils.data
from collections import OrderedDict

from utils import util
from logger.logger import logger

seed_val = 0


class JSON_Handler(torch.utils.data.Dataset):
    """
    Class to process pre-built json files.

    sentences : Wikipedia english texts after parsing and cleaning.
    sentences = {"id1": "wiki_text_1", "id2": "wiki_text_2"}

    classes   : OrderedDict of id to classes.
    classes = {"id1": [class_id_1,class_id_2],"id2": [class_id_2,class_id_10]}

    categories : Dict of class texts.
    categories = {"Computer Science":class_id_1, "Machine Learning":class_id_2}
    """

    def __init__(self, dataset_name: str, data_dir: str = "D:\\Datasets\\Extreme Classification"):
        """
        Initializes the json handler.

        Args:
            data_dir : Path to directory of the dataset.
            dataset_name : Name of the dataset.
        """
        super(JSON_Handler, self).__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir

    def split_data(self, sentences, classes, categories, test_split=0.3, val_split=0.2):
        """
        Splits input data into train, val and test.

        :param data_dir:
        :param dataset_name:
        :param categories:
        :param classes:
        :param sentences:
        :param val_split: Validation split size.
        :param test_split: Test split size.
        :return:
        """
        logger.info("Total number of samples: [{}]".format(len(classes)))
        classes_train, classes_test, sentences_train, sentences_test = \
            util.split_dict(classes, sentences, batch_size=int(len(classes) * test_split))
        logger.info("Test count: [{}]. Remaining count: [{}]".format(len(classes_test),len(classes_train)))
        util.save_json(sentences_test,self.dataset_name + "_sentences_test", file_path=self.data_dir)
        util.save_json(classes_test,self.dataset_name + "_classes_test", file_path=self.data_dir)

        classes_train, classes_val, sentences_train, sentences_val = \
            util.split_dict(classes_train, sentences_train, batch_size=int(len(sentences_train) * val_split))
        logger.info("Validation count: [{}]. Train count: [{}]".format(len(classes_val), len(classes_train)))
        util.save_json(sentences_val,self.dataset_name + "_sentences_val", file_path=self.data_dir)
        util.save_json(classes_val,self.dataset_name + "_classes_val", file_path=self.data_dir)
        util.save_json(sentences_train,self.dataset_name + "_sentences_train", file_path=self.data_dir)
        util.save_json(classes_train,self.dataset_name + "_classes_train", file_path=self.data_dir)

        if isfile(join(self.data_dir,self.dataset_name + "_id2cat_map.json")):
            id2cat_map = util.load_json(self.dataset_name + "_id2cat_map", file_path=self.data_dir)
            # integer keys are converted to string when saving as JSON. Need to convert it back to integer.
            id2cat_map_int = OrderedDict()
            for k, v in id2cat_map.items():
                id2cat_map_int[int(k)] = v
            id2cat_map = id2cat_map_int
        else:
            logger.info("Generating inverted categories.")
            id2cat_map = util.inverse_dict_elm(categories)
            util.save_json(id2cat_map,self.dataset_name + "_id2cat_map", file_path=self.data_dir)

        logger.info("Creating train categories.")
        categories_train = OrderedDict()
        for k, v in classes_train.items():
            for cat_id in v:
                if cat_id not in categories_train:
                    categories_train[cat_id] = id2cat_map[cat_id]
        categories_train = categories_train
        util.save_json(categories_train,self.dataset_name + "_categories_train", file_path=self.data_dir)

        logger.info("Creating validation categories.")
        categories_val = OrderedDict()
        for k, v in classes_val.items():
            for cat_id in v:
                if cat_id not in categories_val:
                    categories_val[cat_id] = id2cat_map[cat_id]
        categories_val = categories_val
        util.save_json(categories_val,self.dataset_name + "_categories_val", file_path=self.data_dir)

        logger.info("Creating test categories.")
        categories_test = OrderedDict()
        for k, v in classes_test.items():
            for cat_id in v:
                if cat_id not in categories_test:
                    categories_test[cat_id] = id2cat_map[cat_id]
        categories_test = categories_test
        util.save_json(categories_test,self.dataset_name + "_categories_test", file_path=self.data_dir)
        return sentences_train, classes_train, categories_train, sentences_val, classes_val, categories_val,  sentences_test, classes_test, categories_test

    def get_data(self):
        """
        Function to get the entire dataset
        """
        return self.sentences, self.classes, self.categories

    def cat2samples(self, classes_dict: dict = None):
        """
        Converts sample : categories to  categories : samples

        :returns: A dictionary of categories to sample mapping.
        """
        cat2id = OrderedDict()
        if classes_dict is None: classes_dict = self.selected_classes
        for k, v in classes_dict.items():
            for cat in v:
                if cat not in cat2id:
                    cat2id[cat] = []
                cat2id[cat].append(k)
        return cat2id

    def load_train(self):
        self.sentences_train, self.classes_train, self.categories_train = self.get_train_data()
        self.selected_sentences, self.selected_classes, self.selected_categories = self.sentences_train, self.classes_train, self.categories_train
        self.remain_sample_ids = list(self.selected_sentences.keys())
        self.cat2id_map = self.cat2samples(self.selected_classes)
        self.remain_cat_ids = list(self.selected_categories.keys())

    def load_val(self):
        self.sentences_val, self.classes_val, self.categories_val = self.get_val_data()
        self.selected_sentences, self.selected_classes, self.selected_categories = self.sentences_val, self.classes_val, self.categories_val
        self.remain_sample_ids = list(self.selected_sentences.keys())
        self.cat2id_map = self.cat2samples(self.selected_classes)
        self.remain_cat_ids = list(self.selected_categories.keys())

    def load_test(self):
        self.sentences_test, self.classes_test, self.categories_test = self.get_test_data()
        self.selected_sentences, self.selected_classes, self.selected_categories = self.sentences_test, self.classes_test, self.categories_test
        self.remain_sample_ids = list(self.selected_sentences.keys())
        self.cat2id_map = self.cat2samples(self.selected_classes)
        self.remain_cat_ids = list(self.selected_categories.keys())

    def get_train_data(self):
        """
        Function to get the entire dataset
        """
        return self.sentences_train, self.classes_train, self.categories_train

    def get_val_data(self):
        """
        Function to get the entire dataset
        """
        if self.sentences_val is None:
            if isfile(join(self.data_dir, self.dataset_name + "_sentences_val.json")):
                self.sentences_val = util.load_json(self.dataset_name + "_sentences_val", file_path=self.data_dir)

        if self.classes_val is None:
            if isfile(join(self.data_dir, self.dataset_name + "_classes_val.json")):
                self.classes_val = util.load_json(self.dataset_name + "_classes_val", file_path=self.data_dir)

        if self.categories_val is None:
            if isfile(join(self.data_dir, self.dataset_name + "_categories_val.json")):
                self.categories_val = util.load_json(self.dataset_name + "_categories_val", file_path=self.data_dir)
        return self.sentences_val, self.classes_val, self.categories_val

    def get_test_data(self):
        """
        Function to get the entire dataset
        """
        if isfile(join(self.data_dir, self.dataset_name + "_sentences_test.json")) \
                and isfile(join(self.data_dir, self.dataset_name + "_classes_test.json")):
            self.sentences_test = util.load_json(self.dataset_name + "_sentences_test", file_path=self.data_dir)
            self.classes_test = util.load_json(self.dataset_name + "_classes_test", file_path=self.data_dir)
            self.categories_test = util.load_json(self.dataset_name + "_categories_test", file_path=self.data_dir)
        return self.sentences_test, self.classes_test, self.categories_test


def main():
    # config = read_config(args)
    cls = JSON_Handler()
    # data_dict = cls.read_html_dir("D:\Datasets\Extreme Classification\html_test")
    sentences_val, classes_val, categories_val = cls.get_val_data()
    util.print_dict(sentences_val)
    util.print_dict(classes_val)
    util.print_dict(categories_val)


if __name__ == '__main__':
    main()
