# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Common_JSON_Handler.
__description__ : Class to handle pre-processed json files.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : Common_JSON_Handler,

__variables__   :

__methods__     :
"""

import gc
from os import makedirs
from os.path import join, isfile
from collections import OrderedDict

from data_loaders import html_loader as html
from data_loaders import json_loader as json
from data_loaders import txt_loader as txt
from utils import util
from logger.logger import logger

seed_val = 0


class Common_JSON_Handler:
    """
    Class to load and prepare and split pre-built json files.

    sentences : Wikipedia english texts after parsing and cleaning.
    sentences = {"id1": "wiki_text_1", "id2": "wiki_text_2"}

    classes   : OrderedDict of id to classes.
    classes = {"id1": [class_id_1,class_id_2],"id2": [class_id_2,class_id_10]}

    categories : Dict of class texts.
    categories = {"Computer Science":class_id_1, "Machine Learning":class_id_2}
    """

    def __init__(self,
                 dataset_name: str,
                 dataset_type="html",
                 data_dir: str = "D:\\Datasets\\Extreme Classification", ):
        """
        Loads train val or test data based on run_mode.

        Args:
            data_dir : Path to directory of the dataset.
            dataset_name : Name of the dataset.
        """
        super(Common_JSON_Handler, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.data_dir = join(data_dir, self.dataset_name)

        self.sentences_selected, self.classes_selected, self.categories_selected = None, None, None

        self.sentences_train, self.classes_train, self.categories_train = None, None, None
        self.sentences_test, self.classes_test, self.categories_test = None, None, None
        self.sentences_val, self.classes_val, self.categories_val = None, None, None

    def load_full_json(self, return_values=False):
        """
        Loads full dataset and splits the data into train, val and test.
        """
        if isfile(join(self.data_dir, self.dataset_name + "_sentences.json")) \
                and isfile(join(self.data_dir, self.dataset_name + "_classes.json")) \
                and isfile(join(self.data_dir, self.dataset_name + "_categories.json")):
            logger.info("Loading pre-processed json files from: [{}]".format(
                join(self.data_dir, self.dataset_name)))
            sentences = util.load_json(self.dataset_name + "_sentences", file_path=self.data_dir)
            classes = util.load_json(self.dataset_name + "_classes", file_path=self.data_dir)
            categories = util.load_json(self.dataset_name + "_categories", file_path=self.data_dir)
            assert len(sentences) == len(classes), \
                "Count of sentences [{0}] and classes [{1}] should match.".format(len(sentences),
                                                                                  len(classes))
        else:
            logger.debug(
                "Loading raw data and creating 3 separate dicts of sentences [id->texts], classes [id->class_ids]"
                " and categories [class_name : class_id].")
            sentences, classes, categories = self.load_raw_data(self.dataset_type)
            logger.debug("Cleaning categories.")
            categories, categories_dup_dict, dup_cat_text_map = util.clean_categories(categories)
            util.save_json(dup_cat_text_map, self.dataset_name + "_dup_cat_text_map", file_path=self.data_dir)
            util.save_json(categories, self.dataset_name + "_categories", file_path=self.data_dir)
            if categories_dup_dict:  # Replace old category ids with new ids if duplicate categories found.
                util.save_json(categories_dup_dict, self.dataset_name + "_categories_dup_dict",
                               file_path=self.data_dir)  # Storing the duplicate categories for future dedup removal.
                classes = util.dedup_data(classes, categories_dup_dict)
            assert len(sentences) == len(classes), \
                "Count of sentences [{0}] and classes [{1}] should match.".format(len(sentences),
                                                                                  len(classes))
            util.save_json(sentences, self.dataset_name + "_sentences", file_path=self.data_dir)
            util.save_json(classes, self.dataset_name + "_classes", file_path=self.data_dir)
            logger.info("Saved sentences [{0}], classes [{1}] and categories [{2}] as json files.".format(
                join(self.data_dir + "_sentences.json"),
                join(self.data_dir + "_classes.json"),
                join(self.data_dir + "_categories.json")))
        # Splitting data into train, validation and test sets.
        self.sentences_train, self.classes_train, self.categories_train, self.sentences_val, self.classes_val, \
        self.categories_val, self.sentences_test, self.classes_test, self.categories_test = \
            self.split_data(sentences=sentences, classes=classes, categories=categories)
        if return_values:
            return sentences, classes, categories
        else:
            sentences, classes, categories = None, None, None  # Remove large dicts and free up memory.
            gc.collect()
        # return self.sentences_train, self.classes_train, self.categories_train, self.sentences_val, self.classes_val,\
        #        self.categories_val, self.sentences_test, self.classes_test, self.categories_test

    def load_raw_data(self, dataset_type=None):
        """
        Loads raw data based on type of dataset.

        :param dataset_type: Type of dataset.
        """
        if dataset_type is None: dataset_type = self.dataset_type
        if dataset_type == "html":
            self.dataset = html.HTMLLoader(dataset_name=self.dataset_name, data_dir=self.data_dir)
        elif dataset_type == "json":
            self.dataset = json.JSONLoader(dataset_name=self.dataset_name, data_dir=self.data_dir)
        elif dataset_type == "txt":
            self.dataset = txt.TXTLoader(dataset_name=self.dataset_name, data_dir=self.data_dir)
        else:
            raise Exception("Dataset type for dataset [{}] not found. \n"
                            "Possible reasons: Dataset not added to the config file.".format(self.dataset_name))
        sentences, classes, categories = self.dataset.get_data()
        return sentences, classes, categories

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
        logger.info("Test count: [{}]. Remaining count: [{}]".format(len(classes_test), len(classes_train)))
        util.save_json(sentences_test, self.dataset_name + "_sentences_test", file_path=self.data_dir)
        util.save_json(classes_test, self.dataset_name + "_classes_test", file_path=self.data_dir)

        classes_train, classes_val, sentences_train, sentences_val = \
            util.split_dict(classes_train, sentences_train, batch_size=int(len(sentences_train) * val_split))
        logger.info("Validation count: [{}]. Train count: [{}]".format(len(classes_val), len(classes_train)))
        util.save_json(sentences_val, self.dataset_name + "_sentences_val", file_path=self.data_dir)
        util.save_json(classes_val, self.dataset_name + "_classes_val", file_path=self.data_dir)
        util.save_json(sentences_train, self.dataset_name + "_sentences_train", file_path=self.data_dir)
        util.save_json(classes_train, self.dataset_name + "_classes_train", file_path=self.data_dir)

        if isfile(join(self.data_dir, self.dataset_name + "_cat_id2text_map.json")):
            cat_id2text_map = util.load_json(self.dataset_name + "_cat_id2text_map", file_path=self.data_dir)
            # Integer keys are converted to string when saving as JSON. Converting back to integer.
            cat_id2text_map_int = OrderedDict()
            for k, v in cat_id2text_map.items():
                cat_id2text_map_int[int(k)] = v
            cat_id2text_map = cat_id2text_map_int
        else:
            logger.info("Generating inverted categories.")
            cat_id2text_map = util.inverse_dict_elm(categories)
            util.save_json(cat_id2text_map, self.dataset_name + "_cat_id2text_map", file_path=self.data_dir)

        logger.info("Creating train categories.")
        categories_train = OrderedDict()
        for k, v in classes_train.items():
            for cat_id in v:
                if cat_id not in categories_train:
                    categories_train[cat_id] = cat_id2text_map[cat_id]
        categories_train = categories_train
        util.save_json(categories_train, self.dataset_name + "_categories_train", file_path=self.data_dir)

        logger.info("Creating validation categories.")
        categories_val = OrderedDict()
        for k, v in classes_val.items():
            for cat_id in v:
                if cat_id not in categories_val:
                    categories_val[cat_id] = cat_id2text_map[cat_id]
        categories_val = categories_val
        util.save_json(categories_val, self.dataset_name + "_categories_val", file_path=self.data_dir)

        logger.info("Creating test categories.")
        categories_test = OrderedDict()
        for k, v in classes_test.items():
            for cat_id in v:
                if cat_id not in categories_test:
                    categories_test[cat_id] = cat_id2text_map[cat_id]
        categories_test = categories_test
        util.save_json(categories_test, self.dataset_name + "_categories_test", file_path=self.data_dir)
        return sentences_train, classes_train, categories_train, sentences_val, classes_val, categories_val, sentences_test, classes_test, categories_test

    def cat2samples(self, classes_dict: dict = None):
        """
        Converts sample : categories to  categories : samples

        :returns: A dictionary of categories to sample mapping.
        """
        cat2samples_map = OrderedDict()
        if classes_dict is None: classes_dict = self.classes_selected
        for k, v in classes_dict.items():
            for cat in v:
                if cat not in cat2samples_map:
                    cat2samples_map[cat] = []
                cat2samples_map[cat].append(k)
        return cat2samples_map

    def get_data(self, load_type="train"):
        """:returns loaded dictionaries based on "load_type" value."""
        if load_type == "train":
            self.sentences_selected, self.classes_selected, self.categories_selected = self.load_train()
        elif load_type == "val":
            self.sentences_selected, self.classes_selected, self.categories_selected = self.load_val()
        elif load_type == "test":
            self.sentences_selected, self.classes_selected, self.categories_selected = self.load_test()
        else:
            raise Exception("Unknown 'load_type': [{}]. \n Available options: ['train','val','test']".format(load_type))
        return self.sentences_selected, self.classes_selected, self.categories_selected

    def load_train(self):
        """Loads and returns training set."""
        logger.debug(join(self.data_dir, self.dataset_name, self.dataset_name + "_sentences_train.json"))
        if self.sentences_train is None:
            if isfile(join(self.data_dir, self.dataset_name + "_sentences_train.json")):
                self.sentences_train = util.load_json(self.dataset_name + "_sentences_train", file_path=self.data_dir)
            else:
                self.load_full_json()

        if self.classes_train is None:
            if isfile(join(self.data_dir, self.dataset_name + "_classes_train.json")):
                self.classes_train = util.load_json(self.dataset_name + "_classes_train", file_path=self.data_dir)
            else:
                self.load_full_json()

        if self.categories_train is None:
            if isfile(join(self.data_dir, self.dataset_name + "_categories_train.json")):
                self.categories_train = util.load_json(self.dataset_name + "_categories_train", file_path=self.data_dir)
            else:
                self.load_full_json()
        gc.collect()

        logger.info("Training data counts:\n\tSentences = [{}],\n\tClasses = [{}],\n\tCategories = [{}]"
                    .format(len(self.sentences_train), len(self.classes_train), len(self.categories_train)))
        return self.sentences_train, self.classes_train, self.categories_train

    def load_val(self):
        """Loads and returns validation set."""
        if self.sentences_val is None:
            if isfile(join(self.data_dir, self.dataset_name + "_sentences_val.json")):
                self.sentences_val = util.load_json(self.dataset_name + "_sentences_val", file_path=self.data_dir)
            else:
                self.load_full_json()

        if self.classes_val is None:
            if isfile(join(self.data_dir, self.dataset_name + "_classes_val.json")):
                self.classes_val = util.load_json(self.dataset_name + "_classes_val", file_path=self.data_dir)
            else:
                self.load_full_json()

        if self.categories_val is None:
            if isfile(join(self.data_dir, self.dataset_name + "_categories_val.json")):
                self.categories_val = util.load_json(self.dataset_name + "_categories_val", file_path=self.data_dir)
            else:
                self.load_full_json()
        gc.collect()

        logger.info("Validation data counts:\n\tSentences = [{}],\n\tClasses = [{}],\n\tCategories = [{}]"
                    .format(len(self.sentences_val), len(self.classes_val), len(self.categories_val)))
        return self.sentences_val, self.classes_val, self.categories_val

    def load_test(self):
        """Loads and returns test set."""
        if self.sentences_test is None:
            if isfile(join(self.data_dir, self.dataset_name + "_sentences_test.json")):
                self.sentences_test = util.load_json(self.dataset_name + "_sentences_test", file_path=self.data_dir)
            else:
                self.load_full_json()

        if self.classes_test is None:
            if isfile(join(self.data_dir, self.dataset_name + "_classes_test.json")):
                self.classes_test = util.load_json(self.dataset_name + "_classes_test", file_path=self.data_dir)
            else:
                self.load_full_json()

        if self.categories_test is None:
            if isfile(join(self.data_dir, self.dataset_name + "_categories_test.json")):
                self.categories_test = util.load_json(self.dataset_name + "_categories_test", file_path=self.data_dir)
            else:
                self.load_full_json()
        gc.collect()

        logger.info("Testing data counts:\n\tSentences = [{}],\n\tClasses = [{}],\n\tCategories = [{}]"
                    .format(len(self.sentences_test), len(self.classes_test), len(self.categories_test)))
        return self.sentences_test, self.classes_test, self.categories_test

    def create_oneclass_data(self, save_dir=None):
        """Creates a dataset of samples which belongs to single class only.

        NOTE: This method is used only for sanity testing using multi-class scenario.
        """
        sentences, classes, _ = self.load_full_json(return_values=True)
        cat_id2text_map = util.load_json(self.dataset_name + "_cat_id2text_map", file_path=self.data_dir)
        sentences_one = OrderedDict()
        classes_one = OrderedDict()
        categories_one = OrderedDict()
        for k, v in classes.items():
            if len(v) == 1:
                classes_one[k] = v
                sentences_one[k] = sentences[k]
                for lbl in classes_one[k]:
                    if lbl not in categories_one:
                        categories_one[cat_id2text_map[str(lbl)]] = lbl

        # Storing one-class classes
        if save_dir is None: save_dir = join(self.data_dir, self.dataset_name + "_onehot")
        makedirs(save_dir, exist_ok=True)
        util.save_json(classes_one,self.dataset_name+"_classes",file_path=save_dir)
        util.save_json(sentences_one, self.dataset_name + "_sentences",file_path=save_dir)
        util.save_json(categories_one, self.dataset_name + "_categories",file_path=save_dir)

        return classes_one, sentences_one, categories_one


def main():
    config = util.load_json('../MNXC.config', ext=False)
    plat = util.get_platform()

    common_handler = Common_JSON_Handler(dataset_type=config["xc_datasets"][config["data"]["dataset_name"]],
                                         dataset_name=config["data"]["dataset_name"],
                                         data_dir=config["paths"]["dataset_dir"][plat])
    classes_one, sentences_one, categories_one = common_handler.create_oneclass_data()

    # util.print_dict(classes_one, count=5)
    # util.print_dict(sentences_one, count=5)
    # util.print_dict(categories_one, count=5)
    logger.debug(len(classes_one))
    logger.debug(len(sentences_one))
    logger.debug(len(categories_one))
    # logger.debug(classes_one)
    # sentences_val, classes_val, categories_val = common_handler.load_val()
    # util.print_dict(sentences_val)
    # util.print_dict(classes_val)
    # util.print_dict(categories_val)


if __name__ == '__main__':
    main()
