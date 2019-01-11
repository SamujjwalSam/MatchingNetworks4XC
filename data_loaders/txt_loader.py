# coding=utf-8

#  coding=utf-8
#  !/usr/bin/python3.6
#
#  """
#  Author : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
#  Version : "0.1"
#  Date : "11/1/19 3:46 PM"
#  Copyright : "Copyright (c) 2019. All rights reserved."
#  Licence : "This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree."
#  Last modified : 11/1/19 3:38 PM.
#  """

# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : TXTLoader.
__description__ : Class to process and load txt files from a directory.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : TXTLoader,

__variables__   :

__methods__     :
"""

import os
import torch.utils.data
from collections import OrderedDict
from smart_open import smart_open as sopen  # Better alternative to Python open().

from utils import util
from logger.logger import logger

seed_val = 0


class TXTLoader(torch.utils.data.Dataset):
    """
    Class to process and load txt files from a directory.

    Datasets: AmazonCat-14K

    sentences : Amazon products title + description after parsing and cleaning.
    sentences = {"id1": "azn_ttl_1", "id2": "azn_ttl_2"}

    classes   : OrderedDict of id to classes.
    classes = {"id1": [class_id_1,class_id_2],"id2": [class_id_2,class_id_10]}

    categories : Dict of class texts.
    categories = {"Computer Science":class_id_1, "Machine Learning":class_id_2}

    samples : {
        "sentences":"",
        "classes":""
        }
    """

    def __init__(self,
                 dataset_name="AmazonCat-13K",
                 data_dir: str = "D:\\Datasets\\Extreme Classification"):
        """
        Initializes the TXT loader.

        Args:
            data_dir : Path to directory containing the txt files.
            dataset_name : Name of the dataset.
        """
        super(TXTLoader, self).__init__()
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(data_dir, self.dataset_name)
        self.raw_txt_dir = os.path.join(self.data_dir, self.dataset_name + "_RawData")
        # self.raw_txt_file = self.dataset_name + "_RawData.txt"
        logger.debug("Dataset name: [{}], Directory: [{}]".format(self.dataset_name, self.data_dir))
        self.sentences, self.classes, self.categories = self.gen_dicts()

    def gen_dicts(self, json_path=None, encoding='UTF-8'):
        """
        Loads the txt files.

        :return:
        """
        classes = OrderedDict()
        categories = OrderedDict()
        cat_idx = 0
        all_classes = self.read_classes(encoding=encoding)
        # logger.debug((len(all_classes)))
        # util.print_dict(all_classes)
        titles = self.read_titles(classes_keys=all_classes.keys(), encoding=encoding)
        descriptions = self.read_desc(classes_keys=all_classes.keys(), encoding=encoding)
        sentences = self.create_sentences(titles,descriptions)
        classes_extra = set(all_classes.keys()).symmetric_difference(set(sentences.keys()))
        if len(classes_extra):
            for k, v in all_classes.items():
                if k not in classes_extra:
                    classes[k] = v
                    for lbl in classes[k]:
                        if lbl not in categories:  # If lbl does not exists in categories already, add it and assign a new category index.
                            categories[lbl] = cat_idx
                            cat_idx += 1
                        classes[k][classes[k].index(lbl)] = categories[lbl]  # Replacing categories text to categories id.
        util.print_dict(sentences)
        util.print_dict(classes)
        util.print_dict(categories)

        return sentences, classes, categories

    def read_classes(self, classes_dir=None, classes_file="categories.txt", encoding='latin-1'):
        """
        Reads the categories.txt file and returns a OrderedDict of id : class ids.

        :param classes_file:
        :param classes_dir:
        :param encoding:
        :return:
        """
        logger.debug("Reads the categories.txt file and returns a OrderedDict of id : class ids.")
        cat_line_phrase = "  "  # Phrase to recognize lines with category information.
        cat_sep_phrase = ", "  # Phrase to separate categories.
        classes = OrderedDict()
        cat_pool = set()
        if classes_dir is None: classes_dir = self.raw_txt_dir
        with sopen(os.path.join(classes_dir, classes_file), encoding=encoding) as raw_cat_ptr:
            sample_idx = raw_cat_ptr.readline().strip()
            for cnt, line in enumerate(raw_cat_ptr):
                if cat_line_phrase in line:
                    cats = line.split(cat_sep_phrase)  # Splliting line based on ', ' to get categories.
                    cats = [x.strip() for x in cats]  # Removing extra characters like: ' ','\n'.
                    cat_pool.update(cats)
                else:
                    classes[sample_idx] = list(cat_pool)
                    cat_pool.clear()
                    sample_idx = line.strip()

        return classes

    def read_titles(self, classes_keys=None, title_path=None, title_file="titles.txt", encoding='latin-1'):
        """
        Reads the titles.txt file and returns a OrderedDict of id : title.

        :param classes_keys: List of classes keys to check only those keys are stored.
        :param title_file:
        :param title_path:
        :param encoding:
        :return:
        """
        logger.debug("Reads the titles.txt file and returns a OrderedDict of id : title.")
        titles = OrderedDict()
        if title_path is None: title_path = os.path.join(self.raw_txt_dir, title_file)
        with sopen(title_path, encoding=encoding) as raw_title_ptr:
            for cnt, line in enumerate(raw_title_ptr):
                line = line.split()
                if classes_keys is None or line[0] in classes_keys:  # Add this sample if corresponding classes exists.
                    titles[line[0].strip()] = " ".join(line[1:]).strip()
        return titles

    def read_desc(self, classes_keys=None, desc_path=None, desc_file="descriptions.txt", encoding='latin-1'):
        """
        Reads the descriptions.txt file and returns a OrderedDict of id : desc.

        :param classes_keys:
        :param desc_file:
        :param desc_path:
        :param encoding:
        :return:
        """
        id_phrase = "product/productId: "  # Phrase to recognize lines with sample id.
        id_remove = 19  # Length of [id_phrase], to be removed from line.
        desc_phrase = "product/description: "  # Phrase to recognize lines with sample description.
        desc_remove = 21  # Length of [desc_phrase], to be removed from line.
        logger.debug("Reads the descriptions.txt file and returns a OrderedDict of id : desc.")
        descriptions = OrderedDict()
        if desc_path is None: desc_path = os.path.join(self.raw_txt_dir, desc_file)
        import itertools
        with sopen(desc_path, encoding=encoding) as raw_desc_ptr:
            for idx_line, desc_line in itertools.zip_longest(
                    *[raw_desc_ptr] * 2):  # Reads multi-line [2] per iteration.
                if id_phrase in idx_line:
                    sample_id = idx_line[id_remove:].strip()
                    if classes_keys is None or sample_id in classes_keys:  # Add this sample if corresponding class exists.
                        if desc_phrase in desc_line:
                            sample_desc = desc_line[desc_remove:].strip()
                        else:
                            sample_desc = None  # Even if 'description' is not found, we are not ignoring the sample as it might still have text in 'title'.
                        descriptions[sample_id] = sample_desc
        return descriptions

    def create_sentences(self, titles, descriptions=None):
        """
        Creates sentences for each sample by using either title or descriptions if only one exists else appends desc to title.

        :param titles:
        :param descriptions:
        :return:
        """
        logger.debug("Creates sentences for each sample by using either title or descriptions if only one exists else appends desc to title.")
        sentences = OrderedDict()
        if descriptions is not None:
            intersect = set(titles.keys()).intersection(set(descriptions.keys()))
            logger.info("[{}] samples have both 'title' and 'description'.".format(len(intersect)))
            for idx in intersect:
                sentences[idx] = titles[idx] + ". \nDESC: " + descriptions[idx]
            sym_dif = set(titles.keys()).symmetric_difference(set(descriptions.keys()))
            if len(sym_dif):
                logger.info("[{}] samples either only have 'title' or 'description'.".format(len(sym_dif)))
                for idx in sym_dif:
                    if idx in titles.keys():
                        sentences[idx] = titles[idx]
                    else:
                        sentences[idx] = descriptions[idx]
        else:
            logger.info("'description' data not provided, only using 'title'.")
            for idx in titles.keys():
                sentences[idx] = titles[idx]
        return sentences

    def get_data(self):
        """
        Function to get the entire dataset
        """
        return self.sentences, self.classes, self.categories

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


def main():
    # config = read_config(args)
    cls = TXTLoader()
    sentences_val, classes_val, categories_val = cls.get_val_data()
    util.print_dict(sentences_val)
    util.print_dict(classes_val)
    util.print_dict(categories_val)


if __name__ == '__main__':
    main()
