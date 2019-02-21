# coding=utf-8
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

from os.path import join, exists, split, isfile
import torch.utils.data
from collections import OrderedDict
from smart_open import smart_open as sopen  # Better alternative to Python open().

from utils import util
from logger.logger import logger
from data_loaders.common_data_handler import JSON_Handler

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
                 run_mode="train",
                 data_dir: str = "D:\\Datasets\\Extreme Classification"):
        """
        Initializes the TXT loader.

        Args:
            data_dir : Path to directory containing the txt files.
            dataset_name : Name of the dataset.
        """
        super(TXTLoader, self).__init__()
        self.dataset_name = dataset_name
        self.data_dir = join(data_dir, self.dataset_name)
        self.raw_txt_dir = join(self.data_dir, self.dataset_name + "_RawData")
        # self.raw_txt_file = self.dataset_name + "_RawData.txt"
        logger.debug("Dataset name: [{}], Directory: [{}]".format(self.dataset_name, self.data_dir))

        self.sentences_train = None
        self.classes_train = None
        self.categories_train = None
        self.sentences_val = None
        self.classes_val = None
        self.categories_val = None
        self.sentences_test = None
        self.classes_test = None
        self.categories_test = None

        logger.debug("Check if processed json file already exists at [{}], then load."
                     .format(join(self.data_dir, self.dataset_name + "_sentences_train.json")))
        # if run_mode == "train" or run_mode == "val" or run_mode == "test":
        #     if isfile(join(self.data_dir, self.dataset_name + "_sentences_"+run_mode+".json")) \
        #             and isfile(join(self.data_dir, self.dataset_name + "_classes_"+run_mode+".json")) \
        #             and isfile(join(self.data_dir, self.dataset_name + "_categories_"+run_mode+".json")):
        #         self.sentences_train = util.load_json(self.dataset_name + "_sentences_"+run_mode, file_path=self.data_dir)
        #         self.classes_train = util.load_json(self.dataset_name + "_classes_"+run_mode, file_path=self.data_dir)
        #         self.categories_train = util.load_json(self.dataset_name + "_categories_"+run_mode, file_path=self.data_dir)
        #     else:
        #         self.load_full_json()
        if run_mode == "train":
            if isfile(join(self.data_dir, self.dataset_name + "_sentences_train.json")) \
                    and isfile(join(self.data_dir, self.dataset_name + "_classes_train.json")) \
                    and isfile(join(self.data_dir, self.dataset_name + "_categories_train.json")):
                self.sentences_train = util.load_json(self.dataset_name + "_sentences_train", file_path=self.data_dir)
                self.classes_train = util.load_json(self.dataset_name + "_classes_train", file_path=self.data_dir)
                self.categories_train = util.load_json(self.dataset_name + "_categories_train", file_path=self.data_dir)
            else:
                self.load_full_json()
        elif run_mode == "val":
            if isfile(join(self.data_dir, self.dataset_name + "_sentences_val.json")) \
                    and isfile(join(self.data_dir, self.dataset_name + "_classes_val.json")) \
                    and isfile(join(self.data_dir, self.dataset_name + "_categories_val.json")):
                self.sentences_val = util.load_json(self.dataset_name + "_sentences_val", file_path=self.data_dir)
                self.classes_val = util.load_json(self.dataset_name + "_classes_val", file_path=self.data_dir)
                self.categories_val = util.load_json(self.dataset_name + "_categories_val", file_path=self.data_dir)
            else:
                self.load_full_json()
        elif run_mode == "test":
            if isfile(join(self.data_dir, self.dataset_name + "_sentences_test.json")) \
                    and isfile(join(self.data_dir, self.dataset_name + "_classes_test.json")) \
                    and isfile(join(self.data_dir, self.dataset_name + "_categories_test.json")):
                self.sentences_test = util.load_json(self.dataset_name + "_sentences_test", file_path=self.data_dir)
                self.classes_test = util.load_json(self.dataset_name + "_classes_test", file_path=self.data_dir)
                self.categories_test = util.load_json(self.dataset_name + "_categories_test", file_path=self.data_dir)
            else:
                self.load_full_json()
        else:
            raise Exception("Unknown running mode: [{}]. \n Available options: ['train','val','test']".format(run_mode))

    def load_full_json(self):
        """
        Loads full dataset and splits the data into train, val and test.
        """
        # logger.debug(join(self.data_dir, self.dataset_name + "_sentences.json"))
        if isfile(join(self.data_dir, self.dataset_name + "_sentences.json")) \
                and isfile(join(self.data_dir, self.dataset_name + "_classes.json")) \
                and isfile(join(self.data_dir, self.dataset_name + "_categories.json")):
            logger.info("Loading pre-processed json files from: [{}]".format(
                join(self.data_dir, self.dataset_name)))
            self.sentences = util.load_json(self.dataset_name + "_sentences", file_path=self.data_dir)
            self.classes = util.load_json(self.dataset_name + "_classes", file_path=self.data_dir)
            self.categories = util.load_json(self.dataset_name + "_categories", file_path=self.data_dir)
            assert len(self.sentences) == len(self.classes), \
                "Count of sentences [{0}] and classes [{1}] should match.".format(len(self.sentences),
                                                                                  len(self.classes))
        else:
            logger.debug(
                "Loading data from TXT files and creating 3 separate dicts of sentences [id->texts], classes [id->class_ids]"
                " and categories [class_name : class_id] from TXT files.")
            sentences, classes, categories = self.load_txts(encoding="latin-1")
            logger.debug((len(sentences), len(classes), len(categories)))
            # logger.debug("Cleaning categories.")
            categories_cleaned, categories_dup_dict, dup_cat_text_map = util.clean_categories(categories)
            self.categories = categories_cleaned
            util.save_json(self.categories, self.dataset_name + "_categories", file_path=self.data_dir)
            util.save_json(dup_cat_text_map, self.dataset_name + "_dup_cat_text_map", file_path=self.data_dir)
            self.n_categories = len(categories)
            if categories_dup_dict:
                util.save_json(categories_dup_dict, self.dataset_name + "_categories_dup_dict",
                               file_path=self.data_dir)  # Storing the duplicate categories for future dedup removal.
                self.classes = util.dedup_data(classes, categories_dup_dict)
            else:
                self.classes = classes
            self.sentences = util.clean_sentences_dict(sentences)
            assert len(self.sentences) == len(self.classes), \
                "Count of sentences [{0}] and classes [{1}] should match.".format(len(self.sentences),
                                                                                  len(self.classes))
            util.save_json(self.sentences, self.dataset_name + "_sentences", file_path=self.data_dir)
            util.save_json(self.classes, self.dataset_name + "_classes", file_path=self.data_dir)
            logger.info("Saved sentences [{0}], classes [{1}] and categories [{2}] as json files.".format(
                join(self.data_dir + "_sentences.json"),
                join(self.data_dir + "_classes.json"),
                join(self.data_dir + "_categories.json")))

        # Splitting data into train, validation and test sets.
        data_loader = JSON_Handler(dataset_name=self.dataset_name, data_dir=self.data_dir)
        self.sentences_train, self.classes_train, self.categories_train, self.sentences_val, self.classes_val, self.categories_val, self.sentences_test, self.classes_test, self.categories_test = data_loader.split_data(
            sentences=self.sentences, classes=self.classes, categories=self.categories)
        self.num_samples = len(self.sentences)

    def load_txts(self, json_path=None, encoding='UTF-8'):
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
        # No need to read descriptions as no sample intersects with all_classes.
        descriptions = self.read_desc(classes_keys=all_classes.keys(), encoding=encoding)
        # logger.debug((len(descriptions)))
        sentences = self.create_sentences(titles,descriptions)
        # logger.debug((len(sentences)))
        classes_extra = set(all_classes.keys()).symmetric_difference(set(sentences.keys()))
        # logger.debug(len(classes_extra))
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
        # logger.debug("Generates the data dictionaries from original json file.")
        cat_line_phrase = "  "  # Phrase to recognize lines with category information.
        # cat_remove = 2  # Length of [cat_phrase], to be removed from line.
        cat_sep_phrase = ", "  # Phrase to separate categories.
        classes = OrderedDict()
        # categories = OrderedDict()
        cat_pool = set()
        cat_idx = 0
        if classes_dir is None: classes_dir = self.raw_txt_dir
        with sopen(join(classes_dir, classes_file), encoding=encoding) as raw_cat_ptr:
            sample_idx = raw_cat_ptr.readline().strip()
            for cnt, line in enumerate(raw_cat_ptr):
                if cat_line_phrase in line:
                    cats = line.split(cat_sep_phrase)  # Splliting line based on ', ' to get categories.
                    cats = [x.strip() for x in cats]  # Removing extra characters like: ' ','\n'.
                    cat_pool.update(cats)
                else:
                    # logger.debug("idx:[{}], cats:[{}]".format(sample_idx,cat_pool))
                    classes[sample_idx] = list(cat_pool)
                    # Generate Categories dict from classes.
                    # for lbl in classes[sample_idx]:
                    #     if lbl not in categories:  # If lbl does not exists in categories already, add it and assign a new category index.
                    #         categories[lbl] = cat_idx
                    #         cat_idx += 1
                    #     classes[sample_idx][classes[sample_idx].index(lbl)] = categories[lbl]  # Replacing categories text to categories id.
                    cat_pool.clear()
                    sample_idx = line.strip()

        return classes#, categories

    def read_titles(self, classes_keys=None, title_path=None, title_file="titles.txt", encoding='latin-1'):
        """
        Reads the titles.txt file and returns a OrderedDict of id : title.

        :param classes_keys: List of classes keys to check only those keys are stored.
        :param title_file:
        :param title_path:
        :param encoding:
        :return:
        """
        # logger.debug("Generates the data dictionaries from original json file.")
        titles = OrderedDict()
        if title_path is None: title_path = join(self.raw_txt_dir, title_file)
        with sopen(title_path, encoding=encoding) as raw_title_ptr:
            for cnt, line in enumerate(raw_title_ptr):
                line = line.split()
                if classes_keys is None or line[
                    0] in classes_keys:  # Only add this sample if corrosponding classes exists.
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
        # logger.debug("Generates the data dictionaries from original json file.")
        descriptions = OrderedDict()
        # no_cat_ids = []
        if desc_path is None: desc_path = join(self.raw_txt_dir, desc_file)
        import itertools
        with sopen(desc_path, encoding=encoding) as raw_desc_ptr:
            for idx_line, desc_line in itertools.zip_longest(
                    *[raw_desc_ptr] * 2):  # Reads multi-line [2] per iteration.
                if id_phrase in idx_line:
                    sample_id = idx_line[id_remove:].strip()
                    if classes_keys is None or sample_id in classes_keys:  # Only add this sample if corrosponding classes exists.
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
        # logger.debug("Generates the data dictionaries from original json file.")
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

    def __len__(self):
        logger.info("Number of samples: [{}] for dataset: [{}]".format(self.num_samples, self.dataset_name))
        return self.num_samples

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
    sentences_val, classes_val, categories_val = cls.get_val_data()
    util.print_dict(sentences_val)
    util.print_dict(classes_val)
    util.print_dict(categories_val)


if __name__ == '__main__':
    main()
