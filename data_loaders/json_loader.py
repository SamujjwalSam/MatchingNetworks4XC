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
from collections import OrderedDict
from smart_open import smart_open as sopen  # Better alternative to Python open().

from utils import util
from logger.logger import logger
# from data_loaders.common_data_handler import JSON_Handler

seed_val = 0


class JSONLoader(torch.utils.data.Dataset):
    """
    Class to process and load json files from data directory.

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
                 dataset_name="AmazonCat-14K",
                 run_mode="train",
                 data_dir: str = "D:\\Datasets\\Extreme Classification"):
        """
        Initializes the JSON loader.

        Args:
            data_dir : Path to directory of the dataset.
            dataset_name : Name of the dataset.
        """
        super(JSONLoader, self).__init__()
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(data_dir, self.dataset_name)
        self.raw_json_dir = os.path.join(self.data_dir, self.dataset_name + "_RawData")
        self.raw_json_file = self.dataset_name + "_RawData.json"
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
                     .format(os.path.join(self.data_dir, self.dataset_name + "_sentences_train.json")))
        # if run_mode == "train" or run_mode == "val" or run_mode == "test":
        #     if os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_sentences_"+run_mode+".json")) \
        #             and os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_classes_"+run_mode+".json")) \
        #             and os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_categories_"+run_mode+".json")):
        #         self.sentences_train = util.load_json(self.dataset_name + "_sentences_"+run_mode, file_path=self.data_dir)
        #         self.classes_train = util.load_json(self.dataset_name + "_classes_"+run_mode, file_path=self.data_dir)
        #         self.categories_train = util.load_json(self.dataset_name + "_categories_"+run_mode, file_path=self.data_dir)
        #     else:
        #         self.load_full_json()
        if run_mode == "train":
            if os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_sentences_train.json")) \
                    and os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_classes_train.json")) \
                    and os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_categories_train.json")):
                self.sentences_train = util.load_json(self.dataset_name + "_sentences_train", file_path=self.data_dir)
                self.classes_train = util.load_json(self.dataset_name + "_classes_train", file_path=self.data_dir)
                self.categories_train = util.load_json(self.dataset_name + "_categories_train", file_path=self.data_dir)
            else:
                self.load_full_json()
        elif run_mode == "val":
            if os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_sentences_val.json")) \
                    and os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_classes_val.json")) \
                    and os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_categories_val.json")):
                self.sentences_val = util.load_json(self.dataset_name + "_sentences_val", file_path=self.data_dir)
                self.classes_val = util.load_json(self.dataset_name + "_classes_val", file_path=self.data_dir)
                self.categories_val = util.load_json(self.dataset_name + "_categories_val", file_path=self.data_dir)
            else:
                self.load_full_json()
        elif run_mode == "test":
            if os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_sentences_test.json")) \
                    and os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_classes_test.json")) \
                    and os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_categories_test.json")):
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
        # logger.debug(os.path.join(self.data_dir, self.dataset_name + "_sentences.json"))
        if os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_sentences.json")) \
                and os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_classes.json")) \
                and os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_categories.json")):
            logger.info("Loading pre-processed json files from: [{}]".format(
                os.path.join(self.data_dir, self.dataset_name)))
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
            sentences, classes, categories, no_cat_ids = self.gen_dicts(json_path=os.path.join(self.raw_json_dir,self.raw_json_file), encoding="UTF-8")
            util.save_json(no_cat_ids, self.dataset_name + "_no_cat_ids",
                           file_path=self.data_dir)  # Storing the ids for which no categories were found.
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
                os.path.join(self.data_dir + "_sentences.json"),
                os.path.join(self.data_dir + "_classes.json"),
                os.path.join(self.data_dir + "_categories.json")))

        # Splitting data into train, validation and test sets.
        data_loader = JSON_Handler(dataset_name=self.dataset_name, data_dir=self.data_dir)
        self.sentences_train, self.classes_train, self.categories_train, self.sentences_val, self.classes_val, self.categories_val, self.sentences_test, self.classes_test, self.categories_test = data_loader.split_data(
            sentences=self.sentences, classes=self.classes, categories=self.categories)
        self.num_samples = len(self.sentences)
        # return self.sentences,self.classes,self.categories

    def gen_dicts(self,json_path=None, encoding='latin-1'):
        """
        Generates the data dictionaries from original json file.

        :param json_path: Path to raw json file.
        :param encoding: Encoding for the raw json file.
        :return: sentences, classes, categories, no_cat_ids
            no_cat_ids: ids for which no categories were found.
        """
        import ast  # As the data is not proper JSON (single-quote instead of double-quote) format, "json" library will not work.

        logger.debug("Generates the data dictionaries from original json file.")
        sentences = OrderedDict()
        classes = OrderedDict()
        categories = OrderedDict()
        no_cat_ids = []
        if json_path is None: json_path = self.raw_json_dir
        with sopen(json_path, encoding=encoding) as raw_json_ptr:
            cat_idx = 0  # Holds the category index.
            for cnt, line in enumerate(raw_json_ptr):
                # Instead of: line_dict = OrderedDict(json.loads(line));
                # Use: import ast; line_dict = ast.literal_eval(line.strip().replace('\n','\\n'));
                line_dict = ast.literal_eval(line.strip().replace('\n','\\n'))
                if "categories" in line_dict:  # Check if "categories" exists.
                    if "title" in line_dict:  # Check if "title" exists, add if True.
                        sentences[line_dict["asin"]] = line_dict["title"]
                        if "description" in line_dict:  # Check if "description" exists and append to "title" with keyword: ". \nDESC: ", if true.
                            sentences[line_dict["asin"]] = sentences[line_dict["asin"]] + ". \nDESC: " + line_dict["description"]
                    else:
                        if "description" in line_dict:  # Check if "description" exists even though "title" does not, use only "description" if true.
                            sentences[line_dict["asin"]] = ". \nDESC: " + line_dict["description"]
                        else:  # Report and skip the sample if neither "title" nor "description" exists.
                            logger.warning("Neither 'title' nor 'description' found for sample id: [{}]. Adding sample to 'no_cat_ids'.".format(line_dict["asin"]))
                            no_cat_ids.append(line_dict["asin"])  # As neither "title" nor "description" exists, adding the id to "no_cat_ids".
                            continue
                    classes[line_dict["asin"]] = line_dict["categories"][0]
                    for lbl in classes[line_dict["asin"]]:
                        if lbl not in categories:  # If lbl does not exists in categories already, add it and assign a new category index.
                            categories[lbl] = cat_idx
                            cat_idx += 1
                        classes[line_dict["asin"]][classes[line_dict["asin"]].index(lbl)] = categories[lbl]  # Replacing categories text to categories id.
                else:  # if "categories" does not exist, then add the id to "no_cat_ids".
                    no_cat_ids.append(line_dict["asin"])

        logger.info("Number of sentences: [{}], classes: [{}] and categories: [{}].".format(len(sentences),len(classes),len(categories)))
        return sentences, classes, categories, no_cat_ids

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
            if os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_sentences_val.json")):
                self.sentences_val = util.load_json(self.dataset_name + "_sentences_val", file_path=self.data_dir)

        if self.classes_val is None:
            if os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_classes_val.json")):
                self.classes_val = util.load_json(self.dataset_name + "_classes_val", file_path=self.data_dir)

        if self.categories_val is None:
            if os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_categories_val.json")):
                self.categories_val = util.load_json(self.dataset_name + "_categories_val", file_path=self.data_dir)
        return self.sentences_val, self.classes_val, self.categories_val

    def get_test_data(self):
        """
        Function to get the entire dataset
        """
        if os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_sentences_test.json")) \
                and os.path.isfile(os.path.join(self.data_dir, self.dataset_name + "_classes_test.json")):
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
    cls = JSONLoader()
    sentences_val, classes_val, categories_val = cls.get_val_data()
    util.print_dict(sentences_val)
    util.print_dict(classes_val)
    util.print_dict(categories_val)


if __name__ == '__main__':
    main()
