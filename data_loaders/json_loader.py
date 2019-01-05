# coding=utf-8

#  coding=utf-8
#  !/usr/bin/python3.6
#
#  """
#  Author : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
#  Version : "0.1"
#  Date : "5/1/19 11:44 AM"
#  Copyright : "Copyright (c) 2019. All rights reserved."
#  Licence : "This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree."
#  Last modified : 5/1/19 11:42 AM.
#  """

# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : JSONLoader.
__description__ : Class to process and load json files from a directory.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
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
        self.sentences, self.classes, self.categories = self.gen_dicts(json_path=os.path.join(self.raw_json_dir,self.raw_json_file), encoding="UTF-8")

    def gen_dicts(self,json_path=None, encoding='latin-1',specials="""_-@*#'"/\\""", replace=' '):
        """
        Generates the data dictionaries from original json file.

        :param replace: Character to replace with.
        :param specials: Characters to clean from sentences.
        :param json_path: Path to raw json file.
        :param encoding: Encoding for the raw json file.
        :return: sentences, classes, categories, no_cat_ids
            no_cat_ids: ids for which no categories were found.
        """
        import ast  # As the data is not proper JSON (single-quote instead of double-quote) format, "json" library will not work.
        from unidecode import unidecode

        logger.debug("Generates the data dictionaries from original json file.")
        sentences = OrderedDict()
        classes = OrderedDict()
        categories = OrderedDict()
        no_cat_ids = []  # To store ids for which no categories were found.

        if json_path is None: json_path = self.raw_json_dir
        with sopen(json_path, encoding=encoding) as raw_json_ptr:
            trans_table = util.make_trans_table(specials=specials, replace=replace)  # Creating mapping to clean sentences.
            cat_idx = 0  # Holds the category index.
            for cnt, line in enumerate(raw_json_ptr):
                # Instead of: line_dict = OrderedDict(json.loads(line));
                # Use: import ast; line_dict = ast.literal_eval(line.strip().replace('\n','\\n'));
                line_dict = ast.literal_eval(line.strip().replace('\n','\\n'))
                if "categories" in line_dict:  # Check if "categories" exists.
                    if "title" in line_dict:  # Check if "title" exists, add if True.
                        sentences[line_dict["asin"]] = unidecode(str(line_dict["title"])).translate(trans_table)
                        if "description" in line_dict:  # Check if "description" exists and append to "title" with keyword: ". \nDESC: ", if true.
                            sentences[line_dict["asin"]] = sentences[line_dict["asin"]] + ". \nDESC: " + unidecode(str(line_dict["description"])).translate(trans_table)
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

        util.save_json(no_cat_ids, self.dataset_name + "_no_cat_ids", file_path=self.data_dir)
        logger.info("Number of sentences: [{}], classes: [{}] and categories: [{}]."
                    .format(len(sentences),len(classes),len(categories)))
        return sentences, classes, categories

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
    cls = JSONLoader()
    categories_val = cls.get_categories()
    util.print_dict(categories_val)


if __name__ == '__main__':
    main()
