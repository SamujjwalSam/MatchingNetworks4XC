# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6 or above
"""
__synopsis__    : WIKIHTMLLoader.
__description__ : Class to process and load html files from a directory.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2018"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : WIKI_HTML_Dataset,

__variables__   :

__methods__     :
"""

import os
import torch.utils.data
from collections import OrderedDict

from utils import util
from logger.logger import logger

seed_val = 0


class WIKI_HTML_Dataset(torch.utils.data.Dataset):
    """
    Class to process and load html files from a directory.

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
    def __init__(self, dataset_name="Wiki10-31k", run_mode="train",data_dir: str = "D:\Datasets\Extreme Classification"):
        """
        Initializes the html loader.

        Args:
            data_dir : Path to the file containing the html files.
            dataset_name : Name of the dataset.
        """
        super(WIKI_HTML_Dataset, self).__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        logger.debug("Dataset name:%s" % self.dataset_name)
        logger.debug("HTML directory:%s" % self.data_dir)
        logger.debug("Check if processed json file already exists at [{}], then load.".format(os.path.join(self.data_dir, dataset_name + "_sentences.json")))

        if run_mode == "train":
            self.sentences_train = util.load_json("_sentences_train", file_path=os.path.join(self.data_dir, dataset_name))
            self.classes_train = util.load_json("_classes_train", file_path=os.path.join(self.data_dir, dataset_name))
        elif run_mode == "val":
            self.sentences_val = util.load_json("_sentences_val", file_path=os.path.join(self.data_dir, dataset_name))
            self.classes_val = util.load_json("_classes_val", file_path=os.path.join(self.data_dir, dataset_name))
        elif run_mode == "test":
            self.sentences_test = util.load_json("_sentences_test", file_path=os.path.join(self.data_dir, dataset_name))
            self.classes_test = util.load_json("_classes_test", file_path=os.path.join(self.data_dir, dataset_name))
        else:
            raise Exception("Unknown running mode: [{}]. \n Available options: ['train','val','test']"
            .format(run_mode))

        if os.path.isfile(os.path.join(self.data_dir, dataset_name + "_sentences.json")) \
                and os.path.isfile(os.path.join(self.data_dir, dataset_name + "_classes.json")) \
                and os.path.isfile(os.path.join(self.data_dir, dataset_name + "_categories.json")):
            logger.info("Loading pre-processed json files from: [{}]".format(os.path.join(self.data_dir, dataset_name + "_sentences.json")))
            self.sentences = util.load_json(dataset_name + "_sentences", file_path=self.data_dir)
            self.classes = util.load_json(dataset_name + "_classes", file_path=self.data_dir)
            self.categories = util.load_json(dataset_name + "_categories", file_path=self.data_dir)
            assert len(self.sentences) == len(self.classes),\
                "Count of sentences [{0}] and classes [{1}] should match.".format(len(self.sentences),len(self.classes))
            self.sentences_train, self.classes_train, self.sentences_test, self.classes_test = self.split_data(self.sentences, self.classes, split_size=0.3)
            self.sentences_train, self.classes_train, self.sentences_val, self.classes_val = self.split_data(self.sentences_train,self.classes_train, split_size=0.3)

            util.save_json(self.sentences_train, dataset_name + "_sentences_train", file_path=os.path.join(self.data_dir, dataset_name))
            util.save_json(self.classes_train, dataset_name + "_classes_train", file_path=os.path.join(self.data_dir, dataset_name))
            util.save_json(self.sentences_val, dataset_name + "_sentences_val", file_path=os.path.join(self.data_dir, dataset_name))
            util.save_json(self.classes_val, dataset_name + "_classes_val", file_path=os.path.join(self.data_dir, dataset_name))
            util.save_json(self.sentences_test, dataset_name + "_sentences_test", file_path=os.path.join(self.data_dir, dataset_name))
            util.save_json(self.classes_test, dataset_name + "_classes_test", file_path=os.path.join(self.data_dir, dataset_name))
        else:
            logger.info("Loading data from HTML files.")
            html_parser = self.get_html_parser()
            self.samples = self.read_html_dir(html_parser)
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

    def split_data(self, data:dict, split_size=0.3):
        """
        Splits a dict
        :param data:
        :param split_size:
        :return:
        """
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(data, test_size=split_size, random_state=seed_val)
        return train, test

    def __getitem__(self, idx):
        # TODO: correct this part. -> Probably not required.
        return (torch.from_numpy(self.sentences[idx].todense().reshape(-1)),
                torch.from_numpy(self.classes[idx].todense().reshape(-1)))

    def get_html_parser(self, alt_text=True, ignore_table=True, decode_errors="ignore", default_alt="<IMG>",
                        ignore_link=True, reference_links=True, bypass_tables=True, ignore_emphasis=True,
                        unicode_snob=True, no_automatic_links=True, no_skip_internal_links=True, single_line_break=True,
                        escape_all=True):
        """
        Returns a html parser with config, based on: https://github.com/Alir3z4/html2text.

        Usage: https://github.com/Alir3z4/html2text/blob/master/docs/usage.md
        logger.debug(html_parser.handle("<p>Hello, <a href='http://earth.google.com/'>world</a>!"))

        ignore_links    : Ignore converting links from HTML
        images_to_alt   : Discard image data, only keep alt text
        ignore_tables   : Ignore table-related tags (table, th, td, tr) while keeping rows.
        decode_errors   : What to do in case an error is encountered. ignore, strict, replace etc.
        default_image_alt: Inserts the given alt text whenever images are missing alt values.
        :return: html2text parser.
        """
        logger.debug("Getting a HTML parser.")
        import html2text  # https://github.com/Alir3z4/html2text
        html_parser = html2text.HTML2Text()
        html_parser.images_to_alt = alt_text  # Discard image data, only keep alt text
        html_parser.ignore_tables = ignore_table  # Ignore table-related tags (table, th, td, tr) while keeping rows.
        html_parser.decode_errors = decode_errors  # Handling decoding error: "ignore", "strict", "replace" etc.
        html_parser.default_image_alt = default_alt  # Inserts the given alt text whenever images are missing alt values.
        html_parser.ignore_links = ignore_link  # Ignore converting links from HTML.
        html_parser.reference_links = reference_links  # Use reference links instead of inline links to create markdown.
        html_parser.bypass_tables = bypass_tables  # Format tables in HTML rather than Markdown syntax.
        html_parser.ignore_emphasis = ignore_emphasis  # Ignore all emphasis formatting in the html.
        html_parser.unicode_snob = unicode_snob  # Use unicode throughout instead of ASCII.
        html_parser.no_automatic_links = no_automatic_links  # Do not use automatic links like http://google.com.
        # html_parser.no_skip_internal_links = no_skip_internal_links  # Turn off skipping of internal links.
        html_parser.single_line_break = single_line_break  # Use a single line break after a block element rather than two.
        html_parser.escape_all = escape_all  # Escape all special characters.
        return html_parser

    def read_html_dir(self, html_parser, encoding="iso-8859-1"):
        """
        Reads all html files in a folder as str and returns a OrderedDict[str(filename)]=str(content).

        :param html_parser:
        :param data_dir: Path to directory of html files.
        """
        # html_parser = get_html_parser()
        data = OrderedDict()
        if os.path.isdir(self.data_dir):
            for i in os.listdir(self.data_dir):
                if os.path.isfile(os.path.join(self.data_dir, i)):
                    # if not os.path.exists(os.path.join(self.data_dir,"txt_files")):
                    # try:
                    # os.makedirs(os.path.join(self.data_dir,"txt_files"))
                    # except OSError as exc: # Guard against race condition
                    # if exc.errno != errno.EEXIST:
                    # raise
                    os.makedirs(os.path.join(self.data_dir, "txt_files"), exist_ok=True)
                    with open(os.path.join(self.data_dir, i), encoding=encoding) as html_ptr:
                        h_content = html_parser.handle(html_ptr.read())
                        # txt_ptr.write(h_content)
                        util.write_file(h_content, i, file_path=os.path.join(self.data_dir, "txt_files"))
                        data[str(i)] = str(h_content).splitlines()
                    # f.close()
        return data

    WIKI_CATEGORIES = ["Categories:", "Hidden categories:"]

    def filter_wiki_categories(self, txt: list):
        """Filters and removes categories from wikipedia text."""
        category_lines = ""
        hid_category_lines = ""
        # filtered_categories = []
        copy_flag = False
        hid_copy_flag = False
        del_start = 0
        # del_end = 0
        remove_first_chars = 12  # Length of "Categories:", to be removed from that line.
        for i, line in enumerate(txt):
            # Categories are written in multiple lines, need to read all lines (till "##### Views").
            # logger.info(line)
            if "Categories:" in line or "Category: " in line:
                del_start = i  # start index of lines to be removed.
                copy_flag = True
                remove_first_chars = 12
            if "Category: " in line:
                del_start = i  # start index of lines to be removed.
                copy_flag = True
                remove_first_chars = 11
            if "Hidden categories:" in line:
                hid_copy_flag = True
                copy_flag = False  # Stop "Categories:" as "Hidden categories:" start.
                # filtered_categories.append(line[19:].split(" | "))
            if "##### Views" in line:
                # logger.info(copy_flag)
                copy_flag = False  # Stop "Categories:" as "##### Views" start.
                hid_copy_flag = False  # Stop "Hidden categories:" as "##### Views" start.
                # del_end = i  # end index of lines to be removed.
            if copy_flag:
                # logger.debug((category_lines,line))
                # logger.debug(txt[i-1])
                category_lines = category_lines + " " + line
            if hid_copy_flag:
                hid_category_lines = hid_category_lines + " " + line
            # del txt[del_start:del_end]  # Delete lines containing category info.
            del txt[del_start:]  # After category info all lines are either copyright or not required.

        filtered_categories = category_lines[remove_first_chars:].split(" | ")
        # filtered_categories.append(hid_category_lines[19:].split(" | ")[0])  # Do not add hidden categories.
        filtered_categories = [cat.strip() for cat in filtered_categories]
        # logger.debug(filtered_categories)
        filtered_categories = list(filter(None, filtered_categories))  # Removing empty items.
        # logger.debug(filtered_categories)
        return txt, filtered_categories

    def filter_wiki(self, datapoints: dict):
        """Filters sentences, classes and categories from wikipedia text.

        :return: Dict of sentences, classes and categories filtered from samples.
        """
        classes = OrderedDict()
        categories = OrderedDict()
        sentences = OrderedDict()
        i = 0
        no_cat_ids = []  # List to store failed parsing cases.
        for id, txt in datapoints.items():
            # logger.debug(type(txt))
            clean_txt, filtered_categories = self.filter_wiki_categories(txt)
            # assert filtered_categories, "No category information was found for id: [{0}].".format(id)
            if filtered_categories:  # Check at least one category was successfully filtered from html file.
                sentences[id] = clean_txt
                # categories_list.append(set(filtered_categories))
                for lbl in filtered_categories:
                    if lbl not in categories:  # If lbl does not exists in categories already, add it and assign a new category index.
                        categories[lbl] = i
                        i += 1
                    if id in classes:  # Check if id exists, append if yes.
                        classes[id].append(categories[lbl])
                    else:  # Create entry for id if does not exist.
                        classes[id] = [categories[lbl]]
            else:  # If no category was found, store the id in a separate place for later inspection.
                logger.warn("No categories[{0}] found for id: [{1}]".format(filtered_categories,id))
                logger.warn("Storing id [{0}] for future use.".format(id))
                no_cat_ids.append(id)
                # return False
        # assert len(self.samples) == len(classes), "Count of samples and classes should match."
        # assert len(self.samples) == len(sentences), "Count of samples and sentences should also match."
        # logger.info(categories)
        # categories = OrderedDict([(k,v) for k,v in categories.items() if len(k)>0])  #
        logger.info(no_cat_ids)
        # logger.info((len(sentences), classes, categories))
        return sentences, classes, categories, no_cat_ids

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
    cls = WIKI_HTML_Dataset()
    data_dict = cls.read_html_dir("D:\Datasets\Extreme Classification\html_test")
    logger.debug(data_dict)
    return False


if __name__ == '__main__':
    main()
