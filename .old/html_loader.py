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

from os.path import join, isfile
import torch.utils.data
from collections import OrderedDict

from utils import util
from logger.logger import logger
from data_loaders.common_data_handler import Common_JSON_Handler

seed_val = 0
WIKI_CATEGORIES = ["Categories:", "Category:", "Hidden categories:"]


class HTMLLoader(torch.utils.data.Dataset):
    """
    Class to process and load html files from a directory.

    Datasets: Wiki10-31K

    sentences : Wikipedia english texts after parsing and cleaning.
    sentences = {"id1": "wiki_text_1", "id2": "wiki_text_2"}

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
                 dataset_name="Wiki10-31K",
                 run_mode="train",
                 data_dir: str = "D:\\Datasets\\Extreme Classification"):
        """
        Initializes the html loader.

        Args:
            data_dir : Path to directory of the dataset.
            dataset_name : Name of the dataset.
        """
        super(HTMLLoader, self).__init__()
        self.dataset_name = dataset_name
        self.data_dir = join(data_dir, self.dataset_name)
        self.raw_html_dir = join(self.data_dir, dataset_name + "_RawData")
        self.raw_txt_dir = join(self.data_dir, "txt_files")
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
        Loads dataset.

        :param dataset_name:
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
            if isdir(self.raw_txt_dir):
                logger.info("Loading data from TXT files.")
                self.samples = self.read_txt_dir(self.raw_txt_dir)
            else:
                logger.info("Could not find TXT files: [{}]".format(self.raw_txt_dir))
                logger.info("Loading data from HTML files.")
                html_parser = self.get_html_parser()
                self.samples = self.read_html_dir(html_parser)
            logger.info("Creating 3 separate dicts of sentences[id->texts], classes[id->class_ids]"
                        "and categories[class_name : class_id] from HTML.")
            sentences, classes, categories, hid_classes, hid_categories, no_cat_ids = self.filter_categories(
                self.samples)
            util.save_json(no_cat_ids, self.dataset_name + "_no_cat_ids",
                           file_path=self.data_dir)  # Storing the ids for which no categories were found.
            util.save_json(hid_classes, self.dataset_name + "_hid_classes",
                           file_path=self.data_dir)  # Storing details of file id to hidden categories map.
            util.save_json(hid_categories, self.dataset_name + "_hid_categories",
                           file_path=self.data_dir)  # Storing dict of hidden categories.
            self.classes = classes
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
        # return self.sentences,self.classes,self.categories

    def read_txt_dir(self, raw_txt_dir, encoding="iso-8859-1"):
        """
        Reads all txt files from [self.raw_txt_dir] folder as str and returns a OrderedDict[str(filename)]=str(content).

        :param raw_txt_dir:
        :param encoding:
        :param html_parser:
        :param data_dir: Path to directory of html files.
        """
        data = OrderedDict()
        if raw_txt_dir is None: raw_txt_dir = self.raw_txt_dir
        logger.debug("Raw TXT path: {}".format(raw_txt_dir))
        if isdir(raw_txt_dir):
            for i in os.listdir(raw_txt_dir):
                if isfile(join(raw_txt_dir, i)) and i.endswith(".txt"):
                    with open(join(raw_txt_dir, i), encoding=encoding) as txt_ptr:
                        data[str(i[:-4])] = str(
                            txt_ptr.read()).splitlines()  # [:-4] to remove the ".txt" from sample id.
        return data

    def __len__(self):
        logger.info("Number of samples: [{}] for dataset: [{}]".format(self.num_samples, self.dataset_name))
        return self.num_samples

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
        logger.info("Getting HTML parser.")
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

        :param encoding:
        :param html_parser:
        :param data_dir: Path to directory of html files.
        """
        # html_parser = get_html_parser()
        data = OrderedDict()
        # logger.debug("Raw HTML path: {}".format(self.raw_html_dir))
        os.makedirs(join(self.data_dir, "txt_files"), exist_ok=True)
        if isdir(self.raw_html_dir):
            for i in os.listdir(self.raw_html_dir):
                if isfile(join(self.raw_html_dir, i)):
                    with open(join(self.raw_html_dir, i), encoding=encoding) as html_ptr:
                        h_content = html_parser.handle(html_ptr.read())
                        util.write_file(h_content, i, file_path=join(self.data_dir, "txt_files"))
                        data[str(i)] = str(h_content).splitlines()
        return data

    def filter_wiki_categories(self, txt: list):
        """Filters and removes categories from wikipedia text."""
        category_lines = ""
        hid_category_lines = ""
        # filtered_categories = []
        copy_flag = False
        hid_copy_flag = False
        del_start = 0
        # del_end = 0
        remove_first_chars = 12  # Length of "Categories:", to be removed from line.
        for i, line in enumerate(txt):
            # Categories are written in multiple lines, need to read all lines (till "##### Views").
            # logger.debug(line)
            if "Categories:" in line or "Category: " in line:
                del_start = i  # start index of lines to be removed.
                copy_flag = True
                remove_first_chars = 12
                # logger.debug("Categories")
            if "Category: " in line:
                del_start = i  # start index of lines to be removed.
                copy_flag = True
                remove_first_chars = 11
                # logger.debug("Category")
            if "Hidden categories:" in line:
                hid_copy_flag = True
                copy_flag = False  # Stop "Categories:" as "Hidden categories:" start.
                # filtered_categories.append(line[19:].split(" | "))
                # logger.debug("Hidden")
            if "##### Views" in line:
                # logger.debug(copy_flag)
                copy_flag = False  # Stop coping "Categories:" as "##### Views" start.
                hid_copy_flag = False  # Stop coping "Hidden categories:" as "##### Views" start.
                # del_end = i  # end index of lines to be removed.
                # logger.debug("Views")
            if copy_flag:
                # logger.debug((category_lines,line))
                # logger.debug(txt[i-1])
                category_lines = category_lines + " " + line
                # logger.debug("copy_flag")
            if hid_copy_flag:
                hid_category_lines = hid_category_lines + " " + line
                # logger.debug("hid_copy_flag")
            # del txt[del_start:del_end]  # Delete lines containing category info. -> below line.
        del txt[del_start:]  # After category info all lines are either copyright or not required.

        filtered_categories = category_lines[remove_first_chars:].split(" | ")
        filtered_hid_categories = (hid_category_lines[19:].split(" | "))  # Do not add hidden categories to categories.
        # logger.debug(filtered_hid_categories)
        filtered_categories = [cat.strip() for cat in filtered_categories]
        filtered_hid_categories = [cat.strip() for cat in filtered_hid_categories]
        # logger.debug(filtered_hid_categories)
        # logger.debug(filtered_categories)
        filtered_categories = list(filter(None, filtered_categories))  # Removing empty items.
        filtered_hid_categories = list(filter(None, filtered_hid_categories))  # Removing empty items.
        # logger.debug(filtered_hid_categories)
        return txt, filtered_categories, filtered_hid_categories

    def filter_categories(self, samples: dict):
        """Filters sentences, classes and categories from wikipedia text.

        :return: Dict of sentences, classes and categories filtered from samples.
        """
        classes = OrderedDict()
        hid_classes = OrderedDict()
        categories = OrderedDict()
        hid_categories = OrderedDict()
        sentences = OrderedDict()
        cat_idx = 0
        hid_cat_idx = 0
        no_cat_ids = []  # List to store failed parsing cases.
        for id, txt in samples.items():
            # logger.debug(type(txt))
            clean_txt, filtered_categories, filtered_hid_categories = self.filter_wiki_categories(txt)
            # assert filtered_categories, "No category information was found for id: [{0}].".format(id)
            if filtered_categories:  # Check at least one category was successfully filtered from html file.
                sentences[id] = clean_txt
                # categories_list.append(set(filtered_categories))
                for lbl in filtered_categories:
                    if lbl not in categories:  # If lbl does not exists in categories already, add it and assign a new category index.
                        categories[lbl] = cat_idx
                        cat_idx += 1
                    if id in classes:  # Check if id exists, append if yes.
                        classes[id].append(categories[lbl])
                    else:  # Create entry for id if does not exist.
                        classes[id] = [categories[lbl]]
            else:  # If no category was found, store the id in a separate place for later inspection.
                logger.warn(
                    "No categories [{0}] found for id: [{1}]. Storing for future use.".format(filtered_categories, id))
                no_cat_ids.append(id)
                # return False
            if filtered_hid_categories:  # Check at least one category was successfully filtered from html file.
                # categories_list.append(set(filtered_categories))
                for lbl in filtered_hid_categories:
                    if lbl not in hid_categories:  # If lbl does not exists in hid_categories already, add it and assign a new hid_category index.
                        hid_categories[lbl] = hid_cat_idx
                        hid_cat_idx += 1
                    if id in hid_classes:  # Check if id exists, append if yes.
                        hid_classes[id].append(hid_categories[lbl])
                    else:  # Create entry for id if does not exist.
                        hid_classes[id] = [hid_categories[lbl]]
            # else:  # If no category was found, store the id in a separate place for later inspection.
            #     logger.warn("No categories[{0}] found for id: [{1}]".format(filtered_hid_categories,id))
            #     logger.warn("Storing id [{0}] for future use.".format(id))
            # no_cat_ids.append(id)
            # return False

        # assert len(self.samples) == len(classes), "Count of samples and classes should match."
        # assert len(self.samples) == len(sentences), "Count of samples and sentences should also match."
        # logger.debug(categories)
        # categories = OrderedDict([(k,v) for k,v in categories.items() if len(k)>0])  #
        logger.debug(no_cat_ids)
        # logger.debug((len(sentences), classes, categories))
        return sentences, classes, categories, hid_classes, hid_categories, no_cat_ids

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
    cls = HTMLLoader()
    # data_dict = cls.read_html_dir("D:\Datasets\Extreme Classification\html_test")
    sentences_val, classes_val, categories_val = cls.get_val_data()
    util.print_dict(sentences_val)
    util.print_dict(classes_val)
    util.print_dict(categories_val)


if __name__ == '__main__':
    main()