# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Utility functions for Matching Networks for Extreme Classification.

__description__ :
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.
__classes__     :
__variables__   :
__methods__     :
"""

import json, unicodedata
import pickle as pk

from os.path import join, exists, isfile
from random import sample, shuffle
from smart_open import smart_open as sopen  # Better alternative to Python open().
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict
from scipy import sparse, math
from unidecode import unidecode

from logger.logger import logger
# from config import configuration as config
# from config import platform as plat
        
        
def print_dict(data, count=5):
    """
    Prints the key and values of a Python dict.

    :param data:
    :param count:
    """
    i = 0
    for k, v in data.items():
        logger.debug("{} : {}".format(k, v))
        i += 1
        if i >= count:
            break


def print_json(json_data, s="", indent=4, sort_keys=True):
    """
    Pretty prints json data.

    :param sort_keys:
    :param indent:
    :param s:
    :param json_data:
    """
    logger.info("[{}] : {}".format(s, json.dumps(json_data, indent=indent, sort_keys=sort_keys)))


def create_batch(X: dict, Y: dict, keys):
    """
    Generates batch from keys.

    :param X:
    :param Y:
    :param keys:
    :return:
    """
    batch_x = OrderedDict()
    batch_y = OrderedDict()
    for k in keys:
        batch_x[k] = X[k]
        batch_y[k] = Y[k]
    return batch_x, batch_y


def create_batch_repeat(X: dict, Y: dict, keys):
    """
    Generates batch from keys.

    :param X:
    :param Y:
    :param keys:
    :return:
    """
    batch_x = []
    batch_y = []
    shuffle(keys)
    for k in keys:
        batch_x.append(X[k])
        batch_y.append(Y[k])
    return batch_x, batch_y


def get_batch_keys(keys: list, batch_size=64, remove_keys=True):
    """
    Randomly selects [batch_size] numbers of key from keys list and remove them from the original list.

    :param remove_keys: Flag to indicate if selected keys should be removed. It should be False for support set selection.
    :param keys:
    :param batch_size:
    :return:
    """
    if len(keys) <= batch_size:
        return keys, keys
    selected_keys = sample(keys, k=batch_size)
    if remove_keys:
        keys_after_remove = []
        for item in keys:
            if item not in selected_keys:
                keys_after_remove.append(item)
        return keys_after_remove, selected_keys
    return keys, selected_keys


def split_dict(classes: dict, sentences: dict, batch_size=64, remove_keys=True):
    """
    Randomly selects [batch_size] numbers of items from dictionary and remove them from the original dict.

    :param remove_keys: Flag to indicate if selected items should be removed. It should be False for support set selection.
    :param classes:
    :param batch_size:
    :return:
    """
    if len(classes) <= batch_size:
        return None, classes, None, sentences
    selected_keys = sample(list(classes.keys()), k=batch_size)
    selected_classes = OrderedDict()
    selected_sentences = OrderedDict()
    if remove_keys:
        for key in selected_keys:
            selected_classes[key] = classes[key]
            del classes[key]
            selected_sentences[key] = sentences[key]
            del sentences[key]
        return classes, selected_classes, sentences, selected_sentences
    else:
        for key in selected_keys:
            selected_classes[key] = classes[key]
            selected_sentences[key] = sentences[key]
    return classes, selected_classes, sentences, selected_sentences


def remove_dup_list(seq, case=False):  # Dave Kirby
    """Removes duplicates from a list. Order preserving"""
    seen = set()
    if case: return [x.lower() for x in seq if
                     x.lower() not in seen and not seen.add(x)]
    return [x for x in seq if x not in seen and not seen.add(x)]


def split_docs(docs, criteria=' '):
    """
    Splits a dict of idx:documents based on [criteria].

    :param docs: idx:documents
    :param criteria:
    :return:
    """
    splited_docs = OrderedDict()
    for idx, doc in docs:
        splited_docs[idx] = doc.split(criteria)
    return splited_docs


def unicodeToAscii(s):
    """
    Turn a Unicode string to plain ASCII. Thanks to http://stackoverflow.com/a/518232/2809427

    :param s:
    :return:
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def make_trans_table(specials="""< >  * ? " / \\ : |""", replace=' '):
    """
    Makes a transition table to replace [specials] chars within [text] with [replace].

    :param specials:
    :param replace:
    :return:
    """
    trans_dict = {chars: replace for chars in specials}
    trans_table = str.maketrans(trans_dict)
    return trans_table


def clean_categories(categories: dict, specials="""_-@""", replace=' '):
    """Cleans categories dict and returns set of cleaned categories and the dict of duplicate categories.

    :param: categories: dict of cat:id
    :param: specials: list of characters to clean.
    :returns:
        category_cleaned_dict : contains categories which are unique after cleaning.
        dup_cat_map : Dict of new category id mapped to old category id. {old_cat_id : new_cat_id}
    """
    category_cleaned_dict = OrderedDict()
    dup_cat_map = OrderedDict()
    dup_cat_text_map = OrderedDict()
    trans_table = make_trans_table(specials=specials, replace=replace)
    for cat, cat_id in categories.items():
        cat_clean = unidecode(str(cat)).translate(trans_table)
        if cat_clean in category_cleaned_dict.keys():
            dup_cat_map[categories[cat]] = category_cleaned_dict[cat_clean]
            dup_cat_text_map[cat] = cat_clean
        else:
            category_cleaned_dict[cat_clean] = cat_id
    return category_cleaned_dict, dup_cat_map, dup_cat_text_map


def clean_sentences_dict(sentences: dict, specials="""_-@*#'"/\\""", replace=' '):
    """Cleans sentences dict and returns dict of cleaned sentences.

    :param: sentences: dict of idx:label
    :returns:
        sents_cleaned_dict : contains cleaned sentences.
    """
    # TODO: Remove all headings with "##"
    # TODO: Remove ### From Wikipedia, the free encyclopedia \n Jump to: navigation, search
    # TODO: Remove Repeated special characters like ####,     , ,, etc.
    sents_cleaned_dict = OrderedDict()
    trans_table = make_trans_table(specials=specials, replace=replace)
    for idx, text in sentences.items():
        sents_cleaned_dict[idx] = unidecode(str(text)).translate(trans_table)
    return sents_cleaned_dict


def clean_sentences(sentences: list, specials="""_-@*#'"/\\""", replace=' '):
    """Cleans sentences dict and returns dict of cleaned sentences.

    :param: sentences: dict of idx:label
    :returns:
        sents_cleaned_dict : contains cleaned sentences.
    """
    # TODO: Remove all headings with "##"
    # TODO: Remove ### From Wikipedia, the free encyclopedia \n Jump to: navigation, search
    # TODO: Remove Repeated special characters like ####,     , ,, etc.
    sents_cleaned_dict = []
    trans_table = make_trans_table(specials=specials, replace=replace)
    for text in sentences:
        sents_cleaned_dict.append(unidecode(str(text)).translate(trans_table))
    return sents_cleaned_dict


def remove_special_chars(text, specials="""<>*?"/\\:|""", replace=' '):
    """
    Replaces [specials] chars from [text] with [replace].

    :param text:
    :param specials:
    :param replace:
    :return:
    """
    text = unidecode(str(text))
    trans_dict = {chars: replace for chars in specials}
    trans_table = str.maketrans(trans_dict)
    return text.translate(trans_table)


def dedup_data(Y: dict, dup_cat_map: dict):
    """
    Replaces category ids in Y if it's duplicate.

    :param Y:
    :param dup_cat_map: {old_cat_id : new_cat_id}
    """
    for k, v in Y.items():
        commons = set(v).intersection(set(dup_cat_map.keys()))
        if len(commons) > 0:
            for dup_key in commons:
                dup_idx = v.index(dup_key)
                v[dup_idx] = dup_cat_map[dup_key]
    return Y


def inverse_dict_elm(labels: dict):
    """
    Inverses key to value of a dict and vice versa. Retains the initial values if key is repeated.

    :param labels:
    :return:
    """
    labels_inv = OrderedDict()
    for k, v in labels.items():
        if v not in labels_inv:  # check if key does not exist.
            labels_inv[int(v)] = k
    return labels_inv


def get_date_time_tag(caller=False):
    from datetime import datetime
    date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = str(date_time)
    if caller:
        tag = caller + "_" + str(date_time)
    return tag


def get_dataset_path():
    """
    Returns dataset path based on OS.

    :return: Returns dataset path based on OS.
    """
    import platform
    from sys import path

    if platform.system() == 'Windows':
        dataset_path = 'D:\\Datasets\\Extreme Classification'
        path.append('D:\\GDrive\\Dropbox\\IITH\\0 Research')
    elif platform.system() == 'Linux':
        # dataset_path = '/raid/ravi'
        dataset_path = '/home/cs16resch01001/datasets/Extreme Classification'
        # sys.path.append('/home/cs16resch01001/codes')
    else:  # OS X returns name 'Darwin'
        dataset_path = '/Users/monojitdey/ml/datasets'
    logger.debug(str(platform.system()) + " os detected.")

    return dataset_path


def get_platform():
    """
    Returns dataset path based on OS.

    :return: Returns dataset path based on OS.
    """
    import platform

    if platform.system() == 'Windows':
        return platform.system()
    elif platform.system() == 'Linux':
        return platform.system()
    else:  # OS X returns name 'Darwin'
        return "OSX"


def save_json(data, filename, file_path='', overwrite=False, indent=2, date_time_tag=''):
    """

    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :param indent:
    :param date_time_tag:
    :return:
    """
    logger.debug("Saving JSON file: [{}]".format(join(file_path, date_time_tag + filename + ".json")))
    if not overwrite and exists(join(file_path, date_time_tag + filename + ".json")):
        logger.warning("File [{}] already exists and Overwrite == False.".format(
            join(file_path, date_time_tag + filename + ".json")))
        return True
    try:
        with sopen(join(file_path, date_time_tag + filename + ".json"), 'w') as json_file:
            try:
                json_file.write(json.dumps(data, indent=indent))
            except Exception as e:
                logger.warning("Writing JSON failed: [{}]".format(e))
                logger.warning(
                    "Writing as string: [{}]".format(join(file_path, date_time_tag + filename + ".json")))
                json_file.write(json.dumps(str(data), indent=indent))
                return True
        json_file.close()
        return True
    except Exception as e:
        logger.warning("Writing JSON file [{}] failed: [{}]".format(join(file_path, filename), e))
        logger.warning("Writing as TXT: [{}]".format(filename + ".txt"))
        write_file(data, filename, date_time_tag=date_time_tag)
        return False


def load_json(filename, file_path='', date_time_tag='', ext=".json", show_path=False):
    """
    Loads json file as python OrderedDict.

    :param ext: Should extension be appended?
    :param filename:
    :param file_path:
    :param date_time_tag:
    :return: OrderedDict
    """
    file_loc = join(file_path, date_time_tag + filename + ext)
    if show_path:
        logger.info("Reading JSON file: [{}]".format(file_loc))
    if exists(join(file_path, date_time_tag + filename + ext)):
        try:
            with sopen(file_loc, encoding="utf-8") as file:
                json_dict = json.load(file)
                json_dict = OrderedDict(json_dict)
                # json_dict = OrderedDict(json.load(file))
            file.close()
            return json_dict
        except Exception as e:
            logger.warning("Could not open file as JSON: [{}]. \n Reason:[{}]".format(file_loc,e))
            with sopen(file_loc, encoding="utf-8") as file:
                json_dict = str(file)
                json_dict = json.loads(json_dict)
                # json_dict = OrderedDict(json_dict)
            return json_dict
    else:
        logger.warning("File does not exist at: [{}]".format(file_loc))
        return False


def read_json_str(filename, file_path='', date_time_tag='', ext="", show_path=False):
    """
    Loads json file as python OrderedDict.

    :param show_path:
    :param ext: Should extension be appended?
    :param filename:
    :param file_path:
    :param date_time_tag:
    :return: OrderedDict
    """
    file_loc = join(file_path, date_time_tag + filename + ext)
    if show_path:
        logger.info("Reading JSON file: [{}]".format(file_loc))
    if exists(join(file_path, date_time_tag + filename + ext)):
        raw_json=open(file_loc, 'r').read()
        json_dict = json.loads(raw_json)
        logger.debug(json_dict)
        return json_dict
    else:
        logger.warning("File does not exist at: [{}]".format(file_loc))
        return False


def write_file(data, filename, file_path='', overwrite=False, mode='w', encoding="utf-8", date_time_tag='',
               verbose=False):
    """

    :param verbose:
    :param encoding:
    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :param mode:
    :param date_time_tag:
    :return:
    """
    if not overwrite and exists(join(file_path, date_time_tag + filename + ".txt")):
        # logger.warning("File [{}] already exists and Overwrite == False.".format(
        #     join(file_path, date_time_tag + filename + ".txt")))
        return True
    with sopen(join(file_path, date_time_tag + filename + ".txt"), mode, encoding=encoding) as text_file:
        if verbose:
            logger.debug("Saving text file: [{}]".format(join(file_path, date_time_tag + filename + ".txt")))
        text_file.write(str(data))
        text_file.write("\n")
        text_file.write("\n")
    text_file.close()


def load_npz(filename, file_path=''):
    """
    Loads numpy objects from npz files.

    :param filename:
    :param file_path:
    :return:
    """
    logger.debug("Reading NPZ file: [{}]".format(join(file_path, filename + ".npz")))
    if isfile(join(file_path, filename + ".npz")):
        npz = sparse.load_npz(join(file_path, filename + ".npz"))
        return npz
    else:
        logger.warning("Could not open file: [{}]".format(join(file_path, filename + ".npz")))
        return False


def save_npz(data, filename, file_path='', overwrite=False):
    """
    Saves numpy objects to file.

    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :return:
    """
    logger.debug("Saving NPZ file: [{}]".format(join(file_path, filename + ".npz")))
    if not overwrite and exists(join(file_path, filename + ".npz")):
        logger.warning(
            "File [{}] already exists and Overwrite == False.".format(join(file_path, filename + ".npz")))
        return True
    try:
        sparse.save_npz(join(file_path, filename + ".npz"), data)
        return True
    except Exception as e:
        logger.warning("Could not write to npz file: [{}]".format(join(file_path, filename + ".npz")))
        logger.warning("Failure reason: [{}]".format(e))
        return False


def save_pickle(data, filename, file_path, overwrite=False):
    """
    Saves python object as pickle file.

    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :return:
    """
    # logger.debug("Method: save_pickle(data, filename, file_path, overwrite=False)")
    logger.debug("Writing to pickle file: [{}]".format(join(file_path, filename + ".pkl")))
    if not overwrite and exists(join(file_path, filename + ".pkl")):
        logger.warning("File [{}] already exists and Overwrite == False.".format(
            join(file_path, filename + ".pkl")))
        return True
    try:
        if isfile(join(file_path, filename + ".pkl")):
            logger.info(
                "Overwriting on pickle file: [{}]".format(join(file_path, filename + ".pkl")))
        with sopen(join(file_path, filename + ".pkl"), 'wb') as pkl_file:
            pk.dump(data, pkl_file)
        pkl_file.close()
        return True
    except Exception as e:
        logger.warning(
            "Could not write to pickle file: [{}]".format(join(file_path, filename + ".pkl")))
        logger.warning("Failure reason: [{}]".format(e))
        return False


def load_pickle(filename, file_path):
    """
    Loads pickle file from files.

    :param filename:
    :param file_path:
    :return:
    """
    # logger.debug("Method: load_pickle(pkl_file)")
    if exists(join(file_path, filename + ".pkl")):
        try:
            logger.debug("Reading pickle file: [{}]".format(join(file_path, filename + ".pkl")))
            with sopen(join(file_path, filename + ".pkl"), 'rb') as pkl_file:
                loaded = pk.load(pkl_file)
            return loaded
        except Exception as e:
            logger.warning(
                "Could not open file: [{}]".format(join(file_path, filename + ".pkl")))
            logger.warning("Failure reason: [{}]".format(e))
            return False
    else:
        logger.warning("File not found at: [{}]".format(join(file_path, filename + ".pkl")))


def main():
    pass


if __name__ == '__main__':
    main()
