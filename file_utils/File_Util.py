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

import json
import pickle as pk
import numpy as np
from random import sample,shuffle
from collections import OrderedDict
from os.path import join,exists,isfile

from scipy import sparse
from smart_open import smart_open as sopen  # Better alternative to Python open().

from logger.logger import logger


def create_batch(X: dict,Y: dict,keys):
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
    return batch_x,batch_y


def create_batch_repeat(X: dict,Y: dict,keys):
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
    return batch_x,batch_y


def get_batch_keys(keys: list,batch_size=64,remove_keys=True):
    """
    Randomly selects [batch_size] numbers of key from keys list and remove them from the original list.

    :param remove_keys: Flag to indicate if selected keys should be removed. It should be False for support set selection.
    :param keys:
    :param batch_size:
    :return:
    """
    if len(keys) <= batch_size:
        return keys,keys
    selected_keys = sample(keys,k=batch_size)
    if remove_keys:
        keys_after_remove = []
        for item in keys:
            if item not in selected_keys:
                keys_after_remove.append(item)
        return keys_after_remove,selected_keys
    return keys,selected_keys


def split_dict(classes: dict,sentences: dict,batch_size=64,remove_keys=True):
    """
    Randomly selects [batch_size] numbers of items from dictionary and remove them from the original dict.

    :param sentences:
    :param remove_keys: Flag to indicate if selected items should be removed. It should be False for support set selection.
    :param classes:
    :param batch_size:
    :return:
    """
    if len(classes) <= batch_size:
        return None,classes,None,sentences
    selected_keys = sample(list(classes.keys()),k=batch_size)
    selected_classes = OrderedDict()
    selected_sentences = OrderedDict()
    if remove_keys:
        for key in selected_keys:
            selected_classes[key] = classes[key]
            del classes[key]
            selected_sentences[key] = sentences[key]
            del sentences[key]
        return classes,selected_classes,sentences,selected_sentences
    else:
        for key in selected_keys:
            selected_classes[key] = classes[key]
            selected_sentences[key] = sentences[key]
    return classes,selected_classes,sentences,selected_sentences


def remove_dup_list(seq,case=False):  # Dave Kirby
    """Removes duplicates from a list. Order preserving"""
    seen = set()
    if case: return [x.lower() for x in seq if
                     x.lower() not in seen and not seen.add(x)]
    return [x for x in seq if x not in seen and not seen.add(x)]


def inverse_dict_elm(labels: dict):
    """
    Inverses key to value of a dict and vice versa. Retains the initial values if key is repeated.

    :param labels:
    :return:
    """
    labels_inv = OrderedDict()
    for k,v in labels.items():
        if v not in labels_inv:  # check if key does not exist.
            labels_inv[int(v)] = k
    return labels_inv


def save_json(data,filename,file_path='',overwrite=False,indent=2,date_time_tag=''):
    """

    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :param indent:
    :param date_time_tag:
    :return:
    """
    logger.info("Saving JSON file: [{}]".format(join(file_path,date_time_tag + filename + ".json")))
    if not overwrite and exists(join(file_path,date_time_tag + filename + ".json")):
        logger.warning("File [{}] already exists and Overwrite == False.".format(
            join(file_path,date_time_tag + filename + ".json")))
        return True
    try:
        with sopen(join(file_path,date_time_tag + filename + ".json"),'w') as json_file:
            try:
                json_file.write(json.dumps(data,indent=indent))
            except Exception as e:
                logger.warning("Writing JSON failed: [{}]".format(e))
                logger.warning(
                    "Writing as string: [{}]".format(join(file_path,date_time_tag + filename + ".json")))
                json_file.write(json.dumps(str(data),indent=indent))
                return True
        json_file.close()
        return True
    except Exception as e:
        logger.warning("Writing JSON file [{}] failed: [{}]".format(join(file_path,filename),e))
        logger.warning("Writing as TXT: [{}]".format(filename + ".txt"))
        write_file(data,filename,date_time_tag=date_time_tag)
        return False


def load_json(filename: str,file_path: str = '',date_time_tag: str = '',ext: str = ".json",
              show_path: bool = False) -> OrderedDict:
    """
    Loads json file as python OrderedDict.

    :param show_path:
    :param ext: Should extension be appended?
    :param filename:
    :param file_path:
    :param date_time_tag:
    :return: OrderedDict
    """
    file_loc = join(file_path,date_time_tag + filename + ext)
    if show_path:
        logger.info("Reading JSON file: [{}]".format(file_loc))
    if exists(join(file_path,date_time_tag + filename + ext)):
        try:
            with sopen(file_loc,encoding="utf-8") as file:
                json_dict = json.load(file)
                json_dict = OrderedDict(json_dict)
                # json_dict = OrderedDict(json.load(file))
            file.close()
            return json_dict
        except Exception as e:
            logger.warning("Could not open file as JSON: [{}]. \n Reason:[{}]".format(file_loc,e))
            with sopen(file_loc,encoding="utf-8") as file:
                json_dict = str(file)
                json_dict = json.loads(json_dict)
                # json_dict = OrderedDict(json_dict)
            return json_dict
    else:
        logger.warning("File does not exist at: [{}]".format(file_loc))
        return False


def read_json_str(filename,file_path='',date_time_tag='',ext="",show_path=False):
    """
    Loads json file as python OrderedDict.

    :param show_path:
    :param ext: Should extension be appended?
    :param filename:
    :param file_path:
    :param date_time_tag:
    :return: OrderedDict
    """
    file_loc = join(file_path,date_time_tag + filename + ext)
    if show_path:
        logger.info("Reading JSON file: [{}]".format(file_loc))
    if exists(join(file_path,date_time_tag + filename + ext)):
        raw_json = open(file_loc,'r').read()
        json_dict = json.loads(raw_json)
        logger.debug(json_dict)
        return json_dict
    else:
        logger.warning("File does not exist at: [{}]".format(file_loc))
        return False


def write_file(data,filename,file_path='',overwrite=False,mode='w',encoding="utf-8",date_time_tag='',
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
    if not overwrite and exists(join(file_path,date_time_tag + filename + ".txt")):
        # logger.warning("File [{}] already exists and Overwrite == False.".format(
        #     join(file_path, date_time_tag + filename + ".txt")))
        return True
    with sopen(join(file_path,date_time_tag + filename + ".txt"),mode,encoding=encoding) as text_file:
        if verbose:
            logger.info("Saving text file: [{}]".format(join(file_path,date_time_tag + filename + ".txt")))
        text_file.write(str(data))
        text_file.write("\n")
        text_file.write("\n")
    text_file.close()


def load_npz(filename,file_path=''):
    """
    Loads numpy objects from npz files.

    :param filename:
    :param file_path:
    :return:
    """
    logger.info("Reading NPZ file: [{}]".format(join(file_path,filename + ".npz")))
    if isfile(join(file_path,filename + ".npz")):
        npz = sparse.load_npz(join(file_path,filename + ".npz"))
        return npz
    else:
        logger.warning("Could not open file: [{}]".format(join(file_path,filename + ".npz")))
        return False


def save_npz(data,filename,file_path='',overwrite=False):
    """
    Saves numpy objects to file.

    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :return:
    """
    logger.info("Saving NPZ file: [{}]".format(join(file_path,filename + ".npz")))
    if not overwrite and exists(join(file_path,filename + ".npz")):
        logger.warning(
            "File [{}] already exists and Overwrite == False.".format(join(file_path,filename + ".npz")))
        return True
    try:
        sparse.save_npz(join(file_path,filename + ".npz"),data)
        return True
    except Exception as e:
        logger.warning("Could not write to npz file: [{}]".format(join(file_path,filename + ".npz")))
        logger.warning("Failure reason: [{}]".format(e))
        return False


def save_pickle(data,filename,file_path,overwrite=False):
    """
    Saves python object as pickle file.

    :param data:
    :param filename:
    :param file_path:
    :param overwrite:
    :return:
    """
    # logger.debug("Method: save_pickle(data, filename, file_path, overwrite=False)")
    logger.info("Writing to pickle file: [{}]".format(join(file_path,filename + ".pkl")))
    if not overwrite and exists(join(file_path,filename + ".pkl")):
        logger.warning("File [{}] already exists and Overwrite == False.".format(
            join(file_path,filename + ".pkl")))
        return True
    try:
        if isfile(join(file_path,filename + ".pkl")):
            logger.info(
                "Overwriting on pickle file: [{}]".format(join(file_path,filename + ".pkl")))
        with sopen(join(file_path,filename + ".pkl"),'wb') as pkl_file:
            pk.dump(data,pkl_file)
        pkl_file.close()
        return True
    except Exception as e:
        logger.warning(
            "Could not write to pickle file: [{}]".format(join(file_path,filename + ".pkl")))
        logger.warning("Failure reason: [{}]".format(e))
        return False


def load_pickle(filename: str,file_path: str) -> np.ndarray:
    """
    Loads pickle file from files.

    :param filename:
    :param file_path:
    :return:
    """
    # logger.debug("Method: load_pickle(pkl_file)")
    if exists(join(file_path,filename + ".pkl")):
        try:
            logger.info("Reading pickle file: [{}]".format(join(file_path,filename + ".pkl")))
            with sopen(join(file_path,filename + ".pkl"),'rb') as pkl_file:
                loaded = pk.load(pkl_file)
            return loaded
        except Exception as e:
            logger.warning(
                "Could not open file: [{}]".format(join(file_path,filename + ".pkl")))
            logger.warning("Failure reason: [{}]".format(e))
            return False
    else:
        logger.warning("File not found at: [{}]".format(join(file_path,filename + ".pkl")))


def main():
    pass


if __name__ == '__main__':
    main()
