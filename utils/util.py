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
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

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
from scipy import sparse, random, math
from unidecode import unidecode

from logger.logger import logger

seed_val = 0
# random.seed(seed=seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed=seed_val)
        
        
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


def read_inputs(dataset_path, dataset, read_test=False):
    """
    Reads input files.

    :param read_test: Bool; to indicate between train and test file
    :param dataset_path:
    :param dataset:
    :return:
    """
    filename = '_train'
    if read_test:  ## If true read test file instead of train file
        filename = '_test'

    file = dataset + filename
    file = join(dataset_path, dataset, dataset, file)
    X, Y, V, E = False, False, False, False
    logger.debug('Reading files from path{0}'.format(file))
    if exists(join(dataset_path, dataset, "X" + filename + ".npz")):
        X = load_npz("X" + filename, file_path=join(dataset_path, dataset))

    if exists(join(dataset_path, dataset, "Y_ints" + filename + ".pkl")):
        Y = load_pickle(filename="Y_ints" + filename, file_path=join(dataset_path, dataset))

    if exists(join(dataset_path, dataset, "V_ints" + filename + ".pkl")):
        V = load_pickle(filename="V_ints" + filename, file_path=join(dataset_path, dataset))

    if exists(join(dataset_path, dataset, "E" + filename + ".pkl")):
        E = load_pickle(filename="E" + filename, file_path=join(dataset_path, dataset))

    if X is False or Y is False or V is False or E is False:
        logger.info('Generating data from train and test files.')
        _, _, _, X, Y, V, E = get_x_y_v_e(join(dataset_path, dataset, dataset, dataset + filename + '.txt'))

        save_npz(X, "X" + filename, file_path=join(dataset_path, dataset), overwrite=False)
        Y_ints = []
        for y_list in Y:
            Y_ints.append([int(i) for i in y_list])
        save_pickle(Y_ints, filename="Y_ints" + filename, file_path=join(dataset_path, dataset))
        save_pickle(Y, filename="Y" + filename, file_path=join(dataset_path, dataset))

        V_ints = []
        for y_list in V:
            V_ints.append(int(y_list))
        save_pickle(V_ints, filename="V_ints" + filename, file_path=join(dataset_path, dataset))
        save_pickle(V, filename="V" + filename, file_path=join(dataset_path, dataset))

        ## Converting [E] to default dict as returned [E] is not pickle serializable
        E_tr_2 = OrderedDict()
        for i, j in E.items():
            E_tr_2[i] = j
        save_pickle(E_tr_2, filename="E" + filename, file_path=join(dataset_path, dataset))
    return X, Y, V, E


def split_data(X, Y, V, split=0.1, label_preserve=False, save_path=get_dataset_path(), seed=seed_val):
    """
    Splits the data into 2 parts.

    :param X:
    :param Y:
    :param V:
    :param split:
    :param label_preserve: if True; splits the data keeping the categories common.
    :param save_path:
    :param seed:
    :return:
    """
    assert (X.shape[0] == len(Y))

    if not label_preserve:
        from sklearn.model_selection import train_test_split
        X_tr, X_val, Y_tr, Y_val = train_test_split(X, Y, test_size=split, random_state=seed)
        return X_tr, Y_tr, X_val, Y_val

    lbl_feature_count = OrderedDict().fromkeys(V)

    for lbl in V:
        for y_list in Y:
            if int(lbl) in y_list:
                if lbl_feature_count[lbl] is None:
                    lbl_feature_count[lbl] = 1
                else:
                    lbl_feature_count[lbl] += 1
    assert (len(lbl_feature_count) == len(V))

    lbl_feature_count_portion = OrderedDict().fromkeys(V)
    for k, val in lbl_feature_count.items():
        lbl_feature_count_portion[k] = int(math.floor(lbl_feature_count[k] * split))
    logger.debug(len(lbl_feature_count_portion))

    X_val = []
    Y_val = []
    X_tr = None
    Y_tr = Y.copy()
    for lbl, count in lbl_feature_count_portion.items():
        for c in range(count):
            for i, y_list in enumerate(Y):
                if lbl in y_list:
                    X_val.append(X[i])
                    X_tr = np.delete(X, i)
                    Y_val.append(Y_tr.pop(i))
                    break
    save_npz(X_tr, "X_tr", file_path=save_path, overwrite=False)
    save_pickle(Y_tr, filename="Y_tr", file_path=save_path)
    save_npz(X_val, "X_val", file_path=save_path, overwrite=False)
    save_pickle(Y_val, filename="Y_val", file_path=save_path)
    return X_tr, Y_tr, X_val, Y_val


def _test_split_val():
    X = np.asarray(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    Y = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 1], [2, 1]]
    V = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    X_tr, Y_tr, X_val, Y_val = split_data(X, Y, V)
    logger.debug(X_tr)
    logger.debug(Y_tr)
    logger.debug(X_val)
    logger.debug(Y_val)


def main(args):
    """

    :param args:
    :return:
    """

    datasets = ['RCV1-2K', 'EURLex-4K', 'AmazonCat-13K', 'AmazonCat-14K', 'Wiki10-31K', 'Delicious-200K',
                'WikiLSHTC-325K', 'Wikipedia-500K', 'Amazon-670K', 'Amazon-3M']
    arff_datasets = ['Corel-374', 'Bibtex_arff', 'Delicious_arff', 'Mediamill_arff', 'Medical', 'Reuters-100_arff']
    datasets = ['RCV1-2K']
    for dataset in datasets:
        train_graph_file = dataset + '_train.txt'
        train_graph_file = join(args.dataset_path, dataset, train_graph_file)

        total_points, feature_dm, number_of_labels, X, Y, V, E = get_x_y_v_e(train_graph_file)

        save_json(V, dataset + '_V_train', join(args.dataset_path, dataset))
        save_json(E, dataset + '_E_train', join(args.dataset_path, dataset), overwrite=True)

        # Collecting some stats about the dataset and graph.
        e_stats, edge_occurances_sorted = edge_stats(E)
        e_stats['singles_train'] = find_single_labels(Y)
        save_json(e_stats, dataset + "_edge_statistics_train")

        plot_occurance(edge_occurances_sorted, plot_name=dataset + '_train_edge_occurances_sorted.jpg', clear=False)
        plot_occurance(edge_occurances_sorted, plot_name=dataset + '_train_edge_occurances_sorted_log.jpg', log=True)

        test_graph_file = dataset + '_test.txt'
        test_graph_file = join(args.dataset_path, dataset, test_graph_file)

        total_points, feature_dm, number_of_labels, X, Y, V, E = get_x_y_v_e(test_graph_file)

        save_json(V, dataset + '_V_test', join(args.dataset_path, dataset))
        save_json(E, dataset + '_E_test', join(args.dataset_path, dataset), overwrite=True)

        # Collecting some stats about the dataset and graph.
        e_stats, edge_occurances_sorted = edge_stats(E)
        e_stats['singles_test'] = find_single_labels(Y)
        save_json(e_stats, dataset + "_edge_statistics_test")

        plot_occurance(edge_occurances_sorted, plot_name=dataset + '_test_edge_occurances_sorted.jpg', clear=False)
        plot_occurance(edge_occurances_sorted, plot_name=dataset + '_test_edge_occurances_sorted_log.jpg', log=True)


if __name__ == '__main__':
    # text = "Ceñía Lo+=r?e~~m ipsum dol;or sit!! amet, consectet..ur ad%"
    # logger.debug(remove_special_chars(text))
    # exit(0)
    """
    sample call: python utils/py /Users/monojitdey/Downloads/Wiki10-31K/Wiki10/wiki10_test.txt
    /Users/monojitdey/Downloads/Wiki10-31K/Wiki10-31K_mappings/wiki10-31K_label_map.txt dataset_path =
    'D:\Datasets\Extreme Classification' dataset_name = 'Wiki10-31K' test_file = 'Wiki10/wiki10_test.txt'
    label_map_file = 'Wiki10-31K_mappings/wiki10-31K_label_map.txt'

    Examples:
      1. python utils/py

      2. python utils/py --node_id 4844

      3. python utils/py --test_file /Users/monojitdey/Downloads/Wiki10-31K/Wiki10/wiki10_test.txt --label_map_file
      /Users/monojitdey/Downloads/Wiki10-31K/Wiki10-31K_mappings/wiki10-31K_label_map.txt

      4. python utils/py --dataset_path /Users/monojitdey/Downloads/ --dataset_name Wiki10-31K --test_file
      /Wiki10/wiki10_test.txt --label_map_file /Wiki10-31K_mappings/wiki10-31K_label_map.txt
      5. python utils/py --dataset_path /Users/monojitdey/Downloads/ --dataset_name Wiki10-31K --test_file
      /Wiki10/wiki10_test.txt --label_map_file /Wiki10-31K_mappings/wiki10-31K_label_map.txt --node_id 4844
    """
    parser = ArgumentParser("Label Sub-graph generator",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve',
                            epilog="Example: python utils/py --dataset_path /Users/monojitdey/Downloads/ "
                                   "--dataset_name Wiki10-31K --test_file /Wiki10/wiki10_train.txt --label_map_file "
                                   "/Wiki10-31K_mappings/wiki10-31K_label_map.txt --node_id 4844 \n")
    parser.add_argument('--dataset_path',
                        help='Path to dataset folder', type=str,
                        default=get_dataset_path())
    parser.add_argument('--dataset_name',
                        help='Name of the dataset to use', type=str,
                        default='all')
    # parser.add_argument('--graph_file',  # required=True,
    #                     help='File path from which graph to be generated',type=str,
    #                     default='AmazonCat-13K/AmazonCat-13K_train.txt')
    # parser.add_argument('--label_map_file',
    #                     help="Label_map file path inside dataset. (If label file is not provided, graph will show "
    #                          "numeric categories only)",
    #                     type=str,
    #                     default='AmazonCat-13K_mappings/AmazonCat-13K_label_map.txt')
    parser.add_argument('--level',
                        help='Number of hops to generate graph', type=int,
                        default=1)
    parser.add_argument('--ignore_deg',
                        help='Ignores nodes with degree >= [ignore_degree]', type=int,
                        default=500)
    parser.add_argument('--node_id',
                        help='ID [Row number on  file] of the root node to generate graph', type=int,
                        default=12854)
    parser.add_argument('--subgraph_count',
                        help='How many subgraphs should be generated in single run', type=int,
                        default=1)
    args = parser.parse_args()

    logger.info("Parameters: [{}]".format(args))
    main(args)
