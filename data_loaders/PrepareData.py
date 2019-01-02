# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6 or above
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Prepares the datasets as per Matching Networks model.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2018"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : PrepareData

__variables__   :

__methods__     :
"""

import os, random
import numpy as np
from random import sample
from sklearn.preprocessing import MultiLabelBinarizer
from collections import OrderedDict

from data_loaders import html_loader as html
from data_loaders import json_loader as json
from data_loaders import txt_loader as txt
from pretrained.TextEncoder import TextEncoder
from logger.logger import logger
from utils import util

# import warnings
# from sklearn.exceptions import Warning
# warnings.filterwarnings(action='ignore', category=UserWarning)

seed_val = 0
random.seed(seed_val)
np.random.seed(seed_val)  # for reproducibility


class PrepareData():
    """Loads datasets and prepare data into proper format."""
    def __init__(self, dataset_type="html", dataset_name="Wiki10-31K", default_load="val",
                 dataset_dir="D:\\Datasets\\Extreme Classification"):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.default_load = default_load

        self.sentences_train = None
        self.classes_train = None
        self.categories_train = None
        self.sentences_val = None
        self.classes_val = None
        self.categories_val = None
        self.sentences_test = None
        self.classes_test = None
        self.categories_test = None
        self.doc2vec_model = None

        if dataset_type == "html":
            self.dataset = html.HTMLLoader(dataset_name=self.dataset_name, data_dir=self.dataset_dir, run_mode=default_load)
        elif dataset_type == "json":
            self.dataset = json.JSONLoader(dataset_name=self.dataset_name, data_dir=self.dataset_dir)
        elif dataset_type == "txt":
            self.dataset = txt.TXTLoader(dataset_name=self.dataset_name, data_dir=self.dataset_dir)
        else:
            raise Exception("Dataset type for dataset [{}] not found. \n"
                            "Possible reasons: Dataset not added in the config file.".format(self.dataset_name))

        if default_load == "train": self.load_train()
        if default_load == "val": self.load_val()
        if default_load == "test": self.load_test()

        self.mlb = MultiLabelBinarizer()

    def cat2samples(self, classes_dict: dict = None):
        """
        Converts sample : categories to  categories : samples

        :returns: A dictionary of categories to sample mapping.
        """
        cat2id = OrderedDict()
        if classes_dict is None: classes_dict = self.selected_classes
        for k, v in classes_dict.items():
            for cat in v:
                if cat not in cat2id:
                    cat2id[cat] = []
                cat2id[cat].append(k)
        return cat2id

    def load_train(self):
        self.sentences_train, self.classes_train, self.categories_train = self.dataset.get_train_data()
        self.selected_sentences, self.selected_classes, self.selected_categories = self.sentences_train, self.classes_train, self.categories_train
        self.remain_sample_ids = list(self.selected_sentences.keys())
        self.cat2id_map = self.cat2samples(self.selected_classes)
        self.remain_cat_ids = list(self.selected_categories.keys())
        logger.info("Training data counts:\n\tSentences = [{}],\n\tClasses = [{}],\n\tCategories = [{}]"
                    .format(len(self.sentences_train), len(self.classes_train), len(self.categories_train)))

    def load_val(self):
        self.sentences_val, self.classes_val, self.categories_val = self.dataset.get_val_data()
        self.selected_sentences, self.selected_classes, self.selected_categories = self.sentences_val, self.classes_val, self.categories_val
        self.remain_sample_ids = list(self.selected_sentences.keys())
        self.cat2id_map = self.cat2samples(self.selected_classes)
        self.remain_cat_ids = list(self.selected_categories.keys())
        logger.info("Validation data counts:\n\tSentences = [{}],\n\tClasses = [{}],\n\tCategories = [{}]"
                    .format(len(self.sentences_val), len(self.classes_val), len(self.categories_val)))

    def load_test(self):
        self.sentences_test, self.classes_test, self.categories_test = self.dataset.get_test_data()
        self.selected_sentences, self.selected_classes, self.selected_categories = self.sentences_test, self.classes_test, self.categories_test
        self.remain_sample_ids = list(self.selected_sentences.keys())
        self.cat2id_map = self.cat2samples(self.selected_classes)
        self.remain_cat_ids = list(self.selected_categories.keys())
        logger.info("Testing data counts:\n\tSentences = [{}],\n\tClasses = [{}],\n\tCategories = [{}]"
                    .format(len(self.sentences_test), len(self.classes_test), len(self.categories_test)))

    def txt2vec(self, sentences:list, mode="chunked", tfidf_avg=False, embedding_dim=300, max_vec_len=10000, num_chunks=10):
        """
        Creates vectors from input_texts based on [mode].

        :param max_vec_len: Maximum vector length of each document.
        :param num_chunks: Number of chunks the input_texts are to be divided. (Applicable only when mode = "chunked")
        :param embedding_dim: Embedding dimension for each word.
        :param sentences:
        :param mode: Decides how to create the vector.
            "chunked" : Partitions the whole text into equal length chunks and concatenates the avg of each chunk.
                        vec_len = num_chunks * embedding_dim
                        [["these", "are"], ["chunks", "."]]

            "sentences" : Same as chunked except each sentence forms a chunk. "\n" is sentence separator.
                        vec_len = max(num_sents, max_len) * embedding_dim
                        ["this", "is", "a", "sentence", "."]
                        NOTE: This will create variable length vectors.

            "concat" : Concatenates the vectors of each word, adds padding to make equal length.
                       vec_len = max(num_words) * embedding_dim

            "word_avg" : Take the average of vectors of all the words.
                       vec_len = embedding_dim
                       [["these"], ["are"], ["words"], ["."]]

            "doc2vec" : Use Gensim Doc2Vec to generate vectors.
                        https://radimrehurek.com/gensim/models/doc2vec.html
                        vec_len = embedding_dim
                        ["this", "is", "a", "document", "."]

        :param tfidf_avg: If tf-idf weighted avg is to be taken or simple.
            True  : Take average based on each words tf-idf value.
            False : Take simple average.

        :returns: Vector length, numpy.ndarray(batch_size, vec_len)
        """
        self.embedding_dim = embedding_dim
        self.text_encoder = TextEncoder()

        sentences = util.clean_sentences(sentences, specials="""_-@*#'"/\\""", replace='')

        if mode == "doc2vec":
            if self.doc2vec_model is None:
                self.doc2vec_model = self.text_encoder.load_doc2vec(sentences, vector_size=self.embedding_dim, window=7,
                                                               seed=seed_val, negative=10,
                                                               doc2vec_dir=self.dataset_dir,
                                                               doc2vec_model_file=self.dataset_name + "_doc2vec")
            vectors_dict = self.text_encoder.get_doc2vecs(sentences, self.doc2vec_model)
            return vectors_dict
        else:
            self.num_chunks = num_chunks
            w2v_model = self.text_encoder.load_word2vec()
            return self.create_doc_vecs(sentences, w2v_model, mode)

    def create_doc_vecs(self, sentences: dict, w2v_model, mode, concat_axis=0):
        """
        Calculates the average of vectors of all words within a chunk and concatenates the chunks.

        :param concat_axis: The axis the vectors should be concatenated.
        :param mode:
        :param w2v_model:
        :param sentences: Dict of texts.
        :returns: Average of vectors of chunks. Dim: embedding_dim.
        """
        oov_words = []  # To hold out-of-vocab words.
        docs_vecs = OrderedDict()
        for idx, doc in sentences.items():
            chunks = self.partition_doc(doc, mode)
            chunks = list(filter(None, chunks))  # Removing empty items.
            for chunk in chunks:
                avg_vec = None
                for word in chunk:
                    if word in w2v_model.vocab:
                        if avg_vec is None:
                            avg_vec = w2v_model[word]
                        else:
                            avg_vec = np.add(avg_vec, avg_vec)
                    else:
                        new_oov_vec = np.random.uniform(-0.5, 0.5, self.embedding_dim)
                        w2v_model.add(word, new_oov_vec)
                        oov_words.append(word)
                        if avg_vec is None:
                            avg_vec = new_oov_vec
                        else:
                            avg_vec = np.add(avg_vec, new_oov_vec)
                chunk_avg_vec = np.divide(avg_vec, float(len(chunk)))
                if idx in docs_vecs:
                    docs_vecs[idx] = np.concatenate((docs_vecs[idx], chunk_avg_vec), axis=concat_axis)
                else:
                    docs_vecs[idx] = chunk_avg_vec
        util.save_json(oov_words, "oov_words")

        return docs_vecs

    def partition_doc(self, sentence, mode, num_chunks=10):
        """
        Divides a document into chunks based on the mode.

        :param num_chunks:
        :param sentence:
        :param mode:
        :param doc_len:
        :return:
        """
        chunks = []
        # TODO: Use better word and sentence tokenizer, i.e. Spacy, NLTK, etc.
        if mode == "concat":
            words = sentence.split(" ")
            for word in words:
                chunks.append(word)
        elif mode == "word_avg":
            chunks = sentence.split(" ")
        elif mode == "sentences":
            chunks = sentence.splitlines()
        elif mode == "chunked":
            splitted_doc = sentence.split()
            doc_len = len(splitted_doc)
            chunk_size = doc_len // num_chunks  # Calculates how large each chunk should be.
            index_start = 0
            for i in range(num_chunks):
                batch_portion = doc_len / (chunk_size * (i + 1))
                if batch_portion > 1.0:
                    index_end = index_start + chunk_size
                else:  # Available data is less than chunk_size
                    index_end = index_start + (doc_len - index_start)
                logger.info('Making chunk of tokens from [{0}] to [{1}]'.format(index_start, index_end))
                chunk = splitted_doc[index_start:index_end]
                chunks.append(chunk)
                index_start = index_end
        else:
            raise Exception("Unknown document partition mode: [{}]. \n"
                            "Available options: ['concat','word_avg','sentences','chunked (Default)']".format(mode))
        chunks = list(filter(None, chunks))  # Removes empty items, like: ""
        return chunks

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1.

        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        logger.debug(("train_shape", self.x_train.shape, "test_shape", self.x_test.shape, "val_shape", self.x_val.shape))
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_val = (self.x_val - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

    def create_multihot(self, batch_classes_dict):
        """
        Creates multi-hot vectors for a batch of data.

        :param batch_classes_dict:
        :return:
        """
        classes_multihot = self.mlb.fit_transform(batch_classes_dict.values())
        return classes_multihot

    def get_support_cats(self, categories_per_set=5):
        """
        Randomly selects [num_cat] number of classes from which support set will be created.

        Will remove the selected classes from self.remain_cat_ids.
        :param categories_per_set: Number of samples to draw.
        :return:
        """
        self.remain_cat_ids, selected_cat_ids = util.get_batch_keys(self.remain_cat_ids, batch_size=categories_per_set,
                                                                    remove_keys=False)
        selected_cat_ids = [int(cat) for cat in selected_cat_ids]  # For some reason returned values are not int, converting to int.
        return selected_cat_ids

    def get_supports(self, support_cat_ids, cat2id_map=None, input_size=300, samples_per_category=4, mode="doc2vec", repeat_mode='append'):
        """
        Returns a batch of feature vectors and multi-hot classes.

        :param repeat_mode: How to repeat sample if available data is less than samples_per_class. ["append (default)", "sample"].
        :param cat2id_map: A dictionary of categories to sample mapping.
        :return: Next batch
        :param min_match: Minimum number of categories should match.
        :param support_cat_ids:
        :param samples_per_category:
        :param mode:
        """
        # cat_selected_ids = OrderedDict()
        selected_ids = []
        if cat2id_map is None: cat2id_map = self.cat2id_map
        for cat in support_cat_ids:
            # logger.debug((cat2id_map,cat,samples_per_category))
            if len(cat2id_map[cat]) == samples_per_category:
                selected_ids = selected_ids + cat2id_map[cat]
            elif len(cat2id_map[cat]) > samples_per_category:  # More than required, sample [samples_per_category] from the list.
                selected_ids = selected_ids + sample(cat2id_map[cat], k=samples_per_category)
            else:  # Less than required, repeat the available classes.
                selected_ids = selected_ids + cat2id_map[cat]
                length = len(cat2id_map[cat])
                if repeat_mode == "append":
                    for i in range(samples_per_category - length):
                        selected_ids.append(cat2id_map[cat][i % length])
                elif repeat_mode == "sample":
                    empty_count = samples_per_category - length
                    for i in range(empty_count):  # Sampling [samples_per_category] number of elements from cat2id_map[cat].
                        selected_ids.append(cat2id_map[cat][sample(range(length), 1)])
                else:
                    raise Exception("Unknown [repeat_mode]: [{}]".format(repeat_mode))
        sentences_batch, classes_batch = util.create_batch_repeat(self.selected_sentences, self.selected_classes, selected_ids)
        # logger.debug(input_size)
        x_target = self.txt2vec(sentences_batch, embedding_dim=input_size, mode=mode)
        y_target_hot = self.mlb.transform(classes_batch)
        return x_target, y_target_hot

    def get_targets(self, target_size=4, mode="doc2vec"):
        """
        Returns a batch of feature vectors and multi-hot classes of randomly selected targets.

        :return: Next batch of targets.
        """
        # remain_sample_ids = random.shuffle(list(sentences.keys()))
        if self.remain_sample_ids is not None:
            _, batch_supports = util.get_batch_keys(self.remain_sample_ids, target_size, remove_keys=False)
            # logger.debug((self.remain_sample_ids, batch_supports))
            support_sentences_batch, support_classes_batch = util.create_batch(self.selected_sentences,
                                                                               self.selected_classes, batch_supports)
            x_support = self.txt2vec(support_sentences_batch, mode=mode)
            y_support_hot = self.create_multihot(support_classes_batch)
            return x_support, y_support_hot
        else:
            logger.warn("No remain_sample_ids.")
            return

    def get_batches(self, batch_size=32,input_size=300,categories_per_set=5, samples_per_category=4, mode="doc2vec"):
        """
        Returns an iterator over data.

        :param input_size: Input embedding dimension.
        :param batch_size:
        :param categories_per_set:
        :param min_match:
        :param samples_per_category:
        :param mode:
        :returns: An iterator over data.
        """
        # logger.debug(self.remain_sample_ids)
        support_cat_ids = self.get_support_cats(categories_per_set=categories_per_set)
        support_cat_ids_list = []  # MultiLabelBinarizer only takes list of lists as input. Need to convert our list of int to list of lists.
        for ids in support_cat_ids:
            support_cat_ids_list.append([ids])
        self.mlb.fit_transform(support_cat_ids_list)  # Fitting the selected classes. Outputs not required.
        x_supports = []
        y_support_hots = []
        x_targets = []
        y_target_hots = []
        for i in range(batch_size):
            x_support, y_support_hot = self.get_supports(support_cat_ids, samples_per_category=samples_per_category, mode=mode, input_size=input_size)
            sel_cat = sample(support_cat_ids, k=1)
            # logger.debug(sel_cat)
            x_target, y_target_hot = self.get_supports(sel_cat, samples_per_category=samples_per_category, mode=mode, input_size=input_size)
            # logger.debug(x_support.shape)
            # logger.debug(y_support_hot.shape)
            x_supports.append(x_support)
            y_support_hots.append(y_support_hot)
            x_targets.append(x_target)
            y_target_hots.append(y_target_hot)
        x_supports = np.stack(x_supports)
        y_support_hots = np.stack(y_support_hots)
        x_targets = np.stack(x_targets)
        y_target_hots = np.stack(y_target_hots)
        return x_supports, y_support_hots, x_targets, y_target_hots


if __name__ == '__main__':
    logger.debug("Preparing Data...")
    cls = PrepareData(default_load="train")
    x_supports, y_support_hots, x_targets, y_target_hots = cls.get_batches(batch_size=32, categories_per_set=5, samples_per_category=4)
    logger.debug(x_supports.shape)
    logger.debug(y_support_hots.shape)
    logger.debug(x_targets.shape)
    logger.debug(y_target_hots.shape)
