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

import numpy as np
import random
from sklearn.preprocessing import MultiLabelBinarizer
from collections import OrderedDict

from data_loaders import html_loader as html
from data_loaders import json_loader as json
from data_loaders import txt_loader as txt
from pretrained.TextEncoder import TextEncoder
from logger.logger import logger
from utils import util

seed_val = 0
random.seed(seed_val)
np.random.seed(seed_val)  # for reproducibility


class PrepareData():
    """Loads datasets and prepare data."""

    def __init__(self, dataset_type="html", dataset_name="Wiki10-31k",
                 dataset_dir="D:\Datasets\Extreme Classification\Wiki10-31k"):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

        if dataset_type == "html":
            self.dataset = html.WIKI_HTML_Dataset(dataset_name=dataset_name, data_dir=self.dataset_dir)
            # sentences = self.dataset.get_sentences()
            # classes = self.dataset.get_classes()
            # categories = self.dataset.get_categories()
            # sentences, classes, categories = self.dataset.get_data()
        elif dataset_type == "json":
            self.dataset = json.JSONLoader(dataset_name=dataset_name, data_dir=self.dataset_dir)
            # (sentences, classes), categories = self.dataset.get_data()
        elif dataset_type == "txt":
            self.dataset = txt.TXTLoader(dataset_name=dataset_name, data_dir=self.dataset_dir)
            # (sentences, classes), categories = self.dataset.get_data()
        else:
            raise Exception(
                "Dataset type for dataset [{}] not found. \n Possible reason: Dataset not added in the config file.".format(
                    dataset_name))

        self.sentences_train = None
        self.classes_train = None
        self.sentences_val = None
        self.classes_val = None
        self.sentences_test = None
        self.classes_test = None
        self.categories = None

    def txt2vec(self, sentences, mode="chunked", tfidf_avg=False, embedding_dim=300, max_vec_len=10000, num_chunks=10):
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
        self.num_chunks = num_chunks
        self.embedding_dim = embedding_dim
        self.text_encoder = TextEncoder()

        sentences = util.clean_sentences(sentences, specials="""_-@*#'"/\\""", replace='')

        if mode == "doc2vec":
            doc2vec_model = self.text_encoder.load_doc2vec(sentences, vector_size=self.embedding_dim, window=7,
                                                           seed=seed_val, negative=10,
                                                           doc2vec_dir=self.dataset_dir,
                                                           doc2vec_model_file=self.dataset_name + "_doc2vec")
            vectors_dict = self.text_encoder.get_doc2vectors(sentences, doc2vec_model)
            return vectors_dict
        else:
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
        oov_words = OrderedDict()  # To hold out-of-vocab words.
        docs_vecs = OrderedDict()
        # logger.debug(sentences)
        # logger.debug(sentences)
        for idx, doc in sentences.items():
            chunks = self.partition_doc(doc, mode)
            # logger.debug(chunks)
            chunks = list(filter(None, chunks))  # Removing empty items.
            # logger.debug(chunks)
            # logger.debug(len(chunks))
            for chunk in chunks:
                # logger.debug(chunk)
                # logger.debug(chunk, w2v_model.vocab)
                avg_vec = None
                for word in chunk:
                    if word in w2v_model.vocab:
                        if avg_vec is None:
                            avg_vec = w2v_model[word]
                        else:
                            avg_vec = np.add(avg_vec, avg_vec)
                        # logger.info(w2v_model[word].shape)
                    else:
                        new_oov_vec = np.random.uniform(-0.5, 0.5, self.embedding_dim)
                        w2v_model.add(word, new_oov_vec)
                        oov_words[word] = new_oov_vec
                        if avg_vec is None:
                            avg_vec = new_oov_vec
                        else:
                            avg_vec = np.add(avg_vec, new_oov_vec)
                        # logger.debug(avg_vec)
                        # logger.debug(avg_vec.shape)
                        # logger.debug(self.embedding_dim)
                        logger.debug(avg_vec)
                chunk_avg_vec = np.divide(avg_vec, float(len(chunk)))
                if idx in docs_vecs:
                    logger.debug((docs_vecs[idx].shape, chunk_avg_vec.shape))
                    docs_vecs[idx] = np.concatenate((docs_vecs[idx], chunk_avg_vec), axis=concat_axis)
                    logger.debug(docs_vecs[idx].shape)
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
        # logger.debug(sentence)
        # TODO: Use better word and sentence tokenizers.
        if mode == "concat":
            words = sentence.split(" ")
            # logger.debug(words)
            for word in words:
                # logger.debug(word)
                chunks.append(word)
            # logger.debug(chunks)
        elif mode == "word_avg":
            chunks = sentence.split(" ")
        elif mode == "sentences":
            chunks = sentence.splitlines()
        elif mode == "chunked":
            splitted_doc = sentence.split()
            # logger.debug(splitted_doc)
            doc_len = len(splitted_doc)
            # logger.debug(doc_len)
            chunk_size = doc_len // num_chunks  # Calculates how large each chunk should be.
            # logger.debug(chunk_size)
            index_start = 0
            for i in range(num_chunks):
                batch_portion = doc_len / (chunk_size * (i + 1))
                # logger.debug(batch_portion)
                if batch_portion > 1.0:
                    index_end = index_start + chunk_size
                else:  # Available data is less than chunk_size
                    # logger.debug("Am I here? {}".format(batch_portion))
                    index_end = index_start + (doc_len - index_start)
                logger.info('Making chunk of tokens from [{0}] to [{1}]'.format(index_start, index_end))
                chunk = splitted_doc[index_start:index_end]
                # logger.debug((len(chunk),(chunk_size)))
                chunks.append(chunk)
                index_start = index_end
        else:
            raise Exception(
                "Unknown document partition mode: [{}]. \n Available options: ['concat','word_avg','sentences','chunked (Default)']".format(
                    mode))
        logger.debug(len(chunks))
        chunks = list(filter(None, chunks))  # Removing empty items.
        logger.debug(len(chunks))
        return chunks

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        print("train_shape", self.x_train.shape, "test_shape", self.x_test.shape, "val_shape", self.x_val.shape)
        # print("before_normalization", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_val = (self.x_val - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std
        # print("after_normalization", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def create_multihot(self, batch_classes_dict):
        """
        Creates multi-hot vectors for a batch of data.
        :param batch_classes_dict:
        :return:
        """
        mlb = MultiLabelBinarizer()
        classes_multihot = mlb.fit_transform(batch_classes_dict.values())
        return classes_multihot

    def get_batch(self, batch_size=64, run_mode="train"):
        """
        Yields a batch of features and classes based on [run_mode].

        :return: Next batch
        """
        if run_mode == "train":
            self.sentences_train, self.classes_train = self.dataset.get_train_data()
            sentences = self.sentences_train
            classes = self.classes_train
        elif run_mode == "val":
            self.sentences_val, self.classes_val = self.dataset.get_val_data()
            sentences = self.sentences_val
            classes = self.classes_val
        elif run_mode == "test":
            self.sentences_test, self.classes_test = self.dataset.get_test_data()
            sentences = self.sentences_test
            classes = self.classes_test
        else:
            raise Exception("Unknown running mode: [{}]. \n Available options: ['train','val','test']".format(run_mode))

        remaining_keys = random.shuffle(sentences.keys())

        num_chunks = len(sentences) // batch_size
        for i in range(num_chunks):
            remaining_keys, batch_keys = self.get_batch_keys(remaining_keys, batch_size)
            sentences_batch, classes_batch = self.create_batch(sentences,classes,batch_keys)
            x_target = self.txt2vec(sentences_batch, mode="chunked")
            y_target_hot = self.create_multihot(classes_batch)
            # x_support = self.txt2vec(x_target, mode="chunked")
            # y_support_hot = self.create_multihot(y_target)
            yield x_target, y_target_hot#, x_support, y_support_hot


if __name__ == '__main__':
    logger.debug("Preparing Data...")
    cls = PrepareData()
    logger.debug(len(cls.sentences))
    vectors = cls.txt2vec(cls.sentences, mode="chunked")
    logger.debug(vectors.shape)
