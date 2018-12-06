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
# import torchvision.transforms as transforms
# import os.path
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from data_loaders import html_loader as html
from data_loaders import json_loader as json
from data_loaders import txt_loader as txt
from pretrained.TextEncoder import TextEncoder
from logger.logger import logger
from utils import util

seed_val = 0
np.random.seed(seed_val)  # for reproducibility


class PrepareData():
    """Loads datasets and prepare data."""

    def __init__(self, dataset_type="html", dataset_name="Wiki10-31k",
                 dataset_dir="D:\Datasets\Extreme Classification\Wiki10-31k"):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir

        if dataset_type == "html":
            html_dataset = html.WIKI_HTML_Dataset(dataset_name=dataset_name,data_dir=self.dataset_dir)
            (sentences, classes), categories = html_dataset.get_data()
        elif dataset_type == "json":
            json_dataset = json.JSONLoader(dataset_name=dataset_name,data_dir=self.dataset_dir)
            (sentences, classes), categories = json_dataset.get_data()
        elif dataset_type == "txt":
            txt_dataset = txt.TXTLoader(dataset_name=dataset_name,data_dir=self.dataset_dir)
            (sentences, classes), categories = txt_dataset.get_data()
        else:
            raise Exception("Dataset type [{}] not found.".format(dataset_name))

        self.sentences = sentences
        self.classes = classes
        self.categories = categories

        mlb = MultiLabelBinarizer()
        self.classes_multihot = mlb.fit_transform(classes.values())

        self.text_encoder = TextEncoder()

        # self.txt2vec(sentences)
        # self.normalization()

    def txt2vec(self, docs, mode="word_avg", tfidf_avg=False, embedding_dim=300, max_vec_len=10000, num_chunks=10):
        """
        Creates vectors from input_texts based on [mode].

        :param max_vec_len: Maximum vector length of each document.
        :param num_chunks: Number of chunks the input_texts are to be divided. (Applicable only when mode = "chunked")
        :param embedding_dim: Embedding dimension for each word.
        :param docs:
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

        if mode == "doc2vec":
            doc2vec_model = self.text_encoder.load_doc2vec(docs, vector_size=self.embedding_dim, window=7, seed=seed_val, negative=10,
                                                           doc2vec_dir=self.dataset_dir, doc2vec_model_file=self.dataset_name+"_doc2vec")
            vectors = self.text_encoder.get_doc2vectors(docs,doc2vec_model)
            return vectors
        else:
            w2v_model = self.text_encoder.load_word2vec()
            return self.take_avg(docs, w2v_model, mode)

    def take_avg(self, docs: dict, w2v_model, mode,concat_axis=1):
        """
        Calculates the average of vectors of all the words in items.

        :param concat_axis: The axis the vectors should be concatenated.
        :param mode:
        :param w2v_model:
        :param docs: Dict of texts.
        :returns: Average of vectors of words. Dim: embedding_dim.
        """
        oov_words = {}  # To hold out-of-vocab words.
        documents = {}
        docs = util.clean_split(docs)
        for idx, doc in docs.items():
            chunks = self.partition_doc(doc, mode)
            logger.debug(chunks)
            logger.debug(len(chunks))
            avg_vec = np.zeros(self.embedding_dim)
            for item in chunks:
                if item in w2v_model.vocab:
                    avg_vec = np.add(avg_vec,w2v_model[item])
                    # logger.info(w2v_model[item].shape)
                else:
                    avg_vec = np.random.uniform(-0.5,0.5,self.embedding_dim)
                    # logger.debug(avg_vec)
                    # logger.debug(avg_vec.shape)
                    # logger.debug(self.embedding_dim)
                    w2v_model.add(item,avg_vec)
                    oov_words[item] = avg_vec
            chunk_vec = np.mean(avg_vec)
            if idx in documents:
                documents[idx] = np.concatenate((documents[idx],chunk_vec),axis=concat_axis)
            else:
                documents[idx] = chunk_vec

        util.save_json(oov_words,"oov_words")
        return documents

    def partition_doc(self, doc, mode, num_chunks=10):
        """
        Divides a document into chunks based on the mode.

        :param num_chunks:
        :param doc:
        :param mode:
        :param doc_len:
        :return:
        """
        chunks = []
        if mode == "concat":
            logger.debug(doc)
            words = doc.split(" ")
            logger.debug(words)
            for word in words:
                logger.debug(word)
                chunks.append(word)
            logger.debug(chunks)
        elif mode == "word_avg":
            chunks = doc.split()
        elif mode == "sentences":
            chunks = doc.split(". ")  # TODO: Clean all \n within sentences.
        elif mode == "chunked":
            chunks = len(doc.split()) / num_chunks  # calculates how long each chunk should be.
        else:
            raise Exception("Unknown Mode: [{}]".format(mode))
        logger.debug(chunks)
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

    def get_batch(self, batch_size=64):
        """
        Get next batch
        :return: Next batch
        """
        x_support_set, y_support_set, x_target, y_target = None, None, None, None
        return x_support_set, y_support_set, x_target, y_target


if __name__ == '__main__':
    logger.debug("Preparing Data...")
    cls = PrepareData()
    logger.debug(len(cls.sentences))
    vectors = cls.txt2vec(cls.sentences,mode="concat")
    logger.debug(vectors.shape)
