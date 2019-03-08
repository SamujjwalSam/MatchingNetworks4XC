# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Class to process and load pretrained models.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : TextEncoder

__variables__   :

__methods__     :
"""

import numpy as np
from os import mkdir
from os.path import join, exists, split
from collections import OrderedDict

from gensim.models import word2vec, doc2vec
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.utils import simple_preprocess

from logger.logger import logger
from config import configuration as config
from config import platform as plat


class TextEncoder:
    """
    Class to process and load pretrained models.

    Supported models: glove, word2vec, fasttext, googlenews, bert, lex, etc.
    """

    def __init__(self, model_type: str = "googlenews", model_dir: str = config["paths"]["pretrain_dir"][plat],
                 embedding_dim: int = config["prep_vecs"]["input_size"]):
        """
        Initializes the pretrain class and checks for paths validity.

        Args:
            model_type : Path to the file containing the html files.
            Supported model_types:
                glove (default)
                word2vec
                fasttext_wiki
                fasttext_crawl
                fasttext_wiki_subword
                fasttext_crawl_subword
                lex_crawl
                lex_crawl_subword
                googlenews
                bert_multi
                bert_large_uncased
        """
        super(TextEncoder, self).__init__()
        self.model_type = model_type
        self.model_dir = model_dir
        self.embedding_dim = embedding_dim
        if model_type == "googlenews":
            filename = "GoogleNews-vectors-negative300.bin"
            binary_file = True
        elif model_type == "glove":
            filename = "glove.6B.300d.txt"
            binary_file = False
        elif model_type == "fasttext_wiki":
            filename = "wiki-news-300d-1M.vec"
            binary_file = False
        elif model_type == "fasttext_crawl":
            filename = "crawl-300d-2M.vec.zip"
            binary_file = False
        elif model_type == "fasttext_wiki_subword":
            filename = "wiki-news-300d-1M-subword.vec.zip"
            binary_file = False
        elif model_type == "fasttext_crawl_subword":
            filename = "crawl-300d-2M-subword.vec.zip"
            binary_file = False
        elif model_type == "lex_crawl":
            filename = "lexvec.commoncrawl.300d.W+C.pos.vectors.gz"
            binary_file = True
        elif model_type == "lex_crawl_subword":
            filename = "lexvec.commoncrawl.ngramsubwords.300d.W.pos.bin.gz"
            binary_file = True
        elif model_type == "bert_multi":
            filename = "BERT_multilingual_L-12_H-768_A-12.zip"
            binary_file = True
        elif model_type == "bert_large_uncased":
            filename = "BERT_large_uncased_L-24_H-1024_A-16.zip"
            binary_file = True
        else:
            raise Exception("Unknown pretrained model type: [{}]".format(model_type))
        # logger.debug("Creating TextEncoder.")
        self.model_file_name = filename
        self.binary = binary_file
        # self.pretrain_model = self.load_word2vec(self.model_dir, model_file_name=self.model_file_name, model_type=model_type)

    def load_doc2vec(self, documents, vector_size=config["prep_vecs"]["input_size"], window=config["prep_vecs"]["window"],
                     min_count=config["prep_vecs"]["min_count"], workers=config["text_process"]["workers"], seed=0,
                     negative=config["prep_vecs"]["negative"], doc2vec_dir=join(config["paths"]["dataset_dir"][plat],config["data"]["dataset_name"]), doc2vec_model_file=config["data"]["dataset_name"] + "_doc2vec",
                     clean_tmp=False, save_model=True):
        """
        Generates vectors from documents.
        https://radimrehurek.com/gensim/models/doc2vec.html

        :param save_model:
        :param clean_tmp: Flag to set if cleaning is to be done.
        :param doc2vec_dir:
        :param doc2vec_model_file: Name of Doc2Vec model.
        :param negative: If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20).
        :param documents:
        :param vector_size:
        :param window:
        :param min_count:
        :param workers:
        :param seed:
        """
        full_model_name = doc2vec_model_file + "_" + str(vector_size) + "_" + str(window) + "_" + str(
            min_count) + "_" + str(negative)
        if exists(join(doc2vec_dir, full_model_name)):
            logger.info("Loading doc2vec model from: [{}]".format(join(doc2vec_dir, full_model_name)))
            doc2vec_model = doc2vec.Doc2Vec.load(join(doc2vec_dir, full_model_name))
        else:
            train_corpus = list(self.read_corpus(documents))
            doc2vec_model = doc2vec.Doc2Vec(train_corpus, vector_size=vector_size, window=window, min_count=min_count,
                                            workers=workers, seed=seed, negative=negative)
            # doc2vec_model.build_vocab(train_corpus)
            doc2vec_model.train(train_corpus, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
            if save_model:
                save_path = get_tmpfile(join(doc2vec_dir, full_model_name))
                doc2vec_model.save(save_path)
                logger.info("Saved doc2vec model to: [{}]".format(save_path))
            if clean_tmp:  # Do this when finished training a model (no more updates, only querying, reduce memory usage)
                doc2vec_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        return doc2vec_model

    def read_corpus(self, documents, tokens_only=False):
        """
        Read the documents, pre-process each line using a simple gensim pre-processing tool and return a list of words. The tag is simply the zero-based line number.

        :param documents: List of documents.
        :param tokens_only:
        """
        for i, line in enumerate(documents):
            if tokens_only:
                yield simple_preprocess(line)
            else:  # For training data, add tags, tags are simply zero-based line number.
                yield doc2vec.TaggedDocument(simple_preprocess(line), [i])

    def get_doc2vecs(self, documents: list, doc2vec_model=None):
        """
        Generates vectors for documents.

        :param doc2vec_model: doc2vec model object.
        :param documents:
        :return:
        """
        if doc2vec_model is None:  # If model is not supplied, need to create model.
            doc2vec_model = self.load_doc2vec(documents)
        doc2vectors = []
        for doc in documents:
            doc2vectors.append(doc2vec_model.infer_vector(doc))  # Infer vector for a new document
        doc2vectors = np.asarray(list(doc2vectors))  # Converting Dict values to Numpy array.
        return doc2vectors

    def load_word2vec(self, model_dir=config["paths"]["pretrain_dir"][plat], model_type='googlenews', encoding='utf-8',
                      model_file_name="GoogleNews-vectors-negative300.bin", newline='\n', errors='ignore'):
        """
        Loads Word2Vec model
        Returns initial weights for embedding layer.

        inputs:
        model_type      # GoogleNews / glove
        embedding_dim    # Word vector dimensionality
        """
        logger.debug("Using [{0}] model from [{1}]".format(model_type, join(model_dir, model_file_name)))
        if model_type == 'googlenews' or model_type == "fasttext_wiki":
            assert (exists(join(model_dir, model_file_name)))
            if exists(join(model_dir, model_file_name + '.bin')):
                try:
                    pretrain_model = FastText.load_fasttext_format(
                        join(model_dir, model_file_name + '.bin'))  # For original fasttext *.bin format.
                except Exception as e:
                    pretrain_model = KeyedVectors.load_word2vec_format(join(model_dir, model_file_name + '.bin'),
                                                                       binary=True)
            else:
                try:
                    pretrain_model = KeyedVectors.load_word2vec_format(join(model_dir, model_file_name),
                                                                       binary=self.binary)
                except Exception as e:  # On exception, trying a different format.
                    logger.debug('Loading original word2vec format failed. Trying Gensim format.')
                    pretrain_model = KeyedVectors.load(join(model_dir, model_file_name))
                pretrain_model.save_word2vec_format(join(model_dir, model_file_name + ".bin"),
                                                    binary=True)  # Save model in binary format for faster loading in future.
                logger.debug("Saved binary model at: [{0}]".format(join(model_dir, model_file_name + ".bin")))
                logger.info(type(pretrain_model))
        elif model_type == 'glove':
            assert (exists(join(model_dir, model_file_name)))
            logger.debug('Loading existing Glove model: [{0}]'.format(join(model_dir, model_file_name)))
            # dictionary, where key is word, value is word vectors
            pretrain_model = OrderedDict()
            for line in open(join(model_dir, model_file_name), encoding=encoding):
                tmp = line.strip().split()
                word, vec = tmp[0], map(float, tmp[1:])
                assert (len(vec) == self.embedding_dim)
                if word not in pretrain_model:
                    pretrain_model[word] = vec
            logger.debug('Found [{}] word vectors.'.format(len(pretrain_model)))
            assert (len(pretrain_model) == 400000)
        elif model_type == "bert_multi":
            # pretrain_model = FastText.load_fasttext_format(join(model_dir,model_file_name))
            # pretrain_model = FastText.load_binary_data (join(model_dir,model_file_name))
            pretrain_model = KeyedVectors.load_word2vec_format(join(model_dir, model_file_name), binary=False)
            # import io
            # fin = io.open(join(model_dir, model_file_name), encoding=encoding, newline=newline,
            #               errors=errors)
            # n, d = map(int, fin.readline().split())
            # pretrain_model = OrderedDict()
            # for line in fin:
            #     tokens = line.rstrip().split(' ')
            #     pretrain_model[tokens[0]] = map(float, tokens[1:])
            """embedding_dict = gensim.models.KeyedVectors.load_word2vec_format(dictFileName, binary=False) embedding_dict.save_word2vec_format(dictFileName+".bin", binary=True) embedding_dict = gensim.models.KeyedVectors.load_word2vec_format(dictFileName+".bin", binary=True)"""
            return pretrain_model
        else:
            raise ValueError('Unknown pretrain model type: %s!' % model_type)

        # logger.debug(pretrain_model["hello"].shape)
        return pretrain_model

    def train_w2v(self, sentence_matrix, vocabulary_inv, embedding_dim=config["prep_vecs"]["input_size"],
                  min_word_count=config["prep_vecs"]["min_count"], context=config["prep_vecs"]["window"]):
        """
        Trains, saves, loads Word2Vec model
        Returns initial weights for embedding layer.

        inputs:
        sentence_matrix # int matrix: num_sentences x max_sentence_len
        vocabulary_inv  # dict {str:int}
        embedding_dim    # Word vector dimensionality
        min_word_count  # Minimum word count
        context         # Context window size
        """
        model_dir = 'word2vec_models'
        model_name = "{:d}features_{:d}minwords_{:d}context".format(embedding_dim, min_word_count, context)
        model_name = join(model_dir, model_name)
        if exists(model_name):
            pretrain_model = word2vec.Word2Vec.load(model_name)
            logger.debug('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
        else:
            # Set values for various parameters
            num_workers = 2  # Number of threads to run in parallel
            downsampling = 1e-3  # Downsample setting for frequent words

            # Initialize and train the model
            logger.debug("Training Word2Vec model...")
            sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
            pretrain_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                               size=embedding_dim,
                                               min_count=min_word_count,
                                               window=context,
                                               sample=downsampling)

            # If we don't plan to train the model any further, calling init_sims will make the model much more memory-efficient.
            pretrain_model.init_sims(replace=True)

            # Saving the model for later use. You can load it later using Word2Vec.load()
            if not exists(model_dir):
                mkdir(model_dir)
            logger.debug('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
            pretrain_model.save(model_name)

        #  add unknown words
        embedding_weights = [np.array([pretrain_model[w] if w in pretrain_model else np.random.uniform(-0.25, 0.25,
                                                                                                       pretrain_model.vector_size)
                                       for w in vocabulary_inv])]
        return embedding_weights

    def get_embedding_matrix(self, vocabulary_inv: dict):
        """
        Generates the embedding matrix.
        :param vocabulary_inv:
        :param embedding_model:
        :return:
        """
        embedding_weights = [self.pretrain_model[w] if w in self.pretrain_model
                             else np.random.uniform(-0.25, 0.25, self.embedding_dim)
                             for w in vocabulary_inv]
        embedding_weights = np.array(embedding_weights).astype('float32')

        return embedding_weights


if __name__ == '__main__':
    logger.debug("Loading pretrained...")
    cls = TextEncoder()
    sentence_obama = 'Obama speaks to the media in Illinois'
    sentence_president = 'The president greets the press in Chicago'

    docs = [sentence_obama, sentence_president]
    doc2vec_model = cls.load_doc2vec(docs, vector_size=10, window=2, negative=2, save_model=False)
    vectors = cls.get_doc2vectors(docs, doc2vec_model)
    logger.debug(vectors)
