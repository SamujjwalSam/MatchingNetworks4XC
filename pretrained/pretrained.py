# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6 or above
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ :
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2018 sam"
__license__     : "Python"

__classes__     : Class,

__variables__   :

__methods__     :

__todo__        : Include Multi-lingual BERT by Google
"""

from gensim.models import word2vec
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
from os.path import join, exists, split
import os
import numpy as np

from logger.logger import logger

"""
To solve MKL problem: Adding <conda-env-root>/Library/bin to the path in the run configuration solves the issue, but adding it to the interpreter paths in the project settings doesn't.
"""


class Pretrain(object):
    """
    Class to process and load pretrained models.

    Supported models: glove, word2vec, fasttext, googlenews, bert, lex, etc.
    """
    def __init__(self, model_type: str = "googlenews", model_dir: str = "D:\Datasets\pretrain", embedding_dim:int =300):
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
        super(Pretrain, self).__init__()
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
            raise NotImplementedError
        logger.info("Creating Pretrain.")
        self.model_file_name = filename
        self.binary = binary_file
        self.pretrain_model = self.load_word2vec(self.model_dir, model_file_name=self.model_file_name,
                                                 model_type=model_type)

    # def load_word2vec(self,model_dir=self.model_dir,file_name=self.model_file_name,binary=binary_file):
        # try:
            # embedding_model = KeyedVectors.load_word2vec_format(os.path.join(model_dir,file_name),binary=binary)
        # except Exception as e: ## Loading a different format.
            # logger.debug('Loading original word2vec format failed. Trying Gensim format.')
            # embedding_model = KeyedVectors.load(os.path.join(model_dir,file_name),binary=binary)
        # return embedding_model

    def get_sim(self,w1:str,w2:str):
        """
        Calculates cosine similarity between two words.
        :param w1: str
        :param w2: str
        :return:
        """
        sim = self.pretrain_model.similarity(w1,w2)
        logger.debug("Similarity between [{0}] and [{1}] is [{2}]".format(w1,w2,sim))
        return sim

    def get_sent_sim(self,s1:str,s2:str):
        """
        Calculates cosine similarity between two sentences.
        :param s1: str = Sentence
        :param s2: str = Sentence
        :return:
        """
        sent_sim = self.pretrain_model.wv.wmdistance(s1.lower().split(), s2.lower().split())  # Need to use .wv here.
        return sent_sim

    def get_sent_sim_list(self,s1:list,s2:list):
        """
        Calculates cosine similarity between two sentences.
        :param s1: str = Sentence
        :param s2: str = Sentence
        :return:
        """
        sent_sim = self.pretrain_model.wv.wmdistance(s1, s2)  # Need to use .wv here.
        return sent_sim

    def load_word2vec(self, model_dir="D:\Datasets\pretrain", model_file_name="GoogleNews-vectors-negative300.bin", model_type='googlenews', encoding='utf-8', newline='\n', errors='ignore'):
        """
        Loads Word2Vec model
        Returns initial weights for embedding layer.

        inputs:
        model_type      # GoogleNews / glove
        embedding_dim    # Word vector dimensionality
        """
        logger.debug("Using [{0}] model from [{1}]".format(model_type, os.path.join(model_dir, model_file_name)))
        if model_type == 'googlenews' or model_type == "fasttext_wiki":
            assert (exists(os.path.join(model_dir, model_file_name)))
            if os.path.exists(os.path.join(model_dir, model_file_name+'.bin')):
                try:
                    pretrain_model = FastText.load_fasttext_format(os.path.join(model_dir, model_file_name+'.bin'))  # For original fasttext *.bin format.
                except Exception as e:
                    pretrain_model = KeyedVectors.load_word2vec_format(os.path.join(model_dir, model_file_name+'.bin'), binary=True)
            else:
                try:
                    pretrain_model = KeyedVectors.load_word2vec_format(os.path.join(model_dir, model_file_name),
                                                                       binary=self.binary)
                except Exception as e:  # On exception, trying a different format.
                    logger.debug('Loading original word2vec format failed. Trying Gensim format.')
                    pretrain_model = KeyedVectors.load(os.path.join(model_dir, model_file_name))
                pretrain_model.save_word2vec_format(os.path.join(model_dir, model_file_name + ".bin"), binary=True)  # Save model in binary format for faster loading in future.
                logger.debug("Saved binary model at: [{0}]".format(os.path.join(model_dir, model_file_name + ".bin")))
                logger.info(type(pretrain_model))
        elif model_type == 'glove':
            assert (exists(os.path.join(model_dir, model_file_name)))
            logger.debug('Loading existing Glove model: [{0}]'.format(os.path.join(model_dir, model_file_name)))
            # dictionary, where key is word, value is word vectors
            pretrain_model = {}
            for line in open(os.path.join(model_dir, model_file_name), encoding=encoding):
                tmp = line.strip().split()
                word, vec = tmp[0], map(float, tmp[1:])
                assert (len(vec) == self.embedding_dim)
                if word not in pretrain_model:
                    pretrain_model[word] = vec
            assert (len(pretrain_model) == 400000)
        elif model_type == "bert_multi":
            # pretrain_model = FastText.load_fasttext_format(os.path.join(model_dir,model_file_name))
            # pretrain_model = FastText.load_binary_data (os.path.join(model_dir,model_file_name))
            pretrain_model = KeyedVectors.load_word2vec_format(os.path.join(model_dir, model_file_name), binary=False)
            # import io
            # fin = io.open(os.path.join(model_dir, model_file_name), encoding=encoding, newline=newline,
            #               errors=errors)
            # n, d = map(int, fin.readline().split())
            # pretrain_model = {}
            # for line in fin:
            #     tokens = line.rstrip().split(' ')
            #     pretrain_model[tokens[0]] = map(float, tokens[1:])
            """embedding_dict = gensim.models.KeyedVectors.load_word2vec_format(dictFileName, binary=False) embedding_dict.save_word2vec_format(dictFileName+".bin", binary=True) embedding_dict = gensim.models.KeyedVectors.load_word2vec_format(dictFileName+".bin", binary=True)"""
            return pretrain_model
        else:
            raise ValueError('Unknown pretrain model type: %s!' % model_type)

        logger.info(type(pretrain_model))
        return pretrain_model

    def train_w2v(self,sentence_matrix, vocabulary_inv, embedding_dim=300,
                  min_word_count=1, context=10):
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
        model_name = "{:d}features_{:d}minwords_{:d}context".format(embedding_dim,
                                                                    min_word_count,
                                                                    context)
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

            # If we don't plan to train the model any further, calling
            # init_sims will make the model much more memory-efficient.
            pretrain_model.init_sims(replace=True)

            # Saving the model for later use. You can load it later using
            # Word2Vec.load()
            if not exists(model_dir):
                os.mkdir(model_dir)
            logger.debug('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
            pretrain_model.save(model_name)

        #  add unknown words
        embedding_weights = [np.array([pretrain_model[w] if w in pretrain_model
                                       else np.random.uniform(-0.25, 0.25,
                                                              pretrain_model.vector_size)
                                       for w in vocabulary_inv])]
        return embedding_weights

    def get_embedding_matrix(self,vocabulary_inv:dict):
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
    cls = Pretrain()
    data_dict = cls.get_sim("hello","hi")
    logger.debug("Word similarity: [{0}]".format(data_dict))

    sentence_obama = 'Obama speaks to the media in Illinois'
    sentence_president = 'The president greets the press in Chicago'
    sent_sim = cls.get_sent_sim(sentence_obama, sentence_president)
    logger.debug("Sentence similarity: [{0}]".format(sent_sim))
