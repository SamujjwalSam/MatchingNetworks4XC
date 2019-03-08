# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Matching Networks for Extreme Classification.
__description__ : Prepares the datasets as per Matching Networks model.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : PrepareData

__variables__   :

__methods__     :
"""

from random import sample
from os.path import join, isfile
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from collections import OrderedDict

from pretrained.TextEncoder import TextEncoder
from logger.logger import logger
from utils import util
from config import configuration as config
from config import platform as plat


class PrepareData:
    """ Prepare data into proper format.

        Converts strings to vectors,
        Converts category ids to multi-hot vectors,
        etc.
    """

    def __init__(self,
                 dataset_loader,
                 dataset_name=config["data"]["dataset_name"],
                 dataset_dir=config["paths"]["dataset_dir"][plat]):
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.dataset_loader = dataset_loader

        self.doc2vec_model = None
        self.categories_all = None
        self.sentences_selected, self.classes_selected, self.categories_selected = None, None, None

        self.mlb = MultiLabelBinarizer()
        dataset_loader.gen_data_stats()

    def cat2samples(self, classes_dict: dict = None):
        """
        Converts sample : categories to categories : samples

        :returns: A dictionary of categories to sample mapping.
        """
        cat2id = OrderedDict()
        if classes_dict is None: classes_dict = self.classes_selected
        for k, v in classes_dict.items():
            for cat in v:
                if cat not in cat2id:
                    cat2id[cat] = []
                cat2id[cat].append(k)
        return cat2id

    def prepare_data(self, load_type='train'):
        """
        Prepares (loads, vectorize, etc) the data provided by param "load_type".

        :param load_type: Which data to load: Options: ['train', 'val', 'test']
        """
        self.sentences_selected, self.classes_selected, self.categories_selected, self.categories_all = \
            self.dataset_loader.get_data(load_type=load_type)
        self.remain_sample_ids = list(self.sentences_selected.keys())
        self.cat2sample_map = self.cat2samples(self.classes_selected)
        self.remain_cat_ids = list(self.categories_selected.keys())
        logger.info("[{}] data counts:\n\tSentences = [{}],\n\tClasses = [{}],\n\tCategories = [{}]"
                    .format(load_type, len(self.sentences_selected), len(self.classes_selected),
                            len(self.categories_selected)))
        ## MultiLabelBinarizer only takes list of lists as input. Need to convert our list of ints to list of lists.
        cat_ids = []
        for cat_id in self.categories_all.values():
            cat_ids.append([cat_id])
        self.mlb.fit(cat_ids)

    def txt2vec(self, sentences: list, vectorizer=config["prep_vecs"]["vectorizer"], tfidf_avg=config["prep_vecs"]["tfidf_avg"]):
        """
        Creates vectors from input_texts based on [vectorizer].

        :param max_vec_len: Maximum vector length of each document.
        :param num_chunks: Number of chunks the input_texts are to be divided. (Applicable only when vectorizer = "chunked")
        :param input_size: Embedding dimension for each word.
        :param sentences:
        :param vectorizer: Decides how to create the vector.
            "chunked" : Partitions the whole text into equal length chunks and concatenates the avg of each chunk.
                        vec_len = num_chunks * input_size
                        [["these", "are"], ["chunks", "."]]

            "sentences" : Same as chunked except each sentence forms a chunk. "\n" is sentence separator.
                        vec_len = max(num_sents, max_len) * input_size
                        ["this", "is", "a", "sentence", "."]
                        NOTE: This will create variable length vectors.

            "concat" : Concatenates the vectors of each word, adds padding to make equal length.
                       vec_len = max(num_words) * input_size

            "word_avg" : Take the average of vectors of all the words.
                       vec_len = input_size
                       [["these"], ["are"], ["words"], ["."]]

            "doc2vec" : Use Gensim Doc2Vec to generate vectors.
                        https://radimrehurek.com/gensim/models/doc2vec.html
                        vec_len = input_size
                        ["this", "is", "a", "document", "."]

        :param tfidf_avg: If tf-idf weighted avg is to be taken or simple.
            True  : Take average based on each words tf-idf value.
            False : Take simple average.

        :returns: Vector length, numpy.ndarray(batch_size, vec_len)
        """
        self.text_encoder = TextEncoder()

        # sentences = util.clean_sentences(sentences, specials="""_-@*#'"/\\""", replace='')

        if vectorizer == "doc2vec":
            if self.doc2vec_model is None:
                self.doc2vec_model = self.text_encoder.load_doc2vec(sentences)
            vectors_dict = self.text_encoder.get_doc2vecs(sentences, self.doc2vec_model)
            return vectors_dict
        else:
            w2v_model = self.text_encoder.load_word2vec()
            return self.create_doc_vecs(sentences=sentences, w2v_model=w2v_model)

    def create_doc_vecs(self, sentences: list, w2v_model, concat_axis=0, input_size=config["prep_vecs"]["input_size"]):
        """
        Calculates the average of vectors of all words within a chunk and concatenates the chunks.

        :param input_size:
        :param concat_axis: The axis the vectors should be concatenated.
        :param sents_chunk_mode:
        :param w2v_model:
        :param sentences: Dict of texts.
        :returns: Average of vectors of chunks. Dim: input_size.
        """
        oov_words = []  ## To hold out-of-vocab words.
        docs_vecs = []
        for i, doc in enumerate(sentences):
            chunks = self.partition_doc(doc)
            chunks = list(filter(None, chunks))  ## Removing empty items.
            for chunk in chunks:  ## Loop to create vector for each chunk to be concatenated.
                avg_vec = None
                for word in chunk:  ## Loop to create average of vectors of all words within a chunk.
                    if word in w2v_model.vocab:
                        if avg_vec is None:
                            avg_vec = w2v_model[word]
                        else:
                            avg_vec = np.add(avg_vec, avg_vec)
                    else:
                        new_oov_vec = np.random.uniform(-0.5, 0.5, input_size)
                        w2v_model.add(word, new_oov_vec)
                        oov_words.append(word)
                        if avg_vec is None:
                            avg_vec = new_oov_vec
                        else:
                            avg_vec = np.add(avg_vec, new_oov_vec)
                docs_vecs[i] = np.concatenate((docs_vecs[i], np.divide(avg_vec, float(len(chunk)))), axis=concat_axis)
        util.save_json(oov_words, "oov_words")

        return docs_vecs

    def partition_doc(self, sentence, sents_chunk_mode=config["text_process"]["sents_chunk_mode"],
                      num_chunks=config["prep_vecs"]["num_chunks"]):
        """
        Divides a document into chunks based on the vectorizer.

        :param num_chunks:
        :param sentence:
        :param sents_chunk_mode:
        :param doc_len:
        :return:
        """
        chunks = []
        # TODO: Use better word and sentence tokenizer, i.e. Spacy, NLTK, etc.
        if sents_chunk_mode == "concat":
            words = sentence.split(" ")
            for word in words:
                chunks.append(word)
        elif sents_chunk_mode == "word_avg":
            chunks = sentence.split(" ")
        elif sents_chunk_mode == "sentences":
            chunks = sentence.splitlines()
        elif sents_chunk_mode == "chunked":
            splitted_doc = sentence.split()
            doc_len = len(splitted_doc)
            chunk_size = doc_len // num_chunks  ## Calculates how large each chunk should be.
            index_start = 0
            for i in range(num_chunks):
                batch_portion = doc_len / (chunk_size * (i + 1))
                if batch_portion > 1.0:
                    index_end = index_start + chunk_size
                else:  ## Available data is less than chunk_size
                    index_end = index_start + (doc_len - index_start)
                logger.info('Making chunk of tokens from [{0}] to [{1}]'.format(index_start, index_end))
                chunk = splitted_doc[index_start:index_end]
                chunks.append(chunk)
                index_start = index_end
        else:
            raise Exception("Unknown document partition mode: [{}]. \n"
                            "Available options: ['concat','word_avg','sentences','chunked (Default)']"
                            .format(sents_chunk_mode))
        chunks = list(filter(None, chunks))  ## Removes empty items, like: ""
        return chunks

    def normalize_inputs(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1.

        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        logger.debug(
            ("train_shape", self.x_train.shape, "test_shape", self.x_test.shape, "val_shape", self.x_val.shape))
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

    def get_support_cats(self, select=config["sampling"]["categories_per_batch"]):
        """
        Randomly selects [categories_per_batch] number of classes from which support set will be created.

        Will remove the selected classes from self.remain_cat_ids.
        :param select: Number of samples to draw.
        :return:
        """
        self.remain_cat_ids, selected_cat_ids = util.get_batch_keys(self.remain_cat_ids,
                                                                    batch_size=select,
                                                                    remove_keys=False)
        selected_cat_ids = [int(cat) for cat in selected_cat_ids]  ## Integer keys are converted to string when saving
        ## as JSON. Converting them back to integer.
        return selected_cat_ids

    def select_samples(self, support_cat_ids, cat2sample_map=None, input_size=300, samples_per_category=4,
                       vectorizer="doc2vec", repeat_mode='append', return_cat_indices=False):
        """
        Returns a batch of feature vectors and multi-hot classes.

        :param return_cat_indices: If true category indices are returned instead of multi-hot vectors. (Used for cross entropy loss)
        :param input_size:
        :param sample_repeat_mode: How to repeat sample if available data is less than samples_per_class. ["append (default)", "sample"].
        :param cat2sample_map: A dictionary of categories to samples mapping.
        :return: Next batch
        :param min_match: Minimum number of categories should match.
        :param support_cat_ids:
        :param select:
        :param vectorizer:
        """
        selected_samples = []
        if cat2sample_map is None: cat2sample_map = self.cat2sample_map
        for cat in support_cat_ids:
            if len(cat2sample_map[cat]) == select:
                selected_samples = selected_samples + cat2sample_map[cat]
            elif len(cat2sample_map[cat]) > select:  ## More than required, sample [supports_per_category]
                ## from the list.
                selected_samples = selected_samples + sample(cat2sample_map[cat], k=select)
            else:  ## Less than required, repeat the available classes.
                selected_samples = selected_samples + cat2sample_map[cat]
                length = len(cat2sample_map[cat])
                if sample_repeat_mode == "append":
                    for i in range(select - length):
                        selected_samples.append(cat2sample_map[cat][i % length])
                elif sample_repeat_mode == "sample":
                    empty_count = select - length
                    for i in range(
                            empty_count):  ## Sampling [supports_per_category] number of elements from cat2sample_map[cat].
                        selected_samples.append(cat2sample_map[cat][sample(range(length), 1)])
                else:
                    raise Exception("Unknown [sample_repeat_mode]: [{}]. \n"
                                    "Available options: ['append','word_avg','sample']".format(sample_repeat_mode))
        sentences_batch, classes_batch = util.create_batch_repeat(self.sentences_selected, self.classes_selected,
                                                                  selected_samples)
        x_target = self.txt2vec(sentences_batch)
        y_target_hot = self.mlb.transform(classes_batch)
        if return_cat_indices:
            y_target_indices = self.mlb.inverse_transform(y_target_hot)
            # y_target_indices = y_target_hot.argmax(1)  ## Returns label indices. Does not work in Multi-Label setting.
            # target_y_np = np.array(y_target_indices)
            # target_y_np = np.asarray([np.asarray(sublist) for sublist in y_target_indices])
            # for cats in y_target_indices:
            #     np.append(target_y_np, np.array(cats))
            # y_target_indices = np.array(y_target_indices)
            return x_target, y_target_hot, y_target_indices
        return x_target, y_target_hot

    def get_batches(self, batch_size=32, supports_per_category=config["sampling"]["supports_per_category"], val=False,
                    targets_per_category=config["sampling"]["targets_per_category"], sample_repeat_mode='append'):
        """
        Returns an iterator over data.

        :param val: Flag to denote if it is Validation run. If true, it's a validation run. Return old values.
        :param targets_per_category:
        :param input_size: Input embedding dimension.
        :param batch_size:
        :param categories_per_batch:
        :param supports_per_category:
        :param vectorizer:
        :returns: An iterator over data.
        """
        if val:  ## If true, it's a validation run. Return stored values.
            logger.debug("Checking if Validation data is stored at: [{}]".format(
                join(self.dataset_dir, self.dataset_name, self.dataset_name + "_supports_x.pkl")))
            if isfile(join(self.dataset_dir, self.dataset_name, self.dataset_name + "_supports_x.pkl")):
                logger.info("Found Validation data at: [{}]".format(
                    join(self.dataset_dir, self.dataset_name, self.dataset_name + "_supports_x.pkl")))
                supports_x = util.load_pickle(self.dataset_name + "_supports_x",
                                              file_path=join(self.dataset_dir, self.dataset_name))
                supports_y_hots = util.load_pickle(self.dataset_name + "_supports_y_hots",
                                                   file_path=join(self.dataset_dir, self.dataset_name))
                targets_x = util.load_pickle(self.dataset_name + "_targets_x",
                                             file_path=join(self.dataset_dir, self.dataset_name))
                targets_y_hots = util.load_pickle(self.dataset_name + "_targets_y_hots",
                                                  file_path=join(self.dataset_dir, self.dataset_name))
                target_cat_indices = util.load_pickle(self.dataset_name + "_target_cat_indices",
                                                      file_path=join(self.dataset_dir, self.dataset_name))
                return supports_x, supports_y_hots, targets_x, targets_y_hots, target_cat_indices
            logger.debug("Validation data not found at: [{}]".format(
                join(self.dataset_dir, self.dataset_name, self.dataset_name + "_supports_x.pkl")))

        support_cat_ids = self.get_support_cats()
        supports_x = []
        supports_y_hots = []
        targets_x = []
        targets_y_hots = []
        target_cat_indices = []
        # target_cat_indices_list1 = []
        # target_cat_indices_list2 = []
        for i in range(batch_size):
            x_support, y_support_hot = self.select_samples(support_cat_ids,
                                                           select=supports_per_category,
                                                           sample_repeat_mode=sample_repeat_mode,
                                                           )
            # sel_cat = sample(support_cat_ids, k=1)
            x_target, y_target_hot, target_cat_indices_batch = \
                self.select_samples(support_cat_ids, select=targets_per_category, return_cat_indices=True)
            supports_x.append(x_support)
            supports_y_hots.append(y_support_hot)
            # support_cat_indices.append(support_cat_indices_batch)
            targets_x.append(x_target)
            targets_y_hots.append(y_target_hot)
            target_cat_indices.append(target_cat_indices_batch)
            # target_cat_indices_list1.append(list(target_cat_indices_batch.tolist()))
        supports_x = np.stack(supports_x)
        supports_y_hots = np.stack(supports_y_hots)
        # support_cat_indices = np.stack(support_cat_indices)
        targets_x = np.stack(targets_x)
        targets_y_hots = np.stack(targets_y_hots)
        # target_cat_indices.append(target_cat_indices)
        # target_cat_indices_list2.append(list(target_cat_indices_list1))

        if val:
            logger.info("Storing Validation data at: [{}]".format(
                join(self.dataset_dir, self.dataset_name, self.dataset_name + "_supports_x.pkl")))
            util.save_pickle(supports_x, filename=self.dataset_name + "_supports_x",
                             file_path=join(self.dataset_dir, self.dataset_name), overwrite=True)
            util.save_pickle(supports_y_hots, self.dataset_name + "_supports_y_hots",
                             file_path=join(self.dataset_dir, self.dataset_name), overwrite=True)
            util.save_pickle(targets_x, self.dataset_name + "_targets_x",
                             file_path=join(self.dataset_dir, self.dataset_name), overwrite=True)
            util.save_pickle(targets_y_hots, self.dataset_name + "_targets_y_hots",
                             file_path=join(self.dataset_dir, self.dataset_name), overwrite=True)
            util.save_pickle(target_cat_indices, self.dataset_name + "_target_cat_indices",
                             file_path=join(self.dataset_dir, self.dataset_name), overwrite=True)
        return supports_x, supports_y_hots, targets_x, targets_y_hots, target_cat_indices  # , target_cat_indices_list2


if __name__ == '__main__':
    logger.debug("Preparing Data...")
    from data_loaders.common_data_handler import Common_JSON_Handler
    plat = util.get_platform()

    data_loader = Common_JSON_Handler(dataset_type=config["xc_datasets"][config["data"]["dataset_name"]],
                                      dataset_name=config["data"]["dataset_name"],
                                      data_dir=config["paths"]["dataset_dir"][plat])

    data_formatter = PrepareData(dataset_loader=data_loader,
                                 dataset_name=config["data"]["dataset_name"],
                                 dataset_dir=config["paths"]["dataset_dir"][plat])

    data_formatter.prepare_data()
    b = [[2],[0,1,2],[1,2],[0]]
    j_sim = data_formatter.cat_jaccard_sim_mat(b)
    logger.info(j_sim)
    j_sim = data_formatter.get_support_cats(select=20)
    logger.info(j_sim)
    j_sim = data_formatter.get_support_cats_jaccard(select=20)
    logger.info(j_sim)
    exit(0)

    supports_x, y_support_hots, targets_x, targets_y_hots = data_formatter.get_batches(batch_size=2,
                                                                                       categories_per_batch=3,
                                                                                       supports_per_category=5)
    logger.debug(supports_x.shape)
    logger.debug(y_support_hots.shape)
    logger.debug(targets_x.shape)
    logger.debug(targets_y_hots.shape)
