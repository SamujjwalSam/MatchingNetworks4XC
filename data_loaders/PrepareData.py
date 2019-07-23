# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Prepares the datasets as per Matching Networks model.

__description__ : Prepares the datasets as per Matching Networks model.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : "0.1"
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root
                  directory of this source tree.
__classes__     : PrepareData
__variables__   :
__methods__     :
"""

import torch
from random import sample
from os.path import join,isfile
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from collections import OrderedDict

# from pretrained.TextEncoder import TextEncoder
from pretrained import TextEncoder
from text_process import Clean_Text
from logger.logger import logger
from file_utils import File_Util
from config import configuration as config
from config import platform as plat
from config import username as user


class PrepareData:
    """ Prepare data into proper format.

        Converts strings to vectors,
        Converts category ids to multi-hot vectors,
        etc.
    """

    def __init__(self,
                 dataset_loader,
                 dataset_name: str = config["data"]["dataset_name"],
                 dataset_dir: str = config["paths"]["dataset_dir"][plat][user]) -> None:
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.dataset_loader = dataset_loader

        self.doc2vec_model = None
        self.categories_all = None
        self.sentences_selected,self.classes_selected,self.categories_selected = None,None,None
        self.clean = Clean_Text()

        self.mlb = MultiLabelBinarizer()
        # dataset_loader.gen_data_stats()
        self.text_encoder = TextEncoder.TextEncoder()

    def cat2samples(self,classes_dict: dict = None):
        """
        Converts sample : categories to categories : samples

        :returns: A dictionary of categories to sample mapping.
        """
        cat2id = OrderedDict()
        if classes_dict is None: classes_dict = self.classes_selected
        for k,v in classes_dict.items():
            for cat in v:
                if cat not in cat2id:
                    cat2id[cat] = []
                cat2id[cat].append(k)
        return cat2id

    def prepare_data(self,load_type: str = 'train',return_loaded: bool = False):
        """
        Prepares (loads, vectorize, etc) the data provided by param "load_type".

        :param return_loaded:
        :param load_type: Which data to load: Options: ['train', 'val', 'test']
        """
        if load_type is 'train':  ## Get the idf dict for train documents but not for others.
            self.sentences_selected,self.classes_selected,self.categories_selected,self.categories_all,self.idf_dict =\
                self.dataset_loader.get_data(load_type=load_type)
        else:
            self.sentences_selected,self.classes_selected,self.categories_selected,self.categories_all =\
                self.dataset_loader.get_data(load_type=load_type)
        self.remain_sample_ids = list(self.sentences_selected.keys())
        self.cat2sample_map = self.cat2samples(self.classes_selected)
        self.remain_cat_ids = list(self.categories_selected.keys())
        logger.info("[{}] data counts:\n\tSentences = [{}],\n\tClasses = [{}],\n\tCategories = [{}]"
                    .format(load_type,len(self.sentences_selected),len(self.classes_selected),
                            len(self.categories_selected)))
        logger.info("Total category count: [{}]".format(len(self.categories_all)))

        ## MultiLabelBinarizer only takes list of lists as input. Need to convert our list of ints to list of lists.
        cat_ids = []
        for cat_id in self.categories_all.values():
            cat_ids.append([cat_id])
        self.mlb.fit(cat_ids)

        # self.idf_dict = self.clean.calculate_idf(docs=self.sentences_selected.values())
        if return_loaded:
            return self.sentences_selected,self.classes_selected,self.categories_selected,self.categories_all

    def txt2vec(self,sentences: list,vectorizer=config["prep_vecs"]["vectorizer"],
                tfidf_avg=config["prep_vecs"]["tfidf_avg"]):
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
        # sentences = util.clean_sentences(sentences, specials="""_-@*#'"/\\""", replace='')
        if vectorizer == "doc2vec":
            if self.doc2vec_model is None:
                self.doc2vec_model = self.text_encoder.load_doc2vec(sentences)
            vectors_dict = self.text_encoder.get_doc2vecs(sentences,self.doc2vec_model)
            return vectors_dict
        elif vectorizer == "word2vec":
            w2v_model = self.text_encoder.load_word2vec()
            sentences = list(filter(None,sentences))  ## Removing empty items.
            return self.create_doc_vecs(sentences,w2v_model=w2v_model)
        else:
            raise Exception("Unknown vectorizer: [{}]. \n"
                            "Available options: ['doc2vec','word2vec']"
                            .format(vectorizer))

    def sum_word_vecs(self,words: list,w2v_model,input_size=config["prep_vecs"]["input_size"],avg=True):
        """ Generates a vector of [input_size] using the [words] and [w2v_model].

        :param avg: If average to be calculated.
        :param words: List of words
        :param w2v_model: Word2Vec model by Gensim
        :param input_size: Dimension of each vector
        :return: Vector of sum of all the words.
        """
        # oov_words_dict = {}  ## To hold out-of-vocab words.
        sum_vec = None
        for i,word in enumerate(words):
            ## Multiply idf of that word with the vector
            try:  ## If word exists in idf dict
                idf = self.idf_dict[word]
            except KeyError as e:  ## If word does not exists in idf_dict, multiply 1
                # logger.info("[{}] not found in idf_dict.".format(word))
                idf = 1
            if word in w2v_model.vocab:  ## If word is present in model
                if sum_vec is None:
                    sum_vec = w2v_model[word] * idf
                else:
                    sum_vec = np.add(sum_vec,w2v_model[word] * idf)
            elif word in PrepareData.oov_words_dict:  ## If word is OOV
                if sum_vec is None:
                    sum_vec = PrepareData.oov_words_dict[word] * idf
                else:
                    sum_vec = np.add(sum_vec,PrepareData.oov_words_dict[word] * idf)
            else:  ## New unknown word, need to create random vector.
                new_oov_vec = np.random.uniform(-0.5,0.5,input_size)
                # w2v_model.add(word, new_oov_vec)  ## For some reason, gensim word2vec.add() not working.
                PrepareData.oov_words_dict[word] = new_oov_vec
                if sum_vec is None:
                    sum_vec = PrepareData.oov_words_dict[word] * idf
                else:
                    sum_vec = np.add(sum_vec,PrepareData.oov_words_dict[word] * idf)
        if avg:
            sum_vec = np.divide(sum_vec,float(len(words)))

        return np.stack(sum_vec)

    oov_words_dict = {}

    def create_doc_vecs(self,sentences: list,w2v_model,use_idf=config["prep_vecs"]["idf"],concat_axis=0,
                        input_size=config["prep_vecs"]["input_size"]):
        """
        Calculates the average of vectors of all words within a chunk and concatenates the chunks.

        :param use_idf: Flag to decide if idf is to be used.
        :param input_size:
        :param concat_axis: The axis the vectors should be concatenated.
        :param sents_chunk_mode:
        :param w2v_model:
        :param sentences: Dict of texts.
        :returns: Average of vectors of chunks. Dim: input_size.
        """
        docs_vecs = []  # List to hold generated vectors.
        for i,doc in enumerate(sentences):
            chunks = self.partition_doc(doc)
            chunks = list(filter(None,chunks))  ## Removing empty items.
            vecs = self.sum_word_vecs(chunks,w2v_model)
            docs_vecs.append(vecs)
            # for chunk in chunks:  ## Loop to create vector for each chunk to be concatenated.
            #     avg_vec = None
            #     for word in chunk:  ## Loop to create average of vectors of all words within a chunk.
            #         if word in w2v_model.vocab:
            #             if avg_vec is None:
            #                 avg_vec = w2v_model[word]
            #             else:
            #                 avg_vec = np.add(avg_vec, w2v_model[word])
            #         else:
            #             if word in oov_words_dict:
            #                 avg_vec = oov_words_dict[word]
            #             else:
            #                 new_oov_vec = np.random.uniform(-0.5, 0.5, input_size)
            #                 # w2v_model.add(word, new_oov_vec)
            #                 oov_words_dict[word] = new_oov_vec
            #                 if avg_vec is None:
            #                     avg_vec = new_oov_vec
            #                 else:
            #                     avg_vec = np.add(avg_vec, new_oov_vec)
            #             oov_words.append(word)
            #     docs_vecs[i] = np.concatenate((docs_vecs[i], np.divide(avg_vec, float(len(chunk)))), axis=concat_axis)

        # File_Util.save_json(PrepareData.oov_words_dict,self.dataset_name+"_oov_words_dict",join(self.dataset_dir,self.dataset_name))
        return np.stack(docs_vecs)

    def partition_doc(self,sentence: str,sents_chunk_mode: str = config["text_process"]["sents_chunk_mode"],
                      num_chunks: int = config["prep_vecs"]["num_chunks"]) -> list:
        """
        Divides a document into chunks based on the vectorizer.

        :param num_chunks:
        :param sentence:
        :param sents_chunk_mode:
        :param doc_len:
        :return:
        """
        chunks = []
        if sents_chunk_mode == "concat":
            words = self.clean.tokenizer_spacy(sentence)
            for word in words:
                chunks.append(word)
        elif sents_chunk_mode == "word_avg":
            chunks = self.clean.tokenizer_spacy(sentence)
        elif sents_chunk_mode == "sentences":
            chunks = self.clean.sents_split(sentence)
        elif sents_chunk_mode == "chunked":
            splitted_doc = self.clean.tokenizer_spacy(sentence)
            doc_len = len(splitted_doc)
            chunk_size = doc_len // num_chunks  ## Calculates how large each chunk should be.
            index_start = 0
            for i in range(num_chunks):
                batch_portion = doc_len / (chunk_size * (i + 1))
                if batch_portion > 1.0:
                    index_end = index_start + chunk_size
                else:  ## Available data is less than chunk_size
                    index_end = index_start + (doc_len - index_start)
                logger.info('Making chunk of tokens from [{0}] to [{1}]'.format(index_start,index_end))
                chunk = splitted_doc[index_start:index_end]
                chunks.append(chunk)
                index_start = index_end
        else:
            raise Exception("Unknown document partition mode: [{}]. \n"
                            "Available options: ['concat','word_avg (Default)','sentences','chunked']"
                            .format(sents_chunk_mode))
        chunks = list(filter(None,chunks))  ## Removes empty items, like: ""
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
            ("train_shape",self.x_train.shape,"test_shape",self.x_test.shape,"val_shape",self.x_val.shape))
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_val = (self.x_val - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

    def create_multihot(self,batch_classes_dict):
        """
        Creates multi-hot vectors for a batch of data.

        :param batch_classes_dict:
        :return:
        """
        classes_multihot = self.mlb.fit_transform(batch_classes_dict.values())
        return classes_multihot

    def get_support_cats(self,select: int = config["sampling"]["categories_per_batch"]) -> list:
        """
        Randomly selects [categories_per_batch] number of classes from which support set will be created.

        Will remove the selected classes from self.remain_cat_ids.
        :param select: Number of samples to draw.
        :return:
        """
        self.remain_cat_ids,selected_cat_ids = File_Util.get_batch_keys(self.remain_cat_ids,
                                                                        batch_size=select,
                                                                        remove_keys=False)
        selected_cat_ids = [int(cat) for cat in selected_cat_ids]  ## Integer keys are converted to string when saving
        ## as JSON. Converting them back to integer.
        return selected_cat_ids

    @staticmethod
    def cat_jaccard_sim_mat(cats_hot: list):
        """
        Calculates Jaccard similarities among categories.

        :param cats_hot: List of lists of categories.
        :return: Matrix of similarities. Matrix of jaccard similarities: number of categories, number of categories
        """
        j_sim = np.zeros((len(cats_hot),len(cats_hot)))
        for i,cat1 in enumerate(cats_hot):
            for j,cat2 in enumerate(cats_hot):
                j_sim[i,j] = len(set(cat1).intersection(set(cat2))) / len(set(cat1).union(set(cat2)))
        return j_sim

    def get_support_cats_jaccard(self,cats=None,select=config["sampling"]["categories_per_batch"]):
        """
        Selects categories such that each category is least similar to previously selected categories.

        :param cats: List of lists of categories.
        :param select: Number of samples to draw.
        :return:
        """
        if cats is None: cats = self.remain_cat_ids
        selected_cats = []
        sel_cat = sample(range(len(cats)),k=1)[0]
        selected_cats.append(sel_cat)
        cat_sim = self.cat_jaccard_sim_mat(cats)
        for i in np.arange(select - 1):
            cat_sim[:,sel_cat] = 1.0
            sel_cat = np.argmin(cat_sim[sel_cat,i:])  ## Symmetric matrix, need to check only upper triangle.
            selected_cats.append(sel_cat)

        return selected_cats

    def select_samples(self,support_cat_ids: list,
                       cat2sample_map: dict = None,
                       select: int = config["sampling"]["supports_per_category"],
                       return_cat_indices: bool = False,
                       sample_repeat_mode: str = config["prep_vecs"]["sample_repeat_mode"]):
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
                selected_samples = selected_samples + sample(cat2sample_map[cat],k=select)
            else:  ## Less than required, repeat the available classes.
                selected_samples = selected_samples + cat2sample_map[cat]
                length = len(cat2sample_map[cat])
                if sample_repeat_mode == "append":
                    for i in range(select - length):
                        selected_samples.append(cat2sample_map[cat][i % length])
                elif sample_repeat_mode == "sample":
                    empty_count = select - length
                    for i in range(
                            empty_count):  ## Sampling [supports_per_category] number of items from cat2sample_map[cat].
                        selected_samples.append(cat2sample_map[cat][sample(range(length),1)])
                else:
                    raise Exception("Unknown [sample_repeat_mode]: [{}]. \n"
                                    "Available options: ['append','word_avg','sample']".format(sample_repeat_mode))
        sentences_batch,classes_batch = File_Util.create_batch_repeat(self.sentences_selected,self.classes_selected,
                                                                      selected_samples)

        # sentences_batch = list(filter(None, sentences_batch))  ## Removing empty items.
        x_target = self.txt2vec(sentences_batch)
        y_target_hot = self.mlb.transform(classes_batch)
        if return_cat_indices:
            ## For Multi-Label, multi-label-margin loss
            # y_target_indices = self.mlb.inverse_transform(y_target_hot)

            ## For Multi-Class, cross-entropy loss
            y_target_indices = y_target_hot.argmax(1)

            # target_y_np = np.array(y_target_indices)
            # target_y_np = np.asarray([np.asarray(sublist) for sublist in y_target_indices])
            # for cats in y_target_indices:
            #     np.append(target_y_np, np.array(cats))
            # y_target_indices = np.array(y_target_indices)
            return x_target,y_target_hot,y_target_indices
        return x_target,y_target_hot

    def get_batches(self,batch_size: int = config["sampling"]["batch_size"],
                    supports_per_category: int = config["sampling"]["supports_per_category"],val: bool = False,
                    targets_per_category: int = config["sampling"]["targets_per_category"]):
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
        target_count = str(config["sampling"]["categories_per_batch"] * supports_per_category) + "_" + str(batch_size)
        if val:  ## If true, it's a validation run. Return stored values.
            logger.info("Checking if Validation data is stored at: [{}]".format(
                join(self.dataset_dir,self.dataset_name,self.dataset_name + "_supports_x_" + target_count + ".pkl")))
            if isfile(join(self.dataset_dir,self.dataset_name,
                           self.dataset_name + "_supports_x_" + target_count + ".pkl")):
                logger.info("Found Validation data at: [{}]".format(
                    join(self.dataset_dir,self.dataset_name,
                         self.dataset_name + "_supports_x_" + target_count + ".pkl")))
                supports_x = File_Util.load_pickle(self.dataset_name + "_supports_x_" + target_count,
                                                   file_path=join(self.dataset_dir,self.dataset_name))
                supports_y_hots = File_Util.load_pickle(self.dataset_name + "_supports_y_hots_" + target_count,
                                                        file_path=join(self.dataset_dir,self.dataset_name))
                targets_x = File_Util.load_pickle(self.dataset_name + "_targets_x_" + target_count,
                                                  file_path=join(self.dataset_dir,self.dataset_name))
                targets_y_hots = File_Util.load_pickle(self.dataset_name + "_targets_y_hots_" + target_count,
                                                       file_path=join(self.dataset_dir,self.dataset_name))
                target_cat_indices = File_Util.load_pickle(self.dataset_name + "_target_cat_indices_" + target_count,
                                                           file_path=join(self.dataset_dir,self.dataset_name))
                return supports_x,supports_y_hots,targets_x,targets_y_hots,target_cat_indices
            logger.info("Validation data not found at: [{}]".format(
                join(self.dataset_dir,self.dataset_name,self.dataset_name + "_supports_x_" + target_count + ".pkl")))

        support_cat_ids = self.get_support_cats()
        supports_x = []
        supports_y_hots = []
        targets_x = []
        targets_y_hots = []
        target_cat_indices = []
        for i in range(batch_size):
            x_support,y_support_hot = self.select_samples(support_cat_ids,
                                                          select=supports_per_category)
            x_target,y_target_hot,target_cat_indices_batch =\
                self.select_samples(support_cat_ids,select=targets_per_category,return_cat_indices=True)
            supports_x.append(x_support)
            supports_y_hots.append(y_support_hot)
            targets_x.append(x_target)
            targets_y_hots.append(y_target_hot)
            target_cat_indices.append(target_cat_indices_batch)
        supports_x = np.stack(supports_x)
        supports_y_hots = np.stack(supports_y_hots)
        targets_x = np.stack(targets_x)
        targets_y_hots = np.stack(targets_y_hots)
        target_cat_indices = np.stack(target_cat_indices)

        if val:
            logger.info("Storing Validation data at: [{}]".format(
                join(self.dataset_dir,self.dataset_name,self.dataset_name + "_supports_x_" + target_count + ".pkl")))
            File_Util.save_pickle(supports_x,filename=self.dataset_name + "_supports_x_" + target_count,
                                  file_path=join(self.dataset_dir,self.dataset_name),overwrite=True)
            File_Util.save_pickle(supports_y_hots,self.dataset_name + "_supports_y_hots_" + target_count,
                                  file_path=join(self.dataset_dir,self.dataset_name),overwrite=True)
            File_Util.save_pickle(targets_x,self.dataset_name + "_targets_x_" + target_count,
                                  file_path=join(self.dataset_dir,self.dataset_name),overwrite=True)
            File_Util.save_pickle(targets_y_hots,self.dataset_name + "_targets_y_hots_" + target_count,
                                  file_path=join(self.dataset_dir,self.dataset_name),overwrite=True)
            File_Util.save_pickle(target_cat_indices,self.dataset_name + "_target_cat_indices_" + target_count,
                                  file_path=join(self.dataset_dir,self.dataset_name),overwrite=True)

        target_cat_indices = np.stack(target_cat_indices)
        return supports_x,supports_y_hots,targets_x,targets_y_hots,target_cat_indices

    def test_d2v_1nn(self):
        """ Checks embedding quality with 1 Nearest Neighbors.

        """
        self.prepare_data(load_type='train')
        test_supports,test_supports_hot,_,_,_ = self.get_batches(batch_size=1,targets_per_category=0)
        test_supports = test_supports.squeeze()
        test_supports_hot = test_supports_hot.squeeze()

        self.prepare_data(load_type='test')
        _,_,test_targets,test_targets_hot,_ = self.get_batches(batch_size=1,supports_per_category=0)
        test_targets = test_targets.squeeze()
        test_targets_hot = test_targets_hot.squeeze()

        self.find_test_supports(test_targets,test_targets_hot,test_supports,test_supports_hot)

    def get_test_data(self,return_cat_indices=True):
        """
        Creates Test data: Selects whole training data as support set.

        :param return_cat_indices: Flag to set if target_indices are to be returned.
        :return: test data.
        """
        ## Using train data as supports
        ## Read and select train data
        self.prepare_data(load_type='train')
        keys = list(self.classes_selected.keys())
        train_sentences,train_classes = File_Util.create_batch_repeat(self.sentences_selected,self.classes_selected,
                                                                      keys)
        ## Vectorize train data
        test_supports = self.txt2vec(train_sentences)
        test_supports_hot = self.mlb.transform(train_classes)

        ## Using test set as targets
        ## Read test data
        self.prepare_data(load_type='test')
        keys = list(self.classes_selected.keys())
        test_sentences,test_classes = File_Util.create_batch_repeat(self.sentences_selected,self.classes_selected,keys)
        ## Vectorize test data
        test_targets = self.txt2vec(test_sentences)
        test_targets_hot = self.mlb.transform(test_classes)

        self.find_test_supports(test_targets,test_targets_hot,test_supports,test_supports_hot)

        if return_cat_indices:
            ## For Multi-Label, multi-label-margin loss
            # y_target_indices = [self.mlb.inverse_transform(test_targets_hot)]

            ## For Multi-Class, cross-entropy loss
            y_target_indices = test_targets_hot.argmax(1)

            # return test_supports, test_supports_hot, test_supports, test_supports_hot, y_target_indices
            return test_supports,test_supports_hot,test_targets,test_targets_hot,y_target_indices

        return test_supports,test_supports_hot,test_targets,test_targets_hot

    @staticmethod
    def find_test_supports(test_targets,test_targets_hot,test_supports,test_supports_hot,k=1):
        """
        Finds supports for testing using doc2vec similarity.

        :param test_targets:
        :param test_targets_hot:
        :param test_supports:
        :param test_supports_hot:
        :param k:
        :return:
        """
        from sklearn.metrics.pairwise import cosine_similarity

        cosine_sim = cosine_similarity(test_targets,test_supports)
        # logger.debug(cosine_sim.shape)
        # logger.debug(cosine_sim)
        most_sim_indices = np.argsort(cosine_sim)[:,-k:]
        # logger.debug(most_sim_indices.shape)
        # logger.debug(most_sim_indices)
        nearest_supports = []
        nearest_supports_hot = []
        nearest_supports_merged = []
        nearest_supports_merged_hot = []
        for i in np.arange(test_targets.shape[0]):
            supports = []
            supports_hot = []
            for j in np.arange(k):
                ind = most_sim_indices[i,j]
                supports.append(test_supports[ind])
                nearest_supports_merged.append(test_supports[ind])
                supports_hot.append(test_supports_hot[ind])
                nearest_supports_merged_hot.append(test_supports_hot[ind])
            supports = np.stack(supports)
            supports_hot = np.stack(supports_hot)
            nearest_supports.append(supports)
            nearest_supports_hot.append(supports_hot)

        ## Count if selected support's labels match with their targets.
        correct_count = 0
        for i in np.arange(test_targets_hot.shape[0]):
            for j in np.arange(nearest_supports_hot[i].shape[0]):  # Support number
                for idx in np.arange(test_targets_hot.shape[1]):
                    # if test_targets_hot[i][idx] != nearest_supports_hot[i][j][idx]: logger.debug("i: {}, j: {},
                    # idx: {},  [{} == {}]".format(i,j,idx,test_targets_hot[i][idx], nearest_supports_hot[i][j][idx]))
                    if test_targets_hot[i][idx] == 1 and nearest_supports_hot[i][j][idx] == 1:
                        # logger.info("correct_count: {}".format(correct_count))
                        correct_count += 1

        logger.info("Number of targets match with their supports: [{}] out of [{}] or [{} %] with [{}] neighbors."
                    .format(correct_count,test_targets_hot.shape[0],100 * (correct_count / test_targets_hot.shape[0]),
                            k))

        # return np.stack(nearest_supports), np.stack(nearest_supports_hot), np.stack(nearest_supports_merged), \
        #        np.stack(nearest_supports_merged_hot)


if __name__ == '__main__':
    logger.debug("Preparing Data...")
    from data_loaders.common_data_handler import Common_JSON_Handler

    data_loader = Common_JSON_Handler(dataset_type=config["xc_datasets"][config["data"]["dataset_name"]],
                                      dataset_name=config["data"]["dataset_name"],
                                      data_dir=config["paths"]["dataset_dir"][plat][user])

    data_formatter = PrepareData(dataset_loader=data_loader,
                                 dataset_name=config["data"]["dataset_name"],
                                 dataset_dir=config["paths"]["dataset_dir"][plat][user])

    # data_formatter.prepare_data()
    # data_formatter.test_d2v_1nn()
    data_formatter.get_test_data()
    # j_sim = data_formatter.test_d2v_1nn()
    # logger.debug(j_sim)
