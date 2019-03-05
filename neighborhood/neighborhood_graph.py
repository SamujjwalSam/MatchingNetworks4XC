# coding=utf-8
# !/usr/bin/python3.6 ## Please use python 3.6
"""
__synopsis__    : Script to create Neighborhood graph.
__description__ : Class to generate neighborhood graph based on label similarity between samples.
__project__     : MNXC
__author__      : Samujjwal Ghosh <cs16resch01001@iith.ac.in>
__version__     : ": 0.1 "
__date__        : "08-11-2018"
__copyright__   : "Copyright (c) 2019"
__license__     : This source code is licensed under the MIT-style license found in the LICENSE file in the root directory of this source tree.

__classes__     : Neighborhood

__variables__   :

__methods__     :
"""

import networkx as nx
from os.path import join, exists
from scipy import *
from queue import Queue  # Python 2.7 does not have this library
from collections import OrderedDict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import util
from pretrained.TextEncoder import TextEncoder
from logger.logger import logger


class Neighborhood:
    """
    Class to generate neighborhood graph based on label similarity between samples.

    Supported models: glove, word2vec, fasttext, googlenews, bert, lex, etc.
    """

    def __init__(self, dataset_name: str, graph_dir: str = "D:\\Datasets\\Extreme Classification",
                 graph_format: str = "graphml", k: int = 10):
        """

        :param dataset_name:
        :param graph_dir:
        :param graph_format:
        :param k:
        """
        super(Neighborhood, self).__init__()
        self.graph_dir = graph_dir
        self.dataset_name = dataset_name
        self.graph_format = graph_format
        self.k = k
        self.classes = util.load_json(
            join(graph_dir, dataset_name, dataset_name + "_text_json", dataset_name + "_classes"))
        self.categories = util.load_json(
            join(graph_dir, dataset_name, dataset_name + "_text_json", dataset_name + "_categories"))
        self.id2cat_map = util.inverse_dict_elm(self.categories)

    def create_V(self):
        """
        Generates the list of vertices.

        :param :
        :return:
        """
        return list(self.categories.keys())

    def test_cosine(self, k=2):
        """

        :param k:
        """
        a = [[0, 1, 1, 0, 0],
             [0, 1, 1, 0, 0],
             [0, 0, 1, 1, 0],
             [1, 0, 0, 1, 0]]
        a = np.ones((2, 2))
        b = np.ones((2, 2))
        pair_cosine = cosine_similarity(a, b)
        logger.debug(pair_cosine)
        logger.debug(pair_cosine.shape)
        exit(0)
        np.fill_diagonal(pair_cosine, 0)
        logger.debug(pair_cosine)
        pair_top_cosine_idx = np.argpartition(pair_cosine, -k)
        logger.debug(pair_top_cosine_idx)
        logger.debug(pair_top_cosine_idx[:, k:])
        logger.debug(type(pair_top_cosine_idx))
        # pair_top_cosine = pair_cosine[[pair_top_cosine_idx]]
        # logger.debug(pair_top_cosine)

    def dict2multihot(self, data_dict: dict = None):
        """
        Converts classes dict (id:[label_ids]) to multi-hot vectors.

        :param data_dict: (id:[label_ids])
        :return:
        """
        if data_dict is None: data_dict = self.classes
        mlb = MultiLabelBinarizer()
        classes_multihot = mlb.fit_transform(data_dict.values())
        logger.debug(classes_multihot.shape)
        logger.debug(type(classes_multihot))
        return classes_multihot

    def topk_sim_idx(self, multihot_data, k: int):
        """
        Finds the top k neighrest neighbors for each sample using cosine similarity.

        :type k: int
        :param multihot_data: matrix of multi-hot vectors [samples * categories].
        :param k:
        :return:
        """
        pair_cosine = cosine_similarity(multihot_data)
        np.fill_diagonal(pair_cosine, 0)  # Remove self-similarity.
        neighbor_idx = np.argpartition(pair_cosine, -k)  # use (-) to partition by largest values.
        neighbor_idx = neighbor_idx[:, -k:]  # last [k] columns are the largest (most similar).
        self.k = k  # Storing to use when saving files.
        assert neighbor_idx.shape[0] == len(multihot_data)
        assert neighbor_idx.shape[1] == k
        return neighbor_idx

    def create_neighborhood_graph(self, neighbor_idx):
        """
        Creates neighborhood graph of samples based on label similarity using Networkx library.

        :param neighbor_idx: Indices of neighbors.
        """
        sample_ids = list(self.classes.keys())  # Get sample indices.
        G = nx.Graph()
        for i, nb_ids in enumerate(neighbor_idx):
            for nb_id in nb_ids:  # Add edge for each neighbor.
                # logger.debug("[{0}] is connnected to [{1}]".format(sample_ids[i], sample_ids[nb_id]))
                if sample_ids[i] != sample_ids[nb_id]:
                    G.add_edge(sample_ids[i], sample_ids[nb_id], label='e' + str(i))
                else:
                    logger.debug("Both same: [{0}] and [{1}]".format(sample_ids[i], sample_ids[nb_id]))
        print("Neighborhood graph: ", G)
        return G

    def load_neighborhood_graph(self, k: int = None):
        """
        Loads the graph file if found else creates neighborhood graph.

        :param k:
        :return:
        """
        if k is None:
            k = self.k
        if exists(join(self.graph_dir, self.dataset_name, self.dataset_name + "_G_" + str(self.k) + ".graphml")):
            G = nx.read_graphml(
                join(self.graph_dir, self.dataset_name, self.dataset_name + "_G_" + str(self.k) + ".graphml"))
            logger.debug("Loaded neighborhood graph from [{0}]".format(
                join(self.graph_dir, self.dataset_name, self.dataset_name + "_G_" + str(self.k) + ".graphml")))
            # stats = util.load_json(join(self.graph_dir,self.dataset_name,self.dataset_name+"_stats_"+str(self.k)))
            stats = self.graph_stats(G)
            # util.save_json(stats, filename=self.dataset_name+"_stats_"+str(self.k),file_path=join(self.graph_dir,self.dataset_name),overwrite=True)
        else:
            data_dict = self.dict2multihot()
            neighbor_idx = self.topk_sim_idx(data_dict, k)
            G = self.create_neighborhood_graph(neighbor_idx)
            stats = self.graph_stats(G)
        logger.debug("Saving neighborhood graph at [{0}]".format(
            join(self.graph_dir, self.dataset_name, self.dataset_name + "_G_" + str(self.k) + ".graphml")))
        nx.write_graphml(G,
                         join(self.graph_dir, self.dataset_name, self.dataset_name + "_G_" + str(self.k) + ".graphml"))
        util.save_json(stats, filename=self.dataset_name + "_stats_" + str(self.k),
                       file_path=join(self.graph_dir, self.dataset_name), overwrite=True)
        logger.debug("Graph stats: [{0}]".format(stats))
        return G, stats

    def graph_stats(self, G):
        """
        Generates and returns graph related statistics.

        :param G: Graph in Netwokx format.
        :return: dict
        """
        G_stats = OrderedDict()
        G_stats["degree_sequence"] = sorted([d for n, d in G.degree()], reverse=True)
        logger.debug("degree_sequence: {0}".format(G_stats["degree_sequence"]))
        G_stats["dmax"] = max(G_stats["degree_sequence"])
        logger.debug("dmax: [{0}]".format(G_stats["dmax"]))
        G_stats["dmin"] = min(G_stats["degree_sequence"])
        logger.debug("dmin: [{0}]".format(G_stats["dmin"]))
        G_stats["info"] = nx.info(G)
        logger.debug("info: [{0}]".format(G_stats["info"]))
        G_stats["node_count"] = nx.number_of_nodes(G)
        logger.debug("node_count: [{0}]".format(G_stats["node_count"]))
        G_stats["edge_count"] = nx.number_of_edges(G)
        logger.debug("edge_count: [{0}]".format(G_stats["edge_count"]))
        G_stats["radius"] = nx.radius(G)
        logger.debug("radius: [{0}]".format(G_stats["radius"]))
        G_stats["diameter"] = nx.diameter(G)
        logger.debug("diameter: [{0}]".format(G_stats["diameter"]))
        G_stats["eccentricity"] = nx.eccentricity(G)
        logger.debug("eccentricity: [{0}]".format(G_stats["eccentricity"]))
        G_stats["center"] = nx.center(G)
        logger.debug("center: [{0}]".format(G_stats["center"]))
        G_stats["periphery"] = nx.periphery(G)
        logger.debug("periphery: [{0}]".format(G_stats["periphery"]))
        G_stats["density"] = nx.density(G)
        logger.debug("density: [{0}]".format(G_stats["density"]))
        G_stats["connected_component_subgraphs"] = nx.connected_component_subgraphs(G)
        logger.debug("connected_component_subgraphs: [{0}]".format(G_stats["connected_component_subgraphs"]))

        return G_stats

    def add_semantic_info(self, E_cats, model_type="glove", embedding_dim=300, alpha=0.5):
        """
        Calculates and stores the semantic similarity between two label texts.

        :param E_cats: Edgedict with category texts.
        :param model_type:
        :param embedding_dim:
        :param alpha: Weitage parameter between "category co-occurrence" and "semantic similarity". High value gives more importance to "category co-occurrence"
        :return:
        TODO: handle multi-word categories and unknown words.
        """
        pretrain_model = TextEncoder(model_type=model_type, embedding_dim=embedding_dim)
        # semantic_sim = OrderedDict()
        for (cat1, cat2) in E_cats.keys():
            E_cats[(cat1, cat2)] = (E_cats[(cat1, cat2)], pretrain_model.get_sim(cat1, cat2),
                                    alpha * E_cats[(cat1, cat2)] + (1 - alpha) * pretrain_model.get_sim(cat1, cat2))
            # semantic_sim[(cat1,cat2)] = alpha*E_cats[(cat1,cat2)]
            # + (1-alpha)*pretrain.similarity(cat1,cat2)
        return E_cats

    def cat2id_map(self, v):
        """
        An utility function to relabel nodes of upcoming graph with textual label names

        :param v: label id (int)
        :return: returns the texual label of the node id [v]
        """
        v = int(v)
        if v in self.id2cat_map:
            return self.id2cat_map[v]
        return str(v)

    def find_single_labels(self):
        """
        Finds the number of samples with only single label.

        """
        single_labels = []
        for i, t in enumerate(self.classes):
            if len(t) == 1:
                single_labels.append(i)
        if single_labels:
            logger.debug(len(single_labels), 'samples has only single category. These categories will not occur in the'
                                             'co-occurrence graph.')
        return len(single_labels)

    def plot_occurance(self, E, plot_name='sample_co-occurance.jpg', clear=True, log=False):
        from matplotlib import pyplot as plt
        plt.plot(E)
        plt.xlabel("Edges")
        if log:
            plt.yscale('log')
        plt.ylabel("Label co-occurance")
        plt.title("Label co-occurance counts")
        plt.savefig(plot_name)
        if clear:
            plt.cla()


def get_label_dict(label_filepath):
    """

    :param label_filepath:
    :return:
    """
    if label_filepath is None:
        return OrderedDict()

    try:
        with open(label_filepath, 'r') as file:
            content = file.read().splitlines()
    except:
        with open(label_filepath, 'r', encoding='latin-1') as file:  # 'latin-1' encoding for old files.
            content = file.read().splitlines()

    label_dict = OrderedDict()
    for i, label in enumerate(content):
        label_dict[i] = str(label)

    return label_dict


def get_subgraph(V, E, label_filepath, dataset_name, level=1, subgraph_count=5, ignore_deg=None, root_node=None):
    """
    # total_points: total number of samples.
    # feature_dm: number of features per datapoint.
    # number_of_labels: total number of categories.
    # X: feature matrix of dimension total_points * feature_dm.
    # classes: list of size total_points. Each element of the list containing categories corresponding to one datapoint.
    # V: list of all categories (nodes).
    # E: dict of edge tuple(node_1,node_2) -> weight, eg. {(1, 4): 1, (2, 7): 3}.
    """

    # get a dict of label -> textual_label
    label_dict = get_label_dict(label_filepath)

    # an utility function to relabel nodes of upcoming graph with textual label names
    def mapping(v):
        """
        An utility function to relabel nodes of upcoming graph with textual label names
        :param v: label id (int)
        :return: returns the texual label of the node id [v]
        """
        v = int(v)
        if v in label_dict:
            return label_dict[v]
        return str(v)

    # build a unweighted graph of all edges
    g = nx.Graph()
    g.add_edges_from(E.keys())

    # Below section will try to build a smaller subgraph from the actual graph for visualization
    subgraph_lists = []
    for sg in range(subgraph_count):
        if root_node is None:
            # select a random vertex to be the root
            np.random.shuffle(V)
            v = V[0]
        else:
            v = root_node

        # two files to write the graph and label information
        # Remove characters like \, /, <, >, :, *, |, ", ? from file names,
        # windows can not have file name with these characters
        label_info_filepath = 'samples/' + str(dataset_name) + '_Info[{}].txt'.format(
            str(int(v)) + '-' + util.remove_special_chars(mapping(v)))
        label_graph_filepath = 'samples/' + str(dataset_name) + '_G[{}].graphml'.format(
            str(int(v)) + '-' + util.remove_special_chars(mapping(v)))
        # label_graph_el = 'samples/'+str(dataset_name)+'_E[{}].el'.format(str(int(v)) + '-' + mapping(v)).replace(' ','_')

        logger.debug('Label:[' + mapping(v) + ']')
        label_info_file = open(label_info_filepath, 'w')
        label_info_file.write('Label:[' + mapping(v) + ']' + "\n")

        # build the subgraph using bfs
        bfs_q = Queue()
        bfs_q.put(v)
        bfs_q.put(0)
        node_check = OrderedDict()
        ignored = []

        sub_g = nx.Graph()
        lvl = 0
        while not bfs_q.empty() and lvl <= level:
            v = bfs_q.get()
            if v == 0:
                lvl += 1
                bfs_q.put(0)
                continue
            elif node_check.get(v, True):
                node_check[v] = False
                edges = list(g.edges(v))
                # label_info_file.write('\nNumber of edges: ' + str(len(edges)) + ' for node: ' + mapping(v) + '[' +
                # str(v) + ']' + '\n')
                if ignore_deg is not None and len(edges) > ignore_deg:
                    # label_info_file.write('Ignoring: [' + mapping(v) + '] \t\t\t degree: [' + str(len(edges)) + ']\n')
                    ignored.append("Ignoring: deg [" + mapping(v) + "] = [" + str(len(edges)) + "]\n")
                    continue
                for uv_tuple in edges:
                    edge = tuple(sorted(uv_tuple))
                    sub_g.add_edge(edge[0], edge[1], weight=E[edge])
                    bfs_q.put(uv_tuple[1])
            else:
                continue

        # relabel the nodes to reflect textual label
        nx.relabel_nodes(sub_g, mapping, copy=False)
        logger.debug('sub_g: [{0}]'.format(sub_g))

        label_info_file.write(str('\n'))
        # Writing some statistics about the subgraph
        label_info_file.write(str(nx.info(sub_g)) + '\n')
        label_info_file.write('density: ' + str(nx.density(sub_g)) + '\n')
        label_info_file.write('list of the frequency of each degree value [degree_histogram]: ' +
                              str(nx.degree_histogram(sub_g)) + '\n')
        for nodes in ignored:
            label_info_file.write(str(nodes) + '\n')
        # subg_edgelist = nx.generate_edgelist(sub_g,label_graph_el)
        label_info_file.close()
        nx.write_graphml(sub_g, label_graph_filepath)

        subgraph_lists.append(sub_g)

        logger.info('Graph generated at: [{0}]'.format(label_graph_filepath))

        if root_node:
            logger.info("Root node provided, will generate only one graph file.")
            break

    return subgraph_lists


def split_data(X, classes, V, split=0.1, label_preserve=False, save_path=util.get_dataset_path(), seed=0):
    """
    Splits the data into 2 parts.

    :param X:
    :param classes:
    :param V:
    :param split:
    :param label_preserve: if True; splits the data keeping the categories common.
    :param save_path:
    :param seed:
    :return:
    """
    assert (X.shape[0] == len(classes))
    # Y_tr_aux = list(classes)
    # Y_val = random.sample(Y_tr_aux,val_portion)

    if not label_preserve:
        from sklearn.model_selection import train_test_split
        X_tr, X_val, Y_tr, Y_val = train_test_split(X, classes, test_size=split, random_state=seed)
        return X_tr, Y_tr, X_val, Y_val

    lbl_feature_count = OrderedDict().fromkeys(V)

    for lbl in V:
        for y_list in classes:
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
    Y_tr = classes.copy()
    for lbl, count in lbl_feature_count_portion.items():
        for c in range(count):
            for i, y_list in enumerate(classes):
                if lbl in y_list:
                    X_val.append(X[i])
                    X_tr = np.delete(X, i)
                    Y_val.append(Y_tr.pop(i))
                    break
    util.save_npz(X_tr, "X_tr", file_path=save_path, overwrite=False)
    util.save_pickle(Y_tr, pkl_file_name="Y_tr", file_path=save_path)
    util.save_npz(X_val, "X_val", file_path=save_path, overwrite=False)
    util.save_pickle(Y_val, pkl_file_name="Y_val", file_path=save_path)
    return X_tr, Y_tr, X_val, Y_val


def _test_split_val():
    X = np.asarray(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    classes = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 1], [2, 1]]
    V = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    X_tr, Y_tr, X_val, Y_val = split_data(X, classes, V)
    logger.debug(X_tr)
    logger.debug(Y_tr)
    logger.debug(X_val)
    logger.debug(Y_val)


def main(dataset_path):
    """

    :param args:
    :return:
    """
    # config = read_config(args)
    cls = Neighborhood(dataset_name="Wiki10-31K")
    data_dict = cls.test_cosine()
    # G, stats = cls.load_neighborhood_graph()
    # stats = cls.plot_occurance(list(stats["degree_sequence"]))
    logger.info("Neighborhood graph statistics: [{0}]".format(data_dict))

    exit(0)

    datasets = ['RCV1-2K', 'EURLex-4K', 'AmazonCat-13K', 'AmazonCat-14K', 'Wiki10-31K', 'Delicious-200K',
                'WikiLSHTC-325K', 'Wikipedia-500K', 'Amazon-670K', 'Amazon-3M']
    arff_datasets = ['Corel-374', 'Bibtex_arff', 'Delicious_arff', 'Mediamill_arff', 'Medical', 'Reuters-100_arff']
    datasets = ['RCV1-2K']
    for dataset in datasets:
        train_graph_file = dataset + '_train.txt'
        # train_graph_file = dataset+'/'+dataset+'_train.txt'
        train_graph_file = join(dataset_path, dataset, train_graph_file)

        # label_map = dataset+'_mappings/'+dataset+'_label_map.txt'
        # label_map_file = join(args.dataset_path,dataset,label_map)

        total_points, feature_dm, number_of_labels, X, classes, V, E = get_cooccurance_dict(train_graph_file)

        util.save_json(V, dataset + '_V_train', join(dataset_path, dataset))
        util.save_json(E, dataset + '_E_train', join(dataset_path, dataset), overwrite=True)

        # Collecting some stats about the dataset and graph.
        e_stats, edge_occurances_sorted = edge_stats(E)
        e_stats['singles_train'] = find_single_labels(classes)
        util.save_json(e_stats, dataset + "_edge_statistics_train")

        plot_occurance(edge_occurances_sorted, plot_name=dataset + '_train_edge_occurances_sorted.jpg', clear=False)
        plot_occurance(edge_occurances_sorted, plot_name=dataset + '_train_edge_occurances_sorted_log.jpg', log=True)

        test_graph_file = dataset + '_test.txt'
        test_graph_file = join(dataset_path, dataset, test_graph_file)

        # label_map = dataset+'_mappings/'+dataset+'_label_map.txt'
        # label_map_file = join(args.dataset_path,dataset,label_map)

        total_points, feature_dm, number_of_labels, X, classes, V, E = get_cooccurance_dict(test_graph_file)

        util.save_json(V, dataset + '_V_test', join(dataset_path, dataset))
        util.save_json(E, dataset + '_E_test', join(dataset_path, dataset), overwrite=True)

        # Collecting some stats about the dataset and graph.
        e_stats, edge_occurances_sorted = edge_stats(E)
        e_stats['singles_test'] = find_single_labels(classes)
        util.save_json(e_stats, dataset + "_edge_statistics_test")

        plot_occurance(edge_occurances_sorted, plot_name=dataset + '_test_edge_occurances_sorted.jpg', clear=False)
        plot_occurance(edge_occurances_sorted, plot_name=dataset + '_test_edge_occurances_sorted_log.jpg', log=True)

    # label_graph_lists = get_subgraph(V,E,label_map_file,dataset_name=dataset,level=args.level,subgraph_count=args.
    # subgraph_count,ignore_deg=args.ignore_deg,root_node=args.node_id)
    return


if __name__ == '__main__':
    main()
