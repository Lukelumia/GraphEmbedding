import itertools
import math
import random

import os
import jsonlines
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .alias import alias_sample, create_alias_table
from .utils import partition_num
from tqdm.auto import tqdm

import multiprocessing

class RandomWalker:
    def __init__(self, G, p=1, q=1, use_rejection_sampling=0, dump_path=None, dump_size=100000):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        :param use_rejection_sampling: Whether to use the rejection sampling strategy in node2vec.
        """
        self.G = G
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling
        self.dump_path = dump_path
        self.dump_size = dump_size

        self.nodes = list(G.nodes())
        self.neighbors = {node: list(G.neighbors(node)) for node in G.nodes()}

    def deepwalk_walk(self, walk_length, start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = self.neighbors[cur]
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def node2vec_walk(self, walk_length, start_node):

        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = self.neighbors[cur]
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],
                                                      alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def node2vec_walk2(self, walk_length, start_node):
        """
        Reference:
        KnightKing: A Fast Distributed Graph Random Walk Engine
        http://madsys.cs.tsinghua.edu.cn/publications/SOSP19-yang.pdf
        """

        def rejection_sample(inv_p, inv_q, nbrs_num):
            upper_bound = max(1.0, max(inv_p, inv_q))
            lower_bound = min(1.0, min(inv_p, inv_q))
            shatter = 0
            second_upper_bound = max(1.0, inv_q)
            if (inv_p > second_upper_bound):
                shatter = second_upper_bound / nbrs_num
                upper_bound = second_upper_bound + shatter
            return upper_bound, lower_bound, shatter

        G = self.G
        alias_nodes = self.alias_nodes
        inv_p = 1.0 / self.p
        inv_q = 1.0 / self.q
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = self.neighbors[cur]
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    upper_bound, lower_bound, shatter = rejection_sample(
                        inv_p, inv_q, len(cur_nbrs))
                    prev = walk[-2]
                    prev_nbrs = set(self.neighbors[prev])
                    while True:
                        prob = random.random() * upper_bound
                        if (prob + shatter >= upper_bound):
                            next_node = prev
                            break
                        next_node = cur_nbrs[alias_sample(
                            alias_nodes[cur][0], alias_nodes[cur][1])]
                        if (prob < lower_bound):
                            break
                        if (prob < inv_p and next_node == prev):
                            break
                        _prob = 1.0 if next_node in prev_nbrs else inv_q
                        if (prob < _prob):
                            break
                    walk.append(next_node)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]


        G = self.G

        # self.nodes = list(G.nodes())
        chunk_size = math.ceil(len(self.nodes) / workers)
        random.shuffle(self.nodes)
        self.node_partitions = {num: chunk for num, chunk in \
                                enumerate(chunks(self.nodes, chunk_size))}

        # return self.simulate_walks2(self, num_walks, walk_length, workers, verbose)
        # old
        # results = Parallel(n_jobs=workers, verbose=verbose, )(
        #     delayed(self._simulate_walks)(num, walk_length) for num in
        #     partition_num(num_walks, workers))

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(num_walks, walk_length, partition) for \
            partition in self.node_partitions.keys())

        walks = list(itertools.chain(*results))

        return walks

    def dump_walks(self, results):
        dump_path = os.path.join(self.dump_path, f'walks_dump.jsonl')
        with jsonlines.open(dump_path, 'a') as writer:
            writer.write_all(results)

    def dump_jsonlines(self, d, path):
        # new_d = {str(key): value for key, value in d.items()}
        # with jsonlines.open(path, 'a') as writer:
        #     writer.write(object)
        pd.DataFrame.from_dict(d.items()).to_csv(path, mode='a', header=False, index=False)

    def load_dumped_alias(self):
        # load nodes
        df = pd.read_csv('data/cache/alias_nodes', converters={0: ast.literal_eval, 1: ast.literal_eval},
                         header=None)
        self.alias_nodes = pd.Series(df[1].values, index=df[0]).to_dict()
        df = None

        if not self.use_rejection_sampling:
            # load alias
            df = pd.read_csv('data/cache/alias_edges', converters={0: ast.literal_eval, 1: ast.literal_eval},
                             header=None)
            self.alias_edges = pd.Series(df[1].values, index=df[0]).to_dict()


    def simulate_walks2(self, num_walks, walk_length, workers=1, verbose=0, batch_size=100000):
        G = self.G

        self.nodes = list(G.nodes())

        jobs = [(num_walks, walk_length, partition_num) for partition_num in self.node_partitions.keys()]

        with multiprocessing.Pool(processes=workers) as p:
            results = p.starmap(self._simulate_walks, jobs)

        # results = list()
        # for batch in np.array_split(jobs, math.ceil(len(jobs)/batch_size)):
        #     result = p.starmap(self._simulate_walks, batch)
        #     if self.dump_path is None:
        #         results.extend(result)
        #     else:
        #         self.dump_walks(result)
        #
        # if self.dump_path is None:
        #     return None

        walks = list(itertools.chain(*results))

        return walks

    def _simulate_walks(self, num_walks, walk_length, partition):
        walks = []
        for _ in range(num_walks):
            random.shuffle(self.node_partitions[partition])
            for num, v in tqdm(enumerate(self.node_partitions[partition]), total=len(self.node_partitions[partition]), desc=f"Simulating walks ({_}/{num_walks})"):
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(
                        walk_length=walk_length, start_node=v))
                elif self.use_rejection_sampling:
                    walks.append(self.node2vec_walk2(
                        walk_length=walk_length, start_node=v))
                else:
                    walks.append(self.node2vec_walk(
                        walk_length=walk_length, start_node=v))

                if num % self.dump_size == 0:
                    if self.dump_path is not None:
                        self.dump_walks(walks)
                        walks = list()
            if self.dump_path is not None:
                self.dump_walks(walks)
                walks = list()

        return walks

    def get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in self.neighbors[v]:
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)



    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.G
        alias_nodes = dict()
        for num, node in tqdm(enumerate(G.nodes()), total=len(G.nodes()), desc='Preprocess - nodes'):
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in self.neighbors[node]]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)
            # if num % 1000 == 0:
        #         self.dump_jsonlines(alias_nodes, './data/cache/alias_nodes')
        #         alias_nodes = dict()
        # self.dump_jsonlines(alias_nodes, './data/cache/alias_nodes')
        # alias_nodes = dict()

        if not self.use_rejection_sampling:
        
            def do_alieas_edge(edges):
                items = list()
                for edge in tqdm(edges):
                    items.append((edge, self.get_alias_edge(edge[0], edge[1])))
                return items

            results = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=1, )(
                delayed(do_alieas_edge)(edges) for edges in np.array_split(G.edges(), multiprocessing.cpu_count()))

            results = list(itertools.chain(*results))
            alias_edges = {key: value for key, value in results}
        
        
            # alias_edges = dict()
            # for num, edge in tqdm(enumerate(G.edges()), total=len(G.edges()), desc='Preprocess - edges'):
                # alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                
            #     if num % 1000 == 0:
            #         self.dump_jsonlines(alias_edges, './data/cache/alias_edges')
            #         alias_edges = dict()
            # self.dump_jsonlines(alias_edges, './data/cache/alias_edges')
            # self.alias_edges = dict()
            self.alias_edges = alias_edges

        self.alias_nodes = alias_nodes
        return


class BiasedWalker:
    def __init__(self, idx2node, temp_path):

        self.idx2node = idx2node
        self.idx = list(range(len(self.idx2node)))
        self.temp_path = temp_path
        pass

    def simulate_walks(self, num_walks, walk_length, stay_prob=0.3, workers=1, verbose=0):

        layers_adj = pd.read_pickle(self.temp_path+'layers_adj.pkl')
        layers_alias = pd.read_pickle(self.temp_path+'layers_alias.pkl')
        layers_accept = pd.read_pickle(self.temp_path+'layers_accept.pkl')
        gamma = pd.read_pickle(self.temp_path+'gamma.pkl')
        walks = []
        initialLayer = 0

        nodes = self.idx  # list(self.g.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self._exec_random_walk(layers_adj, layers_accept, layers_alias,
                                                    v, walk_length, gamma, stay_prob))
        return walks

    def _exec_random_walk(self, graphs, layers_accept, layers_alias, v, walk_length, gamma, stay_prob=0.3):
        initialLayer = 0
        layer = initialLayer

        path = []
        path.append(self.idx2node[v])

        while len(path) < walk_length:
            r = random.random()
            if(r < stay_prob):  # same layer
                v = chooseNeighbor(v, graphs, layers_alias,
                                   layers_accept, layer)
                path.append(self.idx2node[v])
            else:  # different layer
                r = random.random()
                try:
                    x = math.log(gamma[layer][v] + math.e)
                    p_moveup = (x / (x + 1))
                except:
                    print(layer, v)
                    raise ValueError()

                if(r > p_moveup):
                    if(layer > initialLayer):
                        layer = layer - 1
                else:
                    if((layer + 1) in graphs and v in graphs[layer + 1]):
                        layer = layer + 1

        return path


def chooseNeighbor(v, graphs, layers_alias, layers_accept, layer):

    v_list = graphs[layer][v]

    idx = alias_sample(layers_accept[layer][v], layers_alias[layer][v])
    v = v_list[idx]

    return v
