# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
from ge import Node2Vec
import click
import networkx as nx
import pandas as pd
import os
# from cidatakit.utils.logging import setup_logging
import random
import logging


def save_embeddings(embeddings, embedding_path):
    pd.DataFrame(embeddings).T.to_csv(embedding_path, sep=' ')

def add_file_logger(logger, location):
    """
    Add a file handler to a logger object
    so the loggers output is also writen to a file

    :param logger: A logger object from the default logging module
    :param location: A file path location where the log file should be located
    :return:
    """
    fh = logging.FileHandler(location)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)


@click.command()
@click.argument('graph_edgelist_path', type=click.Path())
@click.argument('run_name')
@click.argument('walk_number', default=10, type=click.INT)
@click.argument('walk_length', default=3, type=click.INT)
@click.argument('dimensions', default=64, type=click.INT)
@click.argument('window_size', default=5, type=click.INT)
@click.argument('epochs', default=3, type=click.INT)
@click.argument('p', default=1, type=click.FLOAT)
@click.argument('q', default=1, type=click.FLOAT)
@click.argument('learning_rate', default=0.05, type=click.FLOAT)
@click.argument('log_to_file', default=True, type=click.BOOL)
@click.argument('directional', type=click.BOOL, default=False)
def main(*args, **kwargs):
    node2vec(*args, **kwargs)


def node2vec(graph_edgelist_path, run_name, walk_number, walk_length, dimensions, window_size, epochs, p, q, learning_rate, log_to_file, directional, **kwargs):
    """

    :param graph_edgelist_path:
    :param run_name:
    :param walk_number:
    :param walk_length:
    :param dimensions:
    :param window_size:
    :param epochs:
    :param learning_rate:
    :param log_to_file:
    :param kwargs: Unused paramters
    :return:
    """

    # Step 1 Prep surroundings
    # step 1.1 - Create folder for run
    run_name = run_name + f'_{p}_{q}_{walk_number}_{walk_length}_{dimensions}_{window_size}_{epochs}_{learning_rate}'
    dir_path = os.path.join('.', 'embeddings', run_name)
    embedding_path = os.path.join(dir_path, 'embedding.emb')
    model_path = os.path.join(dir_path, 'model.mod')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Step 1.2 - Create logger object and remove any file loggers which are already attached
    # logger = setup_logging(__name__)
    # logger = setup_logging('src')
    logger = logging.getLogger(__name__)
    random.seed(42)
    logger.info(f'Run: {run_name}')

    if log_to_file:
        add_file_logger(logger, os.path.join(dir_path, 'log.log'))
        add_file_logger(logging.getLogger('gensim.models'), os.path.join(dir_path, 'log.log'))
        add_file_logger(logging.getLogger('ge'), os.path.join(dir_path, 'log.log'))

    # Step 1.3 - Skip if embedding is already created
    if os.path.exists(os.path.join(dir_path, 'embedding.emb')):
        logger.info('Already processed this configuration, skipping this run')
        return

    logger.info('--- Starting new run ---')

    # small: '/home/luke/graph-based-lookalikes/data/preprocessed/networkx_graph.edgelist'
    # big: '/home/luke/graph-based-lookalikes/data/pipeline/networkx_graph.edgelist'

    # Step 2.1 - Load graph
    logger.info('Loading graph')

    if directional:
        logger.info('Using a directed graph')
        graph_type = nx.DiGraph()
    else:
        logger.info('Using an undirected graph')
        graph_type = nx.Graph()

    G = nx.read_edgelist(graph_edgelist_path,
                         create_using=graph_type, nodetype=None, data=[('weight', int)])
    # step2.2 - init model
    logger.info('Initing model')
    workers = multiprocessing.cpu_count() - 1
    model = Node2Vec(G, walk_length=walk_length, num_walks=walk_number, p=p, q=q, workers=workers, use_rejection_sampling=True)

    logger.info('starting training')
    model.train(window_size=window_size, iter=epochs, alpha=learning_rate, embed_size=dimensions, workers=workers)

    logger.info('Saving model')
    model.w2v_model.save(model_path)

    logger.info('saving embedding')
    embeddings = model.get_embeddings()
    save_embeddings(embeddings, embedding_path)



if __name__ == "__main__":
    main()
