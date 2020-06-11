# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os,io
from logging import getLogger
import numpy as np
import torch
from multiprocessing import Pool
from statistics import *

from ..utils import get_nn_avg_dist


DIC_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dictionary', 'crosslingual', 'dictionaries')


logger = getLogger()


def load_identical_char_dico(word2id1, word2id2):
    """
    Build a dictionary of identical character strings.
    """
    pairs = [(w1, w1) for w1 in word2id1.keys() if w1 in word2id2]
    if len(pairs) == 0:
        raise Exception("No identical character strings were found. "
                        "Please specify a dictionary.")

    logger.info("Found %i pairs of identical character strings." % len(pairs))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def load_dictionary(path, word2id1, word2id2):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
    assert os.path.isfile(path)

    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0

    with io.open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            assert line == line.lower()
            parts = line.rstrip().split()
            if len(parts) < 2:
                logger.warning("Could not parse line %s (%i)", line, index)
                continue
            word1, word2 = parts
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.info("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico


def get_word_translation_accuracy(lang1, word2id1, emb1, lang2, word2id2, emb2, method, dico_eval):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """
    if dico_eval == 'default':
        path = os.path.join(DIC_EVAL_PATH, '%s-%s.5000-6500.txt' % (lang1, lang2))
    else:
        path = dico_eval
    dico = load_dictionary(path, word2id1, word2id2)
    dico = dico.cuda() if emb1.is_cuda else dico

    assert dico[:, 0].max() < emb1.size(0)
    assert dico[:, 1].max() < emb2.size(0)

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))

    # inverted softmax
    elif method.startswith('invsm_beta_'):
        beta = float(method[len('invsm_beta_'):])
        bs = 128
        word_scores = []
        for i in range(0, emb2.size(0), bs):
            scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
            word_scores.append(scores.index_select(0, dico[:, 0]))
        scores = torch.cat(word_scores, 1)

    # contextual dissimilarity measure
    elif method.startswith('csls_knn_'):
        # average distances to k nearest neighbors
        knn = method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[dico[:, 0]][:, None])
        scores.sub_(average_dist2[None, :])

    else:
        raise Exception('Unknown method: "%s"' % method)

    results = []
    top_matches = scores.topk(10, 1, True)[1]
    for k in [1, 5, 10]:
        top_k_matches = top_matches[:, :k]
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1).cpu().numpy()
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        logger.info("%i source words - %s - Precision at k = %i: %f" %
                    (len(matching), method, k, precision_at_k))
        results.append(('precision_at_%i' % k, precision_at_k))

    return results

def eval_word_trans(p,i,j,word_per_cpu,dataset,options):
    rightnum = 0
    start = p*word_per_cpu
    end = (p+1)*word_per_cpu
    if end > dataset.max_vocab:
        end = dataset.max_vocab
    # print(p,start,end-1)
    for v in range(start,end):
        if v%1000==0:
            print("v:{}".format(v))
        dists = np.zeros(dataset.max_vocab)
        for u in range(dataset.max_vocab):
            dists[u] = 1-np.dot(dataset.learned_emb[i][v] / np.linalg.norm(dataset.learned_emb[i][v]), dataset.learned_emb[j][u] / np.linalg.norm(dataset.learned_emb[j][u]))
        if np.argmin(dists)==v:
            rightnum += 1
    return rightnum

def get_word_translation_accuracy_for_random(dataset,options):
    logger = getLogger('{}Log'.format(dataset.dataname))
    langnum = dataset.langnum
    cpu_count = options.cpu_count
    word_per_cpu = dataset.max_vocab//(cpu_count-1)+1
    acc = np.zeros((langnum,langnum))
    accs = []
    for i in range(langnum):
        for j in range(langnum):
            print(i,j)
            if i==j:
                acc[i][j]=1
                continue
            multi_args=[(p,i,j,word_per_cpu,dataset,options) for p in range(cpu_count-1)]
            pool = Pool(cpu_count-1)
            rightnums = pool.starmap(eval_word_trans,multi_args)
            rightnum = 0
            for rs in rightnums:
                rightnum += rs
            acc[i][j] = rightnum/dataset.max_vocab
            accs.append(acc[i][j])
            print('acc',acc[i][j])
    logger.info(acc)
    avg = mean(accs)
    logger.info(avg)
    return avg
