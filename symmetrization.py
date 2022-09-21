#!/usr/bin/env python
import re
import sys
import optparse
from typing import Tuple

# option parser
optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/test", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int",
                     help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--iterations", dest="iterations", default=5, type="int",
                     help="Number of iterations to train the IBM Model 1 + EM Algo on (default=1000)")
(opts, _) = optparser.parse_args()

# filepath to the data
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)

START_INDEX = 1


def uniform_trans_prob_initialization(parallel_corpus: list, is_reverse: bool = False) -> Tuple[list, dict, set, set]:
    """
    Function to initialize the translation probabilities by uniform distribution.
    :param: parallel_corpus: The sentences in source and foreign language.
    :param is_reverse: Whether to calculate p(e|f) or p(f|e).
    :return: initial translation probabilities, f_vocab, e_vocab
    """
    parallel_corpus = parallel_corpus[: opts.num_sents]
    e_vocab = set()
    f_vocab = set()
    trans_probs = dict()

    # first create the source and transition vocabs
    for n, (f_sent, e_sent) in enumerate(parallel_corpus):
        # n - sentence number in the parallel corpus
        # f_sent - list containing words from the nth sentence in foreign language
        # e_sent - list containing words from the nth sentence in english language
        for f_word in f_sent:
            f_vocab.add(f_word)
        for e_word in e_sent:
            e_vocab.add(e_word)

    source_vocab_size = len(f_vocab)
    target_vocab_size = len(e_vocab)

    if is_reverse:
        for f_word in f_vocab:
            trans_probs[f_word] = dict()
            for e_word in e_vocab:
                trans_probs[f_word][e_word] = 1 / target_vocab_size

        return parallel_corpus, trans_probs, f_vocab, e_vocab

    # calculating init probs
    for e_word in e_vocab:
        trans_probs[e_word] = dict()
        for f_word in f_vocab:
            trans_probs[e_word][f_word] = 1 / source_vocab_size

    return parallel_corpus, trans_probs, f_vocab, e_vocab


def converge_and_optimize(parallel_corpus: list, f_vocab: set, e_vocab: set, trans_probs: dict,
                          n_iterations: int, is_reverse: bool = False) -> dict:
    """
    Function to converge and optimize the learning for the translation probabilities t(e|f)
    :param parallel_corpus: Sentences in source and target language
    :param f_vocab: The vocab set for source language
    :param e_vocab: The vocab set for target language
    :param trans_probs: The initial translation probabilities for the word pairs
    :param n_iterations: Number of iterations to optimize the cost function
    :param is_reverse: Whether to calculate p(e|f) or p(f|e)
    :return: s_total: The total sum of the cost function
    """
    e_vocab = list(e_vocab)
    f_vocab = list(f_vocab)

    if is_reverse:
        for itr_n in range(n_iterations):
            counts = dict()
            total_f = dict()

            # initializing count(e|f) = 0 for all e, f in vocab
            for f_word in f_vocab:
                counts[f_word] = dict()
                for e_word in e_vocab:
                    counts[f_word][e_word] = 0

            # initializing total(f) for all f in vocab
            for e_word in e_vocab:
                total_f[e_word] = 0

            # normalize the probabilities
            for n, (f_sent, e_sent) in enumerate(parallel_corpus):
                s_total = dict()
                for f_word in f_sent:
                    s_total[f_word] = 0
                    for e_word in e_sent:
                        s_total[f_word] += trans_probs[f_word][e_word]

                # collect counts
                for f_word in f_sent:
                    for e_word in e_sent:
                        counts[f_word][e_word] += trans_probs[f_word][e_word] / s_total[f_word]
                        total_f[e_word] += trans_probs[f_word][e_word] / s_total[f_word]

            # estimate probabilities
            for e_word in e_vocab:
                for f_word in f_vocab:
                    trans_probs[f_word][e_word] = counts[f_word][e_word] / total_f[e_word]

        return trans_probs

    for itr_n in range(n_iterations):
        counts = dict()
        total_f = dict()

        # initializing count(e|f) = 0 for all e, f in vocab
        for e_word in e_vocab:
            counts[e_word] = dict()
            for f_word in f_vocab:
                counts[e_word][f_word] = 0

        # initializing total(f) for all f in vocab
        for f_word in f_vocab:
            total_f[f_word] = 0

        # normalize the probabilities
        for n, (f_sent, e_sent) in enumerate(parallel_corpus):
            s_total = dict()
            for e_word in e_sent:
                s_total[e_word] = 0
                for f_word in f_sent:
                    s_total[e_word] += trans_probs[e_word][f_word]

            # collect counts
            for e_word in e_sent:
                for f_word in f_sent:
                    counts[e_word][f_word] += trans_probs[e_word][f_word] / s_total[e_word]
                    total_f[f_word] += trans_probs[e_word][f_word] / s_total[e_word]

        # estimate probabilities
        for f_word in f_vocab:
            for e_word in e_vocab:
                trans_probs[e_word][f_word] = counts[e_word][f_word] / total_f[f_word]

    return trans_probs


def generate_f2e():
    """
    Function to generate f2e from the data
    :return: f2e matrix
    """
    n_iterations = opts.iterations
    full_parallel_corpus = [[sentence.lower().strip().split() for sentence in pair] for pair in
                            zip(open(f_data), open(e_data))][:opts.num_sents]
    parallel_corpus, trans_probs, f_vocab, e_vocab = uniform_trans_prob_initialization(full_parallel_corpus)
    trans_probs = converge_and_optimize(parallel_corpus, f_vocab, e_vocab, trans_probs, n_iterations)

    # init for f2e
    f2e = {}
    for f_i, f_word in enumerate(f_vocab):
        f2e[f_i] = {}
        # f2e[f_word] = {}
        for e_i, e_word in enumerate(e_vocab):
            f2e[f_i][e_i] = 0
            # f2e[f_word][e_word] = 0

    # creating f2e
    for n, (f_sent, e_sent) in enumerate(full_parallel_corpus):
        for f_i, f_word in enumerate(f_sent):
            # finding the word with the max probability to the given source language word
            max_prob_word, max_prob, max_prob_idx, ct_max = "", 0, 0, 0
            for e_i, e_word in enumerate(e_sent):
                if trans_probs[e_word][f_word] > max_prob:
                    max_prob = trans_probs[e_word][f_word]
                    max_prob_word = e_word
                    max_prob_idx = e_i
                elif trans_probs[e_word][f_word] == max_prob:
                    ct_max += 1
            # f2e[f_word][max_prob_word] = 1
            f2e[f_i][max_prob_idx] = 1
    return f2e


def generate_e2f():
    """
    Function to generate e2f from the data
    :return: e2f matrix
    """
    n_iterations = opts.iterations
    full_parallel_corpus = [[sentence.lower().strip().split() for sentence in pair] for pair in
                            zip(open(f_data), open(e_data))][:opts.num_sents]
    parallel_corpus, trans_probs, f_vocab, e_vocab = uniform_trans_prob_initialization(full_parallel_corpus,
                                                                                       is_reverse=True)
    trans_probs = converge_and_optimize(parallel_corpus, f_vocab, e_vocab, trans_probs, n_iterations, is_reverse=True)

    # init for e2f
    e2f = {}
    for e_i, e_word in enumerate(e_vocab):
        e2f[e_i] = {}
        # e2f[e_word] = {}
        for f_i, f_word in enumerate(f_vocab):
            # e2f[e_word][f_word] = 0
            e2f[e_i][f_i] = 0

    # creating e2f
    for n, (f_sent, e_sent) in enumerate(full_parallel_corpus):
        for e_i, e_word in enumerate(e_sent):
            # finding the word with the max probability to the given source language word
            max_prob_word, max_prob, max_prob_idx, ct_max = "", 0, 0, 0
            for f_i, f_word in enumerate(f_sent):
                if trans_probs[f_word][e_word] > max_prob:
                    max_prob = trans_probs[f_word][e_word]
                    max_prob_word = f_word
                    max_prob_idx = f_i
                elif trans_probs[f_word][e_word] == max_prob:
                    ct_max += 1
            # e2f[e_word][max_prob_word] = 1
            e2f[e_i][max_prob_idx] = 1
    return e2f


def initialize_matrix(rows, columns, value):
    """
    A function that fills a matrix of given dimensions with a value
    """

    result = [[value for _ in range(columns)] for _ in range(rows)]
    return result


def intersection(e2f, f2e):
    """
    Function to find intersection between e2f and f2e
    :param e2f: e2f matrix
    :param f2e: f2e matrix
    :return: intersection of e2f and f2e
    """
    rows = len(e2f)
    columns = len(f2e)
    result = initialize_matrix(rows, columns, False)
    for e in range(rows):
        for f in range(columns):
            result[e][f] = e2f[e][f] and f2e[f][e]

    return result


def union(e2f, f2e):
    """
    Function to find union between e2f and f2e
    :param e2f: e2f matrix
    :param f2e: f2e matrix
    :return: union of e2f and f2e
    """
    rows = len(e2f)
    columns = len(f2e)
    result = initialize_matrix(rows, columns, False)
    for e in range(rows):
        for f in range(columns):
            result[e][f] = e2f[e][f] or f2e[f][e]

    return result


def neighboring_points(e_index, f_index, e_len, f_len):
    """
    A function that returns list of neighboring points in an alignment matrix for a given alignment (pair of indexes).
    """
    result = []

    if e_index > 0:
        result.append((e_index - 1, f_index))
    if f_index > 0:
        result.append((e_index, f_index - 1))
    if e_index < e_len - 1:
        result.append((e_index + 1, f_index))
    if f_index < f_len - 1:
        result.append((e_index, f_index + 1))
    if e_index > 0 and f_index > 0:
        result.append((e_index - 1, f_index - 1))
    if e_index > 0 and f_index < f_len - 1:
        result.append((e_index - 1, f_index + 1))
    if e_index < e_len - 1 and f_index > 0:
        result.append((e_index + 1, f_index - 1))
    if e_index < e_len - 1 and f_index < f_len - 1:
        result.append((e_index + 1, f_index + 1))

    return result


def aligned_e(e, f_len, alignment):
    """
    A function that checks if a given 'english' word is aligned to any foreign word in a given foreign sentence
    """
    for f in range(START_INDEX, f_len):
        if alignment[e][f]:
            return True
    return False


def aligned_f(f, e_len, alignment):
    """
    A function that checks if a given foreign word is aligned to any 'english' word in a given 'english' sentence
    """
    for e in range(START_INDEX, e_len):
        if alignment[e][f]:
            return True
    return False


def grow_diag(union, alignment, e_len, f_len):
    """
    Performing symmetrization of thw word alignments using union and intersection.
    :param union: union of e2f and f2e
    :param alignment: alignment
    :param e_len: len of english
    :param f_len: len of foreign
    :return:
    """
    new_points_added = True
    while new_points_added:
        new_points_added = False
        for e in range(START_INDEX, e_len):
            for f in range(START_INDEX, f_len):
                if alignment[e][f]:
                    for (e_new, f_new) in neighboring_points(e, f, e_len, f_len):
                        if not (aligned_e(e_new, f_len, alignment) and aligned_f(f_new, e_len, alignment)) \
                                and union[e_new][f_new]:
                            alignment[e_new][f_new] = True
                            new_points_added = True


def final(alignment, e2f, f2e, e_len, f_len):
    """
    A function that implements both FINAL(e2f) and FINAL(f2e) steps of GROW-DIAG-FINAL algorithm
    """
    for e in range(START_INDEX, e_len):
        for f in range(START_INDEX, f_len):
            if not (aligned_e(e, f_len, alignment) and aligned_f(f, e_len, alignment)) \
                    and (e2f[e][f] or f2e[f][e]):
                alignment[e][f] = True


def final_e2f(alignment, e2f, e_len, f_len):
    """
    A function that implements FINAL(e2f) step of GROW-DIAG-FINAL algorithm
    """
    for e in range(START_INDEX, e_len):
        for f in range(START_INDEX, f_len):
            if not (aligned_e(e, f_len, alignment) and aligned_f(f, e_len, alignment)) \
                    and e2f[e][f]:
                alignment[e][f] = True


def final_f2e(alignment, f2e, e_len, f_len):
    """
    A function that implements FINAL(f2e) step of GROW-DIAG-FINAL algorithm
    """
    for e in range(START_INDEX, e_len):
        for f in range(START_INDEX, f_len):
            if not (aligned_e(e, f_len, alignment) and aligned_f(f, e_len, alignment)) \
                    and f2e[f][e]:
                alignment[e][f] = True


def grow_diag_final(e2f, f2e, e_len, f_len):
    alignment = intersection(e2f, f2e)
    grow_diag(union(e2f, f2e), alignment, e_len, f_len)
    final(alignment, e2f, f2e, e_len, f_len)
    return alignment


def parse_alignments(alignments_line, values):
    word_alignments_regex = r"(\S+)\s\(\{([\s\d]*)\}\)"
    alignments = re.findall(word_alignments_regex, alignments_line)

    # Initialize matrix with False value for each pair of words
    rows = len(alignments)
    columns = len(values)
    result = initialize_matrix(rows, columns, False)

    # Align words
    for i in range(len(alignments)):
        alignment_values = alignments[i][1].split()
        for alignment in alignment_values:
            result[i][int(alignment)] = True

    return result


def form_alignments(alignments, e_len, f_len):
    result = ''
    for f in range(1, f_len):
        for e in range(1, e_len):
            if alignments[e][f]:
                sys.stdout.write("%i-%i " % (f, e))
        sys.stdout.write("\n")


if __name__ == '__main__':
    f2e = generate_f2e()
    e2f = generate_e2f()

    e_sentence = []
    f_sentence = []

    for e2f_line, f2e_line in zip(e2f, f2e):
        alignments = grow_diag_final(parse_alignments(e2f_line, f_sentence), parse_alignments(f2e_line, e_sentence),
                                     len(e_sentence), len(f_sentence))
        form_alignments(alignments, len(e_sentence), len(f_sentence))
