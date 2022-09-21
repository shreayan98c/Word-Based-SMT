#!/usr/bin/env python
import sys
import optparse

# option parser
optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
# optparser.add_option("-d", "--data", dest="train", default="data/test", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-i", "--iterations", dest="iterations", default=5, type="int",
                     help="Number of iterations to train the IBM Model 1 + EM Algo on (default=1000)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int",
                     help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

# filepath to the data
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)


# implementing IBM Model 1 and EM algorithm
# Input: set of sentence pairs (e, f)
# Output: translation prob. t(e|f)


def uniform_trans_prob_initialization(parallel_corpus: list) -> tuple[list, dict, set, set]:
    """
    Function to initialize the translation probabilities by uniform distribution.
    :param: parallel_corpus: The sentences in source and foreign language.
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

    # calculating init probs
    for e_word in e_vocab:
        trans_probs[e_word] = dict()
        for f_word in f_vocab:
            trans_probs[e_word][f_word] = 1 / source_vocab_size

    return parallel_corpus, trans_probs, f_vocab, e_vocab


def converge_and_optimize(parallel_corpus: list, f_vocab: set, e_vocab: set, trans_probs: dict,
                          n_iterations: int) -> dict:
    """
    Function to converge and optimize the learning for the translation probabilities t(e|f)
    :param parallel_corpus: Sentences in source and target language
    :param f_vocab: The vocab set for source language
    :param e_vocab: The vocab set for target language
    :param trans_probs: The initial translation probabilities for the word pairs
    :param n_iterations: Number of iterations to optimize the cost function
    :return: s_total: The total sum of the cost function
    """
    e_vocab = list(e_vocab)
    f_vocab = list(f_vocab)

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


if __name__ == '__main__':

    n_iterations = opts.iterations
    full_parallel_corpus = [[sentence.lower().strip().split() for sentence in pair] for pair in
                            zip(open(f_data), open(e_data))]#[:opts.num_sents]
    parallel_corpus, trans_probs, f_vocab, e_vocab = uniform_trans_prob_initialization(full_parallel_corpus)
    trans_probs = converge_and_optimize(parallel_corpus, f_vocab, e_vocab, trans_probs, n_iterations)

    # predicting on given data
    for n, (f_sent, e_sent) in enumerate(full_parallel_corpus):
        for f_i, f_word in enumerate(f_sent):
            # finding the word with the max probability to the given source language word
            max_prob_word, max_prob, max_prob_idx, ct_max = "", 0, 0, 0
            for e_i, e_word in enumerate(e_sent):
                if e_word in trans_probs and f_word in trans_probs[e_word]:
                    if trans_probs[e_word][f_word] > max_prob:
                        max_prob_idx = f_i
                        max_prob = trans_probs[e_word][f_word]
                        max_prob_word = e_word
                    elif trans_probs[e_word][f_word] == max_prob:
                        ct_max += 1
                else:
                    # out of vocab for test
                    max_prob_idx = f_i
            sys.stdout.write("%i-%i " % (f_i, max_prob_idx))
        sys.stdout.write("\n")
