#!/usr/bin/env python

import numpy as np
from stanford_bpe import *
def getSentenceFeature(tokens, word_vectors, sentence):
    """
    Obtain the sentence feature for sentiment analysis by averaging its word vectors

    Arguments:
    tokens -- a dictionary that maps words to their indices in the word vector list
    word_vectors -- word vectors (each row) for all tokens
    sentence -- a list of words in the sentence of interest

    Returns:
    sentence_vector -- feature vector for the sentence
    """
    sentence_vector = np.zeros((word_vectors.shape[1],))
    #print(sentence)
    #tokenizer=StanfordSentiment().load_tokenizer()
    #sentence_words=tokenizer.encode(' '.join([x.decode('utf-8') for x in sentence]).replace('-lrb-', '(').replace('-rrb-', ')').replace('ã©','é')).tokens
    #print(sentence_words)
    #print('completed')
    for word in sentence:
        vector = word_vectors[tokens[word],:]
        sentence_vector += vector

    sentence_vector /= len(sentence)
    #print(sentence_vector)
    #print('completed')
    return sentence_vector


def accuracy(y, yhat):
    """ Precision for classifier """
    print(y.shape)
    print(yhat.shape)
    assert(y.shape == yhat.T.shape)
    return np.sum(y == yhat) * 100.0 / y.size
