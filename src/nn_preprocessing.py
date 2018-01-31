import utils
import constants
import re
import pickle as pkl
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# import matplotlib.pyplot as plt

FIX_LEN=40

def tokenize(sentence):
    sentence = sentence.lower()
    return re.findall("[\'\w\d\-\*]+|[^a-zA-Z\d\s]+", sentence)


def vectorize(label_to_idx, word_to_idx, ingred_list, label_list, max_len):
    """
    Each word in the recipe is replaced by its corresponding word index
    :param label_to_idx: Dictionary of Labels to corresponding incdices
    :param word_to_idx: Dictionary of ingredient words to corresponding incdices
    :param ingred_list: List of Ingredients (each element belongs to one recipe)
    :param label_list: List of labels corresponding to the ingred_list
    :param max_len: Max length of recipe
    :return:
    """
    X = []
    Y = []
    keys = set(word_to_idx.keys())
    for i in range(len(label_list)):
        Y.append(label_to_idx[label_list[i]])
        x = []
        for w in ingred_list[i]:
            x.append(word_to_idx[w] if w in keys else 1)
        for j in range(len(x), max_len):
            x.append(0)
        X.append(x)
    return np.asarray(X), np.asarray(Y)



def get_data(data_file, no_labels=False):
    """
    Load data from file, concatenate multiple lines, tokenize and return
    :param data_file: Path to data file
    :param no_labels: True if unlabelled data is being read
    :return: ingred_list, label_list, vocab, labels, max_len (labels omitted if no_labels is set)
    """
    data = utils.read_data(data_file)
    labels = set()
    vocab = set()
    ingred_list = []
    label_list = []
    max_len = 0
    lengths = []
    for item in data:
        if not no_labels:
            labels.add(item['cuisine'])
            label_list.append(item['cuisine'])
        l = []
        for ingred in item["ingredients"]:
            for w in tokenize(ingred):
                vocab.add(w)
                l.append(w)
        ingred_list.append(l)
        if len(l) > max_len:
            max_len = len(l)
        # if len(l) > FIX_LEN:
        #     ingred_list.append(l[:FIX_LEN])
        # else:
        #     ingred_list.append(l)
        # lengths.append(len(l))
        # max_len = FIX_LEN
    print "Vocab Length: " + str(len(vocab))
    print "Ingredient max length: " + str(max_len)
    print "Number of recipes: " + str(len(label_list))
    # print sum(i > 40 for i in lengths)
    if not no_labels:
        return ingred_list, label_list, vocab, labels, max_len
    else:
        return ingred_list, None, vocab, None, max_len



def main():
    ingred_list, label_list, vocab_train, labels, max_len = get_data(constants.TRAIN_FILE)
    ingred_list_dev, label_list_dev, vocab_dev, _, max_len_dev = get_data(constants.DEV_FILE)

    if max_len_dev > max_len:
        max_len = max_len_dev

    vocab = vocab_train.union(vocab_dev)

    label_to_idx = dict()
    idx_to_label = dict()
    for idx, val in enumerate(labels):
        label_to_idx[val] = idx
        idx_to_label[idx] = val

    word_to_idx = dict()
    for idx, val in enumerate(vocab):
        word_to_idx[val] = idx + 2 #Reserve 0 for padding, 1 for unknown
    word_to_idx['__PAD__'] = 0
    word_to_idx['__UNK__'] = 1

    #Build train feature vectors
    X_train, Y_train = vectorize(label_to_idx, word_to_idx, ingred_list, label_list, max_len)
    print "X_train shape: " + str(X_train.shape)
    print "Y_train shape" + str(Y_train.shape)

    #Build dev feature vectors
    X_dev, Y_dev = vectorize(label_to_idx, word_to_idx, ingred_list_dev, label_list_dev, max_len)
    print "X_dev shape: " + str(X_dev.shape)
    print "Y_dev shape" + str(Y_dev.shape)

    utils.write_data(idx_to_label, constants.IDX_TO_LABEL_FILE)
    utils.write_data(word_to_idx, constants.WORD_TO_IDX_FILE)
    utils.write_data(X_train, constants.X_TRAIN_FILE)
    utils.write_data(Y_train, constants.Y_TRAIN_FILE)
    utils.write_data(X_dev, constants.X_DEV_FILE)
    utils.write_data(Y_dev, constants.Y_DEV_FILE)

if __name__ == '__main__':
    main()