import utils
import constants
import re
import pickle as pkl
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def tokenize(sentence):
    sentence = sentence.lower()
    return re.findall("[\'\w\d\-\*]+|[^a-zA-Z\d\s]+", sentence)


def vectorize(label_to_idx, word_to_idx, ingred_list, label_list, max_len):
    """
    Each word in the recipe is replaced by its corresponding word index
    :param label_to_idx: Dictionary of Labels to corresponding indices
    :param word_to_idx: Dictionary of ingredient words to corresponding incdices
    :param ingred_list: List of Ingredients (each element belongs to one recipe)
    :param label_list: List of labels corresponding to the ingred_list (=None if using test set)
    :param max_len: Max length of recipe
    :return:
    """
    X = []
    Y = []
    keys = set(word_to_idx.keys())
    for i in range(len(ingred_list)):
        if label_list is not None:
            Y.append(label_to_idx[label_list[i]])
        x = []
        for w in ingred_list[i]:
            x.append(word_to_idx[w] if w in keys else 1)
        for j in range(len(x), max_len):
            x.append(0)
        X.append(x)
    return np.asarray(X), np.asarray(Y)



def get_data(data_file, no_labels=False, fix_len=False):
    """
    Load data from file, concatenate multiple lines, tokenize and return.

    :param data_file: Path to data file
    :param no_labels: True if unlabelled data is being read
    :param fix_len: Sentences longer than FIX_LEN are clipped if fix_len=true
    :return: ingred_list, label_list, vocab, labels, max_len, id_list(labels omitted if no_labels is set)
    """
    data = utils.read_data(data_file)
    labels, vocab = set(), set()
    label_list, id_list, ingred_list = list(), list(), list()
    max_len = 0
    for item in data:
        id_list.append(item['id'])
        if not no_labels:
            labels.add(item['cuisine'])
            label_list.append(item['cuisine'])
        l = []
        for ingred in item["ingredients"]:
            for w in tokenize(ingred):
                vocab.add(w)
                l.append(w)
        if not fix_len:
            ingred_list.append(l) #To use full length vectors
            if len(l) > max_len:
                max_len = len(l)
        elif len(l) > constants.FIX_LEN:
            ingred_list.append(l[:constants.FIX_LEN]) #Fixing the length to FIX_LEN
        else:
            ingred_list.append(l)
        max_len = constants.FIX_LEN

    if not no_labels:
        return ingred_list, label_list, vocab, labels, max_len, id_list
    else:
        return ingred_list, None, vocab, None, max_len, id_list
    ####################################
    #There are cleaner ways to return so many values. SKipping that in the interest of time
    ####################################


def main():
    ingred_list, label_list_train, vocab_train, labels, max_len_train, _ = get_data(constants.TRAIN_FILE, fix_len=True)
    ingred_list_dev, label_list_dev, vocab_dev, _, max_len_dev, _ = get_data(constants.DEV_FILE, fix_len=True)
    ingred_list_test, _, _, _, max_len_test, id_list_test = get_data(constants.TEST_FILE, no_labels=True, fix_len=True)


    max_len = max([max_len_test, max_len_test, max_len_dev])

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

    #Print stats
    print "Vocab Length: " + str(len(vocab))
    print "Ingredient max length: " + str(max_len)
    print "Number of Train recipes: " + str(len(label_list_train))

    #Build train feature vectors
    X_train, Y_train = vectorize(label_to_idx, word_to_idx, ingred_list, label_list_train, max_len)
    print "X_train shape: " + str(X_train.shape)
    print "Y_train shape" + str(Y_train.shape)

    #Build dev feature vectors
    X_dev, Y_dev = vectorize(label_to_idx, word_to_idx, ingred_list_dev, label_list_dev, max_len)
    print "X_dev shape: " + str(X_dev.shape)
    print "Y_dev shape" + str(Y_dev.shape)

    #Build test set feature vectors
    X_test, _ = vectorize(label_to_idx, word_to_idx, ingred_list_test, None, max_len)
    print "X_test shape: " + str(X_test.shape)


    utils.write_data(idx_to_label, constants.IDX_TO_LABEL_FILE)
    utils.write_data(word_to_idx, constants.WORD_TO_IDX_FILE)
    utils.write_data(X_train, constants.X_TRAIN_FILE)
    utils.write_data(Y_train, constants.Y_TRAIN_FILE)
    utils.write_data(X_dev, constants.X_DEV_FILE)
    utils.write_data(Y_dev, constants.Y_DEV_FILE)
    utils.write_data(X_test, constants.X_TEST_FILE)
    utils.write_data(id_list_test, constants.TEST_ID_LIST_FILE)

if __name__ == '__main__':
    main()