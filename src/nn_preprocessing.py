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
    :param label_to_idx: Dictionary of Labels to corresponding incdices
    :param word_to_idx: Dictionary of ingredient words to corresponding incdices
    :param ingred_list: List of Ingredients (each element belongs to one recipe)
    :param label_list: List of labels corresponding to the ingred_list
    :param max_len: Max length of recipe
    :return:
    """
    X = []
    Y = []
    for i in range(len(label_list)):
        Y.append(label_to_idx[label_list[i]])
        x = [word_to_idx[w] for w in ingred_list[i]]
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

    print "Vocab Length: " + str(len(vocab))
    print "Ingredient max length: " + str(max_len)
    print "Number of recipes: " + str(len(label_list))

    if not no_labels:
        return ingred_list, label_list, vocab, labels, max_len
    else:
        return ingred_list, vocab, max_len


def build_dataset(data_file, sent_len=0):
    """
    Loads data, builds vocab, calls vectorize and returns dataset as a numpy array
    :param data_file: Path to data
    :param sent_len: Max length of sentence
    :return: Vectorized data, Vectorized labels, index to label dict, word to index dict
    """
    ingred_list, label_list, vocab, labels, max_len = get_data(data_file)
    if max_len < sent_len:
        max_len = sent_len
    label_to_idx = dict()
    idx_to_label = dict()
    word_to_idx = dict()
    for idx, val in enumerate(labels):
        label_to_idx[val] = idx
        idx_to_label[idx] = val
    for idx, val in enumerate(vocab):
        word_to_idx[val] = idx + 2 #Reserve 0 for padding, 1 for unknown

    word_to_idx['__PAD__'] = 0
    word_to_idx['__UNK__'] = 1
    X, Y = vectorize(label_to_idx, word_to_idx, ingred_list, label_list, max_len)

    return X, Y, idx_to_label, word_to_idx


def main():
    #Extracting feature vector with only word indices
    X_train, Y_train, idx_to_label, word_to_idx = build_dataset(constants.TRAIN_FILE)
    print "X_train shape: " +str(X_train.shape)
    print "Y_train shape" + str(Y_train.shape)

    #Extracting BOW features
    # ingredients, label_list, max_len = get_data_minimal(constants.TRAIN_FILE)
    # train_count_vect = CountVectorizer()
    # X_train_bow = train_count_vect.fit_transform(ingredients)

    #Extracting dev feature vector with only word indices
    X_dev, Y_dev, _, _ = build_dataset(constants.DEV_FILE, X_train.shape[1])
    print "X_dev shape: " +str(X_dev.shape)
    print "Y_dev shape" + str(Y_dev.shape)

    #Extracting BOW feature vector for dev set
    # ingredients, label_list, max_len = get_data_minimal(constants.DEV_FILE)
    # dev_count_vect = CountVectorizer(vocabulary=train_count_vect.vocabulary_)
    # X_dev_bow = dev_count_vect.fit_transform(ingredients)
    # print X_train_bow.shape
    # print X_dev_bow.shape

    utils.write_data(idx_to_label, constants.IDX_TO_LABEL_FILE)
    utils.write_data(word_to_idx, constants.WORD_TO_IDX_FILE)
    utils.write_data(X_train, constants.X_TRAIN_FILE)
    utils.write_data(Y_train, constants.Y_TRAIN_FILE)
    utils.write_data(X_dev, constants.X_DEV_FILE)
    utils.write_data(Y_dev, constants.Y_DEV_FILE)
    # utils.write_data(X_train_bow.todense(), constants.X_TRAIN_BOW_FILE)
    # utils.write_data(X_dev_bow.todense(), constants.X_DEV_BOW_FILE)

if __name__ == '__main__':
    main()