"""
References: https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
"""

import utils
import constants
from nltk import word_tokenize
import re
import pickle as pkl
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier


def get_data(data_file, no_labels=False):
    """
    :param data_file: Path to data file
    :param no_labels: Set true if reading unlabeled data
    :return: Ingrediants list, Label list (or id_list if unlabeled data)
    """
    data = utils.read_data(data_file)
    ingred_list = []
    label_list = []
    id_list = []
    for item in data:
        id_list.append(item['id'])
        if not no_labels:
            label_list.append(item['cuisine'])
        l = ""
        for ingred in item["ingredients"]:
                l = l + " " + ingred
        ingred_list.append(l)

    if not no_labels:
        return ingred_list, label_list
    else:
        return ingred_list,  id_list


def train():
    """
    Train function to train the classifier
    :return: Returns the classifier object for testing
    """
    ingred_list, label_list = get_data(constants.TRAIN_FILE)
    lr_clf = Pipeline([('vect', CountVectorizer()),
                             ('clf-lr', SGDClassifier(loss='log', penalty='l2',
                             alpha = 1e-4, n_iter = 30, random_state = 42)),])

    lr_clf.fit(ingred_list, label_list)
    ingred_dev, label_dev = get_data(constants.DEV_FILE)
    predicted = lr_clf.predict(ingred_dev)
    print "Validation accuracy: " + str(np.mean(predicted == label_dev))
    print "------------------------------------"
    return lr_clf


def test(clf, topk=3):
    """
    Test function to test the classifier
    :param clf: Trained classifier
    :return:
    """
    ingred_list, id_list = get_data(constants.TEST_FILE, no_labels=True)
    probs = clf.predict_proba(ingred_list)
    best_n = np.argsort(probs, axis=1)
    for i in range(len(id_list)):
        for j in range(1, topk+1):
            print str(id_list[i]) + "," + clf.classes_[best_n[i][-j]] + "," + str(probs[i][best_n[i][-j]])


def main():
    text_clf = train()
    test(text_clf)


if __name__ == '__main__':
    main()
