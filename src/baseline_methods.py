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
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV



def get_data(data_file, no_labels=False):
    data = utils.read_data(data_file)
    ingred_list = []
    label_list = []
    max_len = 0
    for item in data:
        if not no_labels:
            label_list.append(item['cuisine'])
        l = ""
        for ingred in item["ingredients"]:
                l = l + " " + ingred
        ingred_list.append(l)
        if len(l) > max_len:
            max_len = len(l)

    if not no_labels:
        return ingred_list, label_list, max_len
    else:
        return ingred_list,  max_len


def nb():

    text_clf = Pipeline([('vect', CountVectorizer()),
                         # ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB()), ])

    ingred_list, label_list, max_len = get_data(constants.TRAIN_FILE)
    text_clf.fit(ingred_list, label_list)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(ingred_list)
    print X_train_counts

    ingred_dev, label_dev,  max_len = get_data(constants.DEV_FILE)
    predicted = text_clf.predict(ingred_dev)
    print np.mean(predicted == label_dev)


def svm():
    ingred_list, label_list, max_len = get_data(constants.TRAIN_FILE)
    text_clf_svm = Pipeline([('vect', CountVectorizer()),
                             # ('tfidf', TfidfTransformer()),
                             ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                             alpha = 1e-3, n_iter = 10, random_state = 42)),
    ])


    text_clf_svm.fit(ingred_list, label_list)
    ingred_dev, label_dev, max_len = get_data(constants.DEV_FILE)
    predicted_svm = text_clf_svm.predict(ingred_dev)
    print np.mean(predicted_svm == label_dev)





nb()
svm()