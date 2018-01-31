import json
import cPickle as pkl

def read_data(data_file):
    """
    Read from json file
    :param data_file: Pile path
    :return: data
    """
    with open(data_file) as json_data:
        data = json.load(json_data)
    return data

def write_data(data, data_file):
    """
    Write to pickle file
    :param data: Data
    :param data_file: File path
    :return:
    """
    with open(data_file, "wb") as fp:
        pkl.dump(data, fp)
    print "Saved file " + data_file

def read_data_pkl(data_file):
    """
    Read from pickle file
    :param data_file: file path
    :return: data
    """
    with open(data_file, "rb") as fp:
        data = pkl.load(fp)
    return data


def get_accuracy(predictions, labels):
    """
    Returns the accuracy of the predictions, provided the labels
    :param predictions: predictions from the model
    :param labels: gold labels for the respective predictions
    :return:
    """

    return float(sum(predictions == labels).data[0]) / labels.size()[0]