import json
import cPickle as pkl

def read_data(data_file):
    with open(data_file) as json_data:
        data = json.load(json_data)
        return data

def write_data(data, data_file):
    with open(data_file, "wb") as fp:
        pkl.dump(data, fp)
    print "Saved file " + data_file


def get_accuracy(predictions, labels):
    """
    Returns the accuracy of the predictions, provided the labels
    :param predictions: predictions from the model
    :param labels: gold labels for the respective predictions
    :return:
    """

    return float(sum(predictions == labels).data[0]) / labels.size()[0]