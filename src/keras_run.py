import keras
import utils
import constants
from keras_model import RecipeNet
from keras.models import load_model
import sys


def main():
    X_test = utils.read_data_pkl(constants.X_TEST_FILE)
    idx_to_labels = utils.read_data_pkl(constants.IDX_TO_LABEL_FILE)
    test_id_list = utils.read_data_pkl(constants.TEST_ID_LIST_FILE)

    model = RecipeNet()

    if len(sys.argv) > 1:
        model.load()
    else:
        X_train = utils.read_data_pkl(constants.X_TRAIN_FILE)
        Y_train = utils.read_data_pkl(constants.Y_TRAIN_FILE)
        X_dev = utils.read_data_pkl(constants.X_DEV_FILE)
        Y_dev = utils.read_data_pkl(constants.Y_DEV_FILE)
        word_to_idx = utils.read_data_pkl(constants.WORD_TO_IDX_FILE)

        num_labels = len(idx_to_labels)

        Y_train = keras.utils.to_categorical(Y_train, num_labels)
        Y_dev = keras.utils.to_categorical(Y_dev, num_labels)

        model.build_net(len(word_to_idx), num_labels, X_train.shape[1])
        model.fit(X_train, Y_train, X_dev, Y_dev)
        model.save()

    model.predict_outputs(X_test, test_id_list, idx_to_labels)

if __name__ == '__main__':
    main()
