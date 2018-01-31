#Preprocessing constants
FIX_LEN=40


#Source files
TRAIN_FILE = "../data/cuisine.train.json"
DEV_FILE = "../data/cuisine.dev.json"
TEST_FILE = "../data/cuisine.unlabeled.json"

#Preprocessed files
IDX_TO_LABEL_FILE = "../data/idx_to_labels.pkl"
WORD_TO_IDX_FILE = "../data/word_to_idx.pkl"
TEST_ID_LIST_FILE = "../data/test_id_list.pkl"

X_TRAIN_FILE = "../data/x_train.pkl"
Y_TRAIN_FILE = "../data/y_train.pkl"
X_DEV_FILE = "../data/x_dev.pkl"
Y_DEV_FILE = "../data/y_dev.pkl"
X_TEST_FILE = "../data/x_test.pkl"

#Model path
SAVED_MODEL_PATH = "../saved_models/model.h5"

#Neural network parameters
NUM_EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
EMBEDDING_SIZE = 256
DROPOUT = 0.6
KERNEL_SIZES = [2, 3, 4]

ONGPU=False