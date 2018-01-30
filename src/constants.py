TRAIN_FILE = "../data/cuisine.train.json"
DEV_FILE = "../data/cuisine.dev.json"
TEST_FILE = "../data/cuisine.unlabeled.json"

IDX_TO_LABEL_FILE = "../data/idx_to_labels.pkl"
WORD_TO_IDX_FILE = "../data/word_to_idx_file.pkl"

X_TRAIN_FILE = "../data/x_train.pkl"
Y_TRAIN_FILE = "../data/y_train.pkl"
X_DEV_FILE = "../data/x_dev.pkl"
Y_DEV_FILE = "../data/y_dev.pkl"
X_TRAIN_BOW_FILE = "../data/x_train_bow.pkl"
X_DEV_BOW_FILE = "../data/x_dev_bow.pkl"

NUM_EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
EMBEDDING_SIZE = 256
DROPOUT = 0.5
KERNEL_SIZES = [2, 3, 4]

ONGPU=False