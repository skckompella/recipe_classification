from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Merge, Permute, Dropout, Flatten
import numpy as np
import constants


class RecipeNet():
    def __init__(self, vocab_len, num_labels, max_len):
        self.embedding_size = constants.EMBEDDING_SIZE
        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_len, output_dim=self.embedding_size, input_length=max_len))
        self.model.add(Flatten())
        self.model.add(Dense(units=1000, activation='relu'))
        self.model.add(Dropout(constants.DROPOUT))
        self.model.add(Dense(units=500, activation='relu'))
        self.model.add(Dropout(constants.DROPOUT))
        self.model.add(Dense(units=num_labels, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    def fit(self, x_train, y_train, x_dev, y_dev):
        self.model.fit(x_train, y_train,
                       batch_size=constants.BATCH_SIZE,
                       epochs=constants.NUM_EPOCHS,
                       verbose=1,
                       validation_data=(x_dev, y_dev))


    def predict(self, x_test, id_list, idx_to_label, topk=3):
        probs = self.model.predict_proba(x_test, batch_size=constants.BATCH_SIZE)
        best_n = np.argsort(probs, axis=1)
        print best_n
        print id_list
        for i in range(len(id_list)):
            for j in range(1, topk + 1):
                print str(id_list[i]) + "," + idx_to_label[best_n[i][-j]] + "," + str(probs[i][best_n[i][-j]])

    def save(self):
        self.model.save(constants.SAVED_MODEL_PATH)
