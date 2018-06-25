import numpy as np
from utils.data_utils import load_annotations, format_data, format_char_data, cross_val_sets, get_embedding_matrix
from utils.metrics import get_metrics
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.models import Sequential
import pickle

class BiLSTM:

    def __init__(self):
        self.annotations = load_annotations()
        pickle.dump(self.annotations, open('data/annotations.p', 'wb'))
        self.cache = format_data(self.annotations)
        self.char_cache = format_char_data(self.annotations)
        self.cv_sets = cross_val_sets(self.char_cache['padded_char_sents'], self.cache['padded_labels'])
        self.embedding_matrix = get_embedding_matrix(self.cache['word_to_integer'])
        self.model = None

    def fit_model(self, X_train, X_test, X_train_char, X_test_char, y_train, y_test):

        #Extract parameters from the cache
        word_to_integer = self.cache['word_to_integer']
        n_words = self.cache['n_words']
        n_tags = self.cache['n_tags']
        n_chars = self.char_cache['n_chars']
        max_sequence_length = self.cache['max_sequence_length']
        max_word_length = self.char_cache['max_word_length']

        #Define the char-level LSTM

        char_input = Input(shape=(max_sequence_length, max_word_length))
        char_embedding_layer = TimeDistributed(Embedding(input_dim=n_chars + 1, output_dim=10,
                                             input_length=max_word_length, mask_zero=True))

        char_model = char_embedding_layer(char_input)
        char_model = Dropout(0.0)(char_model)
        char_model = TimeDistributed(Bidirectional(LSTM(units=100, return_sequences=False)))(char_model)
        char_model = Dropout(0.4)(char_model)

        #Define word-level lstm
        word_embedding_matrix = get_embedding_matrix(word_to_integer)

        word_embedding_layer = Embedding(input_dim=n_words + 1,
                                    output_dim=200,
                                    weights=[word_embedding_matrix],
                                    input_length=max_sequence_length,
                                    trainable=False)

        word_input = Input(shape=(max_sequence_length,))
        word_model = word_embedding_layer(word_input)
        word_model = Dropout(0.0)(word_model)
        word_model = Bidirectional(LSTM(units=300, return_sequences=True))(word_model)
        word_model = Dropout(0.4)(word_model)

        #Merge models
        concatenated = concatenate([char_model, word_model])
        out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(concatenated)
        model = Model([char_input, word_input], out)
        optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit([X_train_char, X_train], y_train, batch_size=100, epochs=10)
        self.model = model


    def run_cv(self):
        sents = self.char_cache['padded_char_sents']
        labels = self.cache['padded_labels']
        accuracies = []
        f1s = []
        for X_train, X_test, y_train, y_test in cross_val_sets(sents, labels):
            #Train a model
            self.fit_model(X_train, X_test, y_train, y_test)

            #Accuracy metrics
            loss, accuracy = self.model.evaluate(X_test, y_test)
            probs = self.model.predict(X_test)
            predicted = probs.argmax(axis=-1)
            actual = y_test.argmax(axis=-1)
            accuracy, f1 = get_metrics(actual, predicted, self.cache['integer_to_label'])
            print('acc: {}, f1: {}'.format(accuracy, f1))
            accuracies.append(accuracy)
            f1s.append(f1)

        print('Cross-validated accuracy: {}'.format(np.mean(accuracies)))
        print('Cross-validated f1 score: {}'.format(np.mean(f1s)))


    def predict(self, X_test, y_test): pass

    def run(self, cutoff = 0.8):
        sents = self.cache['padded_sents']
        sents_char = self.char_cache['padded_char_sents']
        labels = self.cache['padded_labels']
        # Train a model
        cutoff = int(sents.shape[0]*0.8)
        X_train = sents[:cutoff]
        X_test = sents[cutoff:]
        X_train_char = sents_char[:cutoff]
        X_test_char = sents_char[cutoff:]
        y_train = labels[:cutoff]
        y_test = labels[cutoff:]


        self.fit_model(X_train, X_test, X_train_char, X_test_char, y_train, y_test)


        # Accuracy metrics
        loss, accuracy = self.model.evaluate([X_test_char, X_test], y_test)

        probs = self.model.predict([X_test_char, X_test])

        predicted = probs.argmax(axis=-1)
        actual = y_test.argmax(axis=-1)
        accuracy, f1 = get_metrics(actual, predicted, self.cache['integer_to_label'])
        print('acc: {}, f1: {}'.format(accuracy, f1))


if __name__ == "__main__":
    lstm = BiLSTM()
    lstm.run()






