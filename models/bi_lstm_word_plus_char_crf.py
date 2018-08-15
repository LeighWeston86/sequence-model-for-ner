#Reproducibility
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

#Main imports
import numpy as np
from utils.data_utils import load_annotations, format_data, format_char_data, cross_val_sets, get_embedding_matrix
from utils.metrics import get_metrics
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Input
from keras.optimizers import Adam
from keras.models import Sequential
import pickle
from keras import regularizers
from keras_contrib.layers import CRF

class BiLSTM:

    def __init__(self, annotations = 'data/annotations.p', embeddings = 'data/w2v.txt', embeddings_as_dict = False):
        self.annotations = load_annotations(annotations)
        self.cache = format_data(self.annotations)
        self.char_cache = format_char_data(self.annotations)
        self.cv_sets = cross_val_sets(self.char_cache['padded_char_sents'], self.cache['padded_labels'])
        self.embedding_matrix = get_embedding_matrix(self.cache['word_to_integer'], embeddings, as_dict = embeddings_as_dict)
        self.model = None

    def fit_model(self, X_train, X_test, X_train_char, X_test_char, y_train, y_test):

        #Extract parameters from the cache
        word_to_integer = self.cache['word_to_integer']
        n_words = self.cache['n_words']
        n_tags = self.cache['n_tags']
        n_chars = self.char_cache['n_chars']
        max_sequence_length = self.cache['max_sequence_length']
        max_word_length = self.char_cache['max_word_length']
        print('here', max_sequence_length, max_word_length)

        #Define the char-level LSTM

        char_input = Input(shape=(max_sequence_length, max_word_length))
        char_embedding_layer = TimeDistributed(Embedding(input_dim=n_chars + 1, output_dim=10,
                                             input_length=max_word_length, mask_zero=True))

        char_model = char_embedding_layer(char_input)
        char_model = Dropout(0.0)(char_model)
        char_model = TimeDistributed(Bidirectional(LSTM(units=30 , return_sequences=False)))(char_model)
        char_model = Dropout(0.55)(char_model)

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
        word_model = Bidirectional(LSTM(units=200, return_sequences=True))(word_model)
        word_model = Dropout(0.55)(word_model)

        #Merge models
        concatenated = concatenate([char_model, word_model])
        model = TimeDistributed(Dense(n_tags + 1, activation="relu"))(concatenated)
        #model = Dropout(0.3)(model)

        #Crf as output layer
        crf = CRF(n_tags+1)
        out = crf(model)
        model = Model([char_input, word_input] , out)

        optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss=crf.loss_function, metrics=[crf.accuracy])
        model.fit([X_train_char, X_train], y_train, batch_size=32, epochs=25)
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

    def run(self, cutoff = 0.9):
        sents = self.cache['padded_sents']
        sents_char = self.char_cache['padded_char_sents']
        labels = self.cache['padded_labels']
        # Train a model
        cutoff = int(sents.shape[0]*cutoff)
        X_train = sents[:cutoff]
        X_test = sents[cutoff:]
        X_train_char = sents_char[:cutoff]
        X_test_char = sents_char[cutoff:]
        y_train = labels[:cutoff]
        y_test = labels[cutoff:]

        self.fit_model(X_train, X_test, X_train_char, X_test_char, y_train, y_test)


        # Accuracy metrics for test set
        loss, accuracy = self.model.evaluate([X_test_char, X_test], y_test)

        probs = self.model.predict([X_test_char, X_test])

        predicted = probs.argmax(axis=-1)
        actual = y_test.argmax(axis=-1)
        accuracy, f1, p, r = get_metrics(actual, predicted, self.cache['integer_to_label'])
        print('### Test set metrics ###')
        print('acc: {}, f1: {}'.format(accuracy, f1))
        #tags = ['POL', 'MON']
        #for tag in tags:
        #    accuracy, f1, p, r = get_metrics(actual, predicted, self.cache['integer_to_label'], tag = tag)
        #    print('{} f1: {}; p: {}; r{}'.format(tag, f1, p, r))

        # Accuracy metrics for train set
        loss, accuracy = self.model.evaluate([X_train_char, X_train], y_train)

        probs = self.model.predict([X_train_char, X_train])

        predicted = probs.argmax(axis=-1)
        actual = y_train.argmax(axis=-1)
        accuracy, f1, p, r = get_metrics(actual, predicted, self.cache['integer_to_label'])
        print('### Train set metrics ###')
        print('acc: {}, f1: {}'.format(accuracy, f1))
        #tags = ['POL', 'MON']
        #for tag in tags:
        #    accuracy, f1, p, r = get_metrics(actual, predicted, self.cache['integer_to_label'], tag = tag)
        #    print('{} f1: {}; p: {}; r{}'.format(tag, f1, p, r))


if __name__ == "__main__":
    lstm = BiLSTM()#'data/polymer_data/polymer_ann_formatted.p', 'data/polymer_data/embedding_index_3mil.p')
    lstm.run()






