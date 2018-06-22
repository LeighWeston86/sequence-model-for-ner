import numpy as np
from utils.data_utils import load_annotations, format_data, cross_val_sets, get_embedding_matrix, format_char_data
from utils.metrics import get_metrics
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, concatenate, SpatialDropout1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Input
from keras.optimizers import Adam
from keras_contrib.layers import CRF

class BiLSTM:

    def __init__(self):
        self.annotations = load_annotations('data/annotations.p')
        self.cache = format_data(self.annotations)
        self.cv_sets = cross_val_sets(self.cache['padded_sents'], self.cache['padded_labels'])
        self.embedding_matrix = get_embedding_matrix(self.cache['word_to_integer'])
        self.char_cache = format_char_data(self.annotations,
                                           self.cache['word_to_integer'].keys(),
                                           self.cache['max_sequence_length'])
        self.model = None

    def fit_model(self, X_train, X_test, y_train, y_test, X_char_train, X_char_test):

        #Extract parameters from the cache
        word_to_integer = self.cache['word_to_integer']
        n_words = self.cache['n_words']
        n_tags = self.cache['n_tags']
        max_sequence_length = self.cache['max_sequence_length']
        X_char = self.char_cache['X_char']
        max_len_char = self.char_cache['max_len_char']
        n_chars = self.char_cache['n_chars']

        # (among top max_features most common words)
        batch_size = 32


        #Word input
        word_in = Input(shape=(max_sequence_length,))

        # Word embedding matrix
        embedding_matrix = get_embedding_matrix(word_to_integer)

        # Word Embedding layer
        embedding_layer = Embedding(input_dim=n_words + 1,
                                    output_dim=200,
                                    weights=[embedding_matrix],
                                    input_length=max_sequence_length,
                                    trainable=False)(word_in)

        # input and embeddings for characters
        char_in = Input(shape=(max_sequence_length, max_len_char,))
        emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                                             input_length=max_len_char, mask_zero=True))(char_in)

        # character LSTM to get word encodings by characters
        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                        recurrent_dropout=0.5))(emb_char)

        # main LSTM
        x = concatenate([embedding_layer, char_enc])
        x = SpatialDropout1D(0.3)(x)
        main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                                       recurrent_dropout=0.6))(x)
        out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)

        model = Model([word_in, char_in], out)
        optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])


        model.fit([X_train, X_char_train], y_train,
                            batch_size=32, epochs=1, validation_split=0.2, verbose=1)

        self.model = model



    def run(self, cutoff = 0.8):
        sents = self.cache['padded_sents']
        labels = self.cache['padded_labels']
        # Train a model
        cutoff = int(sents.shape[0]*0.8)
        X_train = sents[:cutoff]
        X_test = sents[cutoff:]
        y_train = labels[:cutoff]
        y_test = labels[cutoff:]
        X_char_train = np.array(self.char_cache['X_char'])[:cutoff]
        X_char_test = np.array(self.char_cache['X_char'])[cutoff:]

        self.fit_model(X_train, X_test, y_train, y_test, X_char_train, X_char_test)

        # Accuracy metrics
        loss, accuracy = self.model.evaluate([X_test, X_char_test], y_test)

        print(accuracy)

        probs = self.model.predict([X_test, X_char_test], batch_size = 32)
        print(probs[0].shape)

        predicted = probs.argmax(axis=-1)
        print(predicted[200])
        actual = y_test.argmax(axis=-1)
        print(actual[200])
        accuracy, f1 = get_metrics(actual, predicted, self.cache['integer_to_label'])
        print('acc: {}, f1: {}'.format(accuracy, f1))


    def run_cv(self):
        sents = self.cache['padded_sents']
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


if __name__ == "__main__":
    lstm = BiLSTM()
    lstm.run()
    #lstm.run_cv()






