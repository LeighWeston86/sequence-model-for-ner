import numpy as np
from utils.data_utils import load_annotations, format_data, cross_val_sets, get_embedding_matrix
from utils.metrics import get_metrics
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Input
from keras.optimizers import Adam

class BiLSTM:

    def __init__(self):
        self.annotations = load_annotations('data/annotations.p')
        self.cache = format_data(self.annotations)
        self.cv_sets = cross_val_sets(self.cache['padded_sents'], self.cache['padded_labels'])
        self.embedding_matrix = get_embedding_matrix(self.cache['word_to_integer'])
        self.model = None

    def fit_model(self, X_train, X_test, y_train, y_test):

        #Extract parameters from the cache
        word_to_integer = self.cache['word_to_integer']
        n_words = self.cache['n_words']
        n_tags = self.cache['n_tags']
        max_sequence_length = self.cache['max_sequence_length']

        # (among top max_features most common words)
        batch_size = 32

        # Embedding matrix
        embedding_matrix = get_embedding_matrix(word_to_integer)

        # Embedding layer
        embedding_layer = Embedding(input_dim=n_words + 1,
                                    output_dim=200,
                                    weights=[embedding_matrix],
                                    input_length=max_sequence_length,
                                    trainable=False)

        input = Input(shape=(max_sequence_length,))
        # model = Embedding(input_dim=n_words, output_dim=50, input_length=maxlen)(input)
        model = embedding_layer(input)
        model = Dropout(0.1)(model)
        model = Bidirectional(LSTM(units=100, return_sequences=True))(model)
        model = Dropout(0.5)(model)
        out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(model)  # softmax output layer
        model = Model(input, out)
        optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, batch_size=100, epochs=10)
        self.model = model


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
    lstm.run_cv()






