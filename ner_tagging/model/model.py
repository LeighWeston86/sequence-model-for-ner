from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, concatenate
from keras.layers.wrappers import TimeDistributed
from keras.models import Model, Input
from keras.optimizers import Adam
from keras_contrib.layers import CRF


class NERTagger(object):
    """
    A long short-tem memory (LSTM) based model that performs named entity recognition for
    materials science. The default model architecture is a bi-directional LSTM at the word and
    character level, with a conditional random fields (CRF) output layer. The char-level LSTM and
    the CRF output layer can be optionally removed."""

    def __init__(self, word_lstm_size=200,
                 char_lstm_size=50,
                 word_embedding_dim=200,
                 char_embedding_dim=20,
                 dropout=0.5):
        self.word_lstm_size = word_lstm_size
        self.char_lstm_size = char_lstm_size
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.dropout = dropout
        self.crf = None
        self.model = None

    def _char_lstm(self, max_sequence_length, max_char_sequence_length, n_chars):
        """
        The character-level LSTM. The model defines a trainable character-embedding layer;
        the character embeddings are fed into a bi-directional LSTM.

        :param max_sequence_length: int; maximum word-sequence length.
        :param max_char_sequence_length: int; maximum character-sequence length.
        :param n_chars: int; size of the character vocabulary
        :return: char-level LSTM model
        """

        char_input = Input(shape=(max_sequence_length, max_char_sequence_length))
        char_embedding_layer = TimeDistributed(Embedding(input_dim=n_chars+2,
                                                         output_dim=20,
                                                         input_length=max_char_sequence_length,
                                                         mask_zero=True))
        char_model = char_embedding_layer(char_input)
        char_model = TimeDistributed(Bidirectional(LSTM(units=50, return_sequences=False)))(char_model)
        return char_input, char_model

    def _word_lstm(self, word_embedding_array, max_sequence_length, n_words, char_model):
        """
        The word-level lstm. The model defines a non-trainable embedding layer that must have a
        pre-trained embedding array as input. The mode is a bi-directional LSTM at that takes the
        word embeddings as inputs at each time step; if self.include_chars is True, the final
        activations from the char-level LSTM are concatenated with the word embeddings before
        being fed in the to main word-level LSTM.

        :param word_embedding_array: np.array; pre-trained word embeddings. The shape of the
        array should be [embedding_size, vocab size]
        :param max_sequence_length: int; maximum character-sequence length
        :param n_words: int; size of the vocabulary
        :param char_model: character level LSTM
        :return: word-level LSTM model
        """

        word_embedding_layer = Embedding(input_dim=n_words+2,
                                         output_dim=self.word_embedding_dim,
                                         weights=[word_embedding_array],
                                         input_length=max_sequence_length,
                                         trainable=False)
        word_input = Input(shape=(max_sequence_length,))
        word_model = word_embedding_layer(word_input)
        word_model = concatenate([char_model, word_model])
        word_model = Dropout(self.dropout)(word_model)
        word_model = Bidirectional(LSTM(units=self.word_embedding_dim, return_sequences=True))(word_model)
        word_model = Dropout(self.dropout)(word_model)
        return word_input, word_model

    def build(self,
              word_embedding_array,
              max_sequence_length,
              n_words,
              max_char_sequence_length,
              n_chars,
              n_tags):
        """
        Builds the model.

        :param word_embedding_array: np.array; pre-trained word embeddings. The shape of the
        array should be [embedding_size, vocab size]
        :param max_sequence_length: int; maximum character-sequence length
        :param n_words: int; size of the vocabulary
        :param max_char_sequence_length: int; maximum character-sequence length.
        :param n_chars: int; size of the character vocabulary
        :param n_tags: int; number of classes for classification
        """

        char_input, char_model = self._char_lstm(max_sequence_length, max_char_sequence_length, n_chars)
        word_input, word_model = self._word_lstm(word_embedding_array, max_sequence_length, n_words, char_model)
        model = TimeDistributed(Dense(n_tags+1, activation="relu"))(word_model)
        crf = CRF(n_tags + 1)
        out = crf(model)
        model = Model([char_input, word_input], out)
        self.crf = crf
        self.model = model

    def fit(self, X_train, X_train_char, y_train, learning_rate=0.01, batch_size=256, num_epochs=10):
        """
        Fits the model.

        :param X_train: np.array; shape should be [n_samples, max_sequence_length]
        :param X_train_char: np.array; shape should be [n_sampes, max_seuence_length, max_char_sequence_length]
        :param y_train: np.array; shape should be [n_samples, num_classes]
        :param learning_rate: float; learning rate for Adam optimizer
        :param batch_size: int; samples per minibatch
        :param num_epochs: int; number of epochs during training
        """
        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
        if self.model is None:
            raise ValueError("The model must be built before fitting, run NERTagger.build()")
        self.model.compile(optimizer=optimizer, loss=self.crf.loss_function, metrics=[self.crf.accuracy])
        self.model.fit([X_train_char, X_train], y_train, batch_size=batch_size, epochs=num_epochs)

    def predict(self, X_test, X_test_char):
        """
        Makes predictions.

        :param X_test: np.array; shape should be [n_samples, max_sequence_length]
        :return: np.array; a 1d array containing the class with highest probability
        """
        probs = self.model.predict([X_test_char, X_test])
        predicted = probs.argmax(axis=-1)
        return predicted
