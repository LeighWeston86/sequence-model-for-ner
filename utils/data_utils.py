import os
from matstract.models.annotation_builder import AnnotationBuilder
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from keras.utils import to_categorical
import numpy as np
from matstract.nlp.ner_features import FeatureGenerator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from numpy import array
from scipy.sparse import hstack
import pandas as pd


import pickle

def load_annotations(annotation_location = None):
    '''
    Load the text with annotations.
    :param annotation_location: string, location of pickled annotations
    :return: list, annotations
    '''
    if annotation_location:
        #Relative annotation location
        annotation_location = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), annotation_location)
        annotations = pickle.load(open(annotation_location, 'rb'))
    else:
        builder = AnnotationBuilder()
        annotations = builder.get_annotations(user=os.environ['ANNOTATOR'])
        annotations = [annotated.to_iob(phrases=True)[0] for annotated in annotations]
        annotations = [[[((word, pos), tag) for word, pos, tag in sent] for sent in doc] for doc in annotations]
    return annotations


def format_data(data, max_sequence_length = None):
    '''
    Format raw annotated text into format for NN model.

    :param data: list of docs, each doc is a list of sents, each sent is a list of tuples
    with ((word, pos), iob)
    :param max_sequence_length: integer, size for sentence padding
    :return: padded sentences and labels, dictionaries for mapping words/labels to integers
    '''

    #Print the sequence stats
    lengths  = [len(sent) for doc in data for sent in doc]
    if not max_sequence_length:
        max_sequence_length = max(lengths)
    print('Max sequence length: {}'.format(max_sequence_length))

    #Flatten out in to sentences
    word_sents = [[word for (word, pos), bio in sent] for doc in data for sent in doc]
    tag_sents = [[bio for (word, pos), bio in sent] for doc in data for sent in doc]


    #Create word-integer mapping
    words = set([word for sent in word_sents for word in sent])
    word_to_integer = {word:n for n, word in enumerate(words, 1)}
    integer_to_word = {n:word for n, word in enumerate(words, 1)}

    #Create label_integer mapping
    labels = set([tag for sent in tag_sents for tag in sent])
    label_to_integer = {label:n for n, label in enumerate(labels, 1)}
    integer_to_label = {n:label for n, label in enumerate(labels, 1)}

    #Convert words to integers
    sents_integer = [[word_to_integer[word] for word in sent] for sent in word_sents]

    #Convert outcomes to integerencoding
    labels_integer = [[label_to_integer[tag] for tag in sent] for sent in tag_sents]

    #Pad the sentences/outcomes, convert outcomes to one_hot
    padded_sents = pad_sequences(sents_integer, maxlen=max_sequence_length)
    padded_labels = pad_sequences(labels_integer, maxlen=max_sequence_length)
    padded_labels = to_categorical(padded_labels)

    #Additional parameters for model
    n_words = len(integer_to_word)
    n_tags = len(integer_to_label)

    cache = {'max_sequence_length': max_sequence_length,
             'padded_sents'       : padded_sents,
             'padded_labels'      : padded_labels,
             'integer_to_word'    : integer_to_word,
             'word_to_integer'    : word_to_integer,
             'integer_to_label'   : integer_to_label,
             'label_to_integer'   : label_to_integer,
             'n_words'            : n_words,
             'n_tags'             : n_tags}

    return cache

def format_char_data(data, max_sent_length = None, max_word_length = 20):

    lengths = [len(sent) for doc in data for sent in doc]
    if not max_sent_length:
        max_sequence_length = max(lengths)
    print('Max sequence length: {}'.format(max_sequence_length))

    # Flatten out in to sentences
    word_sents = [[word for (word, pos), bio in sent] for doc in data for sent in doc]
    tag_sents = [[bio for (word, pos), bio in sent] for doc in data for sent in doc]

    #Get unique words/chars
    words = list(set([word for doc in data for sent in doc for (word, pos), bio in sent]))
    chars = list(set([char for word in words for char in word]))
    n_chars = len(chars)

    #Char to integer mapping
    char_to_integer = {char:n for n, char in enumerate(chars, 1)}
    integer_to_char = {n:char for n, char in enumerate(chars, 1)}

    #Sents to char_integer encoded
    char_sents = [[[char_to_integer[char] for char in word] for word in sent] for sent in word_sents]
    #Add the padding at sent and word level
    padded_char_sents = pad_sequences([[np.squeeze(pad_sequences([[char for char in word]], maxlen=max_word_length))
                          for word in sent]
                         for sent in char_sents], maxlen = max_sent_length)

    cache = {'max_word_length'    : max_word_length,
             'max_sent_length'    : max_sent_length,
             'padded_char_sents'  : padded_char_sents,
             'char_to_integer'    : char_to_integer,
             'integer_to_char'    : integer_to_char,
             'n_chars'            : n_chars}

    return cache


def cross_val_sets(sents, labels, n_cv = 5):
    '''
    Split data for cross validation
    :param sents: list of sentences
    :param labels: list of labels
    :param n_cv: number of folds for cross-validation
    :return: cross-val sets
    '''
    kf = KFold(n_splits=n_cv)
    cv_sets = []
    for train_index, test_index in kf.split(sents):
        # Get the train/test set
        X_train = np.array([sents[idx] for idx in train_index])
        y_train = np.array([labels[idx] for idx in train_index])
        X_test  = np.array([sents[idx] for idx in test_index])
        y_test  = np.array([labels[idx] for idx in test_index])
        cv_sets.append((X_train, X_test, y_train, y_test))
    return cv_sets

def get_embedding_matrix(word_to_integer,
                         embeddings_location = 'data/w2v.txt',
                         embedding_dim = 200,
                         as_dict = False):
    '''
    Uses a word_to_integer mapping and local word embeddings to generate the embedding matrix

    :param word_to_integer: dict, maps words to integers
    :param embeddings_location: string, location of local embeddings
    :param embedding_dim: dimension of the word embedidngs
    :return: embedding matrix
    '''

    #Relative embedding location
    embeddings_location = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), embeddings_location)

    if not as_dict:
        embeddings_index = {}
        with open(embeddings_location) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    else:
        embeddings_index = pickle.load(open(embeddings_location, 'rb'))

    embedding_matrix = np.zeros((len(word_to_integer) + 1, embedding_dim))  #Plus 1 ensures padding element is set ot zero
    for word, i in word_to_integer.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:   #words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix



def get_syntactical_features(word_to_integer, embedding_matrix):
    fg = FeatureGenerator()
    all_features = []
    for word, i in word_to_integer.items():
        all_features.append(fg.syntactical_features(word))
    all_features = pd.DataFrame(all_features)
    cat_array_exists = False
    for col in all_features:
        data_vec = hot_encoder(all_features[col])
        if not cat_array_exists:
            cat_feature_array = data_vec
            cat_array_exists = True
        else:
            cat_feature_array = hstack([cat_feature_array, data_vec])
    cat_feature_array = cat_feature_array.tocsr()
    feature_dict = {}
    for row, word in zip(cat_feature_array, word_to_integer.keys()):
        feature_dict[word] = row
    #Create a syntax matrix
    syntax_matrix = np.zeros((len(word_to_integer) + 1, cat_feature_array.shape[1]))
    for word, i in word_to_integer.items():
        syntax_vector = feature_dict.get(word)
        if syntax_vector is not None:
            syntax_matrix[i] = syntax_vector.todense()

    combined_matrix = np.hstack([embedding_matrix, syntax_matrix])

    return combined_matrix

def hot_encoder(data_vec):
    '''
    Binary encoder for categorical f
    :param array/list containg categoricla features:
    :return: sklean OneHotEncoder vector
    '''
    values = array(data_vec)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=True)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded


def format_char_data_old(data, words, max_len, max_len_char = 20):

    chars = set([w_i for w in words for w_i in w])
    n_chars = len(chars)
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0

    X_char = []
    sents = [[(word, pos, bio) for (word, pos), bio in sent] for doc in data for sent in doc]
    for sentence in sents:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))

    char_cache = {}
    char_cache['X_char'] = X_char
    char_cache['max_len_char'] = max_len_char
    char_cache['n_chars'] = n_chars
    return char_cache


if __name__ == "__main__":
    annotations = load_annotations('annotations.p')
    cache = format_data(annotations)
    cv_set = cross_val_sets(cache['padded_sents'], cache['padded_labels'])


