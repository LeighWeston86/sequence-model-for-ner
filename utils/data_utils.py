import os
from matstract.models.annotation_builder import AnnotationBuilder
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
from keras.utils import to_categorical
import numpy as np
from matstract.nlp.ner_features import FeatureGenerator


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
                         embedding_dim = 200):
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

    embeddings_index = {}
    with open(embeddings_location) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(word_to_integer) + 1, embedding_dim))  #Plus 1 ensures padding element is set ot zero
    for word, i in word_to_integer.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:   #words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def get_syntactical_features(vocabulary):
    fg = FeatureGenerator()
    all_features = []
    for word in vocabulary:
        all_features.append(fg.syntactical_features(word))
    categorical = to_categorical(all_features)
    feature_dict = {}
    for row, word in zip(categorical, vocabulary):
        feature_dict[word] = row
    return feature_dict




if __name__ == "__main__":
    annotations = load_annotations('annotations.p')
    cache = format_data(annotations)
    cv_set = cross_val_sets(cache['padded_sents'], cache['padded_labels'])


