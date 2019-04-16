import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def get_data():
    """
    Gets the data for training and testing the model

    :return: training, development, test, word_cache, char_cache
    """

    # Data paths
    local_dir = os.path.dirname(__file__)
    train_path = os.path.join(local_dir, "../data/train_set.txt")
    dev_path = os.path.join(local_dir, "../data/dev_set.txt")
    test_path = os.path.join(local_dir, "../data/test_set.txt")

    # Load the data
    train_set = get_sents(train_path)
    dev_set = get_sents(dev_path)
    test_set = get_sents(dev_path)

    # Format the data
    word_cache = format_data(train_set)
    char_cache = format_char_data(train_set, word_cache["max_sequence_length"])

    # Get the train set
    training = word_cache["padded_sents"], char_cache["padded_char_sents"], word_cache["padded_labels"]

    # Get the dev set
    development = format_test(dev_set,
                                           word_cache["word_to_integer"],
                                           word_cache["label_to_integer"],
                                           word_cache["max_sequence_length"],
                                           char_cache["char_to_integer"],
                                           char_cache["max_word_length"])
    # Get the test set
    test = format_test(test_set,
                                           word_cache["word_to_integer"],
                                           word_cache["label_to_integer"],
                                           word_cache["max_sequence_length"],
                                           char_cache["char_to_integer"],
                                           char_cache["max_word_length"])

    return training, development, test, word_cache, char_cache


def get_sents(path):
    """
    Loads training/test data from a text file and converts to a list of sentences; each sentence
    is a list of (word, tag) pairs.

    :param path:
    :return:
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f]
    sents = []
    sent = []
    for line in lines:
        if line:
            word, tag = line.strip().split()
            sent.append((word, tag))
        elif not line and sent:
            sents.append(sent)
            sent = []
        else:
            pass
    if sent:
        sents.append(sent)
    return sents

def format_data(data):
    """
    Format raw annotated text into arrays for RNN model.

    :param data: list of sentences; each sentence is a list of tuples of the form (word, tag)
    :param max_sequence_length: integer; size for sentence padding
    :return: word_cache containing data and metadata
    """

    # Get sequence length
    lengths = [len(sent) for sent in data]
    max_sequence_length = max(lengths)

    # Separate words/tags
    word_sents = [[word for word, tag in sent] for sent in data]
    tag_sents = [[tag for word, tag in sent] for sent in data]

    # Create word-integer mapping
    words = set([word for sent in word_sents for word in sent])
    word_to_integer = {word: n for n, word in enumerate(words, 2)}
    word_to_integer["uNk"] = 1
    integer_to_word = {n: word for word, n in word_to_integer.items()}

    # Create label_integer mapping
    labels = set([tag for sent in tag_sents for tag in sent])
    label_to_integer = {label: n for n, label in enumerate(labels, 1)}
    integer_to_label = {n: label for n, label in enumerate(labels, 1)}

    # Convert words to integers
    sents_integer = [[word_to_integer[word] for word in sent] for sent in word_sents]

    # Convert outcomes to integerencoding
    labels_integer = [[label_to_integer[tag] for tag in sent] for sent in tag_sents]

    # Pad the sentences/outcomes, convert outcomes to one_hot
    padded_sents = pad_sequences(sents_integer, maxlen=max_sequence_length)
    padded_labels = pad_sequences(labels_integer, maxlen=max_sequence_length)
    padded_labels = to_categorical(padded_labels)

    # Additional parameters for model
    n_words = len(integer_to_word)
    n_tags = len(integer_to_label)
    cache = {
        "max_sequence_length": max_sequence_length,
        "padded_sents": padded_sents,
        "padded_labels": padded_labels,
        "integer_to_word": integer_to_word,
        "word_to_integer": word_to_integer,
        "integer_to_label": integer_to_label,
        "label_to_integer": label_to_integer,
        "n_words": n_words,
        "n_tags": n_tags
    }

    return cache

def format_char_data(data, max_sent_length, max_word_length=20):
    """
    Formats the data for the character-leve LSTM.

    :param data: list of sentences; each sentence is a list of tuples of the form (word, tag)
    :param max_sent_length: integer; max number of words per sentence
    :param max_word_length: integer; max number of chars per word
    :return: char_cache containing data and metadata
    """

    # Separate words/tags
    word_sents = [[word for word, tag in sent] for sent in data]

    #Get unique words/chars
    words = set([word for sent in word_sents for word in sent])
    chars = list(set([char for word in words for char in word]))
    n_chars = len(chars)

    #Char to integer mapping
    char_to_integer = {char: n for n, char in enumerate(chars, 2)}
    char_to_integer["uNk"] = 1
    integer_to_char = {n: char for char, n in char_to_integer.items()}

    #Sents to char_integer encoded
    char_sents = [[[char_to_integer[char] for char in word] for word in sent] for sent in word_sents]

    #Add the padding at sent and word level
    padded_char_sents = pad_sequences([[np.squeeze(pad_sequences([[char for char in word]], maxlen=max_word_length))
                          for word in sent]
                         for sent in char_sents], maxlen = max_sent_length)

    cache = {
        "max_word_length": max_word_length,
        "padded_char_sents": padded_char_sents,
        "char_to_integer": char_to_integer,
        "integer_to_char": integer_to_char,
        "n_chars": n_chars
    }

    return cache

def format_test(test_set, word_to_integer, label_to_integer, max_sequence_length, char_to_integer, max_word_length):
    """
    Formats data for a test set/

    :param test_set: list of sentences; each sentence is a list of tuples of the form (word, tag)
    :param word_to_integer: dict; maps each word onto a unique integer
    :param label_to_integer: dict; maps each label onto a unique integer
    :param max_sequence_length: int; max number of words in a sentence
    :param char_to_integer: dict; maps each char onto a unique integer
    :param max_word_length: int; max number of chars in a word
    :return: X_test, X_test_char, y_test
    """

    # Separate words and tags
    word_sents = [[word for word, tag in sent] for sent in test_set]
    tag_sents = [[tag for word, tag in sent] for sent in test_set]
    # Convert words to integers
    sents_integer = [[word_to_integer[word] if word in word_to_integer else 1
                      for word in sent] for sent in word_sents]

    # Convert outcomes to integerencoding
    labels_integer = [[label_to_integer[tag] for tag in sent] for sent in tag_sents]

    # Pad the sentences/outcomes, convert outcomes to one_hot
    X_test = pad_sequences(sents_integer, maxlen=max_sequence_length)
    y_test = pad_sequences(labels_integer, maxlen=max_sequence_length)
    y_test = to_categorical(y_test)

    #Sents to char_integer encoded
    char_sents = [[[char_to_integer[char] if char in char_to_integer else 1
                    for char in word] for word in sent] for sent in word_sents]

    #Add the padding at sent and word level
    X_test_char = pad_sequences([[np.squeeze(pad_sequences([[char for char in word]], maxlen=max_word_length))
                          for word in sent]
                         for sent in char_sents], maxlen=max_sequence_length)

    return X_test, X_test_char, y_test

def get_embedding_matrix(word_to_integer, embedding_dim=200):
    """
    Uses a word_to_integer mapping and local word embeddings to generate the embedding matrix

    :param word_to_integer: dict, maps words to integers
    :param embedding_dim: dimension of the word embeddings
    :return: embedding matrix
    """

    #Relative embedding location
    embeddings_location = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../data/w2v.txt")

    # Get the word-to-vector mappings
    embeddings_index = {}
    with open(embeddings_location) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Create the embeddings matrix
    embedding_matrix = np.zeros((len(word_to_integer) + 2, embedding_dim))
    embedding_matrix[1] = np.random.uniform(-1, 1, size=embedding_dim)
    for word, i in word_to_integer.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_chunks(sequence):
    """
    Finds entity chunks.

    :param sequence: list of tokens
    :return: entity chunks
    """
    chunks = []
    chunk_type, chunk_start = None, None
    for idx, seq in enumerate(sequence):
        # Add the first chunk
        if seq == 'O' and chunk_type is not None:
            chunk = (chunk_type, chunk_start, idx)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a new chunk
        elif seq != 'O':
            tok_chunk_class = seq.split('-')[0]
            tok_chunk_type = seq.split('-')[1]
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, idx
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, idx)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, idx
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(sequence))
        chunks.append(chunk)

    return set(chunks)

def get_metrics(actual, predicted, integer_to_label, tag=None):
    """
    Finds accuracy metrics by comparing the actual abd predicted labels.

    :param actual: array; ground-truth labels
    :param predicted: array; predicted labels
    :param integer_to_label: dict; maps integers to tag names
    :param tag: string or None; if not None, will evaluate accuracy for specified tag
    :return: tuple; accuracy, f1 score, precision, recall
    """

    correct_preds, total_correct, total_preds = 0.0, 0.0, 0.0

    accuracies = []
    for _ac, _pred in zip(actual, predicted):

        ac = []
        pred = []
        for a, p in zip(_ac, _pred):
            if a in integer_to_label.keys():
                ac.append(integer_to_label[a])
                pred.append(integer_to_label[p])

        accuracies += [a == p for (a, p) in zip(ac, pred)]

        ac_chunks = set(get_chunks(ac))
        pred_chunks = set(get_chunks(pred))

        if tag:
            ac_chunks = set([(l, b, e) for l, b, e in ac_chunks if l == tag])
            pred_chunks = set([(l, b, e) for l, b, e in pred_chunks if l == tag])

        correct_preds += len(ac_chunks & pred_chunks)
        total_preds += len(pred_chunks)
        total_correct += len(ac_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2*p*r/(p+r) if correct_preds > 0 else 0

    return {
               "accuracy": np.mean(accuracies),
               "f1": f1,
               "precision": p,
               "recall": r
    }
