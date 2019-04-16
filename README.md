A bi-directional LSTM for sequence tagging. This model was developed for Named Entity Recognition (NER) applied to materials science. Details can be found in the following publication: Weston at al., submitted to J. Chem. Inf. Model.

### Usage

##### Data

The materials-science specific training data included in this repository is heavily truncated; to access the full data, contact Leigh Weston at lweston@lbl.gov. To use your own data, replace the training/test sets, and embeddings file with your own data in teh same format. 

Load the data as follows:
```python
from ner_tagging.model.utils import get_data, get_embedding_matrix, get_metrics
word_embedding_dim = 200
training, development, test, word_cache, char_cache = get_data()
embedding_matrix = get_embedding_matrix(word_cache["word_to_integer"], word_embedding_dim)
```

##### Training

To train the model first extract the requirement data:

```python
max_sequence_length = word_cache["max_sequence_length"]
n_words = word_cache["n_words"]
max_char_sequence_length = char_cache["max_word_length"]
n_chars = char_cache["n_chars"]
n_tags = word_cache["n_tags"]
```

The model has to be built before fitting:

```python
model = NERTagger()
model.build(embedding_matrix, max_sequence_length, n_words, max_char_sequence_length, n_chars, n_tags)
model.fit(X_train, X_train_char, y_train, num_epochs=15)
```

##### Predictions and assessment

To assess the model after training, do the following:

```python
predicted = model.predict(X_dev, X_dev_char)
actual = y_dev.argmax(axis=-1)
print(get_metrics(actual, predicted, word_cache["integer_to_label"]))
```




