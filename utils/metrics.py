import numpy as np

def get_chunks(sequence):
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

def get_metrics(actual, predicted, integer_to_label):


    correct_preds, total_correct, total_preds = 0.0, 0.0, 0.0

    accuracies = []
    for _ac, _pred in zip(actual, predicted):

        print(_ac)
        print(_pred)

        ac = []
        pred = []
        for a, p in zip(_ac, _pred):
            if a in integer_to_label.keys():
                ac.append(integer_to_label[a])
                pred.append(integer_to_label[p])

        print(ac)
        print(pred)

        accuracies += [a == p for (a, p) in zip(ac, pred)]

        ac_chunks = set(get_chunks(ac))
        pred_chunks = set(get_chunks(pred))
        correct_preds += len(ac_chunks & pred_chunks)
        total_preds += len(pred_chunks)
        total_correct += len(ac_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2*p*r/(p+r) if correct_preds > 0 else 0

    return np.mean(accuracies), f1

