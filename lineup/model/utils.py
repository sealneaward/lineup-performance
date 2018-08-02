import numpy as np

def _hash(i):
    return i['gameclock'] + i['eid']

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def disentangle_train_val(train, val):
    """
    Given annotations of train/val splits, make sure no overlapping Event
    -> for later detection testing
    """
    new_train = []
    new_val = []
    while len(val) > 0:
        ve = val.pop()
        vh = _hash(ve)
        if vh in [_hash(i) for i in train] + [_hash(i) for i in new_train]:
            new_train.append(ve)
            # to balance, find a unique train_anno to put in val
            while True:
                te = train.pop(0)
                if _hash(te) in [_hash(i) for i in train]:  # not unique, put back
                    train.append(te)
                else:
                    new_val.append(te)
                    break
        else:
            new_val.append(ve)
    new_train += train
    return new_train, new_val

def split(config, annotations, fold_index):
    if config['shuffle']:
        np.random.seed(config['randseed'])
        np.random.shuffle(annotations)
    N = len(annotations)
    val_start = np.round(
        fold_index / config['N_folds'] * N).astype('int32')
    val_end = np.round((fold_index + 1) / config['N_folds'] * N).astype('int32')
    val_annotations = annotations[val_start:val_end]
    train_annotations = annotations[:val_start] + annotations[val_end:]
    return train_annotations, val_annotations

def clean(config, annotations, setting):
    """
    Use config columns to subset annotations to linear interopable features
    """
    annotations = annotations[config[setting]]
    return annotations