import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.data import Dataset, AUTOTUNE, experimental

### Baseline model
def note_transformer(note_list, seq_length=40):
  """Function which takes a list of musical notes,
  and returns numpy arrays of X and y where X contains
  encoded sequences of these notes, and y corresponds
  to the seq_length following notes of that sequence"""

  symb = sorted(list(set(note_list)))

  mapping = dict((c, i) for i, c in enumerate(symb))
  reverse_mapping = dict((i, c) for i, c in enumerate(symb))

  length = seq_length
  features = []
  targets = []
  for i in range(0, len(note_list) - length, 1):
    feature = note_list[i:i + length]
    target = note_list[i + length]
    features.append([mapping[j] for j in feature])
    targets.append(mapping[target])

  X = (np.reshape(features, (len(targets), length, 1)))/ float(len(symb))
  y = to_categorical(targets)

  return X, y


### Multivariate model

def data_split(df,
               train_split=0.7,
               val_split=0.2,
               seed_split=0.1,
               seq_length=40):
    """Function which takes a chronologically
    ordered dataframe, desired sequence length,
    as well as train, validation, seed split sizes,
    and returns train, validation and seed datasets.
    (function ONLY works with input split
    sizes with one decimal place)"""

    num_seq = round(len(df)/(seq_length+1))
    train_df = pd.DataFrame(columns=df.columns)
    val_df = pd.DataFrame(columns=df.columns)
    seed_df = pd.DataFrame(columns=df.columns)

    for i, chunk in enumerate(np.array_split(df, num_seq)):
        if int(str(i)[-1]) < 10*train_split:
            train_df = pd.concat([train_df, chunk])
        elif int(str(i)[-1]) < 10*(train_split+val_split):
            val_df = pd.concat([val_df, chunk])
        else:
            seed_df = pd.concat([seed_df, chunk])

    train_df = train_df.reset_index().drop(columns='index').astype('float64')

    val_df = val_df.reset_index().drop(columns='index').astype('float64')

    seed_df = seed_df.reset_index().drop(columns='index').astype('float64')

    return train_df, val_df, seed_df

def df_to_dataset(df: pd.DataFrame):
  """Transforms a dataframe into a Tensorflow dataset"""

  cols=['pitch', 'step', 'velocity']
  dataset = np.stack([df[col] for col in cols], axis=1)
  dataset = Dataset.from_tensor_slices(dataset)

  return dataset

def create_sequences(dataset: Dataset,
                     seq_length: int = 40
                     ) -> Dataset:
  """Returns TF Dataset of sequence and label examples."""
  seq_length = seq_length+1

  # Take 1 extra for the labels
  windows = dataset.window(seq_length, shift=1, stride=1,
                              drop_remainder=True)

  # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
  flatten = lambda x: x.batch(seq_length, drop_remainder=True)
  sequences = windows.flat_map(flatten)

  # Split the labels
  def split_labels(sequences):
    inputs = sequences[:-1]
    labels_dense = sequences[-1]
    labels = {key:labels_dense[i] for i,key in enumerate(['pitch', 'step', 'velocity'])}

    return inputs, labels

  return sequences.map(split_labels, num_parallel_calls=AUTOTUNE)

def create_batches(sequences,
                   batch_size: int=128) -> Dataset:
  """Returns batched dataset for more efficient
  data extraction during model training."""

  ds = (sequences
        .batch(batch_size, drop_remainder=True)
        .cache()
        .prefetch(experimental.AUTOTUNE))
  return ds
