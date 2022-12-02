from tensorflow.keras.utils import to_categorical

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
