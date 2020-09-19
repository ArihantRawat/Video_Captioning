import numpy as np
from pickle import load
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def create_dataset(feature_src, data_src):
    sent = []
    features = []
    data = open(data_src, encoding='utf-8').read().split('\n')
    for name in data:
        toks = name.split()
        # load video
        vid = load(open(os.path.join(feature_src, toks[0]+'.pkl'), 'rb'))
        features.append(vid)
        sent.append(['\t']+toks[1:]+['\n'])
    features = np.array(features)

    max_len = max([len(s) for s in sent])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sent)
    seqs = tokenizer.texts_to_sequences(sent)
    padded = pad_sequences(seqs, max_len, padding='post').astype(int)
    target_padded = np.zeros(padded.shape)
    for i in range(padded.shape[0]):
        target_padded[i, :-1] = padded[i, 1:]
    return features, padded, target_padded, tokenizer, max_len


def create_embedding_matrix(glove_src, word_index, vocab_size, embd_size):
    vecs = {}
    for line in open(glove_src, encoding='utf-8'):
        toks = line.split()
        vecs[toks[0]] = np.array([float(toks[i]) for i in range(1, len(toks))])
    print(vecs['hello'][:20])

    embd_matrix = np.zeros((vocab_size+1, embd_size))
    random_vec = np.ones((embd_size,))*0.1
    c = 0
    for word, i in word_index.items():
        if i >= vocab_size:
            break
        ev = vecs.get(word)
        if ev is not None:
            embd_matrix[i] = vecs[word]
        else:
            embd_matrix[i], c = random_vec, c+1
    print('Embed not found for {} words'.format(c))
    return embd_matrix


# path = os.getcwd()
# print(path)

# feature_src = os.path.join(path, 'features-small')
# data_src = os.path.join(path, 'dataset', 'cleaned_data.txt')
# features, padded, target_padded, tokenizer, max_len = create_dataset(
#     feature_src, data_src)
# print(features.shape)
# print(padded[0])
# print(target_padded[0])
# print(max_len)
