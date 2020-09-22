from pickle import load
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from numpy import array

path = os.path.dirname(os.getcwd())

# load doc into memory


def load_doc(filename):
    file = open(filename, 'r', encoding="utf8")
    text = file.read()
    file.close()
    return text


def load_descriptions(filename):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id not in descriptions:
            descriptions[image_id] = list()
        # wrap description in tokens
        desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
        descriptions[image_id].append(desc)
    return descriptions

# load video features


def load_video_features(filename):
    all_features = load(open(filename, 'rb'))
    return all_features

# convert a dictionary of clean descriptions to a list of descriptions


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the length of the description with the most words


def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# mapping of input video feature with input text and output predicted word


def create_sequences(tokenizer, max_length, descriptions, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each video identifier
    for key, desc_list in descriptions.items():
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = seq[i]
                X1.append(key)
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)


train_descriptions = load_descriptions(path+"/dataset/cleaned_data.txt")
# print(train_descriptions["hbE29pZh76I_3_8"])
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
# print(vocab_size)
max_length = max_length(train_descriptions)
X1train, X2train, ytrain = create_sequences(
    tokenizer, max_length, train_descriptions, vocab_size)
# print(ytrain)
