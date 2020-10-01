from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
from os import listdir
from pickle import dump
import cv2
import os
import skvideo
import skvideo.io
from tqdm import tqdm
# skvideo.setFFmpegPath('D:\\data\\ffmpeg-20200831-4a11a6f-win64-shared\\bin')


def extract_frames_from_video(filename, num_of_frames):
    videodata = skvideo.io.vread(filename,)
    total_frames = videodata.shape[0]
    sequence = np.linspace(
        0, total_frames, num_of_frames, False, dtype=np.int32)
    videodata = videodata[sequence, :, :, :]
    return videodata


def create_model():
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.input, outputs=model.layers[-2].output)
    # print(model.summary)
    return model


def extract_features_from_video(video, name, model):
    features = []
    i = 0
    for frame in video:
        image = img_to_array(frame)
        image = cv2.resize(image, dsize=(224, 224),
                           interpolation=cv2.INTER_CUBIC)
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features.append(feature.ravel())
        # print('>%s %s' % (name, i+1))
        i = i+1
    features = np.array(features)
    return features


dir_path = os.getcwd()
video_path = os.path.join(dir_path, 'dataset', 'YoutubeClips-small-train')
video_list = os.listdir(video_path)
feature_path = os.path.join(dir_path, 'dataset', 'features-small-train')
if not os.path.exists(feature_path):
    os.mkdir(feature_path)
print(os.path.exists(feature_path))

model = create_model()
for i in tqdm(range(len(video_list)), 'Processing videos'):
    name = video_list[i]

    data = extract_frames_from_video(os.path.join(video_path, name), 20)
    features = extract_features_from_video(data, name, model)
    dump(features, open(os.path.join(
        feature_path, name.split('.')[0]+'.pkl'), 'wb'))
