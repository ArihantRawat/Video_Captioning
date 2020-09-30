import pandas as pd
import string
import csv
import os

path = os.getcwd()
print(path)

# extract descriptions for images


def load_descriptions(csv_path):
    mapping = dict()

    with open(csv_path, encoding='utf-8') as file:
        reader = csv.reader(file)
        for line in reader:
            if len(line) < 1:
                continue
            image_id, image_desc = line[0], line[1]
            image_id = image_id.split('.')[0]
            if image_id not in mapping:
                mapping[image_id] = list()
            mapping[image_id].append(image_desc)

    return mapping


def clean_descriptions(descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] = ' '.join(desc)


# Creating set of videonames
def find_videoName(vids_src):
    files = os.listdir(vids_src)
    video_names = set()
    for f in files:
        video_names.add(f.split('.')[0])
    return video_names

# save descriptions to file, one per line


def save_descriptions(descriptions, outputFileLoc, video_names):
    lines = ["VideoID,Decription"]
    for key, desc_list in descriptions.items():
        if key in video_names:
            for desc in desc_list:
                lines.append(key+'.avi'+','+desc)
    data = '\n'.join(lines)
    file = open(outputFileLoc, 'w', encoding="utf8")
    file.write(data)
    file.close()
    print(len(lines))


csv_path = os.path.join(path, 'dataset', 'MSVD_description_cfile-nodup.csv')
out_csv_path = os.path.join(path, 'dataset', 'cleaned_data.csv')
vids_src = os.path.join(path, "dataset", "YouTubeClips")

descriptions = load_descriptions(csv_path)
# clean descriptions
clean_descriptions(descriptions)
# Storing the name of videos
video_names = find_videoName(vids_src)
# save to file
save_descriptions(descriptions, out_csv_path, video_names)

print(len(video_names))
print(len(pd.read_csv(out_csv_path).iloc[:, 0].unique()))
print(len(pd.read_csv(csv_path).iloc[:, 0].unique()))
