import os
import pandas as pd
import shutil
from tqdm import tqdm
import random


def create_dataset(vids_src, vids_dest, csv_src, csv_dest, n=120, action='copy'):
    df = pd.read_csv(csv_src, encoding='utf8')
    data = {}
    vids_list = []
    for i in tqdm(range(len(df)), 'Reading CSV'):
        if df.iloc[i, 0] in data:
            data[df.iloc[i, 0]].append(df.iloc[i, 1])
        else:
            data[df.iloc[i, 0]] = [df.iloc[i, 1]]
            vids_list.append(df.iloc[i, 0])

    vid_count = len(vids_list)
    print(vid_count)
    indices = random.sample(range(vid_count), n)
    new_vids_list = [vids_list[i] for i in indices]
    indices = []

    for i in tqdm(range(len(df)), 'Reading CSV again!!'):
        if df.iloc[i, 0] in new_vids_list:
            indices.append(i)

    # print(new_vids_list)

    # print(len(indices), len(new_vids_list))
    # print(len(set(indices)))

    if not os.path.isdir(vids_dest):
        os.mkdir(vids_dest)

    for i in tqdm(range(len(new_vids_list)), 'Shifting files'):
        file = new_vids_list[i]
        if action == 'copy':
            shutil.copy2(os.path.join(vids_src, file),
                         os.path.join(vids_dest, file))
        else:
            shutil.move(os.path.join(vids_src, file),
                        os.path.join(vids_dest, file))
    df.iloc[indices, :].to_csv(csv_dest, index=None)
    if action == 'move':
        df = df.drop(indices, axis=0)
        df.to_csv(csv_src, index=None)


# train set

path = os.getcwd()
vids_src = os.path.join(path, 'dataset', 'YouTubeClips')
vids_dest = os.path.join(path, 'dataset', 'YouTubeClips-small-train')
csv_src = os.path.join(path, 'dataset', 'cleaned_data.csv')
csv_dest = os.path.join(path, 'dataset', 'MSVD_train-small.csv')
create_dataset(vids_src, vids_dest, csv_src, csv_dest, 120, 'copy')
print(len(os.listdir(vids_dest)))

# test set
vids_src = vids_dest
vids_dest = os.path.join(path, 'dataset', 'YouTubeClips-small-test')
csv_src = csv_dest
csv_dest = os.path.join(path, 'dataset', 'MSVD_test-small.csv')
create_dataset(vids_src, vids_dest, csv_src, csv_dest, 20, 'move')
print(len(os.listdir(vids_dest)))
