# %%
from numpy.lib.recfunctions import drop_fields
import pandas as pd
import shutil
import os
# %%


def split_dataset(vids_src, csv_src, count, vids_dest, csv_dest):
    vid_ids = set()
    df = pd.read_csv(csv_src)
    if not os.path.isdir(vids_dest):
        os.mkdir(vids_dest)
    i = 0
    drop_rows = []
    while i < len(df):
        id = df.iloc[i, 0]
        if id.startswith('#') or not os.path.isfile(os.path.join(vids_src, id)):
            drop_rows.append(i)
            i = i+1
            continue
        if id not in vid_ids:
            if len(vid_ids) == count:
                break
            vid_ids.add(id)
            shutil.copy(os.path.join(vids_src, id),
                        os.path.join(vids_dest, id))
            print('.', end='')
        i = i+1
    small_df = df.iloc[:i, :].drop(drop_rows)
    small_df.to_csv(csv_dest, index=None)
    print(len(vid_ids))
    print(len(small_df.iloc[:, 0].unique()))
    print('Done')


# %%
dataset = '..\\dataset'
vids_src = os.path.join(dataset, 'YouTubeClips')
vids_dest = os.path.join(dataset, 'YouTubeClips-small')
csv_src = os.path.join(dataset, 'MSVD_description_cfile-nodup.csv')
csv_dest = os.path.join(dataset, 'MSVD-small.csv')

split_dataset(vids_src,
              csv_src,
              120,
              vids_dest, csv_dest)

# %%
items = pd.read_csv(csv_dest).iloc[:, 0].unique()
x = 0
for id in items:
    if os.path.isfile(os.path.join(vids_dest, id)):
        x = x+1
print(x)
# %%


# %%
# %%
