# %%
import shutil
import pandas as pd
import os

# %%


def split_datatset(video_src, csv_src, video_dest, csv_dest, split_count=100):
    df = pd.read_csv(csv_src)
    dict, k, count = {}, 0, 0
    while count < split_count and k < len(df):
        if dict.get(df.iloc[k, 0]) == None:
            dict[df.iloc[k, 0]] = 1
            count = count+1
        k = k + 1
    print(count, len(dict.keys()))
    small_df = df.iloc[:k, :]
    if os.path.isfile(csv_dest):
        os.remove(csv_dest)
    small_df.to_csv(csv_dest, index=None)
    if not os.path.isdir(video_dest):
        os.mkdir(video_dest)

    for name in dict.keys():
        srcfile = os.path.join(video_src, name)
        destfile = os.path.join(video_dest, name)
        if os.path.isfile(srcfile):
            shutil.copy(srcfile, destfile)


# %%
split_datatset('..\\dataset\\YouTubeClips',
               '..\\dataset\\MSVD_description_cfile.csv',
               '..\\dataset\\YouTubeClips_small',
               '..\\dataset\\small_cfile.csv', 600)

# %%
