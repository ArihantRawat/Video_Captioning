# %%
import skvideo.io
import numpy as np
import matplotlib.pyplot as plt

# %%


def extract_frame_from_video(filename, num_of_frames):
    videodata = skvideo.io.vread(filename,)
    total_frames = videodata.shape[0]
    sequence = np.linspace(
        0, total_frames, num_of_frames, False, dtype=np.int32)
    videodata = videodata[sequence, :, :, :]
    return videodata


# %%
extract_frame_from_video(
    '..\\dataset\\YouTubeClips\\_0nX-El-ySo_83_93.avi', 15)

# %%
