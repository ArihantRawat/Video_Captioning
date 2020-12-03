import os
import skvideo
import skvideo.io
import numpy as np
import matplotlib.pyplot as plt


def extract_frames_from_video(filename, num_of_frames):
    videodata = skvideo.io.vread(filename,)
    total_frames = videodata.shape[0]
    sequence = np.linspace(
        0, total_frames, num_of_frames, False, dtype=np.int32)
    videodata = videodata[sequence, :, :, :]
    return videodata


def saveAsImage(frames, filename):
    for i in range(frames.shape[0]):
        plt.subplot(3, frames.shape[0]//3, i+1)
        fig = plt.imshow(frames[i])
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # plt.subplots_adjust(left=0.1, right=0.5)
    plt.axis('off')
    plt.savefig(filename, dpi=2000, bbox_inches='tight', facecolor='w')
    plt.close()


dir_path = os.getcwd()
video_path = os.path.join(
    dir_path, 'dataset', 'YoutubeClips')
image_path = os.path.join(
    dir_path, 'Extras')

if(not os.path.exists(image_path)):
    os.mkdir(image_path)

filename = 'L8h2DazQZJY_0_10.avi'
# filename = '3zgEl-OLFKE_12_15.avi'
# filename = 'nhm_APPwhWk_6_12.avi'
# filename = 'G-M78KIy19E_315_330.avi'


frames = extract_frames_from_video(os.path.join(video_path, filename), 15)

saveAsImage(frames, os.path.join(image_path, filename.split('.')[0]+".png"))
