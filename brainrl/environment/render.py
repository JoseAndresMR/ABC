import matplotlib.pyplot as plt
from matplotlib import animation
import os


def save_frames_as_mp4(frames: list, path: str ='./', filename: str ='gym_animation'):
    """
    Save frames from render as mp4 using ffmpeg.
    
    :params frames: list of frames, as numpy arrays.
    :params path: path to save renders.
    :params filename: name of the file to save (without mp4 extension).
    """
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0,
               frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50)

    anim.save(os.path.join(path, filename) + '.mp4',
              writer='ffmpeg',
              fps=60)
