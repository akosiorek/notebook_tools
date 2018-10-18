import cv2
import os
import io
import base64
from IPython.display import HTML
import tempfile


def save_video(frames, path, fps=30, codec='XVID', overwrite=False, silent=False):
    """Saves video to a file.

    Args:
      frames: np.ndarray of shape [T, H, W, C].
      path: Path to the video file (with extension).
      fps: float.
      codec: codec.
      overwrite: Boolean; overwrites an existing video if it exists if True.
      silent: Boolean; if True it prints an overwrite warning if it is about to overwrite a file.
      """

    if os.path.exists(path):
        if os.path.isfile(path):
            if overwrite:
                if not silent:
                    print 'Overwriting "{}".'.format(path)
                os.remove(path)
            else:
                raise RuntimeError('File "{}" already exists.'.format(path))
        else:
            raise RuntimeError('"{}" already exists and is not a file.'.format(path))

    fourcc = cv2.VideoWriter_fourcc(*codec)
    shape = frames.shape
    T = shape[0]
    frame_size = shape[2], shape[1]

    is_colour = len(shape) == 4 and shape[-1] != 1
    writer = cv2.VideoWriter(path, fourcc, float(fps), frame_size, isColor=is_colour)

    if is_colour:
        frames = frames[..., [2, 1, 0]]

    for t in xrange(T):
        writer.write(frames[t])

    writer.release()
    return path


def play_video_from_file(path):
    """Plays video from file.

    Args:
      path: Path to a video file."""

    video = io.open(path, 'r+b').read()
    encoded = base64.b64encode(video)
    html = HTML(data='''<video alt="video" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii')))
    return html


def display_video(frames, fps=30, codec='XVID'):
    """Displays video in a notebook.

    Args:
      frames: np.ndarray of shape [T, H, W, C]
    """

    vid_buffer = tempfile.NamedTemporaryFile()
    vid = save_video(frames, vid_buffer.name, fps, codec, overwrite=True, silent=True)
    vid = play_video_from_file(vid)
    vid_buffer.close()
    return vid