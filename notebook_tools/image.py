""""""
import cv2
import itertools
import numpy as np


bbox_colors = """#a6cee3
#1f78b4
#b2df8a
#33a02c
#fb9a99
#e31a1c
#fdbf6f
#ff7f00
#cab2d6
#6a3d9a
#ffff99
#b15928""".split()

bbox_colors = [c.strip() for c in bbox_colors]
bbox_colors = bbox_colors[1::2] + bbox_colors[::2]


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i + 2], 16) / 255. for i in (0, 2, 4))


def color_by_id(obj_ids, colors):
    unique_ids = np.unique(obj_ids)[1:]  # remove id == -1
    color_dict = {i: c for i, c in zip(unique_ids, itertools.cycle(colors))}
    color_dict[-1] = 'k'
    return color_dict


def ensure_n_channels(img_array, n_channels=3, batch_dims=None):
    """Ensures that the given image array has the specified number of channels.

    Args:
      img_array: np.ndarray of shape [..., H, W, (C)], where the initial dimensions
        and the channel dimension are optional.
      n_channels: int, the required number of channels.
      batch_dims: integer or None. If not None, it indicates the number of preceeding dimensions
       and helps in inferring current number of channels.

    Returns:
      np.ndarray of shape [..., H, W, n_channels].
    """

    if batch_dims is not None:
        shape = img_array.shape[batch_dims:]
        if len(shape) == 2:
            img_array = img_array[..., np.newaxis]

    c = img_array.shape[-1]
    if c > 3:
        img_array = img_array[..., np.newaxis]
        c = 1

    if c == 1:
        img_array = np.concatenate([img_array] * n_channels, -1)
    elif c != n_channels:
        raise ValueError('Cannot convert c={} channels into c={} channels'.format(c, n_channels))

    return img_array


def draw_bbox(img, coords, colour, width):
    """Draws a rectangle on an image.

    Args:
      imgs: np.ndarray of shape [H, W, (C)].
      coords: 4-tuple of [y, x, h, w] - location of the recntalge.
      colour: 3-tuple of floats for the colour in RGB.
      width: line width of the rectangle.

    Returns:
      np.ndarray of shape [H, W, 3] - the original image with drawn rectangle.
    """

    shape = img.shape
    new_img = img
    if len(shape) == 2:
        new_img = new_img[..., np.newaxis]

    if new_img.shape[-1] == 1:
        new_img = np.tile(new_img, [1] * (len(img.shape) - 1) + [3])

    if new_img is img:
        new_img = img.copy()

    y, x, h, w = coords

    if isinstance(colour, str) and colour.startswith('#'):
        colour = hex_to_rgb(colour)

    cv2.rectangle(new_img, (x, y), (x + w, y + h), colour, width)
    return new_img


def draw_boxes(imgs, coords, presence, colours, obj_ids=None, width=1):
    """Draws bounding boxes on batched image sequences.

    Args:
      imgs: np.ndarray of shape [T, B, H, W, (C)].
      coords: np.ndarray of shape [T, B, K, 4], where K is the number
        of objects and the last dimension are (y, x, h, w) of the bbox.
    presence: np.ndarray of [T, B, K] binary variables indicating presence of object.
    colours: K-element list of 3-tuples of colour values.
    width: float, line width for drawing the boxes.

    Returns:
      np.ndarray of shape [T, B, H, W, 3] of images with drawn bounding boxes.
    """

    if obj_ids is None:

        def get_color(t, b, k):
            return colours[k]

    else:
        color_dict = color_by_id(colours, obj_ids)

        def get_color(t, b, k):
            return color_dict[obj_ids[t, b, k]]

    imgs = ensure_n_channels(imgs, 3)
    imgs_with_boxes = np.zeros_like(imgs)

    for t in xrange(imgs.shape[0]):
        for b in xrange(imgs.shape[1]):
            img = imgs[t, b]

            for k in xrange(presence.shape[-1]):
                if presence[t, b, k]:
                    c = get_color(t, b, k)
                    img = draw_bbox(img, coords[t, b, k], c, width)

            imgs_with_boxes[t, b] = img

    return imgs_with_boxes


def collate_frames(frames, W, H, cell_size, border_width, border_color):
    """Collates frames into a grid.

    Args:
      frames: np.ndarray of shape [T, B, H, W, (C)].
      H: int, height of the grid.
      W: int, width of the grid.
      cell_size: tuple of ints, resolution of each grid cell.
      border_width: int, width of the border between cells.
      border_color: float \in [0., 1.] or int \in {1, ..., 255}, brightness of the border between cells.
    """

    if len(frames.shape) == 4:
        frames = frames[..., np.newaxis]

    T, B, h, w, c = frames.shape

    if cell_size is None:
        cell_size = (h, w)

    if frames.dtype in (np.float32, np.float64):
        frames = (frames * 255.).astype(np.uint8)

    if border_color < 1.:
        border_color = int(np.round(border_color * 256))

    total_width = W * cell_size[0] + (W - 1) * border_width
    total_height = H * cell_size[1] + (H - 1) * border_width

    collated = np.ones([T, total_width, total_height, c], dtype=np.uint8) * border_color

    for t in xrange(T):
        for y in xrange(H):
            for x in xrange(W):

                st_y = y * (cell_size[1] + border_width)
                ed_y = st_y + cell_size[1]
                st_x = x * (cell_size[0] + border_width)
                ed_x = st_x + cell_size[0]

                img = frames[t, y * W + x]
                if img.shape[:2] != cell_size:
                    img = cv2.resize(img, cell_size[::-1])
                    if len(img.shape) == 2:
                        img = img[..., np.newaxis]

                collated[t, st_x:ed_x, st_y:ed_y] = img

    return collated