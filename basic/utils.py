#  Copyright Université de Rouen Normandie (1), tutelle du laboratoire LITIS (1)
#  contributors :
#  - Denis Coquenet
#  - Thomas Constum
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.
import os
import json

import wandb
import cv2
import numpy as np
import torch
from torch.distributions.uniform import Uniform


def has_glyph(font, glyph):
    for table in font['cmap'].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False

def randint(low, high):
    """
    call torch.randint to preserve random among dataloader workers
    """
    return int(torch.randint(low, high, (1, )))


def rand():
    """
    call torch.rand to preserve random among dataloader workers
    """
    return float(torch.rand((1, )))


def rand_uniform(low, high):
    """
    call torch uniform to preserve random among dataloader workers
    """
    return float(Uniform(low, high).sample())


def pad_sequences_1D(data, padding_value):
    """
    Pad data with padding_value to get same length
    """
    x_lengths = [len(x) for x in data]
    longest_x = max(x_lengths)
    padded_data = np.ones((len(data), longest_x)).astype(np.int32) * padding_value
    for i, x_len in enumerate(x_lengths):
        padded_data[i, :x_len] = data[i][:x_len]
    return padded_data

def pad_sequences_1D_to_given_length(data, padding_value, longest_x):
    """
    Pad data with padding_value to get same length
    """
    x_lengths = [len(x) for x in data]
    padded_data = torch.ones((len(data), longest_x),dtype=torch.int64, device=data[0].device) * padding_value
    for i, x_len in enumerate(x_lengths):
        padded_data[i, :x_len] = data[i][:x_len]
    return padded_data

def resize_max(img, max_width=None, max_height=None):
    if max_width is not None and img.shape[1] > max_width:
        ratio = max_width / img.shape[1]
        new_h = int(np.floor(ratio * img.shape[0]))
        new_w = int(np.floor(ratio * img.shape[1]))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if max_height is not None and img.shape[0] > max_height:
        ratio = max_height / img.shape[0]
        new_h = int(np.floor(ratio * img.shape[0]))
        new_w = int(np.floor(ratio * img.shape[1]))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    return img


def pad_images(data, padding_value, padding_mode="br"):
    """
    data: list of numpy array
    mode: "br"/"tl"/"random" (bottom-right, top-left, random)
    """
    x_lengths = [x.shape[0] for x in data]
    y_lengths = [x.shape[1] for x in data]
    longest_x = max(x_lengths)
    longest_y = max(y_lengths)
    padded_data = np.ones((len(data), longest_x, longest_y, data[0].shape[2])) * padding_value
    for i, xy_len in enumerate(zip(x_lengths, y_lengths)):
        x_len, y_len = xy_len
        if padding_mode == "br":
            padded_data[i, :x_len, :y_len, ...] = data[i]
        elif padding_mode == "tl":
            padded_data[i, -x_len:, -y_len:, ...] = data[i]
        elif padding_mode == "random":
            xmax = longest_x - x_len
            ymax = longest_y - y_len
            xi = randint(0, xmax) if xmax >= 1 else 0
            yi = randint(0, ymax) if ymax >= 1 else 0
            padded_data[i, xi:xi+x_len, yi:yi+y_len, ...] = data[i]
        else:
            raise NotImplementedError("Undefined padding mode: {}".format(padding_mode))
    return padded_data


def pad_image(image, padding_value, new_height=None, new_width=None, pad_width=None, pad_height=None, padding_mode="br", return_position=False):
    """
    data: list of numpy array
    mode: "br"/"tl"/"random" (bottom-right, top-left, random)
    """
    if pad_width is not None and new_width is not None:
        raise NotImplementedError("pad_with and new_width are not compatible")
    if pad_height is not None and new_height is not None:
        raise NotImplementedError("pad_height and new_height are not compatible")

    h, w, c = image.shape
    pad_width = pad_width if pad_width is not None else max(0, new_width - w) if new_width is not None else 0
    pad_height = pad_height if pad_height is not None else max(0, new_height - h) if new_height is not None else 0

    if not (pad_width == 0 and pad_height == 0):
        padded_image = np.ones((h+pad_height, w+pad_width, c)) * padding_value
        if padding_mode == "br":
            hi, wi = 0, 0
        elif padding_mode == "tl":
            hi, wi = pad_height, pad_width
        elif padding_mode == "random":
            hi = randint(0, pad_height) if pad_height >= 1 else 0
            wi = randint(0, pad_width) if pad_width >= 1 else 0
        else:
            raise NotImplementedError("Undefined padding mode: {}".format(padding_mode))
        padded_image[hi:hi + h, wi:wi + w, ...] = image
        output = padded_image
    else:
        hi, wi = 0, 0
        output = image

    if return_position:
        return output, [[hi, hi+h], [wi, wi+w]]
    return output


def pad_image_width_right(img, new_width, padding_value):
    """
    Pad img to right side with padding value to reach new_width as width
    """
    h, w, c = img.shape
    pad_width = max((new_width - w), 0)
    pad_right = np.ones((h, pad_width, c), dtype=img.dtype) * padding_value
    img = np.concatenate([img, pad_right], axis=1)
    return img


def pad_image_width_left(img, new_width, padding_value):
    """
    Pad img to left side with padding value to reach new_width as width
    """
    h, w, c = img.shape
    pad_width = max((new_width - w), 0)
    pad_left = np.ones((h, pad_width, c), dtype=img.dtype) * padding_value
    img = np.concatenate([pad_left, img], axis=1)
    return img


def pad_image_width_random(img, new_width, padding_value, max_pad_left_ratio=1):
    """
    Randomly pad img to left and right sides with padding value to reach new_width as width
    """
    h, w, c = img.shape
    pad_width = max((new_width - w), 0)
    max_pad_left = int(max_pad_left_ratio*pad_width)
    pad_left = randint(0, min(pad_width, max_pad_left)) if pad_width != 0 and max_pad_left > 0 else 0
    pad_right = pad_width - pad_left
    pad_left = np.ones((h, pad_left, c), dtype=img.dtype) * padding_value
    pad_right = np.ones((h, pad_right, c), dtype=img.dtype) * padding_value
    img = np.concatenate([pad_left, img, pad_right], axis=1)
    return img


def pad_image_height_random(img, new_height, padding_value, max_pad_top_ratio=1):
    """
    Randomly pad img top and bottom sides with padding value to reach new_width as width
    """
    h, w, c = img.shape
    pad_height = max((new_height - h), 0)
    max_pad_top = int(max_pad_top_ratio*pad_height)
    pad_top = randint(0, min(pad_height, max_pad_top)) if pad_height != 0 and max_pad_top > 0 else 0
    pad_bottom = pad_height - pad_top
    pad_top = np.ones((pad_top, w, c), dtype=img.dtype) * padding_value
    pad_bottom = np.ones((pad_bottom, w, c), dtype=img.dtype) * padding_value
    img = np.concatenate([pad_top, img, pad_bottom], axis=0)
    return img


def pad_image_height_bottom(img, new_height, padding_value):
    """
    Pad img to bottom side with padding value to reach new_height as height
    """
    h, w, c = img.shape
    pad_height = max((new_height - h), 0)
    pad_bottom = np.ones((pad_height, w, c)) * padding_value
    img = np.concatenate([img, pad_bottom], axis=0)
    return img


class NpEncoder(json.JSONEncoder):
    """
    A custom Encoder that allow to save dicts containing numpy values
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def create_fake_batch_names(folder_path, batch_size):
    files_list = [folder_path+'/'+img_name for img_name in os.listdir(folder_path)]

    batch_file_lists = []
    for i in range((len(files_list) // batch_size)+1):
        batch_file_lists.append(files_list[i * batch_size: (i + 1) * batch_size])

    batch_file_lists = [batch for batch in batch_file_lists if len(batch) > 0]

    return batch_file_lists

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def init_wandb(projet_name, exp_id, params, dataset_name, params_to_log=None):
    if params_to_log is None:
        params_to_log = {
            "learning_rate": params["training_params"]["optimizers"]["all"]["args"]["lr"],
            "encoder": params["model_params"]['models']["encoder"],
            "dataset": dataset_name,
            "worker_per_gpu": params["training_params"]["nb_gpu"],
            "curr_step": params['dataset_params']["config"]["synthetic_data"]["curr_step"],
            'batch_size': params["training_params"]["batch_size"],
            "transfer_learning": params["model_params"]["transfer_learning"],
            'constraints': params["dataset_params"]["config"]["constraints"],
        }
    wandb.init(
        # set the wandb project where this run will be logged
        project=projet_name,
        # dir='wandb',
        id=exp_id,
        # track hyperparameters and run metadata
        config=params_to_log,
        resume=True,
        sync_tensorboard=True
    )