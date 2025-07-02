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

"""
This module contains the DatasetManager class, which is responsible for loading and managing datasets for training, validation, and testing.
It also loads external datasets for synthetic data generation.
"""

from pathlib import Path
import random
import pickle as pkl
import os
import copy

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
from skimage import transform as transform_skimage

from basic.transforms import apply_data_augmentation
from basic.utils import pad_images
from Datasets.dataset_formatters.utils_dataset import natural_sort
from OCR.document_OCR.daniel.synth_doc.wiki_text import Wikipedia
class DatasetManager:

    def __init__(self, params):
        self.params = params
        self.dataset_class = params["dataset_class"]
        self.img_padding_value = params["config"]["padding_value"]

        self.my_collate_function = None

        self.train_dataset = None
        self.valid_datasets = dict()
        self.test_datasets = dict()

        self.train_loader = None
        self.valid_loaders = dict()
        self.test_loaders = dict()

        self.train_sampler = None
        self.valid_samplers = dict()
        self.test_samplers = dict()

        self.generator = torch.Generator()
        if 'deterministic' in params['config'] and params['config']['deterministic']:
            self.generator.manual_seed(0)

        self.batch_size = {
            "train": self.params["batch_size"],
            "valid": self.params["valid_batch_size"] if "valid_batch_size" in self.params else self.params["batch_size"],
            "test": self.params["test_batch_size"] if "test_batch_size" in self.params else 1,
        }

    def apply_specific_treatment_after_dataset_loading(self, dataset):
        raise NotImplementedError

    def load_datasets(self):
        """
        Load training and validation datasets
        """
        self.train_dataset = self.dataset_class(self.params, "train", self.params["train"]["name"], self.get_paths_and_sets(self.params["train"]["datasets"]))
        if not "mean" in self.params["config"] or not "std" in self.params["config"]:
            self.params["config"]["mean"], self.params["config"]["std"] = self.train_dataset.compute_std_mean()

        self.my_collate_function = self.train_dataset.collate_function(self.params["config"])
        self.apply_specific_treatment_after_dataset_loading(self.train_dataset)

        for custom_name in self.params["valid"].keys():
            self.valid_datasets[custom_name] = self.dataset_class(self.params, "valid", custom_name, self.get_paths_and_sets(self.params["valid"][custom_name]))
            self.apply_specific_treatment_after_dataset_loading(self.valid_datasets[custom_name])

    def load_samplers(self):
        """
        Load training and validation data samplers
        """
        for custom_name in self.valid_datasets.keys():
            self.valid_samplers[custom_name] = None

    def load_dataloaders(self):
        """
        Load training and validation data loaders
        """
        if 'MULTI_LINGUAL' in self.params['train']['name'] and 'multi_sampling' in self.params['config'] and self.params['config']['multi_sampling']:
            weights = []
            for ds_name, dataset_bounds in self.train_dataset.index_datasets.items():
                nb_data_in_bounds = dataset_bounds[1]-dataset_bounds[0] + 1
                if 'multi_sampling_weights' in self.params['config']: # cutsomized dataset proportions
                    weights.extend([self.params['config']['multi_sampling_weights'][ds_name]]*nb_data_in_bounds)
                else:
                    weights.extend([1/nb_data_in_bounds]*nb_data_in_bounds) # each dataset has the same weight

            self.train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(self.train_dataset.samples),
            )
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size["train"],
                                       shuffle=True if self.train_sampler is None else False,
                                       drop_last=False,
                                       sampler=self.train_sampler,
                                       num_workers=self.params["num_gpu"]*self.params["worker_per_gpu"],
                                       pin_memory=True,
                                       collate_fn=self.my_collate_function,
                                       worker_init_fn=self.seed_worker if ('deterministic' in self.params['config'] and self.params['config']['deterministic']) else None,
                                       generator=self.generator)

        for key in self.valid_datasets.keys():
            self.valid_loaders[key] = DataLoader(self.valid_datasets[key],
                                                 batch_size=self.batch_size["valid"],
                                                 sampler=self.valid_samplers[key],
                                                 shuffle=False,
                                                 num_workers=self.params["num_gpu"]*self.params["worker_per_gpu"],
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 collate_fn=self.my_collate_function,
                                                 worker_init_fn=self.seed_worker,
                                                 generator=self.generator)

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def generate_test_loader(self, custom_name, sets_list):
        """
        Load test dataset, data sampler and data loader
        """
        if custom_name in self.test_loaders.keys():
            return
        paths_and_sets = list()
        for set_info in sets_list:
            paths_and_sets.append({
                "path": self.params["datasets"][set_info[0]],
                "set_name": set_info[1]
            })
        self.test_datasets[custom_name] = self.dataset_class(self.params, "test", custom_name, paths_and_sets)
        self.apply_specific_treatment_after_dataset_loading(self.test_datasets[custom_name])
        self.test_samplers[custom_name] = None
        self.test_loaders[custom_name] = DataLoader(self.test_datasets[custom_name],
                                                    batch_size=self.batch_size["test"],
                                                    sampler=self.test_samplers[custom_name],
                                                    shuffle=False,
                                                    num_workers=self.params["num_gpu"]*self.params["worker_per_gpu"],
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    collate_fn=self.my_collate_function,
                                                    worker_init_fn=self.seed_worker,
                                                    generator=self.generator)

    def remove_test_dataset(self, custom_name):
        del self.test_datasets[custom_name]
        del self.test_samplers[custom_name]
        del self.test_loaders[custom_name]

    def remove_valid_dataset(self, custom_name):
        del self.valid_datasets[custom_name]
        del self.valid_samplers[custom_name]
        del self.valid_loaders[custom_name]

    def remove_train_dataset(self):
        self.train_dataset = None
        self.train_sampler = None
        self.train_loader = None

    def remove_all_datasets(self):
        self.remove_train_dataset()
        for name in list(self.valid_datasets.keys()):
            self.remove_valid_dataset(name)
        for name in list(self.test_datasets.keys()):
            self.remove_test_dataset(name)

    def get_paths_and_sets(self, dataset_names_folds):
        paths_and_sets = list()
        for dataset_name, fold in dataset_names_folds:
            path = self.params["datasets"][dataset_name]
            paths_and_sets.append({
                "path": path,
                "set_name": fold
            })
        return paths_and_sets


class GenericDataset(Dataset):
    """
    Main class to handle dataset loading
    """

    def __init__(self, params, set_name, custom_name, paths_and_sets):
        self.params = params
        self.name = custom_name
        self.set_name = set_name
        self.mean = np.array(params["config"]["mean"]) if "mean" in params["config"].keys() else None
        self.std = np.array(params["config"]["std"]) if "std" in params["config"].keys() else None

        if set_name in ['valid', 'test']:
            self.load_in_memory = self.params["config"]["load_in_memory_inference"] if "load_in_memory_inference" in self.params["config"] else True
        else:
            self.load_in_memory = self.params["config"]["load_in_memory"] if "load_in_memory" in self.params["config"] else True
        labels_name = self.params["config"]["labels_name"] if "labels_name" in self.params["config"] else "labels.pkl"
        self.samples = self.load_samples(paths_and_sets, load_in_memory=self.load_in_memory, labels_name=labels_name, self_training=self.params["config"].get("self_training", False))

        if "MULTI_LINGUAL" in self.name:
            self.index_datasets = self.params["config"]['index_datasets']

        if "other_samples" in self.params["config"] and 'train' in self.set_name:
            # loads an external corpus to feed the text with synthetic data
            other_samples_batch_size = self.params["config"]['other_samples_batch_size'] if 'other_samples_batch_size' in self.params["config"] else 1
            self.other_samples = self.load_other_samples(self.params["config"]['other_samples'], batch_size=other_samples_batch_size)
        else:
            self.other_samples = {}

        if "multi_samples" in self.params["config"] and 'train' in self.set_name:
            self.multi_samples = self.load_multi_samples(self.params["config"]['multi_samples'], config=self.params["config"])
        else:
            self.multi_samples = {}

        if self.load_in_memory:
            self.apply_preprocessing(params["config"]["preprocessings"])

        self.padding_value = params["config"]["padding_value"]
        if self.padding_value == "mean":
            if self.mean is None:
                _, _ = self.compute_std_mean()
            self.padding_value = self.mean
            self.params["config"]["padding_value"] = self.padding_value


        print('Datased images mean value', self.mean)
        print('Dataset images std', self.std)
        self.curriculum_config = None
        self.training_info = None

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(path):
        with Image.open(path) as pil_img:
            img = np.array(pil_img)
            ## grayscale images
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=2)
        return img

    @staticmethod
    def load_samples(paths_and_sets, load_in_memory=True, labels_name='labels.pkl', self_training=False):
        """
        Load images and labels
        """
        samples = list()
        for path_and_set in paths_and_sets:
            path = path_and_set["path"]
            set_name = path_and_set["set_name"]
            with open(os.path.join(path, labels_name), "rb") as f:
                info = pkl.load(f)
                gt = info["ground_truth"][set_name]
                if 'MULTI_LINGUAL' in path:
                    sorted_keys = gt.keys()
                else:
                    sorted_keys = natural_sort(gt.keys())

                if ('MULTI_LINGUAL' in path or 'EXOPOPP' in path) and "named_entities" in path and set_name == 'train' and not self_training:
                    iam_ner_samples = []

                for filename in sorted_keys:
                    name = os.path.join(os.path.basename(path), set_name, filename)
                    full_path = os.path.join(path, set_name, filename)
                    if not Path(full_path).exists():
                        print('Image path does not exist:', full_path)
                    else:
                        if isinstance(gt[filename], dict) and "text" in gt[filename]:
                            label = gt[filename]["text"]
                        else:
                            label = gt[filename]
                        samples.append({
                            "name": name,
                            "label": label,
                            "unchanged_label": label,
                            "path": full_path,
                            "nb_cols": 1 if "nb_cols" not in gt[filename] else gt[filename]["nb_cols"]
                        })
                        if load_in_memory:
                            samples[-1]["img"] = GenericDataset.load_image(full_path)

                        if 'MULTI_LINGUAL' in path:
                            samples[-1]['dataset'] = gt[filename]['dataset']
                        elif 'EXOPOPP' in path and 'named_entities' in path:
                            samples[-1]['dataset'] = gt[filename].get('dataset','EXOPOPP')
                        elif 'IAM' in path and 'named_entities' in path:
                            samples[-1]['dataset'] = 'IAM_NER'

                        if type(gt[filename]) is dict:
                            if "lines" in gt[filename].keys():
                                samples[-1]["raw_line_seg_label"] = gt[filename]["lines"]
                            if "paragraphs" in gt[filename].keys():
                                samples[-1]["paragraphs_label"] = gt[filename]["paragraphs"]
                            if "pages" in gt[filename].keys():
                                samples[-1]["pages_label"] = gt[filename]["pages"]

                        if ('MULTI_LINGUAL' in path or 'EXOPOPP' in path) and "named_entities" in path and set_name == 'train' and 'dataset' in gt[filename] and gt[filename]['dataset'] in ['IAM', 'EXOPOPP'] and not self_training:
                            iam_ner_samples.append(copy.copy(samples[-1]))
                            iam_ner_samples[-1]["label"] = gt[filename]["text-ner"]
                            iam_ner_samples[-1]["unchanged_label"] = gt[filename]["text-ner"]
                            if gt[filename]['dataset'] == 'IAM':
                                iam_ner_samples[-1]['dataset'] = 'IAM_NER'
                            elif gt[filename]['dataset'] == 'EXOPOPP':
                                iam_ner_samples[-1]['dataset'] = 'EXOPOPP_NER'

        if ('MULTI_LINGUAL' in path or 'EXOPOPP' in path) and "named_entities" in path and set_name == 'train' and not self_training:
            samples.extend(iam_ner_samples)

        return samples

    @staticmethod
    def load_other_samples(dataset_path, labels_name='labels.pkl', batch_size=1, config=None):
        """
        Load labels from another dataset than the one used for training.
        This is useful for synthetical data generation when the model overfits the labels.

        dataset_path: The path to the dataset.
        labels_name: The name of the labels file, defaults to 'labels.pkl'
        batch_size: The batch size, defaults to 1

        Returns:
            The loaded samples
        """
        char_rpl_dict = {
            '!':'',
            '%':'',
            '*':'',
            '+':'',
            '=':'',
            '>':'',
            '@':'',
            '\\':'',
            '{':'',
            '}':'',
            '²':'',
            'À':'A',
            'Ç':'C',
            'É':'E',
            'Ê':'E',
            'Ô':'O',
            'œ':'oe',
            '€':''
        } # dictionary to replace special characters
        samples = list()
        if 'huggingface' in dataset_path: # text corpus from huggingface
            if ":" in dataset_path:
                samples = Wikipedia(cache_path=dataset_path.split(':')[1])
            else:
                samples = Wikipedia()
            samples = DataLoader(
                        samples,
                        batch_size=batch_size,
                        shuffle=True, #set to True to have the data reshuffled at every epoch
                        drop_last=True,
                        pin_memory=True,
                        generator=None
            )
        elif isinstance(dataset_path,list):
            datasets = []
            for path in dataset_path:
                if path.endswith('.pkl'):
                    with open(path, "rb") as f:
                        samples = pkl.load(f)
                else:
                    with open(path) as f: # plain text file
                        samples = f.readlines()

                    random.shuffle(samples)

                if isinstance(samples, dict):
                    samples = samples.values()
                samples = [{'source': Path(path).stem, 'text': sample} for sample in samples]
                datasets.append(samples)

            samples = torch.utils.data.ConcatDataset(datasets)

            weights = []
            for i in range(len(datasets)):
                weight = 1/len(datasets[i])
                if config and 'read_sampling_weights' in config:
                    weight = weight * list(config['read_sampling_weights'].values())[i]
                    # customized dataset proportions
                weights.extend([weight]*len(datasets[i])) # each dataset has the same weight

            weighted_sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(samples),
            )

            samples = DataLoader(
                    samples,
                    batch_size=batch_size,
                    drop_last=True,
                    sampler=weighted_sampler,
                    pin_memory=True,
                    generator=None
            )

        elif 'wikipedia' in dataset_path:
            with open(dataset_path) as f: # plain text file
                samples = f.readlines()

            random.shuffle(samples)
            samples = DataLoader(
                        samples,
                        batch_size=1,
                        shuffle=True,
                        drop_last=True,
                        pin_memory=True,
                        generator=None
                )
        else:
            with open(os.path.join(dataset_path, labels_name), "rb") as f:
                info = pkl.load(f)
                gt = info["ground_truth"]['train']
                for filename in natural_sort(gt.keys()):
                    name = os.path.join(os.path.basename(dataset_path), 'train', filename)
                    if isinstance(gt[filename], dict) and "text" in gt[filename]:
                        label = gt[filename]
                    else:
                        label = gt[filename]
                    for char_to_rpl, rpl_char in char_rpl_dict.items():
                        for pg_dict in label['paragraphs']:
                            pg_dict['label']=pg_dict['label'].replace(char_to_rpl,rpl_char)
                        label['text']=label['text'].replace(char_to_rpl,rpl_char)
                    samples.append({
                        "name": name,
                        "pages_label": [label],
                    })
        return samples

    @staticmethod
    def load_multi_samples(dataset_dict, config=None):
        """
        Load labels from several external corpus to be used for synthetic data
        to train a multidataset model.
        """
        loaded_corpus = {}

        multi_samples = {}
        for dataset_name, wiki_path in dataset_dict.items():
            if isinstance(wiki_path,list):
                alread_loaded = wiki_path[0] in loaded_corpus
            else:
                alread_loaded = wiki_path in loaded_corpus
            if not alread_loaded:
                if isinstance(wiki_path,list): # it is the case when using multiple corpus for READ
                    datasets = []
                    for path in wiki_path:
                        if path.endswith('.pkl'):
                            with open(path, "rb") as f:
                                samples = pkl.load(f)
                        else:
                            with open(path) as f:
                                samples = f.readlines()

                            random.shuffle(samples)

                        if isinstance(samples, dict):
                            samples = samples.values()
                        samples = [{'source': Path(path).stem, 'text': sample} for sample in samples]
                        datasets.append(samples)

                    samples = torch.utils.data.ConcatDataset(datasets)
                    weights = []
                    for i in range(len(datasets)):
                        weight = 1/len(datasets[i])
                        if config:
                            if 'read_sampling_weights' in config:
                                weight = weight * list(config['read_sampling_weights'].values())[i]
                                # customized dataset proportions when mixing several corpus for the READ dataset
                        weights.extend([weight]*len(datasets[i]))

                    weighted_sampler = torch.utils.data.WeightedRandomSampler(
                        weights=weights,
                        num_samples=len(samples),
                    )

                    multi_samples[dataset_name] = DataLoader(
                            samples,
                            batch_size=1,
                            drop_last=True,
                            sampler=weighted_sampler,
                            pin_memory=True,
                            generator=None
                    )
                else:
                    loaded_corpus[wiki_path] = dataset_name
                    if 'sentences' in wiki_path:
                        with open(wiki_path) as f:
                            samples = f.readlines()

                        random.shuffle(samples)
                    else:
                        if ':' in wiki_path:
                            wiki_path = wiki_path.split(':')[1]
                        samples = Wikipedia(cache_path=wiki_path)

                    multi_samples[dataset_name] = DataLoader(
                                samples,
                                batch_size=1,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                generator=None
                    )
            else:
                multi_samples[dataset_name] = multi_samples[loaded_corpus[wiki_path]]
        return multi_samples

    def apply_preprocessing(self, preprocessings):
        for i in range(len(self.samples)):
            self.samples[i] = apply_preprocessing(self.samples[i], preprocessings)

    def compute_std_mean(self):
        """
        Compute cumulated variance and mean of whole dataset
        """
        if self.mean is not None and self.std is not None:
            return self.mean, self.std
        if not self.load_in_memory:
            sample = self.samples[0].copy()
            sample["img"] = self.get_sample_img(0)
            img = apply_preprocessing(sample, self.params["config"]["preprocessings"])["img"]
        else:
            img = self.get_sample_img(0)
        _, _, c = img.shape
        dataset_sum = np.zeros((c,))
        nb_pixels = 0

        for i in range(len(self.samples)):
            if not self.load_in_memory:
                sample = self.samples[i].copy()
                sample["img"] = self.get_sample_img(i)
                img = apply_preprocessing(sample, self.params["config"]["preprocessings"])["img"]
            else:
                img = self.get_sample_img(i)
            try:
                dataset_sum += np.sum(img, axis=(0, 1))
            except Exception as e:
                print(e)
                print(self.samples[i]["name"])
            nb_pixels += np.prod(img.shape[:2])
        mean = dataset_sum / nb_pixels
        diff = np.zeros((c,))
        for i in range(len(self.samples)):
            if not self.load_in_memory:
                sample = self.samples[i].copy()
                sample["img"] = self.get_sample_img(i)
                img = apply_preprocessing(sample, self.params["config"]["preprocessings"])["img"]
            else:
                img = self.get_sample_img(i)
            diff += [np.sum((img[:, :, k] - mean[k]) ** 2) for k in range(c)]
        std = np.sqrt(diff / nb_pixels)

        self.mean = mean
        self.std = std
        return mean, std

    def apply_data_augmentation(self, img):
        """
        Apply data augmentation strategy on the input image
        """
        augs = [self.params["config"][key] if key in self.params["config"].keys() else None for key in ["augmentation", "valid_augmentation", "test_augmentation"]]
        for aug, set_name in zip(augs, ["train", "valid", "test"]):
            if aug and self.set_name == set_name:
                return apply_data_augmentation(img, aug)
        return img, list()

    def get_sample_img(self, i):
        """
        Get image by index
        """
        if self.load_in_memory:
            return self.samples[i]["img"]
        else:
            return GenericDataset.load_image(self.samples[i]["path"])

    def denormalize(self, img):
        """
        Get original image, before normalization
        """
        return img * self.std + self.mean

def preprocess_img(sample, dataset_params):
    if "normalize" in dataset_params["config"] and dataset_params["config"]["normalize"]:
        sample["img"] = (sample["img"] - dataset_params["config"]['mean']) / dataset_params["config"]['std']

    sample["img_shape"] = sample["img"].shape
    sample["img_reduced_shape"] = np.ceil(sample["img_shape"] / dataset_params['reduce_dims_factor']).astype(
        int
    )

    return sample

def load_unlabelled_img_batch(batch_img_list, dataset_params):
    """
    Load a batch of unlabelled images in order to perform inference on it.
    """
    batch_data = [load_unlabelled_img(img_path) for img_path in batch_img_list]

    batch_data = [apply_preprocessing(sample, dataset_params["config"]["preprocessings"]) for sample in batch_data]

    batch_data = [preprocess_img(batch_img, dataset_params) for batch_img in batch_data]

    batch_dict = {}
    imgs = [batch_data[i]["img"] for i in range(len(batch_data))]
    batch_dict['imgs_reduced_shape'] = [batch_data[i]["img_reduced_shape"] for i in range(len(batch_data))]
    batch_dict['names'] = [batch_data[i]["name"] for i in range(len(batch_data))]

    img_padding_value = dataset_params["config"]["padding_value"] if "padding_value" in dataset_params ["config"] else 0
    padding_mode = dataset_params["config"]["padding_mode"] if "padding_mode" in dataset_params["config"] else "br"
    imgs = pad_images(imgs, padding_value=img_padding_value, padding_mode=padding_mode)
    batch_dict['imgs'] = torch.tensor(imgs).float().permute(0, 3, 1, 2)

    return batch_dict

def load_unlabelled_img(full_path):
    return {'img': GenericDataset.load_image(full_path), 'name': full_path}

def apply_preprocessing(sample, preprocessings):
    """
    Apply preprocessings on each sample
    """
    resize_ratio = [1, 1]
    img = sample["img"]
    for preprocessing in preprocessings:

        if preprocessing["type"] == "dpi": # use skimage instead
            ratio = preprocessing["target"] / preprocessing["source"]
            temp_img = img
            h, w, c = temp_img.shape

            if temp_img.shape[2] == 2:
                channel_axis = None
            else:
                channel_axis = 2
            temp_img = transform_skimage.rescale(temp_img,ratio,3,anti_aliasing=True,preserve_range=True,channel_axis=channel_axis)

            if len(temp_img.shape) == 2:
                temp_img = np.expand_dims(temp_img, axis=2)
            img = temp_img

            resize_ratio = [ratio, ratio]

        if preprocessing["type"] == "to_grayscaled":
            temp_img = img
            if len(temp_img.shape) == 2:
                temp_img = np.expand_dims(temp_img, axis=2)
            h, w, c = temp_img.shape
            if c == 3:
                img = np.expand_dims(
                    0.2125 * temp_img[:, :, 0] + 0.7154 * temp_img[:, :, 1] + 0.0721 * temp_img[:, :, 2],
                    axis=2).astype(np.uint8)

        if preprocessing["type"] == "to_RGB":
            temp_img = img
            h, w, c = temp_img.shape
            if c == 1:
                img = np.concatenate([temp_img, temp_img, temp_img], axis=2)

        if preprocessing["type"] == "resize":
            keep_ratio = preprocessing["keep_ratio"]
            max_h, max_w = preprocessing["max_height"], preprocessing["max_width"]
            temp_img = img
            h, w, c = temp_img.shape

            ratio_h = max_h / h if max_h else 1
            ratio_w = max_w / w if max_w else 1
            if keep_ratio:
                ratio_h = ratio_w = min(ratio_w, ratio_h)
            new_h = min(max_h, int(h * ratio_h))
            new_w = min(max_w, int(w * ratio_w))
            temp_img = cv2.resize(temp_img, (new_w, new_h))
            if len(temp_img.shape) == 2:
                temp_img = np.expand_dims(temp_img, axis=2)

            img = temp_img
            resize_ratio = [ratio_h, ratio_w]

        if preprocessing["type"] == "fixed_height":
            new_h = preprocessing["height"]
            temp_img = img
            h, w, c = temp_img.shape
            ratio = new_h / h
            temp_img = cv2.resize(temp_img, (int(w*ratio), new_h))
            if len(temp_img.shape) == 2:
                temp_img = np.expand_dims(temp_img, axis=2)
            img = temp_img
            resize_ratio = [ratio, ratio]

    if resize_ratio != [1, 1] and "raw_line_seg_label" in sample:
        for li in range(len(sample["raw_line_seg_label"])):
            for side, ratio in zip((["bottom", "top"], ["right", "left"]), resize_ratio):
                for s in side:
                    sample["raw_line_seg_label"][li][s] = sample["raw_line_seg_label"][li][s] * ratio

    sample["img"] = img
    sample["resize_ratio"] = resize_ratio
    return sample
