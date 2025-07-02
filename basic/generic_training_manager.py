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
This module contains the implementation of the GenericTrainingManager class.
The GenericTrainingManager is responsible for managing the training process of a generic model.
It handles the initialization of hardware configuration, loading of datasets, and model weights.
It also provides methods for creating output folders, initializing paths, and loading the dataset.
"""

import copy
import json
import os
import sys
from datetime import date
from time import time

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.init import kaiming_uniform_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.file_utils import ModelOutput

import wandb
from basic.metric_manager import MetricManager
from basic.scheduler import DropoutScheduler
from basic.utils import EarlyStopper, NpEncoder
from OCR.ocr_utils import LM_ind_to_str_subwords


class GenericTrainingManager:
    """
    The GenericTrainingManager class manages the training process of a generic model.
    """

    def __init__(self, params):
        """
        Initializes a new instance of the GenericTrainingManager class.

        Args:
            params (dict): The parameters for the training process.
        """
        self.type = None
        self.params = params
        self.dropout_scheduler = None
        self.models = {}
        self.begin_time = None
        self.dataset = None
        self.dataset_name = list(self.params["dataset_params"]["datasets"].values())[0]
        self.paths = None
        self.latest_step = 0
        self.latest_epoch = -1
        self.latest_batch = 0
        self.total_batch = 0
        self.grad_acc_step = 0
        self.latest_train_metrics = dict()
        self.latest_valid_metrics = dict()
        self.curriculum_info = dict()
        self.curriculum_info["latest_valid_metrics"] = dict()
        self.phase = None
        self.max_mem_usage_by_epoch = list()
        self.losses = list()
        self.lr_values = list()

        self.scaler = None

        self.optimizers = dict()
        self.optimizers_named_params_by_group = dict()
        self.lr_schedulers = dict()
        self.best = None
        self.writer = None
        self.metric_manager = dict()

        self.init_hardware_config()
        self.init_paths()
        if not params.get('inference_mode', False):
            self.load_dataset()
        self.params["model_params"]["use_amp"] = self.params["training_params"]["use_amp"]

        if 'MULTI' in self.dataset_name and not params.get('inference_mode', False):
            self.bests = {valid_set_name: None for valid_set_name in self.dataset.valid_datasets}

        if hasattr(self.dataset,'train_dataset') and hasattr(self.dataset.train_dataset,'subword_tokenizer'):
            self.params['model_params']['subword_tokenizer'] = self.dataset.train_dataset.subword_tokenizer
            self.params['model_params']['subword_tokenizer'].__name__ = 'subword_tokenizer'

    def init_paths(self):
        """
        Create output folders for results and checkpoints.
        """
        output_path = os.path.join("outputs", self.params["training_params"]["output_folder"])
        os.makedirs(output_path, exist_ok=True)
        checkpoints_path = os.path.join(output_path, "checkpoints")
        os.makedirs(checkpoints_path, exist_ok=True)
        results_path = os.path.join(output_path, "results")
        os.makedirs(results_path, exist_ok=True)

        self.paths = {
            "results": results_path,
            "checkpoints": checkpoints_path,
            "output_folder": output_path
        }

    def load_dataset(self):
        """
        Load datasets, data samplers, and data loaders.
        """
        self.params["dataset_params"]["batch_size"] = self.params["training_params"]["batch_size"]
        if "valid_batch_size" in self.params["training_params"]:
            self.params["dataset_params"]["valid_batch_size"] = self.params["training_params"]["valid_batch_size"]
        if "test_batch_size" in self.params["training_params"]:
            self.params["dataset_params"]["test_batch_size"] = self.params["training_params"]["test_batch_size"]
        self.params["dataset_params"]["num_gpu"] = self.params["training_params"]["nb_gpu"]
        self.params["dataset_params"]["worker_per_gpu"] = 4 if "worker_per_gpu" not in self.params["dataset_params"] else self.params["dataset_params"]["worker_per_gpu"]
        self.dataset = self.params["dataset_params"]["dataset_manager"](self.params["dataset_params"])
        self.dataset.load_datasets()
        self.dataset.load_samplers()
        self.dataset.load_dataloaders()

    def init_hardware_config(self):
        """
        Initialize the hardware configuration.
        """
        # Debug mode
        if self.params["training_params"]["force_cpu"]:
            self.params["training_params"]["use_amp"] = False
        # Manage Distributed Data Parallel & GPU usage
        self.manual_seed = 1111 if "manual_seed" not in self.params["training_params"].keys() else \
        self.params["training_params"]["manual_seed"]
        if self.params["training_params"]["force_cpu"]:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
        self.params["model_params"]["device"] = self.device.type
        print("##################")
        print("Available GPUS: {}".format(self.params["training_params"]["nb_gpu"]))
        for i in range(self.params["training_params"]["nb_gpu"]):
            print("Rank {}: {} {}".format(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i)))
        print("##################")
        print("Local GPU:")
        if self.device.type != "cpu":
            print("{} {}".format(torch.cuda.get_device_name(), torch.cuda.get_device_properties(self.device)))
        else:
            print("WORKING ON CPU !\n")
        print("##################")

    def load_model(self, reset_optimizer=False, strict=True):
        """
        Load model weights from scratch or from checkpoints.
        """
        # Instantiate Model
        for i, model_name in enumerate(self.params["model_params"]["models"].keys()):
            self.models[model_name] = self.params["model_params"]["models"][model_name](self.params["model_params"])
            self.models[model_name].to(self.device)  # To GPU or CPU

        # Handle curriculum dropout
        if "dropout_scheduler" in self.params["model_params"]:
            func = self.params["model_params"]["dropout_scheduler"]["function"]
            T = self.params["model_params"]["dropout_scheduler"]["T"]
            self.dropout_scheduler = DropoutScheduler(self.models, func, T)

        self.scaler = GradScaler(enabled=self.params["training_params"]["use_amp"])

        # Check if checkpoint exists
        checkpoint = self.get_checkpoint()
        if checkpoint is not None:
            self.load_existing_model(checkpoint, strict=strict)
        else:
            self.init_new_model()

        if not self.params.get('inference_mode', False):
            self.load_optimizers(checkpoint, reset_optimizer=reset_optimizer)

        print("LOADED EPOCH: {}\n".format(self.latest_epoch), flush=True)

    def get_checkpoint(self):
        """
        Seek if checkpoint exist, return None otherwise
        """
        for filename in os.listdir(self.paths["checkpoints"]):
            if '_NER' in filename:
                if self.params["training_params"]["load_epoch"] == filename.split('_NER')[0]+"_NER":
                    return torch.load(os.path.join(self.paths["checkpoints"], filename))
            elif self.params["training_params"]["load_epoch"] in filename:
                return torch.load(os.path.join(self.paths["checkpoints"], filename))
        return None

    def load_existing_model(self, checkpoint, strict=True):
        """
        Load information and weights from previous training
        """
        self.load_save_info(checkpoint)
        self.latest_epoch = checkpoint["epoch"]
        if "step" in checkpoint:
            self.latest_step = checkpoint["step"]
        if "best" in checkpoint:
            self.best = checkpoint["best"]

        if 'MULTI' in self.dataset_name and "bests" in checkpoint:
            self.bests = checkpoint["bests"]

        if not self.params.get('inference_mode', False):
            if "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if "dropout_scheduler" in checkpoint:
                self.dropout_scheduler.step_num = checkpoint["dropout_scheduler"]['step_num']
                self.dropout_scheduler.update_dropout_rate()

            if 'training_info' in checkpoint:
                self.dataset.train_dataset.training_info = checkpoint['training_info'] # Load model weights from past training

        for model_name in self.models.keys():
            self.models[model_name].load_state_dict(checkpoint["{}_state_dict".format(model_name)], strict=strict)

    def init_new_model(self):
        """
        Initialize a new model.

        This method initializes a new model by performing the following steps:
        1. Specific weights initialization if exists.
        2. Handle transfer learning instructions.

        Note: This method assumes that the `self.models` attribute is a dictionary
        containing the models to be initialized.

        Raises:
            RuntimeError: If an error occurs while loading the pretrained weights.
        """
        # Specific weights initialization if exists
        for model_name in self.models.keys():
            try:
                self.models[model_name].init_weights()
            except:
                pass

        # Handle transfer learning instructions
        if self.params["model_params"]["transfer_learning"]:
            # Iterates over models
            for model_name in self.params["model_params"]["transfer_learning"].keys():
                if self.params["model_params"]["transfer_learning"][model_name]:
                    state_dict_name, path, learnable, strict = self.params["model_params"]["transfer_learning"][model_name]
                    # Loading pretrained weights file
                    checkpoint = torch.load(path)
                    if model_name == 'decoder' and "bart_path" in self.params['model_params'] and 'decoder.model.lm_head.weight' in checkpoint['decoder_state_dict']:
                        # The goal is to handle properly the case where the vocabulary size of the
                        # pre-trained model is different from the one of the current model.
                        vocab_size = self.models[model_name].state_dict()['decoder.model.lm_head.weight'].shape[0]
                        same_vocab_size = vocab_size == checkpoint['decoder_state_dict']['decoder.model.lm_head.weight'].shape[0]
                        same_pos_emb_size = self.models['decoder'].state_dict()['decoder.model.model.decoder.embed_positions.weight'].shape[0] == checkpoint['decoder_state_dict']['decoder.model.model.decoder.embed_positions.weight'].shape[0]
                        print('BART transfer learning')
                        print('Same vocab size:', same_vocab_size)
                        print('Same positional embedding size:', same_pos_emb_size)
                        bart_state_dict = checkpoint['decoder_state_dict']
                        new_bart_state_dict = self.models['decoder'].state_dict()
                        if not same_vocab_size or not same_pos_emb_size: # the position embedding or decision layer does not have the same size
                            if not same_vocab_size:
                                if 'vocab' in checkpoint:
                                    max_common_index = 0
                                    max_common_index_found = False
                                    corresp_dict = {}
                                    new_vocab = self.dataset.train_dataset.subword_tokenizer.get_vocab()
                                    # we search for tokens with the same index in the two vocabularies
                                    for token, index in new_vocab.items():
                                        if token in checkpoint['vocab'] and index == checkpoint['vocab'][token] and not max_common_index_found:
                                            max_common_index = index
                                        else:
                                            max_common_index_found = True
                                            if token in checkpoint['vocab']:
                                                corresp_dict[index] = checkpoint['vocab'][token]

                            for x in new_bart_state_dict:
                                if not same_pos_emb_size and x.endswith("embed_positions.weight"):
                                    new_bart_state_dict[x] = torch.nn.Parameter(
                                        self.models[model_name].decoder.resize_bart_abs_pos_emb(
                                            bart_state_dict[x],
                                            new_bart_state_dict['decoder.model.model.decoder.embed_positions.weight'].shape[0]
                                        )
                                    )
                                elif not same_vocab_size and (x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight")):
                                    # the layers that depend on the vocabulary size are the embedding layer and the decision layer
                                    if "replace_dict_ind_to_str" in self.params['dataset_params']['config']:
                                        new_vocab_size = new_bart_state_dict['decoder.model.model.decoder.embed_tokens.weight'].shape[0]
                                        old_indexes = [old_index for new_index, old_index in self.params['dataset_params']['config']["replace_dict_ind_to_str"].items() if new_index < new_vocab_size]
                                        new_bart_state_dict['decoder.model.model.decoder.embed_tokens.weight'] = bart_state_dict['decoder.model.model.decoder.embed_tokens.weight'][old_indexes,:]
                                    elif not 'vocab' in checkpoint:
                                        if vocab_size < bart_state_dict[x].shape[0]:
                                            new_bart_state_dict[x] = bart_state_dict[x][: vocab_size, :]
                                        elif vocab_size > bart_state_dict[x].shape[0]:
                                            new_bart_state_dict[x][: bart_state_dict[x].shape[0], :] = bart_state_dict[x]
                                    else:
                                        new_bart_state_dict[x][: max_common_index, :] = bart_state_dict[x][: max_common_index, :]

                                        for new_index, old_index in corresp_dict.items():
                                            new_bart_state_dict[x][new_index] = bart_state_dict[x][old_index]
                                else:
                                    new_bart_state_dict[x] = bart_state_dict[x]
                        else:
                            new_bart_state_dict = bart_state_dict
                        self.models['decoder'].load_state_dict(new_bart_state_dict)
                        print("transfered weights for {}".format(state_dict_name), flush=True)
                    else:
                        try:
                            # Load pretrained weights for model
                            self.models[model_name].load_state_dict(checkpoint["{}_state_dict".format(state_dict_name)], strict=strict)
                            print("transfered weights for {}".format(state_dict_name), flush=True)
                        except RuntimeError as e:
                            print(e, flush=True)
                            # if error, try to load each parts of the model (useful if only few layers are different)
                            if 'init_blocks.0.conv1.weight' in checkpoint["{}_state_dict".format(state_dict_name)]:
                                weights_new_model_first_layer = self.models['encoder'].init_blocks[0].conv1.weight
                                weights_checkpoint_first_layer = checkpoint["{}_state_dict".format(state_dict_name)]['init_blocks.0.conv1.weight']
                                if weights_new_model_first_layer.shape != weights_checkpoint_first_layer.shape and weights_new_model_first_layer[:,0,:,:].shape == weights_checkpoint_first_layer[:,0,:,:].shape:
                                    # if the error is due to the number of channels, we can load the weights channel by channel
                                    if weights_new_model_first_layer.shape[1] == 1 and weights_checkpoint_first_layer.shape[1] == 3:
                                        # RGB to grayscale
                                        self.models[model_name].load_state_dict(
                                            {'init_blocks.0.conv1.weight': checkpoint["{}_state_dict".format(state_dict_name)]['init_blocks.0.conv1.weight'][:,1:2,:,:]}, strict=False
                                        )
                                        self.models[model_name].load_state_dict(
                                            {'init_blocks.0.conv1.bias': checkpoint["{}_state_dict".format(state_dict_name)]['init_blocks.0.conv1.bias']}, strict=False
                                        )
                                    elif weights_new_model_first_layer.shape[1] == 3 and weights_checkpoint_first_layer.shape[1] == 1:
                                        # grayscale to RGB
                                        ckpt_weights = checkpoint["{}_state_dict".format(state_dict_name)]['init_blocks.0.conv1.weight'].repeat(1,3,1,1)
                                        self.models[model_name].load_state_dict(
                                            {'init_blocks.0.conv1.weight': ckpt_weights}, strict=False
                                        )

                            for key in checkpoint["{}_state_dict".format(state_dict_name)].keys():
                                try:
                                    # for pre-training of decision layer
                                    if "end_conv" in key and "transfered_charset" in self.params["model_params"]:
                                        self.adapt_decision_layer_to_old_charset(model_name, key, checkpoint, state_dict_name)
                                    else:
                                        self.models[model_name].load_state_dict(
                                            {key: checkpoint["{}_state_dict".format(state_dict_name)][key]}, strict=False)
                                except RuntimeError as e:
                                    print(e, flush=True)
                    # Set parameters no trainable
                    if not learnable:
                        self.set_model_learnable(self.models[model_name].module, False)

    def adapt_decision_layer_to_old_charset(self, model_name, key, checkpoint, state_dict_name):
        """
        Transfer learning of the decision learning in case of close charsets between pre-training and training
        """
        pretrained_chars = list()
        weights = checkpoint["{}_state_dict".format(state_dict_name)][key]
        new_size = list(weights.size())
        new_size[0] = len(self.dataset.charset) + self.params["model_params"]["additional_tokens"]

        new_weights = torch.zeros(new_size, device=weights.device, dtype=weights.dtype)
        if not "bias" in key:
            kaiming_uniform_(new_weights, nonlinearity="relu")

        # initialization of the new weights:
        old_charset = checkpoint["charset"] if "charset" in checkpoint else self.params["model_params"]["old_charset"]

        for i, c in enumerate(self.dataset.charset):
            if c in old_charset:
                new_weights[i] = weights[old_charset.index(c)]
                pretrained_chars.append(c)

        if "transfered_charset_last_is_ctc_blank" in self.params["model_params"] and self.params["model_params"]["transfered_charset_last_is_ctc_blank"]:
            new_weights[-1] = weights[-1]
            pretrained_chars.append("<blank>")
        checkpoint["{}_state_dict".format(state_dict_name)][key] = new_weights

        self.models[model_name].load_state_dict({key: checkpoint["{}_state_dict".format(state_dict_name)][key]}, strict=False)


    def load_optimizers(self, checkpoint, reset_optimizer=False):
        """
        Load the optimizer of each model
        """
        for model_name in self.models.keys():
            new_params = dict()
            if checkpoint and "optimizer_named_params_{}".format(model_name) in checkpoint:
                self.optimizers_named_params_by_group[model_name] = checkpoint["optimizer_named_params_{}".format(model_name)]
                # for progressively growing models
                for name, param in self.models[model_name].named_parameters():
                    existing = False
                    for gr in self.optimizers_named_params_by_group[model_name]:
                        if name in gr:
                            gr[name] = param
                            existing = True
                            break
                    if not existing:
                        new_params.update({name: param})
            else:
                self.optimizers_named_params_by_group[model_name] = [dict(), ]
                self.optimizers_named_params_by_group[model_name][0].update(self.models[model_name].named_parameters())

            # Instantiate optimizer
            if not self.params["model_params"]['transfer_learning'] or (
                    self.params["model_params"]['transfer_learning'] and self.params["model_params"]['transfer_learning'][model_name]
                    and self.params["model_params"]['transfer_learning'][model_name][2]
                ):
                self.reset_optimizer(model_name)

            # Handle learning rate schedulers
            if "lr_schedulers" in self.params["training_params"] and self.params["training_params"]["lr_schedulers"]:
                key = "all" if "all" in self.params["training_params"]["lr_schedulers"] else model_name
                if key in self.params["training_params"]["lr_schedulers"]:
                    self.lr_schedulers[model_name] = self.params["training_params"]["lr_schedulers"][key]["class"]\
                        (self.optimizers[model_name], **self.params["training_params"]["lr_schedulers"][key]["args"])

            # Load optimizer state from past training
            if checkpoint and not reset_optimizer and model_name in self.optimizers:
                self.optimizers[model_name].load_state_dict(checkpoint["optimizer_{}_state_dict".format(model_name)])
                # Load optimizer scheduler config from past training if used
                if "lr_schedulers" in self.params["training_params"] and self.params["training_params"]["lr_schedulers"] \
                        and "lr_scheduler_{}_state_dict".format(model_name) in checkpoint.keys():
                    self.lr_schedulers[model_name].load_state_dict(checkpoint["lr_scheduler_{}_state_dict".format(model_name)])

            # for progressively growing models, keeping learning rate
            if checkpoint and new_params:
                self.optimizers_named_params_by_group[model_name].append(new_params)
                self.optimizers[model_name].add_param_group({"params": list(new_params.values())})

    @staticmethod
    def set_model_learnable(model, learnable=True):
        for p in list(model.parameters()):
            p.requires_grad = learnable

    def save_model(self, epoch, name, keep_weights=False):
        """
        Save model weights and training info for curriculum learning or learning rate for instance
        """
        to_del = []

        def should_be_deleted(weights_filename):
            return '_'.join(weights_filename.split('_')[:-1]) == name

        for filename in os.listdir(self.paths["checkpoints"]):
            if should_be_deleted(filename):
                to_del.append(os.path.join(self.paths["checkpoints"], filename))
        path = os.path.join(self.paths["checkpoints"], "{}_{}.pt".format(name, epoch))
        content = {
            'optimizers_named_params': self.optimizers_named_params_by_group,
            'epoch': epoch,
            'step': self.latest_step,
            "scaler_state_dict": self.scaler.state_dict(),
            'best': self.best,
            "charset": self.dataset.charset,
            'dropout_scheduler': {
                'step_num': self.dropout_scheduler.step_num
            },
            'training_info': self.dataset.train_dataset.training_info
        }
        if 'MULTI' in self.dataset_name:
            content['bests'] = self.bests
        for model_name in self.optimizers:
            content['optimizer_{}_state_dict'.format(model_name)] = self.optimizers[model_name].state_dict()
        for model_name in self.lr_schedulers:
            content["lr_scheduler_{}_state_dict".format(model_name)] = self.lr_schedulers[model_name].state_dict()
        content = self.add_save_info(content)
        for model_name in self.models.keys():
            content["{}_state_dict".format(model_name)] = self.models[model_name].state_dict()
        torch.save(content, path)
        if not keep_weights:
            for path_to_del in to_del:
                if path_to_del != path:
                    os.remove(path_to_del)

    def reset_optimizers(self):
        """
        Reset learning rate of all optimizers
        """
        for model_name in self.models.keys():
            self.reset_optimizer(model_name)

    def reset_optimizer(self, model_name):
        """
        Reset optimizer learning rate for given model
        """
        params = list(self.optimizers_named_params_by_group[model_name][0].values())
        key = "all" if "all" in self.params["training_params"]["optimizers"] else model_name
        self.optimizers[model_name] = self.params["training_params"]["optimizers"][key]["class"](params, **self.params["training_params"]["optimizers"][key]["args"])
        for i in range(1, len(self.optimizers_named_params_by_group[model_name])):
            self.optimizers[model_name].add_param_group({"params": list(self.optimizers_named_params_by_group[model_name][i].values())})

    def save_params(self):
        """
        Output text file containing a summary of all hyperparameters chosen for the training
        """
        def compute_nb_params(module):
            return sum([np.prod(p.size()) for p in list(module.parameters())])

        def class_to_str_dict(my_dict):
            for key in my_dict.keys():
                if callable(my_dict[key]):
                    my_dict[key] = my_dict[key].__name__
                elif isinstance(my_dict[key], np.ndarray):
                    my_dict[key] = my_dict[key].tolist()
                elif isinstance(my_dict[key], dict):
                    my_dict[key] = class_to_str_dict(my_dict[key])
            return my_dict

        path = os.path.join(self.paths["results"], "params")
        if os.path.isfile(path):
            return
        params = copy.deepcopy(self.params)
        params = class_to_str_dict(params)
        params["date"] = date.today().strftime("%d/%m/%Y")
        total_params = 0
        for model_name in self.models.keys():
            current_params = compute_nb_params(self.models[model_name])
            params["model_params"]["models"][model_name] = [params["model_params"]["models"][model_name], "{:,}".format(current_params)]
            total_params += current_params
        params["model_params"]["total_params"] = "{:,}".format(total_params)

        params["hardware"] = dict()
        if self.device != "cpu":
            for i in range(self.params["training_params"]["nb_gpu"]):
                params["hardware"][str(i)] = "{} {}".format(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i))
        else:
            params["hardware"]["0"] = "CPU"
        params["software"] = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)

    def backward_loss(self, loss, retain_graph=False):
        self.scaler.scale(loss).backward(retain_graph=retain_graph)

    def step_optimizers(self, names=None):
        for model_name in self.optimizers:
            if names and model_name not in names:
                continue
            if "gradient_clipping" in self.params["training_params"] and model_name in self.params["training_params"]["gradient_clipping"]["models"]:
                self.scaler.unscale_(self.optimizers[model_name])
                torch.nn.utils.clip_grad_norm_(
                    self.models[model_name].parameters(),
                    self.params["training_params"]["gradient_clipping"]["max"],
                    self.params["training_params"]["gradient_clipping"]["error_if_nonfinite"]
                )
            self.scaler.step(self.optimizers[model_name])
        self.scaler.update()
        self.latest_step += 1

    def zero_optimizers(self, set_to_none=True):
        for model_name in self.optimizers:
            self.zero_optimizer(model_name, set_to_none)

    def zero_optimizer(self, model_name, set_to_none=True):
        self.optimizers[model_name].zero_grad(set_to_none=set_to_none)

    def train(self):
        """
        Trains the model using the specified training parameters.

        This method performs the training process by iterating over epochs and mini-batches of data.
        It updates the models, computes metrics, and logs the results.

        Returns:
            None
        """
        # init tensorboard file and output param summary file
        self.writer = SummaryWriter(self.paths["results"])
        self.save_params()
        # init variables
        self.begin_time = time()
        focus_metric_name = self.params["training_params"]["focus_metric"]
        nb_epochs = self.params["training_params"]["max_nb_epochs"]
        interval_save_weights = self.params["training_params"]["interval_save_weights"]
        metric_names = self.params["training_params"]["train_metrics"]
        tagging_mode = self.params["dataset_params"]["tagging_mode"] if 'tagging_mode' in self.params["dataset_params"] else 'end_with-hierarchy'

        if 'early_stopping' in self.params["training_params"] and self.params["training_params"]['early_stopping']:
            early_stopper = EarlyStopper(patience=self.params["training_params"]['patience'], min_delta=self.params["training_params"]['delta'])

        display_values = None
        # init curriculum learning
        if "curriculum_learning" in self.params["training_params"].keys() and self.params["training_params"]["curriculum_learning"]:
            self.init_curriculum()

        self.total_training_metrics = {'edit_chars': [], 'nb_chars': []}
        if not self.dataset.train_dataset.training_info:
            self.dataset.train_dataset.training_info = {'epoch': 0, 'step': 0, 'previous-nb-lines':0}
        # perform epochs
        for num_epoch in range(self.latest_epoch+1, nb_epochs):
            self.dataset.train_dataset.training_info['epoch'] = self.latest_epoch
            self.dataset.train_dataset.training_info['step'] = self.latest_step
            self.phase = "train"
            # Check maximum training time stop condition
            if self.params["training_params"]["max_training_time"] and time() - self.begin_time > self.params["training_params"]["max_training_time"]:
                break
            # set models trainable
            for model_name in self.models.keys():
                self.models[model_name].train()
            self.latest_epoch = num_epoch
            if self.dataset.train_dataset.curriculum_config:
                self.dataset.train_dataset.curriculum_config["epoch"] = self.latest_epoch
            # init epoch metrics values
            datasets = [self.dataset.train_loader.dataset.name]
            if 'MULTI' in datasets[0]:
                datasets = self.params['dataset_params']['synth_eval_loaders']
            self.metric_manager["train"] = MetricManager(metric_names=metric_names, dataset_name=self.dataset_name, tagging_mode=tagging_mode, datasets=datasets)

            len_epoch = len(self.dataset.train_loader.dataset) if not self.params["training_params"].get('eval_inside_epoch', False) else self.params["training_params"]['eval_inside_epoch_interval']
            with tqdm(total=len_epoch) as pbar:
                pbar.set_description("EPOCH {}/{}".format(num_epoch, nb_epochs))
                # iterates over mini-batch data
                for ind_batch, batch_data in enumerate(self.dataset.train_loader):
                    self.latest_batch = ind_batch + 1
                    self.total_batch += 1

                    if self.params["training_params"].get('eval_inside_epoch', False) and ind_batch % self.params["training_params"]['eval_inside_epoch_interval'] == self.params["training_params"]['eval_inside_epoch_interval']-1:
                        break

                    batch_values = self.train_batch(batch_data, metric_names)

                    batch_values['names']=batch_data['names']
                    batch_metrics = self.metric_manager["train"].compute_metrics(batch_values, metric_names)
                    batch_metrics["names"] = batch_data["names"]
                    batch_metrics["ids"] = batch_data["ids"]

                    # Update learning rate via scheduler if one is used
                    if self.params["training_params"]["lr_schedulers"]:
                        for model_name in self.models:
                            key = "all" if "all" in self.params["training_params"]["lr_schedulers"] else model_name
                            if model_name in self.lr_schedulers and ind_batch % self.params["training_params"]["lr_schedulers"][key]["step_interval"] == 0:
                                self.lr_schedulers[model_name].step(len(batch_metrics["names"]))
                                if "lr" in metric_names:
                                    self.writer.add_scalar("lr_{}".format(model_name), self.lr_schedulers[model_name].lr, self.lr_schedulers[model_name].step_num)
                    # Update dropout scheduler if used
                    if self.dropout_scheduler:
                        self.dropout_scheduler.step(len(batch_metrics["names"]))
                        self.dropout_scheduler.update_dropout_rate()

                    # Add batch metrics values to epoch metrics values
                    self.metric_manager["train"].update_metrics(batch_metrics)
                    display_values = self.metric_manager["train"].get_display_values()
                    pbar.set_postfix(values=str(display_values))
                    pbar.update(len(batch_data["names"]))

            if 'use_wandb' in self.params["training_params"] and self.params["training_params"]['use_wandb']:
                wandb.log(display_values)

            self.total_training_metrics['edit_chars'] = (self.total_training_metrics['edit_chars'] + self.metric_manager["train"].epoch_metrics['edit_chars'])[-8000:]
            self.total_training_metrics['nb_chars'] = (self.total_training_metrics['nb_chars'] + self.metric_manager["train"].epoch_metrics['nb_chars'])[-8000:]
            # log metrics in tensorboard file
            for key in display_values.keys():
                self.writer.add_scalar('{}_{}'.format(self.params["dataset_params"]["train"]["name"], key), display_values[key], num_epoch)
            self.latest_train_metrics = display_values

            # evaluate and compute metrics for valid sets
            if self.params["training_params"]["eval_on_valid"] and num_epoch % self.params["training_params"]["eval_on_valid_interval"] == 0:
                for valid_set_name in self.dataset.valid_loaders.keys():
                    if self.params['dataset_params']['config']['synthetic_data'] and (
                        self.params['dataset_params']['config']['synthetic_data'].get('force_synth_eval', False) or (
                            'max_nb_lines' in self.params['dataset_params']['config']['synthetic_data'] and self.params['dataset_params']['config']['synthetic_data'] and self.dataset.train_dataset.get_syn_max_lines()<self.params['dataset_params']['config']['synthetic_data']['max_nb_lines']
                            )
                        ):
                        if 'synth_eval_loaders' in self.params['dataset_params']:
                            valid_set_name = valid_set_name.replace('valid','synth-eval')
                            source_dataset = valid_set_name.split('-synth-eval')[0]
                            if 'synth_eval_loaders' in self.params['dataset_params'] and source_dataset not in self.params['dataset_params']['synth_eval_loaders']:
                                break
                        else:
                            valid_set_name = valid_set_name.replace('valid','train')
                    # evaluate set and compute metrics
                    eval_values = self.evaluate(valid_set_name)

                    self.latest_valid_metrics = eval_values
                    # log valid metrics in tensorboard file
                    for key in eval_values.keys():
                        self.writer.add_scalar('{}_{}'.format(valid_set_name, key), eval_values[key], num_epoch)
                    if 'MULTI' in self.dataset_name:
                        expected_metric_value = self.params["training_params"]["expected_metric_value"]
                        if 'focus_metrics' in self.params["training_params"]:
                            focus_metric_name = self.params["training_params"]["focus_metrics"][valid_set_name.split('-')[0]]

                        if 'expected_metric_values' in self.params["training_params"]:
                            expected_metric_value = self.params["training_params"]["expected_metric_values"][valid_set_name.split('-')[0]]

                        if valid_set_name in self.params["training_params"]["set_name_focus_metrics"] and (self.bests[valid_set_name] is None or \
                                (eval_values[focus_metric_name] <= self.bests[valid_set_name] and expected_metric_value == "low") or\
                                (eval_values[focus_metric_name] >= self.bests[valid_set_name] and expected_metric_value == "high")):
                            weights_suffix = "best-" + valid_set_name.split('-')[0]
                            self.save_model(epoch=num_epoch, name=weights_suffix)
                            self.bests[valid_set_name] = eval_values[focus_metric_name]
                    else:
                        if valid_set_name == self.params["training_params"]["set_name_focus_metric"] and (self.best is None or \
                                (eval_values[focus_metric_name] <= self.best and self.params["training_params"]["expected_metric_value"] == "low") or\
                                (eval_values[focus_metric_name] >= self.best and self.params["training_params"]["expected_metric_value"] == "high")):
                            self.save_model(epoch=num_epoch, name="best")
                            self.best = eval_values[focus_metric_name]

                    if 'early_stopping' in self.params["training_params"] and self.params["training_params"]['early_stopping'] and early_stopper.early_stop(eval_values['cer']):
                        print(f"Early stopping activated after {early_stopper.counter} epochs")
                        return None

            ###### Handle curriculum learning update
            if self.dataset.train_dataset.curriculum_config:
                self.check_and_update_curriculum()

            if "curriculum_model" in self.params["model_params"] and self.params["model_params"]["curriculum_model"]:
                self.update_curriculum_model()

            # save model weights
            self.save_model(epoch=num_epoch, name="last")
            if interval_save_weights and num_epoch % interval_save_weights == 0:
                self.save_model(epoch=num_epoch, name="weigths", keep_weights=True)
            self.writer.flush()

    def evaluate(self, set_name, **kwargs):
        """
        Main loop for validation during training.

        Args:
            set_name (str): The name of the dataset to evaluate on.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the display values of the evaluated metrics.
        """
        self.phase = "eval"
        print(set_name)
        if ('train' in set_name or 'synth-eval' in set_name) and 'selftrain' not in set_name:
            # evaluate the model on the synthetic dataset
            loader = self.dataset.train_loader
            max_nb_imgs = 50
            if 'synth-eval' in set_name:
                loader.dataset.synth_dataset = set_name.split('-synth-eval')[0]
        else:
            loader = self.dataset.valid_loaders[set_name]
            max_nb_imgs = 0
        # Set models in eval mode
        for model_name in self.models.keys():
            self.models[model_name].eval()
        metric_names = copy.deepcopy(self.params["training_params"]["eval_metrics"])
        if not 'NER' in set_name and 'MULTI' in list(self.params['dataset_params']['datasets'].keys())[0]:
            for metric_to_remove in ['f1', 'stricter_f1']:
                if metric_to_remove in metric_names:
                    metric_names.remove(metric_to_remove)

        display_values = None

        tagging_mode = self.params["dataset_params"].get("tagging_mode",'end_with-hierarchy')
        if 'tagging_modes' in self.params["dataset_params"]:
            tagging_mode = self.params["dataset_params"]["tagging_modes"].get(set_name.split('-')[0], tagging_mode)

        # initialize epoch metrics
        self.metric_manager[set_name] = MetricManager(metric_names, dataset_name=self.dataset_name, tagging_mode=tagging_mode)


        with tqdm(total=max_nb_imgs if max_nb_imgs else len(loader.dataset)) as pbar:
            pbar.set_description("Evaluation E{}".format(self.latest_epoch))
            with torch.no_grad():
                # iterate over batch data
                for ind_batch, batch_data in enumerate(loader):
                    if max_nb_imgs and max_nb_imgs <= ind_batch*loader.batch_size:
                        break
                    self.latest_batch = ind_batch + 1
                    # eval batch data and compute metrics
                    batch_values = self.evaluate_batch(batch_data, metric_names)
                    batch_values['names']=batch_data['names']
                    batch_metrics = self.metric_manager[set_name].compute_metrics(batch_values, metric_names)
                    batch_metrics["names"] = batch_data["names"]
                    batch_metrics["ids"] = batch_data["ids"]

                    # add batch metrics to epoch metrics
                    self.metric_manager[set_name].update_metrics(batch_metrics)
                    display_values = self.metric_manager[set_name].get_display_values()

        if 'synth-eval' in set_name:
            loader.dataset.synth_dataset = None

        return display_values

    def predict(self, custom_name, sets_list, metric_names, output=False):
        """
        The main loop for evaluation outside of training.
        Perform prediction on the given dataset.

        Args:
            custom_name (str): The name of the custom dataset.
            sets_list (list): A list of split sets to be used for prediction.
            metric_names (list): A list of metric names to be computed.
            output (bool, optional): Whether to output the metrics values. Defaults to False.
        """
        self.phase = "predict"
        metric_names = metric_names.copy()
        self.dataset.generate_test_loader(custom_name, sets_list)
        loader = self.dataset.test_loaders[custom_name]
        # Set models in eval mode
        for model_name in self.models.keys():
            self.models[model_name].eval()

        if 'tagging_modes' in self.params["dataset_params"]:
            set_name = sets_list[0][1].split('-')[1]
            tagging_mode = self.params["dataset_params"]["tagging_modes"].get(set_name, 'after')
        else:
            tagging_mode = self.params["dataset_params"].get("tagging_mode",'after')
        pred_dict = {}

        if 'map_cer' in metric_names or 'loer' in metric_names:
            layout_type = custom_name.split('-')[-1]
        else:
            layout_type = None
        # initialize epoch metrics
        self.metric_manager[custom_name] = MetricManager(metric_names, self.dataset_name,tagging_mode=tagging_mode, layout_type=layout_type)

        with tqdm(total=len(loader.dataset)) as pbar:
            pbar.set_description("Prediction")
            with torch.no_grad():
                for ind_batch, batch_data in enumerate(loader):
                    # iterates over batch data
                    self.latest_batch = ind_batch + 1
                    # eval batch data and compute metrics
                    batch_values = self.evaluate_batch(batch_data, metric_names, custom_name)
                    batch_values['names']=batch_data['names']
                    img_pred = {name : pred for name,pred in zip(batch_values['names'], batch_values['str_x'])}
                    pred_dict = {**pred_dict, **img_pred}
                    batch_metrics = self.metric_manager[custom_name].compute_metrics(batch_values, metric_names)
                    batch_metrics["ids"] = batch_data["ids"]

                    # add batch metrics to epoch metrics
                    self.metric_manager[custom_name].update_metrics(batch_metrics)
                    display_values = self.metric_manager[custom_name].get_display_values()

                    pbar.set_postfix(values=str(display_values))
                    pbar.update(len(batch_data["names"]))

        self.dataset.remove_test_dataset(custom_name)
        # output metrics values if requested
        if output:
            if "pred" in metric_names:
                self.output_pred(custom_name)
            metrics = self.metric_manager[custom_name].get_display_values(output=True)
            path = os.path.join(self.paths["results"], "predict_{}_{}.txt".format(custom_name, self.latest_epoch))
            with open(path, "w") as f:
                for metric_name in metrics.keys():
                    f.write("{}: {}\n".format(metric_name, metrics[metric_name]))

            for metric_name in metrics.keys():
                if metric_name not in self.metric_manager[custom_name].epoch_metrics:
                    self.metric_manager[custom_name].epoch_metrics[metric_name] = metrics[metric_name]
            path = os.path.join(self.paths["results"], "predict_{}_{}.json".format(custom_name, self.latest_epoch))
            with open(path, "w") as f:
                json.dump(self.metric_manager[custom_name].epoch_metrics, f, ensure_ascii=False, cls=NpEncoder)

            path_pred = os.path.join(
                self.paths["results"],
                "{}_{}_{}.json".format("inference", custom_name, self.latest_epoch),
            )
            with open(path_pred, "w") as f:
                json.dump(pred_dict, f, sort_keys=True)

    def predict_one_unlabelled_image(self, batch_data, max_length=400, output_scores=False, start_token=0):
        """
        Predicts the labels for a batch of unlabelled images.

        Args:
            batch_data (dict): A dictionary containing the batch data, including the images.

        Returns:
            dict: A dictionary containing the predicted labels for each image in the batch.
        """
        """
        Evaluate a batch of data using the trained model.

        Args:
            batch_data (dict): A dictionary containing the batch data.
            metric_names (list): A list of metric names to compute.

        Returns:
            dict: A dictionary containing the evaluation results.
        """
        self.models["encoder"].eval()
        self.models["decoder"].eval()

        with torch.no_grad():
            with autocast(enabled=False):
                x = batch_data["imgs"].to(self.device)

                b = x.size(0)
                prediction_len = torch.zeros((b, ), dtype=torch.int, device=self.device)
                predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * start_token

                features = self.models["encoder"](x)

                pos_features = self.models["decoder"].features_updater.get_pos_features(features)
                last_hidden_state = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(0, 2, 1)

                if self.device.type != "cuda":
                    last_hidden_state = last_hidden_state.to(torch.float32)

                encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state, attentions=None)

                if len(encoder_outputs.last_hidden_state.size()) == 1:
                    encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(0)

                decoder_output = self.models["decoder"].decoder.model.generate(
                    decoder_input_ids=predicted_tokens,
                    encoder_outputs=encoder_outputs,
                    max_length=max_length,
                    early_stopping=True,
                    pad_token_id=self.models["decoder"].decoder.tokenizer.pad_token_id,
                    eos_token_id=self.models["decoder"].decoder.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[self.models["decoder"].decoder.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                    output_attentions=False,
                    output_scores=output_scores
                )
                predicted_tokens = decoder_output['sequences']
                predicted_tokens = predicted_tokens[:, 1:]

                prediction_len = predicted_tokens.shape[1]-torch.sum(torch.eq(predicted_tokens,1),axis=1)
                predicted_tokens = [predicted_tokens[i, :prediction_len[i]-1] for i in range(b)]

                if output_scores:
                    all_logits = decoder_output.scores # shape: (seq-len, batch-size, vocab-size)
                    confidence_scores = [
                        [
                            float(torch.softmax(logits[index_batch], dim=-1).max()) for logits in all_logits[1:]
                        ][:prediction_len[index_batch]-1] for index_batch in range(b)
                    ] # shape: (batch-size, seq-len)

                if 'config' in self.params['dataset_params']:
                    rpl_dict = self.params['dataset_params']['config'].get('replace_dict_ind_to_str', None)
                else:
                    rpl_dict = None
                str_x = [LM_ind_to_str_subwords(self.params['dataset_params']['subword_tokenizer'],t.detach().cpu().numpy(), rpl_dict=rpl_dict) for t in predicted_tokens]

                values = {
                    "str_x": str_x,
                }

                if output_scores:
                    values['confidence'] = confidence_scores

        return values

    def output_pred(self, name):
        path = os.path.join(self.paths["results"], "pred_{}_{}.txt".format(name, self.latest_epoch))
        pred = "\n".join(self.metric_manager[name].get("pred"))
        with open(path, "w") as f:
            f.write(pred)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    def train_batch(self, batch_data, metric_names):
        raise NotImplementedError

    def evaluate_batch(self, batch_data, metric_names, set_name=None):
        raise NotImplementedError

    def init_curriculum(self):
        raise NotImplementedError

    def update_curriculum(self):
        raise NotImplementedError

    def add_checkpoint_info(self, load_mode="last", **kwargs):
        for filename in os.listdir(self.paths["checkpoints"]):
            if load_mode in filename:
                checkpoint_path = os.path.join(self.paths["checkpoints"], filename)
                checkpoint = torch.load(checkpoint_path)
                for key in kwargs.keys():
                    checkpoint[key] = kwargs[key]
                torch.save(checkpoint, checkpoint_path)
            return
        self.save_model(self.latest_epoch, "last")

    def load_save_info(self, info_dict):
        """
        Load curriculum info from saved model info
        """
        if "curriculum_config" in info_dict.keys():
            self.dataset.train_dataset.curriculum_config = info_dict["curriculum_config"]

    def add_save_info(self, info_dict):
        """
        Add curriculum info to model info to be saved
        """
        info_dict["curriculum_config"] = self.dataset.train_dataset.curriculum_config
        return info_dict
