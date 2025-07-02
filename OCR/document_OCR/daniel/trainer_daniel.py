# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
import time
import json
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.nn.functional import softmax
from transformers.file_utils import ModelOutput
from tqdm import tqdm
from PIL import Image

from basic.utils import pad_sequences_1D_to_given_length
from OCR.ocr_manager import OCRManager
from OCR.ocr_utils import LM_ind_to_str, LM_ind_to_str_subwords


class Manager(OCRManager):
    """
    Manager class for OCR training.

    Args:
        params (dict): Parameters for the OCR training.

    Attributes:
        dataset (OCRDataset): The OCR dataset.
    """

    def __init__(self, params):
        super(Manager, self).__init__(params)

    def load_save_info(self, info_dict):
        if "curriculum_config" in info_dict.keys():
            if not self.params.get('inference_mode', False) and self.dataset.train_dataset is not None:
                self.dataset.train_dataset.curriculum_config = info_dict["curriculum_config"]

    def add_save_info(self, info_dict):
        info_dict["curriculum_config"] = self.dataset.train_dataset.curriculum_config
        return info_dict

    def get_init_hidden(self, batch_size):
        num_layers = 1
        hidden_size = self.params["model_params"]["enc_dim"]
        return torch.zeros(num_layers, batch_size, hidden_size), torch.zeros(num_layers, batch_size, hidden_size)

    def apply_teacher_forcing(self, y, y_len, error_rate):
        """
        Add synthetical errors in the ground truth with a rate of 'error_rate' for each character. (teacher forcing)

        Args:
            y (Tensor): The ground truth labels.
            y_len (list): The lengths of the labels.
            error_rate (float): The error rate for adding synthetic errors.

        Returns:
            Tensor: The modified ground truth labels with synthetic errors.
            list: The updated lengths of the labels.
        """
        y_error = y.clone()
        for b in range(len(y_len)):
            for i in range(1, y_len[b]):
                if np.random.rand() < error_rate and y[b][i] != self.dataset.tokens["pad"]:
                    y_error[b][i] = np.random.randint(0, len(self.dataset.train_dataset.vocab))
        return y_error, y_len


    def apply_teacher_forcing_custom(self, y, y_len, error_rate):
        """
        Add synthetical errors in the ground truth with a rate of 'error_rate' for each character. (teacher forcing)

        Args:
            y (Tensor): The ground truth labels.
            y_len (list): The lengths of the labels.
            error_rate (float): The error rate for adding synthetic errors.

        Returns:
            Tensor: The modified ground truth labels with synthetic errors.
            list: The updated lengths of the labels.
        """
        special_tokens_error_rate = error_rate / 3

        if '\n' in self.dataset.train_dataset.vocab or '<\n>' in self.dataset.train_dataset.vocab:
            if '\n' in self.dataset.train_dataset.vocab:
                breakline_char = '\n'
            else:
                breakline_char = '<\n>'

            if "replace_dict_str_to_ind" in self.params['dataset_params']['config']:
                index_break_line = self.params['dataset_params']['config']["replace_dict_str_to_ind"][self.dataset.train_dataset.vocab[breakline_char]]
            else:
                index_break_line = self.dataset.train_dataset.vocab[breakline_char]
        else:
            index_break_line = -10

        chars_in_subst_dict = list(self.params["training_params"]['subword_subst_dict'].keys())

        y_error = y.clone()
        for b in range(len(y_len)): # for each document in the batch
            for i in range(1, y_len[b]): # for each character in the label
                if y[b][i] != self.dataset.tokens["pad"] and y[b][i] != index_break_line:
                    # if the current character is not the padding token or the break line token
                    rand_value = np.random.rand()
                    current_char = str(int(y[b][i]))
                    # if the character is not in the substitution dictionary or
                    # if the substitution dictionary is empty for this character:
                    if current_char not in self.params["training_params"]['subword_subst_dict'] or not self.params["training_params"]['subword_subst_dict'][current_char]:
                        if rand_value < error_rate/3:
                            y_error[b][i] = int(np.random.choice(chars_in_subst_dict,1)[0])
                    else:
                        if int(y[b][i]) in self.params["training_params"].get('teacher_forcing_special_tokens', []):
                            tmp_error_rate = special_tokens_error_rate
                        else:
                            tmp_error_rate = error_rate

                        if rand_value < tmp_error_rate:
                            # in this case, we replace the char/subword by a random char/subword in the values of the substitution dictionary
                            # eg replaced with a subword/char that is likely to be mistaken for the current subword/char
                            y_error[b][i] = np.random.choice(
                                self.params["training_params"]['subword_subst_dict'][current_char],
                                1
                            )[0]

        return y_error, y_len


    def train_batch(self, batch_data, metric_names):
        """
        Train a batch of data.

        Args:
            batch_data (dict): A dictionary containing the batch data.
            metric_names (list): A list of metric names.

        Returns:
            tuple: A tuple containing the average loss and metric values.
        """

        sum_loss = 0
        x = batch_data["imgs"].to(self.device)

        y = batch_data["labels"].to(self.device)
        y_len = [y_len_i-1 for y_len_i in batch_data["labels_len"]]

        # add errors in teacher forcing
        if self.params["training_params"].get('char_error_injection', False):
            error_rate = self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"] + min(self.latest_step, self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]) * (self.params["training_params"]["teacher_forcing_scheduler"]["max_error_rate"]-self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"]) / self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]
            simulated_y_pred, y_len = self.apply_teacher_forcing(y, y_len, error_rate)
        elif "teacher_forcing_error_rate" in self.params["training_params"] and self.params["training_params"]["teacher_forcing_error_rate"] is not None:
            error_rate = self.params["training_params"]["teacher_forcing_error_rate"]
            simulated_y_pred, y_len = self.apply_teacher_forcing_custom(y, y_len, error_rate)
        elif "teacher_forcing_scheduler" in self.params["training_params"]:
            error_rate = self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"] + min(self.latest_step, self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]) * (self.params["training_params"]["teacher_forcing_scheduler"]["max_error_rate"]-self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"]) / self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]
            simulated_y_pred, y_len = self.apply_teacher_forcing_custom(y, y_len, error_rate)
        else:
            simulated_y_pred = y

        with autocast(enabled=self.params["training_params"]["use_amp"]):
            raw_features = self.models["encoder"](x)

            with autocast(enabled=False):
                features_size = raw_features.size()
                b, c, _, _ = features_size

                pos_features = self.models["decoder"].features_updater.get_pos_features(raw_features)
                # use 2d positionnal embedding is specified or does nothing
                features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)

                same_len_y_pred_input_ids = [seq[:seq_len] for seq, seq_len in zip(simulated_y_pred, batch_data["labels_len"])]
                try:
                    input_ids = pad_sequences_1D_to_given_length(same_len_y_pred_input_ids, padding_value=1, longest_x=self.params["training_params"]["max_char_prediction"])
                except Exception as e:
                    print(batch_data["raw_labels"])
                    raise e

                same_len_labels = [seq[1:seq_len] for seq, seq_len in zip(y, batch_data["labels_len"])]
                try:
                    labels = pad_sequences_1D_to_given_length(same_len_labels, padding_value=-100, longest_x=self.params["training_params"]["max_char_prediction"])
                except Exception as e:
                    print(batch_data["raw_labels"])
                    raise e

                model_output = self.models["decoder"](
                    input_ids=input_ids,
                    encoder_hidden_states=features.permute(1, 0, 2),
                    labels=labels,
                )

            loss_ce = model_output['loss']
            pred = model_output['logits'].permute(0,2,1)

            c = 0

            sum_loss += loss_ce
            with autocast(enabled=False):
                self.backward_loss(sum_loss)
                self.step_optimizers()
                self.zero_optimizers()

            predicted_tokens = torch.argmax(pred, dim=1).detach().cpu().numpy()
            predicted_tokens = [predicted_tokens[i, :y_len[i]-1] for i in range(b)]
            if 'use_subwords' in self.params["training_params"] and self.params["training_params"]['use_subwords']:
                if 'config' in self.params['dataset_params']:
                    rpl_dict = self.params['dataset_params']["config"].get('replace_dict_ind_to_str', None)
                else:
                    rpl_dict = None
                str_x = [LM_ind_to_str_subwords(self.dataset.train_dataset.subword_tokenizer,t, rpl_dict=rpl_dict) for t in predicted_tokens]
            else:
                str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]

        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "loss": sum_loss.item(),
            "loss_ce": loss_ce.item(),
            "syn_max_lines": self.dataset.train_dataset.get_syn_max_lines() if self.params["dataset_params"]["config"]["synthetic_data"] else 0,
            'total_mem_forward': c
        }

        if 'use_subwords' in self.params["training_params"] and self.params["training_params"]['use_subwords']:
            values['str_y'] = []
            for i in range(len(batch_data['raw_labels'])):
                if 'use_subwords_bart' in self.params['dataset_params']["config"]["constraints"]:
                    values['str_y'].append(
                    ''.join(self.dataset.train_dataset.subword_tokenizer.decode(
                            self.dataset.train_dataset.subword_tokenizer.encode(
                                batch_data['raw_labels'][i]
                            )[1:-1] # remove start of sequence token
                        ))
                )
                else:
                    values['str_y'].append(' '.join(self.dataset.train_dataset.subword_tokenizer.encode(batch_data['raw_labels'][i]).tokens).replace(' ##',''))

        return values

    def evaluate_batch(self, batch_data, metric_names, set_name=None):
        """
        Evaluate a batch of data using the trained model.

        Args:
            batch_data (dict): A dictionary containing the batch data.
            metric_names (list): A list of metric names to compute.

        Returns:
            dict: A dictionary containing the evaluation results.
        """
        x = batch_data["imgs"].to(self.device)
        if 'max_char_inferences' in self.params["training_params"] and set_name:
            max_length = self.params["training_params"]["max_char_inferences"].get(set_name.split('-')[-1], self.models["decoder"].decoder.max_position_embeddings)
        else:
            max_length = self.params["training_params"].get("max_char_inference",self.models["decoder"].decoder.max_position_embeddings)
        if 'NER' in batch_data['names'][0]:
            datasets = ['/'.join('_'.join(elem.split('_')[1:]).split('/')[:-1]) for elem in batch_data['names']]
        else:
            datasets = ['_'.join(elem.split('_')[1:-1]) for elem in batch_data['names']]

        if 'MULTI' in self.dataset_name:
            if not "mono_start" in self.params['dataset_params']["config"]['constraints']:
                dataset_to_start_token = {
                    'data':self.dataset.tokens["start"],
                    'IAM':self.dataset.train_dataset.vocab['<s-IAM>'],
                    'RIMES':self.dataset.train_dataset.vocab['<s-RIMES>'],
                    'READ_2016':self.dataset.train_dataset.vocab['<s-READ>'],
                    'READ':self.dataset.train_dataset.vocab['<s-READ>'],
                    'EXOPOPP':self.dataset.tokens["start"],
                }
                if '<s-IAM_NER>' in self.dataset.train_dataset.vocab:
                    dataset_to_start_token['IAM_NER'] = self.dataset.train_dataset.vocab['<s-IAM_NER>']
                    dataset_to_start_token['EXOPOPP_NER'] = self.dataset.train_dataset.vocab['<s-EXOPOPP_NER>']
            else:
                dataset_to_start_token = {
                    'data':self.dataset.tokens["start"],
                    'IAM':self.dataset.tokens["start"],
                    'RIMES':self.dataset.tokens["start"],
                    'READ_2016':self.dataset.tokens["start"],
                    'READ':self.dataset.tokens["start"],
                    'EXOPOPP':self.dataset.tokens["start"],
                }
                if '<s-IAM_NER>' in self.dataset.train_dataset.vocab:
                    dataset_to_start_token['IAM_NER'] = self.dataset.train_dataset.vocab['<s-IAM_NER>']
                if '<s-EXOPOPP_NER>' in self.dataset.train_dataset.vocab:
                    dataset_to_start_token['EXOPOPP_NER'] = self.dataset.train_dataset.vocab['<s-EXOPOPP_NER>']

            if 'LINGUAL' in datasets[0]: # evaluation on real data
                if batch_data['names'][0].split('/')[1] == 'train':
                    datasets = []
                    for elem in batch_data['names']:
                        if elem.endswith('.jpeg'):
                            datasets.append('READ_2016')
                        elif elem.split('/')[-1].startswith('train') and elem.endswith('png'):
                            datasets.append('RIMES')
                        elif 'IAM_NER' in elem:
                            datasets.append('IAM_NER')
                        elif 'EXOPOPP_NER' in elem:
                            datasets.append('EXOPOPP_NER')
                        elif 'EXOPOPP' in elem:
                            datasets.append('EXOPOPP')
                        else:
                            datasets.append('IAM')
                else:
                    datasets = [elem.split('/')[1].split('-')[1] for elem in batch_data['names']]
        elif 'EXOPOPP' in self.dataset_name:
            dataset_to_start_token = {
                'data':self.dataset.tokens["start"],
                'EXOPOPP':self.dataset.tokens["start"],
            }
            if '<s-EXOPOPP_NER>' in self.dataset.train_dataset.vocab:
                dataset_to_start_token['EXOPOPP_NER'] = self.dataset.train_dataset.vocab['<s-EXOPOPP_NER>']
            if 'page_sem' in datasets[0]: # evaluation on real data
                if 'named_entities' in datasets[0]:
                    datasets = ['EXOPOPP_NER' for _ in batch_data['names']]
                else:
                    datasets = ['EXOPOPP' for _ in batch_data['names']]
        elif 'named_entities' in self.dataset_name: # IAM NER
            dataset_to_start_token = {
                    'data':self.dataset.tokens["start"],
                    'IAM':self.dataset.tokens["start"],
                    'RIMES':self.dataset.tokens["start"],
                    'READ_2016':self.dataset.tokens["start"],
                    'READ':self.dataset.tokens["start"],
                    'IAM_NER': self.dataset.train_dataset.vocab['<s-IAM_NER>']
                }
            datasets = ['IAM_NER' for _ in batch_data['names']]

        start_time = time.time()
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            prediction_len = torch.zeros((b, ), dtype=torch.int, device=self.device)
            if 'forced_start_token' in self.params['model_params']:
                start_token = self.dataset.tokens.get(self.params['model_params']['forced_start_token'], self.dataset.train_dataset.vocab[self.params['model_params']['forced_start_token']])
                predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * start_token
            elif 'MULTI' in self.dataset_name:
                predicted_tokens=torch.ones((b, 1), dtype=torch.long, device=self.device)
                for k in range(len(batch_data['names'])):
                    predicted_tokens[k] = dataset_to_start_token[datasets[k]]
            elif 'named_entities' in self.dataset_name and 'EXOPOPP' in self.dataset_name and 'selftrain' not in self.dataset_name:
                predicted_tokens=torch.ones((b, 1), dtype=torch.long, device=self.device)
                for k in range(len(batch_data['names'])):
                    predicted_tokens[k] = dataset_to_start_token[datasets[k]]
            elif 'named_entities' in self.dataset_name and not 'EXOPOPP' in self.dataset_name:
                predicted_tokens=torch.ones((b, 1), dtype=torch.long, device=self.device)
                for k in range(len(batch_data['names'])):
                    predicted_tokens[k] = dataset_to_start_token[datasets[k]]
            else:
                predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * self.dataset.tokens["start"]

            features = self.models["encoder"](x)

            pos_features = self.models["decoder"].features_updater.get_pos_features(features)
            last_hidden_state = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(0, 2, 1)

            if self.device.type != "cuda":
                last_hidden_state = last_hidden_state.to(torch.float32)

            encoder_outputs = ModelOutput(last_hidden_state=last_hidden_state, attentions=None)

            if len(encoder_outputs.last_hidden_state.size()) == 1:
                encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.unsqueeze(0)

            output_scores = 'map_cer' in metric_names or 'loer' in metric_names

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
                confidence_scores = [[] for _ in range(b)]
                all_logits = decoder_output.scores # shape: (seq-len, batch-size, vocab-size)
                subword_confidence_scores = [
                    [
                        float(softmax(logits[index_batch], dim=-1).max()) for logits in all_logits[1:]
                    ][:prediction_len[index_batch]-1] for index_batch in range(b)
                ] # shape: (batch-size, seq-len)
                # we duplicate each confidence score to match the number of characters in the sequence

                added_tokens = self.dataset.train_dataset.subword_tokenizer.get_added_vocab().keys()

                sentences = []
                for index_batch in range(b):
                    conf_per_char = []
                    sentence = ''
                    for index_token, proba in enumerate(subword_confidence_scores[index_batch]):
                        current_char = self.dataset.train_dataset.reversed_vocab[int(predicted_tokens[index_batch][index_token])]

                        current_char = current_char.replace('<\n>','▁\n').replace('▁?', '?').replace('▁!', '!')
                        if index_token > 0 and current_char in added_tokens:
                            current_char = '▁'+current_char

                        if index_token < len(subword_confidence_scores[index_batch])-1:
                            next_current_char = self.dataset.train_dataset.reversed_vocab[int(predicted_tokens[index_batch][index_token+1])]
                            if (
                                (current_char == '▁' and (next_current_char in [',', '.','?','<\n>'] or next_current_char.startswith('.'))) or
                                (current_char == '▁\'' and next_current_char.startswith('▁')) or
                                (current_char == '▁' and next_current_char.startswith('▁'))
                            ):
                                current_char = ''

                            if current_char == '▁\n' and not next_current_char.startswith('▁') and next_current_char != '1' and next_current_char not in added_tokens:
                                current_char = '▁\n▁'

                            if current_char in ['▁ȳ', '▁1', "▁ ̄"] and (
                                not (next_current_char.startswith('▁') or next_current_char in added_tokens)
                            ):
                                if not (next_current_char in [',', '.','?'] or next_current_char.startswith('.')):
                                    current_char = current_char + '▁'

                        sentence += current_char
                        conf_per_char.extend([proba]*len(current_char))

                    sentences.append(sentence)
                    confidence_scores[index_batch].extend(conf_per_char)

            else:
                confidence_scores = None

            if 'use_subwords' in self.params["training_params"] and self.params["training_params"]['use_subwords']:
                if 'config' in self.params['dataset_params']:
                    rpl_dict = self.params['dataset_params']['config'].get('replace_dict_ind_to_str', None)
                else:
                    rpl_dict = None
                str_x = [LM_ind_to_str_subwords(self.dataset.train_dataset.subword_tokenizer,t.detach().cpu().numpy(), rpl_dict=rpl_dict) for t in predicted_tokens]
                if output_scores:
                    str_x = [str_x_elem.replace('  ',' ') for str_x_elem in str_x]
            else:
                str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]

        process_time = time.time() - start_time

        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "confidence_score": confidence_scores,
            "time": process_time,
        }

        if 'use_subwords' in self.params["training_params"] and self.params["training_params"]['use_subwords']:
            values['str_y'] = []
            for i in range(len(batch_data['raw_labels'])):
                if 'use_subwords_bart' in self.params['dataset_params']["config"]["constraints"]:
                    values['str_y'].append(
                    ''.join(self.dataset.train_dataset.subword_tokenizer.decode(
                            self.dataset.train_dataset.subword_tokenizer.encode(
                                batch_data['raw_labels'][i]
                            )[1:-1]
                        ))
                )
                else:
                    values['str_y'].append(' '.join(self.dataset.train_dataset.subword_tokenizer.encode(batch_data['raw_labels'][i]).tokens).replace(' ##',''))

        return values


def run(params, mode='train', dataset_names=('IAM'), metrics=["cer", "wer"], set_names=("test"), multi=False, synth_output_folder='synth_data', nb_samples=50, nb_steps=200000):
    model = Manager(params)

    def get_metrics_for_dataset(name):
        if name in ['RIMES', 'READ_2016', 'EXOPOPP']:
            return metrics + ["map_cer", "map_cers", "loer"]
        elif "NER" in name:
            return metrics + ["f1"]
        return metrics.copy()

    def evaluate_single_model():
        model.params["training_params"]["load_epoch"] = "best"
        model.load_model(reset_optimizer=True)
        for dataset_name in dataset_names:
            for set_name in set_names:
                print(f"Evaluating {dataset_name} on {set_name} set")
                model.predict(
                    f"{dataset_name}-{set_name}",
                    [(dataset_name, set_name)],
                    get_metrics_for_dataset(dataset_name),
                    output=True
                )

    def evaluate_multiple_models():
        best_weights = [f"best-{ds_name}" for ds_name in dataset_names]
        for (dataset_name, best_weight) in zip(dataset_names, best_weights):
            print(f"Dataset: {dataset_name}")
            model.params["training_params"]["load_epoch"] = best_weight
            model.load_model(reset_optimizer=True)
            target_set_names = [f"{subset}-{dataset_name}" for subset in set_names]
            for set_name in target_set_names:
                model.predict(
                    f"MULTI_LINGUAL-{set_name}",
                    [('MULTI_LINGUAL', set_name)],
                    get_metrics_for_dataset(dataset_name),
                    output=True
                )

    def offline_synth_data_generation(synth_output_folder, nb_samples, nb_steps):
        # visualize synthetic documents
        # synth_output_folder = os.path.join(os.getcwd(), synth_output_folder)
        model.dataset.train_dataset.training_info = {"step" : nb_steps}
        model.dataset.train_dataset.training_info['previous-nb-lines'] = 15
        Path(synth_output_folder).mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(nb_samples)):
            sample = model.dataset.train_dataset.generate_synthetic_page_sample()
            img = np.squeeze(sample['img'], axis=2)
            Image.fromarray(img).save(synth_output_folder + '/'+str(i)+'.jpg')

            with open(synth_output_folder+'/'+str(i)+'.json','w') as f:
                json.dump({'label':sample['label']},f)

    if mode == 'train':
        model.load_model()
        print('Added vocab:', model.dataset.train_dataset.subword_tokenizer.get_added_vocab())
        model.train()
    elif mode == 'eval':
        if multi:
            evaluate_multiple_models()
        else:
            evaluate_single_model()
    elif mode == 'synth':
        offline_synth_data_generation(synth_output_folder, nb_samples, nb_steps)
    else:
        raise ValueError("`mode` should be either 'train', 'eval' or 'synth'.")
