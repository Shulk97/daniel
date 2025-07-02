"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import os
from typing import List, Optional, Union
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartConfig, MBartForCausalLM, XLMRobertaTokenizer, BartTokenizer, VisionEncoderDecoderModel, BartForCausalLM
from transformers.file_utils import ModelOutput
from OCR.document_OCR.daniel.positional_encoding import FeaturesUpdater
from OCR.ocr_dataset_manager import CustomXLMRobertaTokenizer
import traceback

class GlobalBARTDecoder(nn.Module):
    def __init__(
            self, params
        ):
        """
        Initialize the BART model.

        Args:
            params (dict): A dictionary containing the parameters for the BART model.

        Returns:
            None
        """

        super().__init__()
        nb_layers = params['bart_layers'] if 'bart_layers' in params else 4

        if 'subword_tokenizer' in params:
            tokenizer = params['subword_tokenizer']
        else:
            tokenizer = None

        if 'custom_vocab' in params: # used by BART with light vocabulary
            with open(params['custom_vocab'], 'rb') as f:
                self.custom_vocab = pkl.load(f)
                self.vocab_size = len(self.custom_vocab)

                for token in params['subword_tokenizer'].added_tokens_encoder:
                    if token not in self.custom_vocab and token not in ['<pad>', '<mask>', '<s_iitcdip>', '<s_synthdog>']:
                        self.custom_vocab[token] = len(self.custom_vocab)
                        self.vocab_size += 1

        else:
            self.custom_vocab = None
            self.vocab_size = None

        self.decoder = BARTDecoder(
            decoder_layer=nb_layers,
            max_position_embeddings=params["max_char_prediction"],
            name_or_path=params['bart_path'] if 'bart_path' in params else "naver-clova-ix/donut-base",
            monolingual=False,
            use_line_break=params['use_line_break'] if 'use_line_break' in params else False,
            tokenizer=tokenizer,
            vocab_size=self.vocab_size,
            use_special_line_break=params.get('use_special_line_break',False),
        )

        if 'bart_freeze' in params and params['bart_freeze']:
            for layer in self.decoder.model.model.decoder.layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.features_updater = FeaturesUpdater(params)

    def forward(self, input_ids,encoder_hidden_states,labels):
        return self.decoder(input_ids=input_ids,encoder_hidden_states=encoder_hidden_states,labels=labels)

class BARTDecoder(nn.Module):
    """
    Donut Decoder based on Multilingual BART
    Set the initial weights and configuration with a pretrained multilingual BART model,
    and modify the detailed configurations as a Donut decoder

    Args:
        decoder_layer:
            Number of layers of BARTDecoder
        max_position_embeddings:
            The maximum sequence length to be trained
        name_or_path:
            Name of a pretrained model name either registered in huggingface.co. or saved in local,
            otherwise, `hyunwoongko/asian-bart-ecjk` will be set (using `transformers`)
    """

    def __init__(
        self, decoder_layer: int, max_position_embeddings: int, name_or_path: Union[str, bytes, os.PathLike] = None, load_pretrained_weights=True, monolingual=False, use_line_break=False, tokenizer=None, vocab_size=None, use_special_line_break=False
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = max_position_embeddings

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            if 'bart-base' in name_or_path:
                self.tokenizer = BartTokenizer.from_pretrained(name_or_path)
            else:
                if use_special_line_break:
                    self.tokenizer = CustomXLMRobertaTokenizer.from_pretrained(name_or_path)
                    self.tokenizer.add_special_tokens({'additional_special_tokens': ['<\n>']})
                else:
                    self.tokenizer = XLMRobertaTokenizer.from_pretrained(
                        "hyunwoongko/asian-bart-ecjk" if not name_or_path else name_or_path
                    )


            if use_line_break and not use_special_line_break:
                self.tokenizer.add_tokens(["\n"])

        self.vocab_size = len(self.tokenizer) if not vocab_size else vocab_size

        self.model = MBartForCausalLM(
            config=MBartConfig(
                is_decoder=True, # Whether the model is used as decoder or not (in which case it's used as an encoder)
                is_encoder_decoder=False, # Whether the model is used as an encoder/decoder or not.
                add_cross_attention=True, # Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
                # that can be used as decoder models within the [`EncoderDecoderModel`] class
                decoder_layers=self.decoder_layer,
                max_position_embeddings=self.max_position_embeddings,# The maximum sequence length that this model might ever be used with.
                #Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
                vocab_size=self.vocab_size,
                scale_embedding=True, # Scale embeddings by diving by sqrt(d_model).
                add_final_layer_norm=True,
            )
        )
        self.model.forward = self.forward  #  to get cross attentions and utilize `generate` function

        self.model.config.is_encoder_decoder = True  # to get cross-attention
        self.add_special_tokens(["<sep/>"])  # <sep/> is used for representing a list in a JSON
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id # indice du token de padding
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference

        # weight init with asian-bart
        if load_pretrained_weights:
            if monolingual:
                bart_state_dict = BartForCausalLM.from_pretrained(name_or_path).state_dict()
            else:
                if not name_or_path:
                    name_or_path = "hyunwoongko/asian-bart-ecjk"
                if name_or_path == "hyunwoongko/asian-bart-ecjk" or name_or_path=='facebook/bart-base':
                    bart_state_dict = MBartForCausalLM.from_pretrained(name_or_path).state_dict()
                else:
                    bart_state_dict = VisionEncoderDecoderModel.from_pretrained(name_or_path).decoder.state_dict()

            new_bart_state_dict = self.model.state_dict()
            for x in new_bart_state_dict:
                if x.endswith("embed_positions.weight"):
                    new_bart_state_dict[x] = torch.nn.Parameter(
                        self.resize_bart_abs_pos_emb(
                            bart_state_dict[x],
                            self.max_position_embeddings
                            + 2,
                        )
                    )
                elif x.endswith("embed_tokens.weight") or x.endswith("lm_head.weight"):
                    if self.vocab_size < bart_state_dict[x].shape[0]:
                        new_bart_state_dict[x] = bart_state_dict[x][: self.vocab_size, :]
                    elif self.vocab_size > bart_state_dict[x].shape[0]:
                        new_bart_state_dict[x][: bart_state_dict[x].shape[0], :] = bart_state_dict[x]
                else:
                    new_bart_state_dict[x] = bart_state_dict[x]
            self.model.load_state_dict(new_bart_state_dict)

    def add_special_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings
        """
        newly_added_num = self.tokenizer.add_special_tokens({"additional_special_tokens": sorted(set(list_of_tokens))})
        if newly_added_num > 0:
            self.model.resize_token_embeddings(self.vocab_size)

    def prepare_inputs_for_inference(self, input_ids: torch.Tensor, encoder_outputs: torch.Tensor, past_key_values=None, past=None, use_cache: bool = None, attention_mask: torch.Tensor = None):
        """
        Args:
            input_ids: (batch_size, sequence_lenth)
        Returns:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, embedding_dim)
        """
        if past is not None:
            past_key_values = past
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
        }
        return output

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = None,
    ):
        """
        A forward fucntion to get cross attentions and utilize `generate` function

        Source:
        https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L1669-L1810

        Args:
            input_ids: (batch_size, sequence_length)
            attention_mask: (batch_size, sequence_length)
            encoder_hidden_states: (batch_size, sequence_length, hidden_size)

        Returns:
            loss: (1, )
            logits: (batch_size, sequence_length, hidden_dim)
            hidden_states: (batch_size, sequence_length, hidden_size)
            decoder_attentions: (batch_size, num_heads, sequence_length, sequence_length)
            cross_attentions: (batch_size, num_heads, sequence_length, sequence_length)
        """
        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        try:
            outputs = self.model.model.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        except Exception as e:
            print("input_ids", min(input_ids), max(input_ids))
            print(e)
            tb = traceback.format_exc()
            print(tb)

        logits = self.model.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Resize position embeddings
        Truncate if sequence length of Bart backbone is greater than given max_length,
        else interpolate to max_length
        """
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 0)
            )
        return weight
