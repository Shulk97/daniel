# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
from basic.scheduler import exponential_dropout_scheduler
from OCR.document_OCR.daniel.models_bart import GlobalBARTDecoder

default_model_cfg = {
    "models": {
        "decoder": GlobalBARTDecoder, # architecture of the decoder
    },
    "transfered_charset": True,  # Transfer learning of the decision layer based on charset of the line HTR model
    "additional_tokens": 1,  # for decision layer = [<eot>, ], only for transfered charset
    "input_channels": 1,  # number of channels of input image
    "dropout": 0.5,  # dropout rate for encoder
    "enc_dim": 1024,  # dimension of extracted features
    "bart_layers":4, # number of layers in the decoder
    "bart_path":"naver-clova-ix/donut-base", # path of the pretrained BART model
    "use_line_break":False, # whether to use regular line breaks (= "\n")
    "use_special_line_break":True, # whether to use special line breaks (= "<\n>")
    "add_layout_tokens_in_charset": True, # add layout tokens of every datasets in the vocab
    "rimes_extra_subwords": True, # add extra subwords of the RIMES dataset, mostly accents
    "h_max": 500,  # maximum height for encoder output (for 2D positional embedding)
    "w_max": 1000,  # maximum width for encoder output (for 2D positional embedding)
    "l_max": 15000,  # max predicted sequence (for 1D positional embedding)
    "use_2d_pe": True,  # use 2D positional embedding
    "use_1d_pe": True,  # use 1D positional embedding
    "dropout_scheduler": { # scheduler for the dropout rate
        "function": exponential_dropout_scheduler,
        "T": 5e4,
    }
}