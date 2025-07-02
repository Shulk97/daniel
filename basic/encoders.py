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
import torch
from torch.nn import Module

from basic.layers import ConvBlock, DSCBlock


class FCN_Encoder_Widther(Module):
    def __init__(self, params):
        super(FCN_Encoder_Widther, self).__init__()

        self.dropout = params["dropout"]
        self.use_checkpointing = params['use_checkpointing'] if 'use_checkpointing' in params else False
        self.init_blocks = torch.nn.Sequential(*[
            ConvBlock(params["input_channels"], 32, stride=(1, 1), dropout=self.dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 256, stride=(2, 2), dropout=self.dropout),
            ConvBlock(256, 512, stride=(2, 1), dropout=self.dropout),
            ConvBlock(512, 512, stride=(2, 1), dropout=self.dropout),
        ])
        self.blocks = torch.nn.Sequential(*[
            DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
            DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
            DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
            DSCBlock(512, 1024, stride=(1, 1), dropout=self.dropout),
        ])

    def forward(self, x):
        x = self.init_blocks(x)
        x = self.blocks(x)
        return x

class FCN_Encoder_WidtherFeature(Module):
    """
    Same as FCN_Encoder_Widther but with a total stride of (16, 8) instead of (32, 8)
    """
    def __init__(self, params):
        super(FCN_Encoder_WidtherFeature, self).__init__()

        self.dropout = params["dropout"]
        self.init_blocks = torch.nn.Sequential(*[
            ConvBlock(params["input_channels"], 32, stride=(1, 1), dropout=self.dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 256, stride=(2, 2), dropout=self.dropout),
            ConvBlock(256, 512, stride=(2, 1), dropout=self.dropout),
            ConvBlock(512, 512, stride=(1, 1), dropout=self.dropout),
        ])
        self.blocks = torch.nn.Sequential(*[
            DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
            DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
            DSCBlock(512, 512, stride=(1, 1), dropout=self.dropout),
            DSCBlock(512, 1024, stride=(1, 1), dropout=self.dropout),
        ])

    def forward(self, x):
        x = self.init_blocks(x)
        x = self.blocks(x)
        return x
