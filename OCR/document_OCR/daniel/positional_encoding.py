#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in Python  whose purpose is to
#  provide public implementation of deep learning works, in pytorch.
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

class PositionalEncoding1D(Module):
    """
        1D Positional Encoding (used to encode the output embedding obtained from predictions of the previous step)
    """
    def __init__(self, dim, len_max, device):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, len_max), device=device, requires_grad=False)

        # divide the complete range of sin and cosinus in multiple steps using 'div'
        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        l_pos = torch.arange(0., len_max)
        self.pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0) # even dimensions of the embedding
        # PE 1D (x, 2k) = sin(w_k · x)
        self.pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0) # odd dimensions of the embedding
        # PE 1D (x, 2k + 1) = cos(w_k · x)

    def forward(self, x, start):
        """
        Add 1D positional encoding to x
        x: (B, C, L)
        start: index for x[:,:, 0]
        """
        # we add the positionnal embedding to the input
        if isinstance(start, int):
            return x + self.pe[:, :, start:start+x.size(2)].to(x.device)
        else:
            for i in range(x.size(0)):
                x[i] = x[i] + self.pe[0, :, start[i]:start[i]+x.size(2)]
            return x


class PositionalEncoding2D(Module):
    # since we work on images, we have to use a 2D positionnal embedding
    def __init__(self, dim, h_max, w_max, device):
        super(PositionalEncoding2D, self).__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim # d_model
        self.pe = torch.zeros((1, dim, h_max, w_max), device=device, requires_grad=False)

        div = torch.exp(-torch.arange(0., dim // 2, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        # w_k = 1/10000 2k/d_model
        w_pos = torch.arange(0., w_max)
        h_pos = torch.arange(0., h_max)
        # ∀k ∈ [0, d_model /4]
        # even values for x and y:
        self.pe[:, :dim // 2:2, :, :] = torch.sin(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        # PE 2D (x, y, 2k)
        self.pe[:, 1:dim // 2:2, :, :] = torch.cos(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        # PE 2D (x, y, 2k + 1)
        # odd values for x and y:
        self.pe[:, dim // 2::2, :, :] = torch.sin(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        # PE 2D (x, y, d model /2 + 2k)
        self.pe[:, dim // 2 + 1::2, :, :] = torch.cos(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        #PE 2D (x, y, d model /2 + 2k + 1)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: (B, C, H, W)
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

    def get_pe_by_size(self, h, w, device):
        return self.pe[:, :, :h, :w].to(device)

class FeaturesUpdater(Module):
    """
    Module that handle 2D positional encoding
    """
    def __init__(self, params):
        super(FeaturesUpdater, self).__init__()
        self.enc_dim = params["enc_dim"]
        self.enc_h_max = params["h_max"]
        self.enc_w_max = params["w_max"]
        self.pe_2d = PositionalEncoding2D(self.enc_dim, self.enc_h_max, self.enc_w_max, params["device"])
        self.use_2d_positional_encoding = "use_2d_pe" not in params or params["use_2d_pe"]

    def get_pos_features(self, features):
        if self.use_2d_positional_encoding:
            return self.pe_2d(features)
        return features
