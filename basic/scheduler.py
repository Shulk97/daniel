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

from torch.nn import Dropout, Dropout2d
import numpy as np

def nb_lines(steps:list, curr_step:int):
    steps.sort()
    higher = False
    last_nlines = 1
    for i, (steps_i, nb_l_i) in enumerate(steps[:-1]):
        nb_l_seg = steps[i+1][1]
        p = nb_l_seg-nb_l_i+1
        step_next = steps[i+1][0]
        lspace = np.linspace(steps_i, step_next, p, dtype=int)
        for s_i, s in enumerate(lspace):
            nlines = nb_l_i+s_i
            if curr_step > s:
                higher = True
                last_nlines = nlines
            if curr_step <= s and higher:
                return last_nlines
    return last_nlines
class DropoutScheduler:

    def __init__(self, models, function, T=1e5):
        """
        T: number of gradient updates to converge
        """

        self.teta_list = list()
        self.init_teta_list(models)
        self.function = function
        self.T = T
        self.step_num = 0

    def step(self):
        self.step(1)

    def step(self, num):
        self.step_num += num

    def init_teta_list(self, models):
        for model_name in models.keys():
            self.init_teta_list_module(models[model_name])

    def init_teta_list_module(self, module):
        for child in module.children():
            if isinstance(child, Dropout) or isinstance(child, Dropout2d):
                self.teta_list.append([child, child.p])
            else:
                self.init_teta_list_module(child)

    def update_dropout_rate(self):
        for (module, p) in self.teta_list:
            module.p = self.function(p, self.step_num, self.T)


def exponential_dropout_scheduler(dropout_rate, step, max_step):
    return dropout_rate * (1 - np.exp(-10 * step / max_step))


def exponential_scheduler(init_value, end_value, step, max_step):
    step = min(step, max_step-1)
    return init_value - (init_value - end_value) * (1 - np.exp(-10*step/max_step))


def linear_scheduler(init_value, end_value, step, max_step):
    return init_value + step * (end_value - init_value) / max_step