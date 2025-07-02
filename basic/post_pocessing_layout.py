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

import numpy as np

from Datasets.dataset_formatters.read2016_formatter import SEM_MATCHING_TOKENS as READ_MATCHING_TOKENS
from Datasets.dataset_formatters.rimes_formatter import SEM_MATCHING_TOKENS as RIMES_MATCHING_TOKENS
from Datasets.dataset_formatters.exo_popp_formatter import SEM_MATCHING_TOKENS as EXOPOPP_MATCHING_TOKENS


class PostProcessingModule:
    """
    Forward pass post processing
    Add/remove layout tokens only to:
     - respect token hierarchy
     - complete/remove unpaired tokens
    """

    def __init__(self):
        self.prediction = None
        self.confidence = None
        self.num_op = 0

    def post_processing(self):
        raise NotImplementedError

    def post_process(self, prediction, confidence_score=None):
        """
        Apply dataset-specific post-processing
        """
        self.prediction = list(prediction)
        self.confidence = list(confidence_score) if confidence_score is not None else None
        if self.confidence is not None:
            assert len(self.prediction) == len(self.confidence)
        return self.post_processing()

    def insert_label(self, index, label):
        """
        Insert token at specific index. The associated confidence score is set to 0.
        """
        self.prediction.insert(index, label)
        if self.confidence is not None:
            self.confidence.insert(index, 0)
        self.num_op += 1

    def del_label(self, index):
        """
        Remove the token at a specific index.
        """
        del self.prediction[index]
        if self.confidence is not None:
            del self.confidence[index]
        self.num_op += 1


class PostProcessingModuleREAD(PostProcessingModule):
    """
    Specific post-processing for the READ 2016 dataset at single-page and double-page levels
    """
    def __init__(self):
        super(PostProcessingModuleREAD, self).__init__()

        self.matching_tokens = READ_MATCHING_TOKENS
        self.reverse_matching_tokens = dict()
        for key in self.matching_tokens:
            self.reverse_matching_tokens[self.matching_tokens[key]] = key

    def post_processing_page_labels(self):
        """
        Correct tokens of page detection.
        """
        ind = 0
        while ind != len(self.prediction):
            # Label must start with a begin-page token
            if ind == 0 and self.prediction[ind] != "ⓟ":
                self.insert_label(0, "ⓟ")
                continue
            # There cannot be tokens out of begin-page end-page scope: begin-page must be preceded by end-page
            if self.prediction[ind] == "ⓟ" and ind != 0 and self.prediction[ind - 1] != "Ⓟ":
                self.insert_label(ind, "Ⓟ")
                continue
            # There cannot be tokens out of begin-page end-page scope: end-page must be followed by begin-page
            if self.prediction[ind] == "Ⓟ" and ind < len(self.prediction) - 1 and self.prediction[ind + 1] != "ⓟ":
                self.insert_label(ind + 1, "ⓟ")
            ind += 1
        # Label must start with a begin-page token even for empty prediction
        if len(self.prediction) == 0:
            self.insert_label(0, "ⓟ")
            ind += 1
        # Label must end with a end-page token
        if self.prediction[-1] != "Ⓟ":
            self.insert_label(ind, "Ⓟ")

    def post_processing(self):
        """
        Correct tokens of page number, section, body and annotations.
        """
        self.post_processing_page_labels()
        ind = 0
        begin_token = None
        in_section = False
        while ind != len(self.prediction):
            # each tags must be closed while changing page
            if self.prediction[ind] == "Ⓟ":
                if begin_token is not None:
                    self.insert_label(ind, self.matching_tokens[begin_token])
                    begin_token = None
                    ind += 1
                elif in_section:
                    self.insert_label(ind, self.matching_tokens["ⓢ"])
                    in_section = False
                    ind += 1
                else:
                    ind += 1
                continue
            # End token is removed if the previous begin token does not match with it
            if self.prediction[ind] in "ⓃⒶⒷ":
                if begin_token == self.reverse_matching_tokens[self.prediction[ind]]:
                    begin_token = None
                    ind += 1
                else:
                    self.del_label(ind)
                continue
            if self.prediction[ind] == "Ⓢ":
                # each sub-tags must be closed while closing section
                if in_section:
                    if begin_token is None:
                        in_section = False
                        ind += 1
                    else:
                        self.insert_label(ind, self.matching_tokens[begin_token])
                        begin_token = None
                        ind += 2
                else:
                    self.del_label(ind)
                continue
            if self.prediction[ind] == "ⓢ":
                # A sub-tag must be closed before opening a section
                if begin_token is not None:
                    self.insert_label(ind, self.matching_tokens[begin_token])
                    begin_token = None
                    ind += 1
                # A section must be closed before opening a new one
                elif in_section:
                    self.insert_label(ind, "Ⓢ")
                    in_section = False
                    ind += 1
                else:
                    in_section = True
                    ind += 1
                continue
            if self.prediction[ind] == "ⓝ":
                # Page number cannot be in section: a started section must be closed
                if begin_token is None:
                    if in_section:
                        in_section = False
                        self.insert_label(ind, "Ⓢ")
                        ind += 1
                    begin_token = self.prediction[ind]
                    ind += 1
                else:
                    self.insert_label(ind, self.matching_tokens[begin_token])
                    begin_token = None
                    ind += 1
                continue
            if self.prediction[ind] in "ⓐⓑ":
                # Annotation and body must be in section
                if begin_token is None:
                    if in_section:
                        begin_token = self.prediction[ind]
                        ind += 1
                    else:
                        in_section = True
                        self.insert_label(ind, "ⓢ")
                        ind += 1
                # Previous sub-tag must be closed
                else:
                    self.insert_label(ind, self.matching_tokens[begin_token])
                    begin_token = None
                    ind += 1
                continue
            ind += 1
        res = "".join(self.prediction)
        if self.confidence is not None:
            return res, np.array(self.confidence)
        return res

class PostProcessingModuleEXOPOPP(PostProcessingModule):
    """
    Specific post-processing for the EXOPOPP dataset at single-page and double-page levels
    """
    def __init__(self):
        super(PostProcessingModuleEXOPOPP, self).__init__()

        self.matching_tokens = EXOPOPP_MATCHING_TOKENS
        self.reverse_matching_tokens = dict() # Get begin token from end token
        for key in self.matching_tokens:
            self.reverse_matching_tokens[self.matching_tokens[key]] = key

    def post_processing_page_labels(self):
        """
        Correct tokens of page detection. (OK pour EXO-popp, rien à changer)
        """
        ind = 0
        while ind != len(self.prediction):
            # Label must start with a begin-page token
            if ind == 0 and self.prediction[ind] != "ⓟ":
                self.insert_label(0, "ⓟ")
                continue
            # There cannot be tokens out of begin-page end-page scope: begin-page must be preceded by end-page
            if self.prediction[ind] == "ⓟ" and ind != 0 and self.prediction[ind - 1] != "Ⓟ":
                self.insert_label(ind, "Ⓟ")
                continue
            # There cannot be tokens out of begin-page end-page scope: end-page must be followed by begin-page
            if self.prediction[ind] == "Ⓟ" and ind < len(self.prediction) - 1 and self.prediction[ind + 1] != "ⓟ":
                self.insert_label(ind + 1, "ⓟ")
            ind += 1
        # Label must start with a begin-page token even for empty prediction
        if len(self.prediction) == 0:
            self.insert_label(0, "ⓟ")
            ind += 1
        # Label must end with a end-page token
        if self.prediction[-1] != "Ⓟ":
            self.insert_label(ind, "Ⓟ")

    def post_processing(self):
        """
        Correct tokens of page number, section, body and annotations.
        """
        exopopp_to_multi_dict = {
            "ⓟ":'ⓚ', # page ExoPOPP
            "ⓑ":'ⓗ', # paragraph ExoPOPP
            "ⓝ":'ⓙ', # margin-names M-POPP
            "ⓘ":'ⓘ', # margin-infos ExoPOPP
            "ⓜ":'ⓜ', # marriage act ExoPOPP

            "Ⓟ":'Ⓚ', # closing page ExoPOPP
            "Ⓑ":'Ⓗ', # closing paragraph ExoPOPP
            "Ⓝ":'Ⓙ', # closing margin-names ExoPOPP
            "Ⓘ":'Ⓘ', # closing margin infos ExoPOPP
            "Ⓜ":'Ⓜ', # closing marriage act ExoPOPP
        }
        for i in range(len(self.prediction)):
            for old_tkn, new_tkn in exopopp_to_multi_dict.items():
                self.prediction[i] = self.prediction[i].replace(new_tkn, old_tkn)

        self.post_processing_page_labels()
        ind = 0
        begin_token = None
        in_section = False
        while ind != len(self.prediction):
            # each tags must be closed while changing page
            if self.prediction[ind] == "Ⓟ":
                if begin_token is not None:
                    self.insert_label(ind, self.matching_tokens[begin_token])
                    begin_token = None
                    ind += 1
                elif in_section:
                    self.insert_label(ind, self.matching_tokens["ⓜ"])
                    in_section = False
                    ind += 1
                else:
                    ind += 1
                continue
            # End token is removed if the previous begin token does not match with it
            if self.prediction[ind] in "ⓃⒷⒾⓈ": # marriage certificate can span over two pages
                if begin_token == self.reverse_matching_tokens[self.prediction[ind]]:
                    begin_token = None
                    ind += 1
                else: #
                    self.del_label(ind)
                continue
            if self.prediction[ind] == "Ⓜ":
                # each sub-tags must be closed while closing section
                if in_section:
                    if begin_token is None:
                        in_section = False
                        ind += 1
                    else:
                        self.insert_label(ind, self.matching_tokens[begin_token])
                        begin_token = None
                        ind += 2
                else:
                    self.del_label(ind)
                continue
            if self.prediction[ind] == "ⓜ":
                # A sub-tag must be closed before opening a section
                if begin_token is not None:
                    self.insert_label(ind, self.matching_tokens[begin_token])
                    begin_token = None
                    ind += 1
                # A section must be closed before opening a new one
                elif in_section:
                    self.insert_label(ind, "Ⓜ")
                    in_section = False
                    ind += 1
                else:
                    in_section = True
                    ind += 1
                continue
            if self.prediction[ind] in "ⓝⓑⓘⓢ":
                # Annotation and body must be in section
                if begin_token is None:
                    if in_section:
                        begin_token = self.prediction[ind]
                        ind += 1
                    else:
                        in_section = True
                        self.insert_label(ind, "ⓜ")
                        ind += 1
                # Previous sub-tag must be closed
                else:
                    self.insert_label(ind, self.matching_tokens[begin_token])
                    begin_token = None
                    ind += 1
                continue
            ind += 1
        res = "".join(self.prediction)
        if self.confidence is not None:
            return res, np.array(self.confidence)
        return res
class PostProcessingModuleRIMES(PostProcessingModule):
    """
    Specific post-processing for the RIMES dataset at page level
    """
    def __init__(self):
        super(PostProcessingModuleRIMES, self).__init__()
        self.matching_tokens = RIMES_MATCHING_TOKENS
        self.reverse_matching_tokens = dict()
        for key in self.matching_tokens:
            self.reverse_matching_tokens[self.matching_tokens[key]] = key

    def post_processing(self):
        multi_to_rimes_dict = {
            "ⓒ": "ⓑ",  # body
            "ⓞ": "ⓞ",# opening
            "ⓡ": "ⓡ",# recipient
            "ⓕ": "ⓢ",# sender and Reference
            "ⓦ": "ⓦ",# where, when
            "ⓨ": "ⓨ",# why
            "ⓔ": "ⓟ", # post-scriptum,
            "Ⓒ" : "Ⓑ",  # body
            "Ⓞ": "Ⓞ",# opening
            "Ⓡ": "Ⓡ",# recipient
            "Ⓕ": "Ⓢ",# sender and Reference
            "Ⓦ": "Ⓦ",# where, when
            "Ⓨ": "Ⓨ",# why
            "Ⓔ": "Ⓟ", # post-scriptum,
        }
        for i in range(len(self.prediction)):
            for new_tkn, old_tkn in multi_to_rimes_dict.items():
                self.prediction[i] = self.prediction[i].replace(new_tkn, old_tkn)

        ind = 0
        begin_token = None
        while ind != len(self.prediction):
            char = self.prediction[ind]
            # a tag must be closed before starting a new one
            if char in self.matching_tokens.keys():
                if begin_token is None:
                    ind += 1
                else:
                    self.insert_label(ind, self.matching_tokens[begin_token])
                    ind += 2
                begin_token = char
                continue
            # an end token without prior corresponding begin token is removed
            elif char in self.matching_tokens.values():
                if begin_token == self.reverse_matching_tokens[char]:
                    ind += 1
                    begin_token = None
                else:
                    self.del_label(ind)
                continue
            else:
                ind += 1
        # a tag must be closed
        if begin_token is not None:
            self.insert_label(ind+1, self.matching_tokens[begin_token])
        res = "".join(self.prediction)
        if self.confidence is not None:
            return res, np.array(self.confidence)
        return res
