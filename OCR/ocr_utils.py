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

def LM_str_to_ind(labels, string):
    """
    Converts a string to a list of indices based on the given labels.

    Args:
        labels (list): The list of labels used for mapping characters to indices.
        string (str): The input string to be converted.

    Returns:
        list: A list of indices representing the characters in the input string.
    """
    return [labels.index(c) for c in string]

def LM_str_to_ind_subwords(tokenizer, string, is_bart=False, rpl_dict=None):
    """
    Converts a string into a list of subword indices using the given tokenizer.

    Args:
        tokenizer (Tokenizer): The tokenizer used to encode the string.
        string (str): The input string to be converted.
        is_bart (bool, optional): Whether the tokenizer is for BART model. Defaults to False.
        rpl_dict (dict, optional): A dictionary for replacing subword indices. Defaults to None.

    Returns:
        list: A list of subword indices representing the input string.
    """

    if is_bart:
        subwords_ids = tokenizer.encode(string)
        if rpl_dict is not None:
            subwords_ids = [rpl_dict[i] for i in subwords_ids]
        return subwords_ids
    else:
        subwords_ids = tokenizer.encode(string).ids
        if rpl_dict is not None:
            subwords_ids = [rpl_dict[i] for i in subwords_ids]
        return subwords_ids

def LM_ind_to_str(labels, ind, oov_symbol=None):
    """
    Converts a list of indices to a string representation using a given list of labels.

    Args:
        labels (list): The list of labels.
        ind (list): The list of indices to be converted.
        oov_symbol (str, optional): The symbol to be used for out-of-vocabulary indices. Defaults to None.

    Returns:
        str: The string representation of the indices.
    """
    if oov_symbol is not None:
        res = []
        for i in ind:
            if i < len(labels):
                res.append(labels[i])
            else:
                res.append(oov_symbol)
    else:
        res = [labels[i] for i in ind]
    return "".join(res)

def LM_ind_to_str_subwords(tokenizer, inds, rpl_dict=None):
    """
    Converts a list of subwords indices to a string using the provided tokenizer.

    Args:
        tokenizer (Tokenizer): The tokenizer used to decode the token indices.
        inds (list): The list of token indices.
        oov_symbol (str, optional): The symbol to use for out-of-vocabulary tokens. Defaults to None.
        rpl_dict (dict, optional): A dictionary mapping token indices to replacement indices. Defaults to None.

    Returns:
        str: The string representation of the subwords.

    """
    if rpl_dict is not None:
        inds = [rpl_dict[ind] for ind in inds]

    return tokenizer.decode(inds).replace(' ##', '').replace('<pad>', '')
