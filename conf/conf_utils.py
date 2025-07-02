# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
def merge_config_dicts(A: dict, B: dict) -> dict:
    """
    Recursively merges two dictionaries A and B.
    If a key is present in both dictionaries:
        - If both values are dictionaries, perform a recursive merge.
        - Otherwise, the value from B overrides the one from A.

    :param A: Original dictionary
    :param B: Dictionary to merge (has priority)
    :return: New merged dictionary
    """
    result = A.copy()
    for key, value_B in B.items():
        if key in result:
            value_A = result[key]
            if isinstance(value_A, dict) and isinstance(value_B, dict):
                # Recursive merge
                result[key] = merge_config_dicts(value_A, value_B)
            else:
                # Simple override
                result[key] = value_B
        else:
            result[key] = value_B
    return result

def complete_dataset_params(dataset_params, model_params, training_params):
    dataset_params['use_line_break'] = model_params.get('use_line_break',False)
    dataset_params['use_special_line_break'] = model_params.get('use_special_line_break',False)
    dataset_params['config']['rimes_extra_subwords'] = model_params.get('rimes_extra_subwords',False)
    dataset_params["max_char_prediction"] = training_params["max_char_prediction"]

    if "add_NEs_in_charset" in model_params:
        model_params['add_NEs_in_charset'] = model_params['add_NEs_in_charset']
    if "add_layout_tokens_in_charset" in model_params:
        dataset_params['add_layout_tokens_in_charset'] = model_params['add_layout_tokens_in_charset']
    if 'bart_path' in model_params:
        dataset_params['bart_path'] = model_params['bart_path']

    return dataset_params