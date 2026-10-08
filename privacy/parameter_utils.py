"""Parameter selection, flattening, and fixed-point helpers."""

import numpy as np #type: ignore
import torch #type: ignore


def get_classifier_param_keys(state_dict, model_name=None, target="classifier_bias"):
    del model_name
    keys = list(state_dict)
    bias_candidates = ["head.bias", "fc.bias", "classifier.bias", "classifier.1.bias", "classifier.6.bias"]
    pairs = [("head.weight", "head.bias"), ("fc.weight", "fc.bias"),
             ("classifier.weight", "classifier.bias"), ("classifier.1.weight", "classifier.1.bias"),
             ("classifier.6.weight", "classifier.6.bias")]
    if target == "classifier_bias":
        for bias in bias_candidates:
            if bias in state_dict:
                return [bias]
        one_dimensional = [key for key in keys if state_dict[key].ndim == 1]
        if not one_dimensional:
            raise ValueError("Could not identify classifier bias in state_dict.")
        return [one_dimensional[-1]]
    if target == "classifier_head":
        for weight, bias in pairs:
            found = [key for key in (weight, bias) if key in state_dict]
            if found:
                return found
        weights = [key for key in keys if state_dict[key].ndim == 2]
        if not weights:
            raise ValueError("Could not identify classifier head weight in state_dict.")
        weight = weights[-1]
        bias = weight[:-7] + ".bias" if weight.endswith(".weight") else None
        return [weight] + ([bias] if bias in state_dict else [])
    raise ValueError(f"Unsupported target: {target}")


def flatten_selected_params(state_dict, param_keys):
    parts, metadata = [], []
    for key in param_keys:
        array = state_dict[key].detach().cpu().numpy()
        flat = array.reshape(-1)
        parts.append(flat)
        metadata.append((key, array.shape, flat.size))
    return np.concatenate(parts).astype(np.float64), metadata


def restore_selected_params_from_flat(state_dict, flat_vector, metadata, device=None, dtype_map=None):
    new_state = {key: value.clone() for key, value in state_dict.items()}
    offset = 0
    for key, shape, count in metadata:
        chunk = flat_vector[offset:offset + count].reshape(shape)
        offset += count
        dtype = new_state[key].dtype if dtype_map is None else dtype_map.get(key, new_state[key].dtype)
        tensor = torch.tensor(chunk, dtype=dtype)
        new_state[key] = tensor.to(device) if device is not None else tensor
    return new_state


def fixed_point_encode(values, scale, mod):
    return np.mod(np.round(values * scale).astype(np.int64), mod)


def fixed_point_decode(values, scale, mod):
    signed = np.array(values, dtype=np.int64)
    signed[signed > mod // 2] -= mod
    return signed.astype(np.float64) / float(scale)


def average_decoded_vectors(decoded_sum_vector, num_clients):
    return decoded_sum_vector / float(num_clients)
