import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got: {value}")


def checkpoint_prefix_mode(checkpoint_keys, model_keys):
    checkpoint_keys = set(checkpoint_keys)
    model_keys = set(model_keys)

    if checkpoint_keys == model_keys:
        return "direct"

    if checkpoint_keys and all(key.startswith("module.") for key in checkpoint_keys):
        stripped_keys = {key[len("module."):] for key in checkpoint_keys}
        if stripped_keys == model_keys:
            return "strip"

    if model_keys and all(key.startswith("module.") for key in model_keys):
        inner_keys = {key[len("module."):] for key in model_keys}
        if checkpoint_keys == inner_keys:
            return "inner"

    raise ValueError("checkpoint and model state_dict keys are incompatible")
