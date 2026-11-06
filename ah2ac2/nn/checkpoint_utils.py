"""Compatibility helpers for current and historical VAE parameter wrappers."""


def unwrap_vae_params(params):
    """Return an inner VAE parameter tree when its wrapper is recognized.

    Existing serialized checkpoints retain their original parameter names.
    Unwrapped parameter trees are returned unchanged.
    """
    for key in ("VAE_0", "GAMMAVAE_0"):
        if key in params:
            return params[key]
    return params
