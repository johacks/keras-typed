import json
import os
from typing import Literal, Optional, Union

# Type definitions.
KerasFloatXType = Union[
    Literal["bfloat16"], Literal["float16"], Literal["float32"], Literal["float64"]
]
KerasImageDataFormat = Union[Literal["channels_first"], Literal["channels_last"]]
KerasBackend = Union[Literal["tensorflow"], Literal["torch"], Literal["jax"]]


# The type of float to use throughout a session.
_FLOATX: KerasFloatXType = "float32"

# Epsilon fuzz factor used throughout the codebase.
_EPSILON: float = 1e-7

# Default image data format, one of "channels_last", "channels_first".
_IMAGE_DATA_FORMAT: KerasImageDataFormat = "channels_last"

# Default backend: TensorFlow.
_BACKEND: KerasBackend = "tensorflow"


def floatx() -> Union[KerasFloatXType]:
    """Return the default float type, as a string.

    E.g. `'bfloat16'`, `'float16'`, `'float32'`, `'float64'`.

    Returns:
        str, the current default float type.

    Example:
    >>> keras.config.floatx()
    'float32'

    """
    return _FLOATX


def set_floatx(value: KerasFloatXType) -> None:
    """Set the default float dtype.

    Note: It is not recommended to set this to `"float16"` for training,
    as this will likely cause numeric stability issues.
    Instead, mixed precision, which leverages
    a mix of `float16` and `float32`. It can be configured by calling
    `keras.mixed_precision.set_dtype_policy('mixed_float16')`.

    Args:
        value: str; `'bfloat16'`, `'float16'`, `'float32'`, or `'float64'`.

    Examples:
    >>> keras.config.floatx()
    'float32'

    >>> keras.config.set_floatx("float64")
    >>> keras.config.floatx()
    'float64'

    >>> # Set it back to float32
    >>> keras.config.set_floatx("float32")

    Raises:
        ValueError: In case of invalid value.
    """
    global _FLOATX
    accepted_dtypes = {"bfloat16", "float16", "float32", "float64"}
    if value not in accepted_dtypes:
        raise ValueError(f"Unknown `floatx` value: {value}. " f"Expected one of {accepted_dtypes}")
    _FLOATX = str(value)


def epsilon() -> float:
    """Return the value of the fuzz factor used in numeric expressions.

    Returns:
        A float.

    Example:
    >>> keras.config.epsilon()
    1e-07
    """
    return _EPSILON


def set_epsilon(value: float) -> None:
    """Set the value of the fuzz factor used in numeric expressions.

    Args:
        value: float. New value of epsilon.

    Examples:
    >>> keras.config.epsilon()
    1e-07

    >>> keras.config.set_epsilon(1e-5)
    >>> keras.config.epsilon()
    1e-05

    >>> # Set it back to the default value.
    >>> keras.config.set_epsilon(1e-7)

    """
    global _EPSILON
    _EPSILON = value


def image_data_format() -> KerasImageDataFormat:
    """Return the default image data format convention.

    Returns:
        A string, either `'channels_first'` or `'channels_last'`.

    Example:
    >>> keras.config.image_data_format()
    'channels_last'

    """
    return _IMAGE_DATA_FORMAT


def standardize_data_format(data_format: Optional[str] = None) -> KerasImageDataFormat:
    if data_format is None:
        return image_data_format()
    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    return data_format


def set_image_data_format(data_format: KerasImageDataFormat) -> None:
    """Set the value of the image data format convention.

    Args:
        data_format: string. `'channels_first'` or `'channels_last'`.

    Examples:
    >>> keras.config.image_data_format()
    'channels_last'

    >>> keras.config.set_image_data_format("channels_first")
    >>> keras.config.image_data_format()
    'channels_first'

    >>> # Set it back to `'channels_last'`
    >>> keras.config.set_image_data_format("channels_last")

    Raises:
        ValueError: In case of invalid value.
    """
    global _IMAGE_DATA_FORMAT
    _IMAGE_DATA_FORMAT = standardize_data_format(data_format)


def enable_flash_attention() -> None:
    """Enable flash attention.

    Flash attention offers performance optimization for attention layers,
    making it especially useful for large language models (LLMs) that
    benefit from faster and more memory-efficient attention computations.

    Once enabled, supported layers like `MultiHeadAttention` will **attempt** to
    use flash attention for faster computations. By default, this feature is
    enabled.

    Note that enabling flash attention does not guarantee it will always be
    used. Typically, the inputs must be in `float16` or `bfloat16` dtype, and
    input layout requirements may vary depending on the backend.
    """
    from tkeras._core.backend.common import global_state

    global_state.set_global_attribute("flash_attention", None)


def disable_flash_attention() -> None:
    """Disable flash attention.

    Flash attention offers performance optimization for attention layers,
    making it especially useful for large language models (LLMs) that
    benefit from faster and more memory-efficient attention computations.

    Once disabled, supported layers like `MultiHeadAttention` will not
    use flash attention for faster computations.
    """
    from tkeras._core.backend.common import global_state

    global_state.set_global_attribute("flash_attention", False)


def is_flash_attention_enabled() -> Optional[bool]:
    """Checks whether flash attention is globally enabled in Keras.

    Flash attention is a performance-optimized method for computing attention
    in large models, such as transformers, allowing for faster and more
    memory-efficient operations. This function checks the global Keras
    configuration to determine if flash attention is enabled for compatible
    layers (e.g., `MultiHeadAttention`).

    Note that enabling flash attention does not guarantee it will always be
    used. Typically, the inputs must be in `float16` or `bfloat16` dtype, and
    input layout requirements may vary depending on the backend.

    Returns:
        `False` if disabled; otherwise, it indicates that it is enabled.
    """
    from tkeras._core.backend.common import global_state

    return global_state.get_global_attribute("flash_attention", default=None)


# Set Keras base dir path given KERAS_HOME env variable, if applicable.
# Otherwise either ~/.keras or /tmp.
if "KERAS_HOME" in os.environ:
    _KERAS_DIR = os.environ.get("KERAS_HOME")
else:
    _keras_base_dir = os.path.expanduser("~")
    if not os.access(_keras_base_dir, os.W_OK):
        _keras_base_dir = "/tmp"
    _KERAS_DIR = os.path.join(_keras_base_dir, ".keras")


def keras_home() -> str:
    # Private accessor for the keras home location.
    return _KERAS_DIR


# Attempt to read Keras config file.
_config_path = os.path.expanduser(os.path.join(_KERAS_DIR, "keras.json"))
if os.path.exists(_config_path):
    try:
        with open(_config_path) as f:
            _config = json.load(f)
    except ValueError:
        _config = {}
    _floatx = _config.get("floatx", floatx())
    assert _floatx in {"float16", "float32", "float64"}
    _epsilon = _config.get("epsilon", epsilon())
    assert isinstance(_epsilon, float)
    _backend = _config.get("backend", _BACKEND)
    _image_data_format = _config.get("image_data_format", image_data_format())
    assert _image_data_format in {"channels_last", "channels_first"}

    set_floatx(_floatx)
    set_epsilon(_epsilon)
    set_image_data_format(_image_data_format)
    _BACKEND = _backend

# Save config file, if possible.
if not os.path.exists(_KERAS_DIR):
    try:
        os.makedirs(_KERAS_DIR)
    except OSError:
        # Except permission denied and potential race conditions
        # in multi-threaded environments.
        pass

if not os.path.exists(_config_path):
    _config = {
        "floatx": floatx(),
        "epsilon": epsilon(),
        "backend": _BACKEND,
        "image_data_format": image_data_format(),
    }
    try:
        with open(_config_path, "w") as f:
            f.write(json.dumps(_config, indent=4))
    except IOError:
        # Except permission denied.
        pass

# Set backend based on KERAS_BACKEND flag, if applicable.
if "KERAS_BACKEND" in os.environ:
    _backend = os.environ["KERAS_BACKEND"]
    if _backend:
        _BACKEND = _backend


if _BACKEND != "tensorflow":
    # If we are not running on the tensorflow backend, we should stop tensorflow
    # from using all available GPU memory. See
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def backend() -> KerasBackend:
    """Publicly accessible method for determining the current backend.

    Returns:
        String, the name of the backend Keras is currently using. One of
            `"tensorflow"`, `"torch"`, or `"jax"`.

    Example:
    >>> keras.config.backend()
    'tensorflow'

    """
    return _BACKEND
