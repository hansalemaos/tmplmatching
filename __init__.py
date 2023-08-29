import json
import multiprocessing
import os
import pathlib
import random
import re
import tempfile
import webbrowser
from collections import defaultdict
from multiprocessing.sharedctypes import RawArray
import pandas as pd
import mss
from downloadunzip import extract
from list_all_files_recursively import get_folder_file_complete_path
from tolerant_isinstance import isinstance_tolerant
from touchtouch import touch
import cv2
from a_cv_imwrite_imread_plus import open_image_in_cv, save_cv_image
from functools import cache, reduce
from collections import deque
from intersectioncalc import find_rectangle_intersections

import math
from a_cv2_easy_resize import add_easy_resize_to_cv2

add_easy_resize_to_cv2()
import inspect
import functools
import sys
import warnings
from collections.abc import Iterable
import numexpr
import numpy as np

haystackpic = []
needlepics = []
minimum_match = 0.5

tempmodule = sys.modules[__name__]
tempmodule.allftconvolve = {}
import scipy

tempmodule.allftconvolve["normal"] = scipy.signal.fftconvolve
usecupy = False
try:
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve as fftconvolv

    tempmodule.allftconvolve["cp"] = fftconvolv
except Exception as fe:
    print(fe)

try:
    import pyfftw

    scipy.fft.set_backend(pyfftw.interfaces.scipy_fft)
    pyfftw.interfaces.cache.enable()
except Exception as fe:
    print(fe)


class skimage_deprecation(Warning):
    """Create our own deprecation class, since Python >= 2.7
    silences deprecations by default.

    """

    pass


def _get_stack_rank(func):
    """Return function rank in the call stack."""
    if _is_wrapped(func):
        return 1 + _get_stack_rank(func.__wrapped__)
    else:
        return 0


def _is_wrapped(func):
    return "__wrapped__" in dir(func)


def _get_stack_length(func):
    """Return function call stack length."""
    return _get_stack_rank(func.__globals__.get(func.__name__, func))


class _DecoratorBaseClass:
    """Used to manage decorators' warnings stacklevel.

    The `_stack_length` class variable is used to store the number of
    times a function is wrapped by a decorator.

    Let `stack_length` be the total number of times a decorated
    function is wrapped, and `stack_rank` be the rank of the decorator
    in the decorators stack. The stacklevel of a warning is then
    `stacklevel = 1 + stack_length - stack_rank`.
    """

    _stack_length = {}

    def get_stack_length(self, func):
        return self._stack_length.get(func.__name__, _get_stack_length(func))


class change_default_value(_DecoratorBaseClass):
    """Decorator for changing the default value of an argument.

    Parameters
    ----------
    arg_name: str
        The name of the argument to be updated.
    new_value: any
        The argument new value.
    changed_version : str
        The package version in which the change will be introduced.
    warning_msg: str
        Optional warning message. If None, a generic warning message
        is used.

    """

    def __init__(self, arg_name, *, new_value, changed_version, warning_msg=None):
        self.arg_name = arg_name
        self.new_value = new_value
        self.warning_msg = warning_msg
        self.changed_version = changed_version

    def __call__(self, func):
        parameters = inspect.signature(func).parameters
        arg_idx = list(parameters.keys()).index(self.arg_name)
        old_value = parameters[self.arg_name].default

        stack_rank = _get_stack_rank(func)

        if self.warning_msg is None:
            self.warning_msg = (
                f"The new recommended value for {self.arg_name} is "
                f"{self.new_value}. Until version {self.changed_version}, "
                f"the default {self.arg_name} value is {old_value}. "
                f"From version {self.changed_version}, the {self.arg_name} "
                f"default value will be {self.new_value}. To avoid "
                f"this warning, please explicitly set {self.arg_name} value."
            )

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            stacklevel = 1 + self.get_stack_length(func) - stack_rank
            if len(args) < arg_idx + 1 and self.arg_name not in kwargs.keys():
                # warn that arg_name default value changed:
                warnings.warn(self.warning_msg, FutureWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return fixed_func


class remove_arg(_DecoratorBaseClass):
    """Decorator to remove an argument from function's signature.

    Parameters
    ----------
    arg_name: str
        The name of the argument to be removed.
    changed_version : str
        The package version in which the warning will be replaced by
        an error.
    help_msg: str
        Optional message appended to the generic warning message.

    """

    def __init__(self, arg_name, *, changed_version, help_msg=None):
        self.arg_name = arg_name
        self.help_msg = help_msg
        self.changed_version = changed_version

    def __call__(self, func):
        parameters = inspect.signature(func).parameters
        arg_idx = list(parameters.keys()).index(self.arg_name)
        warning_msg = (
            f"{self.arg_name} argument is deprecated and will be removed "
            f"in version {self.changed_version}. To avoid this warning, "
            f"please do not use the {self.arg_name} argument. Please "
            f"see {func.__name__} documentation for more details."
        )

        if self.help_msg is not None:
            warning_msg += f" {self.help_msg}"

        stack_rank = _get_stack_rank(func)

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            stacklevel = 1 + self.get_stack_length(func) - stack_rank
            if len(args) > arg_idx or self.arg_name in kwargs.keys():
                # warn that arg_name is deprecated
                warnings.warn(warning_msg, FutureWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return fixed_func


def docstring_add_deprecated(func, kwarg_mapping, deprecated_version):
    """Add deprecated kwarg(s) to the "Other Params" section of a docstring.

    Parameters
    ---------
    func : function
        The function whose docstring we wish to update.
    kwarg_mapping : dict
        A dict containing {old_arg: new_arg} key/value pairs as used by
        `deprecate_kwarg`.
    deprecated_version : str
        A major.minor version string specifying when old_arg was
        deprecated.

    Returns
    -------
    new_doc : str
        The updated docstring. Returns the original docstring if numpydoc is
        not available.
    """
    if func.__doc__ is None:
        return None
    try:
        from numpydoc.docscrape import FunctionDoc, Parameter
    except ImportError:
        # Return an unmodified docstring if numpydoc is not available.
        return func.__doc__

    Doc = FunctionDoc(func)
    for old_arg, new_arg in kwarg_mapping.items():
        desc = [
            f"Deprecated in favor of `{new_arg}`.",
            "",
            f".. deprecated:: {deprecated_version}",
        ]
        Doc["Other Parameters"].append(
            Parameter(name=old_arg, type="DEPRECATED", desc=desc)
        )
    new_docstring = str(Doc)

    # new_docstring will have a header starting with:
    #
    # .. function:: func.__name__
    #
    # and some additional blank lines. We strip these off below.
    split = new_docstring.split("\n")
    no_header = split[1:]
    while not no_header[0].strip():
        no_header.pop(0)

    # Store the initial description before any of the Parameters fields.
    # Usually this is a single line, but the while loop covers any case
    # where it is not.
    descr = no_header.pop(0)
    while no_header[0].strip():
        descr += "\n    " + no_header.pop(0)
    descr += "\n\n"
    # '\n    ' rather than '\n' here to restore the original indentation.
    final_docstring = descr + "\n    ".join(no_header)
    # strip any extra spaces from ends of lines
    final_docstring = "\n".join([line.rstrip() for line in final_docstring.split("\n")])
    return final_docstring


class deprecate_kwarg(_DecoratorBaseClass):
    """Decorator ensuring backward compatibility when argument names are
    modified in a function definition.

    Parameters
    ----------
    kwarg_mapping: dict
        Mapping between the function's old argument names and the new
        ones.
    deprecated_version : str
        The package version in which the argument was first deprecated.
    warning_msg: str
        Optional warning message. If None, a generic warning message
        is used.
    removed_version : str
        The package version in which the deprecated argument will be
        removed.

    """

    def __init__(
        self, kwarg_mapping, deprecated_version, warning_msg=None, removed_version=None
    ):
        self.kwarg_mapping = kwarg_mapping
        if warning_msg is None:
            self.warning_msg = (
                "`{old_arg}` is a deprecated argument name " "for `{func_name}`. "
            )
            if removed_version is not None:
                self.warning_msg += (
                    f"It will be removed in " f"version {removed_version}. "
                )
            self.warning_msg += "Please use `{new_arg}` instead."
        else:
            self.warning_msg = warning_msg

        self.deprecated_version = deprecated_version

    def __call__(self, func):
        stack_rank = _get_stack_rank(func)

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            stacklevel = 1 + self.get_stack_length(func) - stack_rank

            for old_arg, new_arg in self.kwarg_mapping.items():
                if old_arg in kwargs:
                    #  warn that the function interface has changed:
                    warnings.warn(
                        self.warning_msg.format(
                            old_arg=old_arg, func_name=func.__name__, new_arg=new_arg
                        ),
                        FutureWarning,
                        stacklevel=stacklevel,
                    )
                    # Substitute new_arg to old_arg
                    kwargs[new_arg] = kwargs.pop(old_arg)

            # Call the function with the fixed arguments
            return func(*args, **kwargs)

        if func.__doc__ is not None:
            newdoc = docstring_add_deprecated(
                func, self.kwarg_mapping, self.deprecated_version
            )
            fixed_func.__doc__ = newdoc
        return fixed_func


class channel_as_last_axis:
    """Decorator for automatically making channels axis last for all arrays.

    This decorator reorders axes for compatibility with functions that only
    support channels along the last axis. After the function call is complete
    the channels axis is restored back to its original position.

    Parameters
    ----------
    channel_arg_positions : tuple of int, optional
        Positional arguments at the positions specified in this tuple are
        assumed to be multichannel arrays. The default is to assume only the
        first argument to the function is a multichannel array.
    channel_kwarg_names : tuple of str, optional
        A tuple containing the names of any keyword arguments corresponding to
        multichannel arrays.
    multichannel_output : bool, optional
        A boolean that should be True if the output of the function is not a
        multichannel array and False otherwise. This decorator does not
        currently support the general case of functions with multiple outputs
        where some or all are multichannel.

    """

    def __init__(
        self,
        channel_arg_positions=(0,),
        channel_kwarg_names=(),
        multichannel_output=True,
    ):
        self.arg_positions = set(channel_arg_positions)
        self.kwarg_names = set(channel_kwarg_names)
        self.multichannel_output = multichannel_output

    def __call__(self, func):
        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            channel_axis = kwargs.get("channel_axis", None)

            if channel_axis is None:
                return func(*args, **kwargs)

            # TODO: convert scalars to a tuple in anticipation of eventually
            #       supporting a tuple of channel axes. Right now, only an
            #       integer or a single-element tuple is supported, though.
            if np.isscalar(channel_axis):
                channel_axis = (channel_axis,)
            if len(channel_axis) > 1:
                raise ValueError("only a single channel axis is currently supported")

            if channel_axis == (-1,) or channel_axis == -1:
                return func(*args, **kwargs)

            if self.arg_positions:
                new_args = []
                for pos, arg in enumerate(args):
                    if pos in self.arg_positions:
                        new_args.append(np.moveaxis(arg, channel_axis[0], -1))
                    else:
                        new_args.append(arg)
                new_args = tuple(new_args)
            else:
                new_args = args

            for name in self.kwarg_names:
                kwargs[name] = np.moveaxis(kwargs[name], channel_axis[0], -1)

            # now that we have moved the channels axis to the last position,
            # change the channel_axis argument to -1
            kwargs["channel_axis"] = -1

            # Call the function with the fixed arguments
            out = func(*new_args, **kwargs)
            if self.multichannel_output:
                out = np.moveaxis(out, -1, channel_axis[0])
            return out

        return fixed_func


class deprecated:
    """Decorator to mark deprecated functions with warning.

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.

    Parameters
    ----------
    alt_func : str
        If given, tell user what function to use instead.
    behavior : {'warn', 'raise'}
        Behavior during call to deprecated function: 'warn' = warn user that
        function is deprecated; 'raise' = raise error.
    removed_version : str
        The package version in which the deprecated function will be removed.
    """

    def __init__(self, alt_func=None, behavior="warn", removed_version=None):
        self.alt_func = alt_func
        self.behavior = behavior
        self.removed_version = removed_version

    def __call__(self, func):
        alt_msg = ""
        if self.alt_func is not None:
            alt_msg = f" Use ``{self.alt_func}`` instead."
        rmv_msg = ""
        if self.removed_version is not None:
            rmv_msg = f" and will be removed in version {self.removed_version}"

        msg = f"Function ``{func.__name__}`` is deprecated{rmv_msg}.{alt_msg}"

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if self.behavior == "warn":
                func_code = func.__code__
                warnings.simplefilter("always", skimage_deprecation)
                warnings.warn_explicit(
                    msg,
                    category=skimage_deprecation,
                    filename=func_code.co_filename,
                    lineno=func_code.co_firstlineno + 1,
                )
            elif self.behavior == "raise":
                raise skimage_deprecation(msg)
            return func(*args, **kwargs)

        # modify doc string to display deprecation warning
        doc = "**Deprecated function**." + alt_msg
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + "\n\n    " + wrapped.__doc__

        return wrapped


def get_bound_method_class(m):
    """Return the class for a bound method."""
    return m.im_class if sys.version < "3" else m.__self__.__class__


def safe_as_int(val, atol=1e-3):
    """
    Attempt to safely cast values to integer format.

    Parameters
    ----------
    val : scalar or iterable of scalars
        Number or container of numbers which are intended to be interpreted as
        integers, e.g., for indexing purposes, but which may not carry integer
        type.
    atol : float
        Absolute tolerance away from nearest integer to consider values in
        ``val`` functionally integers.

    Returns
    -------
    val_int : NumPy scalar or ndarray of dtype `np.int64`
        Returns the input value(s) coerced to dtype `np.int64` assuming all
        were within ``atol`` of the nearest integer.

    Notes
    -----
    This operation calculates ``val`` modulo 1, which returns the mantissa of
    all values. Then all mantissas greater than 0.5 are subtracted from one.
    Finally, the absolute tolerance from zero is calculated. If it is less
    than ``atol`` for all value(s) in ``val``, they are rounded and returned
    in an integer array. Or, if ``val`` was a scalar, a NumPy scalar type is
    returned.

    If any value(s) are outside the specified tolerance, an informative error
    is raised.

    Examples
    --------
    >>> safe_as_int(7.0)
    7

    >>> safe_as_int([9, 4, 2.9999999999])
    array([9, 4, 3])

    >>> safe_as_int(53.1)
    Traceback (most recent call last):
        ...
    ValueError: Integer argument required but received 53.1, check inputs.

    >>> safe_as_int(53.01, atol=0.01)
    53

    """
    mod = np.asarray(val) % 1  # Extract mantissa

    # Check for and subtract any mod values > 0.5 from 1
    if mod.ndim == 0:  # Scalar input, cannot be indexed
        if mod > 0.5:
            mod = 1 - mod
    else:  # Iterable input, now ndarray
        mod[mod > 0.5] = 1 - mod[mod > 0.5]  # Test on each side of nearest int

    try:
        np.testing.assert_allclose(mod, 0, atol=atol)
    except AssertionError:
        raise ValueError(
            f"Integer argument required but received " f"{val}, check inputs."
        )

    return np.round(val).astype(np.int64)


def check_shape_equality(*images):
    """Check that all images have the same shape"""
    image0 = images[0]
    if not all(image0.shape == image.shape for image in images[1:]):
        raise ValueError("Input images must have the same dimensions.")
    return


def slice_at_axis(sl, axis):
    """
    Construct tuple of slices to slice an array in the given dimension.

    Parameters
    ----------
    sl : slice
        The slice for the given dimension.
    axis : int
        The axis to which `sl` is applied. All other dimensions are left
        "unsliced".

    Returns
    -------
    sl : tuple of slices
        A tuple with slices matching `shape` in length.

    Examples
    --------
    >>> slice_at_axis(slice(None, 3, -1), 1)
    (slice(None, None, None), slice(None, 3, -1), Ellipsis)
    """
    return (slice(None),) * axis + (sl,) + (...,)


def reshape_nd(arr, ndim, dim):
    """Reshape a 1D array to have n dimensions, all singletons but one.

    Parameters
    ----------
    arr : array, shape (N,)
        Input array
    ndim : int
        Number of desired dimensions of reshaped array.
    dim : int
        Which dimension/axis will not be singleton-sized.

    Returns
    -------
    arr_reshaped : array, shape ([1, ...], N, [1,...])
        View of `arr` reshaped to the desired shape.

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> arr = rng.random(7)
    >>> reshape_nd(arr, 2, 0).shape
    (7, 1)
    >>> reshape_nd(arr, 3, 1).shape
    (1, 7, 1)
    >>> reshape_nd(arr, 4, -1).shape
    (1, 1, 1, 7)
    """
    if arr.ndim != 1:
        raise ValueError("arr must be a 1D array")
    new_shape = [1] * ndim
    new_shape[dim] = -1
    return np.reshape(arr, new_shape)


def check_nD(array, ndim, arg_name="image"):
    """
    Verify an array meets the desired ndims and array isn't empty.

    Parameters
    ----------
    array : array-like
        Input array to be validated
    ndim : int or iterable of ints
        Allowable ndim or ndims for the array.
    arg_name : str, optional
        The name of the array in the original function.

    """
    array = np.asanyarray(array)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if array.ndim not in ndim:
        raise ValueError(
            msg_incorrect_dim % (arg_name, "-or-".join([str(n) for n in ndim]))
        )


def convert_to_float(image, preserve_range):
    """Convert input image to float image with the appropriate range.

    Parameters
    ----------
    image : ndarray
        Input image.
    preserve_range : bool
        Determines if the range of the image should be kept or transformed
        using img_as_float. Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Notes
    -----
    * Input images with `float32` data type are not upcast.

    Returns
    -------
    image : ndarray
        Transformed version of the input.

    """
    if image.dtype == np.float16:
        return image.astype(np.float32)
    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype.char not in "df":
            image = image.astype(float)
    else:
        image = img_as_float(image)
    return image


def _validate_interpolation_order(image_dtype, order):
    """Validate and return spline interpolation's order.

    Parameters
    ----------
    image_dtype : dtype
        Image dtype.
    order : int, optional
        The order of the spline interpolation. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.

    Returns
    -------
    order : int
        if input order is None, returns 0 if image_dtype is bool and 1
        otherwise. Otherwise, image_dtype is checked and input order
        is validated accordingly (order > 0 is not supported for bool
        image dtype)

    """

    if order is None:
        return 0 if image_dtype == bool else 1

    if order < 0 or order > 5:
        raise ValueError("Spline interpolation order has to be in the " "range 0-5.")

    if image_dtype == bool and order != 0:
        raise ValueError(
            "Input image dtype is bool. Interpolation is not defined "
            "with bool data type. Please set order to 0 or explicitly "
            "cast input image to another data type."
        )

    return order


def _to_np_mode(mode):
    """Convert padding modes from `ndi.correlate` to `np.pad`."""
    mode_translation_dict = dict(nearest="edge", reflect="symmetric", mirror="reflect")
    if mode in mode_translation_dict:
        mode = mode_translation_dict[mode]
    return mode


def _to_ndimage_mode(mode):
    """Convert from `numpy.pad` mode name to the corresponding ndimage mode."""
    mode_translation_dict = dict(
        constant="constant",
        edge="nearest",
        symmetric="reflect",
        reflect="mirror",
        wrap="wrap",
    )
    if mode not in mode_translation_dict:
        raise ValueError(
            f"Unknown mode: '{mode}', or cannot translate mode. The "
            f"mode should be one of 'constant', 'edge', 'symmetric', "
            f"'reflect', or 'wrap'. See the documentation of numpy.pad for "
            f"more info."
        )
    return _fix_ndimage_mode(mode_translation_dict[mode])


def _fix_ndimage_mode(mode):
    # SciPy 1.6.0 introduced grid variants of constant and wrap which
    # have less surprising behavior for images. Use these when available
    grid_modes = {"constant": "grid-constant", "wrap": "grid-wrap"}
    return grid_modes.get(mode, mode)


new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    "g": np.float64,  # np.float128 ; doesn't exist on windows
    "G": np.complex128,  # np.complex256 ; doesn't exist on windows
}


def _supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.

    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.

    Parameters
    ----------
    input_dtype : np.dtype or Iterable of np.dtype
        The input dtype. If a sequence of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `np.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.

    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
    """
    if isinstance(input_dtype, Iterable) and not isinstance(input_dtype, str):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == "c":
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def identity(image, *args, **kwargs):
    """Returns the first argument unmodified."""
    return image


def as_binary_ndarray(array, *, variable_name):
    """Return `array` as a numpy.ndarray of dtype bool.

    Raises
    ------
    ValueError:
        An error including the given `variable_name` if `array` can not be
        safely cast to a boolean array.
    """
    array = np.asarray(array)
    if array.dtype != bool:
        if np.any((array != 1) & (array != 0)):
            raise ValueError(
                f"{variable_name} array is not of dtype boolean or "
                f"contains values other than 0 and 1 so cannot be "
                f"safely cast to boolean array."
            )
    return np.asarray(array, dtype=bool)


def _window_sum_2d(image, window_shape):
    window_sum = np.cumsum(image, axis=0)
    window_sum = window_sum[window_shape[0] : -1] - window_sum[: -window_shape[0] - 1]

    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (
        window_sum[:, window_shape[1] : -1] - window_sum[:, : -window_shape[1] - 1]
    )

    return window_sum


def _window_sum_3d(image, window_shape):
    window_sum = _window_sum_2d(image, window_shape)

    window_sum = np.cumsum(window_sum, axis=2)
    window_sum = (
        window_sum[:, :, window_shape[2] : -1]
        - window_sum[:, :, : -window_shape[2] - 1]
    )

    return window_sum


def match_template(
    image,
    template,
    pad_input=False,
    mode="constant",
    constant_values=0,
):
    """Match a template to a 2-D or 3-D image using normalized correlation.

    The output is an array with values between -1.0 and 1.0. The value at a
    given position corresponds to the correlation coefficient between the image
    and the template.

    For `pad_input=True` matches correspond to the center and otherwise to the
    top-left corner of the template. To find the best match you must search for
    peaks in the response (output) image.

    Parameters
    ----------
    image : (M, N[, D]) array
        2-D or 3-D input image.
    template : (m, n[, d]) array
        Template to locate. It must be `(m <= M, n <= N[, d <= D])`.
    pad_input : bool
        If True, pad `image` so that output is the same size as the image, and
        output values correspond to the template center. Otherwise, the output
        is an array with shape `(M - m + 1, N - n + 1)` for an `(M, N)` image
        and an `(m, n)` template, and matches correspond to origin
        (top-left corner) of the template.
    mode : see `numpy.pad`, optional
        Padding mode.
    constant_values : see `numpy.pad`, optional
        Constant values used in conjunction with ``mode='constant'``.

    Returns
    -------
    output : array
        Response image with correlation coefficients.

    Notes
    -----
    Details on the cross-correlation are presented in [1]_. This implementation
    uses FFT convolutions of the image and the template. Reference [2]_
    presents similar derivations but the approximation presented in this
    reference is not used in our implementation.

    References
    ----------
    .. [1] J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light
           and Magic.
    .. [2] Briechle and Hanebeck, "Template Matching using Fast Normalized
           Cross Correlation", Proceedings of the SPIE (2001).
           :DOI:`10.1117/12.421129`

    Examples
    --------
    >>> template = np.zeros((3, 3))
    >>> template[1, 1] = 1
    >>> template
    array([[0., 0., 0.],
           [0., 1., 0.],
           [0., 0., 0.]])
    >>> image = np.zeros((6, 6))
    >>> image[1, 1] = 1
    >>> image[4, 4] = -1
    >>> image
    array([[ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.]])
    >>> result = match_template(image, template)
    >>> np.round(result, 3)
    array([[ 1.   , -0.125,  0.   ,  0.   ],
           [-0.125, -0.125,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.125,  0.125],
           [ 0.   ,  0.   ,  0.125, -1.   ]])
    >>> result = match_template(image, template, pad_input=True)
    >>> np.round(result, 3)
    array([[-0.125, -0.125, -0.125,  0.   ,  0.   ,  0.   ],
           [-0.125,  1.   , -0.125,  0.   ,  0.   ,  0.   ],
           [-0.125, -0.125, -0.125,  0.   ,  0.   ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.125,  0.125,  0.125],
           [ 0.   ,  0.   ,  0.   ,  0.125, -1.   ,  0.125],
           [ 0.   ,  0.   ,  0.   ,  0.125,  0.125,  0.125]])
    """

    image_shape = image.shape
    pad_width = tuple((width, width) for width in template.shape)
    if mode == "constant":
        image = np.pad(
            image, pad_width=pad_width, mode=mode, constant_values=constant_values
        )
    else:
        image = np.pad(image, pad_width=pad_width, mode=mode)
    nuim = numexpr.evaluate("image ** 2", global_dict={}, local_dict={"image": image})
    if image.ndim == 2:
        image_window_sum = _window_sum_2d(image, template.shape)
        image_window_sum2 = _window_sum_2d(nuim, template.shape)
    elif image.ndim == 3:
        image_window_sum = _window_sum_3d(image, template.shape)
        image_window_sum2 = _window_sum_3d(nuim, template.shape)

    template_mean = template.mean()
    template_volume = math.prod(template.shape)
    # template_ssd = np.sum((template - template_mean) ** 2)
    template_ssd = numexpr.evaluate(
        "sum((template - template_mean) ** 2)",
        global_dict={},
        local_dict={"template": template, "template_mean": template_mean},
    )

    if not usecupy:
        if image.ndim == 2:
            xcorr = tempmodule.allftconvolve["normal"](
                image, template[::-1, ::-1], mode="valid"
            )[1:-1, 1:-1]
        elif image.ndim == 3:
            xcorr = tempmodule.allftconvolve["normal"](
                image, template[::-1, ::-1, ::-1], mode="valid"
            )[1:-1, 1:-1, 1:-1]
    else:
        if image.ndim == 2:
            xcorr2 = cp.array(
                tempmodule.allftconvolve["cp"](
                    cp.array(image), cp.array(template[::-1, ::-1]), mode="valid"
                )[1:-1, 1:-1]
            )
        elif image.ndim == 3:
            xcorr2 = cp.array(
                tempmodule.allftconvolve["cp"](
                    cp.array(image), cp.array(template[::-1, ::-1, ::-1]), mode="valid"
                )[1:-1, 1:-1, 1:-1]
            )

        xcorr = xcorr2.get()
    numerator = numexpr.evaluate(
        "xcorr - image_window_sum * template_mean",
        global_dict={},
        local_dict={
            "xcorr": xcorr,
            "image_window_sum": image_window_sum,
            "template_mean": template_mean,
        },
    )
    denominator = image_window_sum2
    numexpr.evaluate(
        "image_window_sum * image_window_sum",
        out=image_window_sum,
        global_dict={},
        local_dict={
            "image_window_sum": image_window_sum,
        },
    )
    numexpr.evaluate(
        "image_window_sum / template_volume",
        out=image_window_sum,
        global_dict={},
        local_dict={
            "image_window_sum": image_window_sum,
            "template_volume": template_volume,
        },
    )
    numexpr.evaluate(
        "denominator - image_window_sum",
        out=denominator,
        global_dict={},
        local_dict={"denominator": denominator, "image_window_sum": image_window_sum},
    )
    numexpr.evaluate(
        "denominator * template_ssd",
        out=denominator,
        global_dict={},
        local_dict={"denominator": denominator, "template_ssd": template_ssd},
    )

    denominator[
        numexpr.evaluate(
            "denominator < 0", global_dict={}, local_dict={"denominator": denominator}
        )
    ] = 0
    numexpr.evaluate(
        "sqrt(denominator)",
        out=denominator,
        global_dict={},
        local_dict={"denominator": denominator},
    )
    response = np.zeros_like(xcorr, dtype=np.float64)

    va = np.finfo(np.float64).eps
    mask = numexpr.evaluate(
        "denominator > va",
        global_dict={},
        local_dict={"denominator": denominator, "va": va},
    )
    nummask = numerator[mask]
    demask = denominator[mask]
    response[mask] = numexpr.evaluate(
        "nummask / demask",
        global_dict={},
        local_dict={"nummask": nummask, "demask": demask},
    )

    if pad_input:
        return response[
            tuple(
                (
                    slice((d0 := template.shape[i] - 1 // 2), d0 + image_shape[i])
                    for i in range(template.ndim)
                )
            )
        ]
    else:
        return response[
            tuple(
                (
                    slice(
                        (d0 := template.shape[i] - 1 // 2),
                        d0 + image_shape[i] - template.shape[i] + 1,
                    )
                    for i in range(template.ndim)
                )
            )
        ]


def convert_to_normal_dict(di):
    if isinstance_tolerant(di, defaultdict):
        di = {k: convert_to_normal_dict(v) for k, v in di.items()}
    return di


def groupBy(key, seq, continue_on_exceptions=True, withindex=True, withvalue=True):
    indexcounter = -1

    def execute_f(k, v):
        nonlocal indexcounter
        indexcounter += 1
        try:
            return k(v)
        except Exception as fa:
            if continue_on_exceptions:
                return "EXCEPTION: " + str(fa)
            else:
                raise fa

    # based on https://stackoverflow.com/a/60282640/15096247
    if withvalue:
        return convert_to_normal_dict(
            reduce(
                lambda grp, val: grp[execute_f(key, val)].append(
                    val if not withindex else (indexcounter, val)
                )
                or grp,
                seq,
                defaultdict(list),
            )
        )
    return convert_to_normal_dict(
        reduce(
            lambda grp, val: grp[execute_f(key, val)].append(indexcounter) or grp,
            seq,
            defaultdict(list),
        )
    )


def get_screenshot(monitor=0):
    with mss.mss() as sct:
        a = np.array(sct.grab(sct.monitors[monitor]))
    return open_image_in_cv(a, channels_in_output=3)


def get_tmpfile(suffix=".bin"):
    tfp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    filename = tfp.name
    filename = os.path.normpath(filename)
    tfp.close()
    touch(filename)
    return filename


def tempfolder():
    tempfolder = tempfile.TemporaryDirectory()
    tempfolder.cleanup()
    if not os.path.exists(tempfolder.name):
        os.makedirs(tempfolder.name)

    return tempfolder.name


def start_annotation_tool():
    dirname = os.path.normpath(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "tempannotations")
    )

    annotationtoolfiles = [
        x
        for x in [
            "canvas.min.js",
            "filesaver.min.js",
            "jszip.min.js",
            "ybat.css",
            "ybat.html",
            "ybat.js",
        ]
        if os.path.exists(os.path.join(dirname, x))
    ]
    if len(annotationtoolfiles) == 6:
        main_path_stringuri = pathlib.Path(os.path.join(dirname, "ybat.html")).as_uri()
        webbrowser.open(main_path_stringuri)
        print(
            "The annotation tool requires a class list like:\n\n\nclasses.txt\n\nbutton1\nbutton2\ntextfield1\ntextfield2"
        )
        print(
            "\n\nClick on: Save COCO when you are done and copy the file name of the generated file"
        )
    else:
        print(
            f"The annotation tool could not be found! Download it here:\nhttps://github.com/drainingsun/ybat\n\nPut it the folder of this module:\n{dirname}"
        )
    print(dirname)


def cropimage(img, coords):
    if sum(coords) == 0:
        return img
    return img[coords[1] : coords[1] + coords[3], coords[0] : coords[0] + coords[2]]


def save_screenshot():
    tmpfile = get_tmpfile(suffix=".png")
    screenshot = get_screenshot()
    cv2.imwrite(tmpfile, screenshot)
    return tmpfile


def resize_image(img, scale_percent=100, interpolation=cv2.INTER_LANCZOS4):
    width = int(img.shape[1] / 100 * scale_percent)
    height = int(img.shape[0] / 100 * scale_percent)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=interpolation)
    return resized


def extract_images_from_annotations(
    folder,
    zipfile,
    screenshot,
    min_percentage=40,
    max_percentage=140,
    interpolation=cv2.INTER_LANCZOS4,
):
    # start_annotation_tool()
    tmpfold = tempfolder()
    screenshot = open_image_in_cv(screenshot, channels_in_output=3)
    extract(filepath=zipfile, dest_dir=tmpfold)
    jsonfile = [
        x.path
        for x in get_folder_file_complete_path(tmpfold)
        if x.ext.lower() == ".json"
    ][0]
    with open(jsonfile, mode="r", encoding="utf-8") as f:
        jsondata = json.loads(f.read())
    dataclasses = [
        [
            cropimage(screenshot, x["bbox"]),
            x["id"],
            *[y["name"] for y in jsondata["categories"] if y["id"] == x["id"]],
        ]
        for x in jsondata["annotations"]
    ]
    dataclassesgroup = groupBy(
        key=lambda q: q[-1],
        seq=dataclasses,
        continue_on_exceptions=True,
        withindex=True,
        withvalue=True,
    )
    savedimages = []
    for key, item in dataclassesgroup.items():
        for _item in item:
            d = 0

            for item2 in _item:
                c = 0

                if isinstance_tolerant(item2, list):
                    _im, _id, _name = item2
                    save_cv_image(
                        g := os.path.normpath(
                            os.path.join(
                                folder, f"{_name}___{d}___{str(c).zfill(6)}___100.png"
                            )
                        ),
                        _im,
                    )
                    c += 1
                    savedimages.append(g)
                    if min_percentage < 100:
                        for smaller in range(min_percentage, 100):
                            imres = resize_image(
                                _im, scale_percent=smaller, interpolation=interpolation
                            )
                            save_cv_image(
                                g := os.path.normpath(
                                    os.path.join(
                                        folder,
                                        f"{_name}___{d}___{str(c).zfill(6)}___{str(smaller).zfill(3)}.png",
                                    )
                                ),
                                imres,
                            )
                            c += 1
                            savedimages.append(g)
                    if max_percentage > 101:
                        for bigger in range(101, max_percentage):
                            imres = resize_image(
                                _im, scale_percent=bigger, interpolation=interpolation
                            )
                            save_cv_image(
                                g := os.path.normpath(
                                    os.path.join(
                                        folder,
                                        f"{_name}___{d}___{str(c).zfill(6)}___{str(bigger).zfill(3)}.png",
                                    )
                                ),
                                imres,
                            )
                            c += 1
                            savedimages.append(g)
            d += 1
    return savedimages


@cache
def openneedle(pa):
    bi = open_image_in_cv(pa, channels_in_output=3)
    b, g, r = np.mean(bi[..., 0]), np.mean(bi[..., 1]), np.mean(bi[..., 2])
    return (b, g, r), np.ascontiguousarray(open_image_in_cv(bi, channels_in_output=2))


def load_needle_images(folder, min_percentage=80, max_percentage=110):
    data = [
        x.file.split("___", maxsplit=3) + [x.path]
        for x in get_folder_file_complete_path(folder)
        if re.match(r".*?___\d___\d{6}___\d{3}\.png", x.file)
    ]
    data2 = [
        [x[0], int(x[1]), int(x[2]), int(x[3].split(".", maxsplit=1)[0]), x[4]]
        for x in data
        if len(x) == 5
    ]
    data3 = [
        [
            *x,
            *openneedle(x[-1])
            # .astype(np.float64)
            ,
        ]
        for x in data2
        if x[3] <= max_percentage and x[3] >= min_percentage
    ]
    data3 = [
        x[:-1]
        + [RawArray(np.ctypeslib.as_ctypes_type(x[-1].dtype), x[-1].flatten())]
        + [x[-1].shape]
        for x in data3
    ]

    return data3


def _apply_matching_template(
    haystack: np.ndarray,
    needle: np.ndarray,
):
    reas = match_template(
        haystack,
        needle,
        pad_input=False,
        mode="constant",
        constant_values=0,
    )
    return reas


def find_needles(
    haystack,
    needles,
    minimum_match=0.5,
    processes=4,
    percent=50,
    max_diff_r=30,
    max_diff_g=30,
    max_diff_b=30,
    use_cupy=False,
    max_color_difference_factor=5,
    screenshot=None,
):
    d = open_image_in_cv(haystack, channels_in_output=2)
    d = d.astype(np.float64)
    r = RawArray(np.ctypeslib.as_ctypes_type(d.dtype), d.flatten())
    allresults = []

    neely2 = [
        needles[q : q + math.ceil(len(needles) / processes)]
        for q in range(0, len(needles), math.ceil(len(needles) / processes))
    ]
    with multiprocessing.Pool(
        processes=processes,
        initializer=init_multi,
        initargs=(r, d.shape, neely2, minimum_match, use_cupy),
    ) as pool:
        senhas = pool.imap_unordered(apply_matching_template, range(len(neely2)))
        for senha in senhas:
            allresults.extend(senha)
    df = pd.DataFrame(allresults)

    allfound = []
    df = find_rectangle_intersections(
        df,
        columns=("aa_startx", "aa_starty", "aa_endx", "aa_endy"),
        new_column="aa_intersecting",
        dtype=np.int32,
        convert_to_tuples=True,
    )
    for name, group in df.groupby("aa_intersecting"):
        allfound.append(
            group.sort_values(
                by=["aa_match", "aa_width", "aa_height"],
                ascending=[False, False, False],
            )[:1]
        )
    df = (
        pd.concat(allfound, ignore_index=True)
        .sort_values(by=["aa_class_name", "aa_match"], ascending=[True, False])
        .reset_index(drop=True)
    )
    changes = [
        "aa_startx",
        "aa_starty",
        "aa_endx",
        "aa_endy",
        "aa_max",
        "aa_max_x",
        "aa_max_y",
        "aa_width",
        "aa_height",
    ]
    for change in changes:
        ar = df[change].__array__()
        df[change] = (
            numexpr.evaluate(
                "((ar / percent) * 100)",
                global_dict={},
                local_dict={"ar": ar, "percent": percent},
            )
        ).astype(int)

    u = df.apply(
        lambda x: pd.Series(
            [
                [
                    r := cropimage(
                        screenshot,
                        (x["aa_startx"], x["aa_starty"], x["aa_width"], x["aa_height"]),
                    )
                ],
                np.mean(r[..., 0]),
                np.mean(r[..., 1]),
                np.mean(r[..., 2]),
            ][1:]
        ),
        axis=1,
    ).rename(columns={0: "aa_r0", 1: "aa_g0", 2: "aa_b0"})

    df = pd.concat([df, u], axis=1)

    df = (
        df.loc[
            numexpr.evaluate(
                """(((ceil1 / ceil2 <= max_color_difference_factor) & (ceil3 / ceil4 <= max_color_difference_factor) & (ceil5 / ceil6 <= max_color_difference_factor) & (ceil2 / ceil1 <= max_color_difference_factor) & (ceil4 / ceil3 <= max_color_difference_factor) & (ceil6 / ceil5 <= max_color_difference_factor)))""",
                global_dict={},
                local_dict={
                    "ceil1": np.abs(np.ceil(df.aa_b)),
                    "ceil2": np.abs(np.ceil(df.aa_b0)),
                    "ceil3": np.abs(np.ceil(df.aa_g)),
                    "ceil4": np.abs(np.ceil(df.aa_g0)),
                    "ceil5": np.abs(np.ceil(df.aa_r)),
                    "ceil6": np.abs(np.ceil(df.aa_r0)),
                    "max_color_difference_factor": max_color_difference_factor,
                },
            )
            & numexpr.evaluate(
                "(abs(aa_b - aa_b0) < max_diff_b) & (abs(aa_g - aa_g0) < max_diff_g) & (abs(aa_r - aa_r0) < max_diff_r)",
                global_dict={},
                local_dict={
                    "aa_b": df.aa_b,
                    "aa_b0": df.aa_b0,
                    "max_diff_b": max_diff_b,
                    "aa_g": df.aa_g,
                    "aa_g0": df.aa_g0,
                    "max_diff_g": max_diff_g,
                    "aa_r": df.aa_r,
                    "aa_r0": df.aa_r0,
                    "max_diff_r": max_diff_r,
                },
            )
        ]
        .reset_index(drop=True)
        .copy()
    )
    (df.aa_b) = (df.aa_b).astype(np.uint8)
    (df.aa_b0) = (df.aa_b0).astype(np.uint8)
    (df.aa_g) = (df.aa_g).astype(np.uint8)
    (df.aa_g0) = (df.aa_g0).astype(np.uint8)
    (df.aa_r) = (df.aa_r).astype(np.uint8)
    (df.aa_r0) = (df.aa_r0).astype(np.uint8)
    return df


def init_multi(arr, shape, neely2, minimum, usecup):
    global minimum_match, usecupy
    usecupy = usecup
    minimum_match = minimum
    haystackpic.append(np.frombuffer(arr, dtype=np.float64).reshape(shape))
    needlepics.extend(neely2)


def apply_matching_template(arrs):
    allresis = []
    for n in needlepics[arrs]:
        arr = np.frombuffer(n[-2], dtype=np.uint8).reshape(n[-1])
        vari = _apply_matching_template(
            haystackpic[-1],
            arr,
        )
        ma1 = numexpr.evaluate(
            "max(vari,axis=1)", global_dict={}, local_dict={"vari": vari}
        )
        ma0 = numexpr.evaluate(
            "max(vari,axis=0)", global_dict={}, local_dict={"vari": vari}
        )
        xxx = (vari, ma0, ma1, *n[:-2], n[-1])
        result = xxx[0]
        yy, xx = np.unravel_index(np.argmax(result), result.shape)
        maxval = result[yy, xx]
        if maxval < minimum_match:
            continue
        be = numexpr.evaluate(
            f"(result > {minimum_match})", global_dict={}, local_dict={"result": result}
        )
        be1 = result[be]
        be2 = np.where(be)
        starty, startx = be2
        endy = starty + xxx[-1][0]
        endx = startx + xxx[-1][-1]
        for a, b, c, d, e in zip(startx, starty, endx, endy, be1):
            allresis.append(
                {
                    "aa_match": e,
                    "aa_startx": a,
                    "aa_starty": b,
                    "aa_endx": c,
                    "aa_endy": d,
                    "aa_max": maxval,
                    "aa_max_x": xx,
                    "aa_max_y": yy,
                    "aa_width": n[-1][1],
                    "aa_height": n[-1][0],
                    "aa_r": n[-3][0],
                    "aa_g": n[-3][1],
                    "aa_b": n[-3][2],
                    "aa_file": n[-4],
                    "aa_class_name": n[0],
                    "aa_class": n[1],
                    "aa_index": n[2],
                    "aa_percentage": n[3],
                }
            )

    return allresis


def take_screenshot_and_find(
    needles,
    screenshot=None,
    minimum_match=0.5,
    monitor=1,
    percent=50,
    processes=4,
    interpolation=cv2.INTER_LANCZOS4,
    max_diff_r=30,
    max_diff_g=30,
    max_diff_b=30,
    use_cupy=False,
    max_color_difference_factor=5,
):
    if isinstance_tolerant(screenshot, None):
        d = get_screenshot(monitor=monitor)
        screenshot = d.copy()
    else:
        d = screenshot.copy()
    d = cv2.easy_resize_image(
        d, width=None, height=None, percent=percent, interpolation=interpolation
    )
    try:
        df = find_needles(
            d,
            needles,
            minimum_match=minimum_match,
            processes=processes,
            percent=percent,
            max_diff_r=max_diff_r,
            max_diff_g=max_diff_g,
            max_diff_b=max_diff_b,
            use_cupy=use_cupy,
            max_color_difference_factor=max_color_difference_factor,
            screenshot=screenshot,
        )
    except Exception as fe:
        print(fe)
        df = pd.DataFrame(
            columns=[
                "aa_match",
                "aa_startx",
                "aa_starty",
                "aa_endx",
                "aa_endy",
                "aa_max",
                "aa_max_x",
                "aa_max_y",
                "aa_width",
                "aa_height",
                "aa_r",
                "aa_g",
                "aa_b",
                "aa_file",
                "aa_class_name",
                "aa_class",
                "aa_index",
                "aa_percentage",
                "aa_intersecting",
                "aa_r0",
                "aa_g0",
                "aa_b0",
            ]
        )

    return df


def draw_results(df, image, fontdistance1=-10):
    for key, item in df.iterrows():
        x_1_match0 = int(item.aa_startx)
        y_1_match0 = int(item.aa_starty)
        height_haystack0 = int(item.aa_endy)
        width_haystack0 = int(item.aa_endx)
        r_, g_, b_ = (
            random.randrange(50, 255),
            random.randrange(50, 255),
            random.randrange(50, 255),
        )
        image = cv2.rectangle(
            image,
            (x_1_match0, y_1_match0),
            (width_haystack0, height_haystack0),
            color=(0, 0, 0),
            thickness=10,
        )
        image = cv2.rectangle(
            image,
            (x_1_match0, y_1_match0),
            (width_haystack0, height_haystack0),
            color=(r_, g_, b_),
            thickness=5,
        )
        image = cv2.putText(
            image,
            h := f"{key} "
            + str(item.aa_class_name)
            + " ["
            + str(round(item.aa_match, 2))
            + "]",
            (x_1_match0, y_1_match0 - fontdistance1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            4,
        )
        image = cv2.putText(
            image,
            h,
            (x_1_match0, y_1_match0 - fontdistance1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (r_, g_, b_),
            2,
        )
    return image


class TemplateMatching:
    def __init__(
        self,
        screenshot_buffer=10,
        use_cupy=False,
    ):
        self.use_cupy = use_cupy
        self.screenshot_buffer = deque([], screenshot_buffer)
        self.needles = []

    def get_screenshot(self, monitor=1):
        self.screenshot_buffer.append(get_screenshot(monitor=monitor))
        return self

    def save_last_screenshot(self, path):
        save_cv_image(path, self.screenshot_buffer[-1])
        return self

    def start_annotation_tool(self):
        p = save_screenshot()
        print(f"Screenshot saved: {p}")
        start_annotation_tool()
        return self

    def extract_images_from_annotations(
        self,
        folder,
        zipfile,
        screenshot,
        min_percentage=40,
        max_percentage=140,
        interpolation=cv2.INTER_LANCZOS4,
    ):
        return extract_images_from_annotations(
            folder,
            zipfile,
            screenshot,
            min_percentage=min_percentage,
            max_percentage=max_percentage,
            interpolation=interpolation,
        )

    def load_needle_images(self, folder, min_percentage=80, max_percentage=110):
        self.needles.extend(
            load_needle_images(
                folder, min_percentage=min_percentage, max_percentage=max_percentage
            )
        )
        return self

    def find_needles(
        self,
        haystack=None,
        percent=100,
        minimum_match=0.5,
        processes=4,
        max_diff_r=30,
        max_diff_g=30,
        max_diff_b=30,
        max_color_difference_factor=2,
        interpolation=cv2.INTER_LANCZOS4,
    ):
        if isinstance_tolerant(haystack, None):
            haystack = self.screenshot_buffer[-1]
        else:
            haystack = open_image_in_cv(haystack, channels_in_output=3)

        return take_screenshot_and_find(
            self.needles,
            screenshot=haystack,
            minimum_match=minimum_match,
            monitor=1,
            percent=percent,
            processes=processes,
            interpolation=interpolation,
            max_diff_r=max_diff_r,
            max_diff_g=max_diff_g,
            max_diff_b=max_diff_b,
            use_cupy=self.use_cupy,
            max_color_difference_factor=max_color_difference_factor,
        )

    def take_screenshot_and_find_needles(
        self,
        minimum_match=0.8,
        monitor=1,
        percent=100,
        processes=4,
        interpolation=cv2.INTER_LANCZOS4,
        max_diff_r=30,
        max_diff_g=30,
        max_diff_b=30,
        max_color_difference_factor=2,
    ):
        self.get_screenshot(monitor=monitor)
        return take_screenshot_and_find(
            self.needles,
            screenshot=self.screenshot_buffer[-1],
            minimum_match=minimum_match,
            monitor=monitor,
            percent=percent,
            processes=processes,
            interpolation=interpolation,
            max_diff_r=max_diff_r,
            max_diff_g=max_diff_g,
            max_diff_b=max_diff_b,
            use_cupy=self.use_cupy,
            max_color_difference_factor=max_color_difference_factor,
        )

    def draw_results(self, df, haystack=None, fontdistance=-10):
        if isinstance_tolerant(haystack, None):
            haystack = self.screenshot_buffer[-1]
        else:
            haystack = open_image_in_cv(haystack, channels_in_output=3)
        return draw_results(df, haystack.copy(), fontdistance1=-fontdistance)
