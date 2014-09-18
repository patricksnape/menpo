import itertools
import numpy as np
scipy_gaussian_filter = None  # expensive

from .base import ndfeature, winitfeature
from .windowiterator import WindowIterator


@ndfeature
def gradient(pixels):
    r"""
    Calculates the gradient of an input image. The image is assumed to have
    channel information on the last axis. In the case of multiple channels,
    it returns the gradient over each axis over each channel as the last axis.

    Parameters
    ----------
    pixels : `ndarray`, shape (X, Y, ..., Z, C)
        An array where the last dimension is interpreted as channels. This
        means an N-dimensional image is represented by an N+1 dimensional
        array.

    Returns
    -------
    gradient : ndarray, shape (X, Y, ..., Z, C * length([X, Y, ..., Z]))
        The gradient over each axis over each channel. Therefore, the
        last axis of the gradient of a 2D, single channel image, will have
        length `2`. The last axis of the gradient of a 2D, 3-channel image,
        will have length `6`, he ordering being [Rd_x, Rd_y, Gd_x, Gd_y,
        Bd_x, Bd_y].

    """
    grad_per_dim_per_channel = [np.gradient(g) for g in
                                np.rollaxis(pixels, -1)]
    # Flatten out the separate dims
    grad_per_channel = list(itertools.chain.from_iterable(
        grad_per_dim_per_channel))
    # Add a channel axis for broadcasting
    grad_per_channel = [g[..., None] for g in grad_per_channel]
    # Concatenate gradient list into an array (the new_image)
    return np.concatenate(grad_per_channel, axis=-1)


@ndfeature
def gaussian_filter(pixels, sigma):
    global scipy_gaussian_filter
    if scipy_gaussian_filter is None:
        from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    output = np.empty(pixels.shape)
    for dim in range(pixels.shape[2]):
        scipy_gaussian_filter(pixels[..., dim], sigma, output=output[..., dim])
    return output


@winitfeature
def hog(pixels, mode='dense', algorithm='dalaltriggs', num_bins=9,
        cell_size=8, block_size=2, signed_gradient=True, l2_norm_clip=0.2,
        window_height=1, window_width=1, window_unit='blocks',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding=True, verbose=False):
    r"""
    Computes a 2-dimensional HOG features image with k number of channels, of
    size `(M, N, C)` and data type `np.float`.

    Parameters
    ----------
    mode : 'dense' or 'sparse'
        The 'sparse' case refers to the traditional usage of HOGs, so default
        parameters values are passed to the ImageWindowIterator.
        The sparse case of 'dalaltriggs' algorithm sets the window height
        and width equal to block size and the window step horizontal and
        vertical equal to cell size. The sparse case of 'zhuramanan'
        algorithm sets the window height and width equal to 3 times the
        cell size and the window step horizontal and vertical equal to cell
        size. In the 'dense' case, the user can change the window_height,
        window_width, window_unit, window_step_vertical,
        window_step_horizontal, window_step_unit, and padding to completely
        customize the HOG calculation.

    window_height : float
        Defines the height of the window for the ImageWindowIterator
        object. The metric unit is defined by window_unit.

    window_width : float
        Defines the width of the window for the ImageWindowIterator object.
        The metric unit is defined by window_unit.

    window_unit : 'blocks' or 'pixels'
        Defines the metric unit of the window_height and window_width
        parameters for the ImageWindowIterator object.

    window_step_vertical : float
        Defines the vertical step by which the window is moved, thus it
        controls the features density. The metric unit is defined by
        window_step_unit.

    window_step_horizontal : float
        Defines the horizontal step by which the window is moved, thus it
        controls the features density. The metric unit is defined by
        window_step_unit.

    window_step_unit : 'pixels' or 'cells'
        Defines the metric unit of the window_step_vertical and
        window_step_horizontal parameters.

    padding : bool
        Enables/disables padding for the close-to-boundary windows in the
        ImageWindowIterator object. When padding is enabled,
        the out-of-boundary pixels are set to zero.

    algorithm : 'dalaltriggs' or 'zhuramanan'
        Specifies the algorithm used to compute HOGs.

    cell_size : float
        Defines the cell size in pixels. This value is set to both the width
        and height of the cell. This option is valid for both algorithms.

    block_size : float
        Defines the block size in cells. This value is set to both the width
        and height of the block. This option is valid only for the
        'dalaltriggs' algorithm.

    num_bins : float
        Defines the number of orientation histogram bins. This option is
        valid only for the 'dalaltriggs' algorithm.

    signed_gradient : bool
        Flag that defines whether we use signed or unsigned gradient angles.
        This option is valid only for the 'dalaltriggs' algorithm.

    l2_norm_clip : float
        Defines the clipping value of the gradients' L2-norm. This option is
        valid only for the 'dalaltriggs' algorithm.

    constrain_landmarks : bool
        Flag that if enabled, it constrains landmarks that ended up outside of
        the features image bounds.

    verbose : bool
        Flag to print HOG related information.

    Raises
    -------
    ValueError
        HOG features mode must be either dense or sparse
    ValueError
        Algorithm must be either dalaltriggs or zhuramanan
    ValueError
        Number of orientation bins must be > 0
    ValueError
        Cell size (in pixels) must be > 0
    ValueError
        Block size (in cells) must be > 0
    ValueError
        Value for L2-norm clipping must be > 0.0
    ValueError
        Window height must be >= block size and <= image height
    ValueError
        Window width must be >= block size and <= image width
    ValueError
        Window unit must be either pixels or blocks
    ValueError
        Horizontal window step must be > 0
    ValueError
        Vertical window step must be > 0
    ValueError
        Window step unit must be either pixels or cells

    """
    # Parse options
    if mode not in ['dense', 'sparse']:
        raise ValueError("HOG features mode must be either dense or sparse")
    if algorithm not in ['dalaltriggs', 'zhuramanan']:
        raise ValueError("Algorithm must be either dalaltriggs or zhuramanan")
    if num_bins <= 0:
        raise ValueError("Number of orientation bins must be > 0")
    if cell_size <= 0:
        raise ValueError("Cell size (in pixels) must be > 0")
    if block_size <= 0:
        raise ValueError("Block size (in cells) must be > 0")
    if l2_norm_clip <= 0.0:
        raise ValueError("Value for L2-norm clipping must be > 0.0")
    if mode == 'dense':
        if window_unit not in ['pixels', 'blocks']:
            raise ValueError("Window unit must be either pixels or blocks")
        window_height_temp = window_height
        window_width_temp = window_width
        if window_unit == 'blocks':
            window_height_temp = window_height * block_size * cell_size
            window_width_temp = window_width * block_size * cell_size
        if (window_height_temp < block_size * cell_size or
            window_height_temp > pixels.shape[0]):
            raise ValueError("Window height must be >= block size and <= "
                             "image height")
        if (window_width_temp < block_size*cell_size or
            window_width_temp > pixels.shape[1]):
            raise ValueError("Window width must be >= block size and <= "
                             "image width")
        if window_step_horizontal <= 0:
            raise ValueError("Horizontal window step must be > 0")
        if window_step_vertical <= 0:
            raise ValueError("Vertical window step must be > 0")
        if window_step_unit not in ['pixels', 'cells']:
            raise ValueError("Window step unit must be either pixels or cells")

    # Correct input image_data
    pixels = np.asfortranarray(pixels)
    pixels *= 255.

    # Dense case
    if mode == 'dense':
        # Iterator parameters
        if algorithm == 'dalaltriggs':
            algorithm = 1
            if window_unit == 'blocks':
                block_in_pixels = cell_size * block_size
                window_height = np.uint32(window_height * block_in_pixels)
                window_width = np.uint32(window_width * block_in_pixels)
            if window_step_unit == 'cells':
                window_step_vertical = np.uint32(window_step_vertical *
                                                 cell_size)
                window_step_horizontal = np.uint32(window_step_horizontal *
                                                   cell_size)
        elif algorithm == 'zhuramanan':
            algorithm = 2
            if window_unit == 'blocks':
                block_in_pixels = 3 * cell_size
                window_height = np.uint32(window_height * block_in_pixels)
                window_width = np.uint32(window_width * block_in_pixels)
            if window_step_unit == 'cells':
                window_step_vertical = np.uint32(window_step_vertical *
                                                 cell_size)
                window_step_horizontal = np.uint32(window_step_horizontal *
                                                   cell_size)
        iterator = WindowIterator(pixels, window_height, window_width,
                                  window_step_horizontal,
                                  window_step_vertical, padding)
    # Sparse case
    else:
        # Create iterator
        if algorithm == 'dalaltriggs':
            algorithm = 1
            window_size = cell_size * block_size
            step = cell_size
        else:
            algorithm = 2
            window_size = 3 * cell_size
            step = cell_size
        iterator = WindowIterator(pixels, window_size, window_size, step,
                                  step, False)
    # Print iterator's info
    if verbose:
        print(iterator)
    # Compute HOG
    return iterator.HOG(algorithm, num_bins, cell_size, block_size,
                        signed_gradient, l2_norm_clip, verbose)

    # store parameters
    # hog_image.hog_parameters = {'mode': mode, 'algorithm': algorithm,
    #                             'num_bins': num_bins,
    #                             'cell_size': cell_size,
    #                             'block_size': block_size,
    #                             'signed_gradient': signed_gradient,
    #                             'l2_norm_clip': l2_norm_clip,
    #
    #                             'window_height': window_height,
    #                             'window_width': window_width,
    #                             'window_unit': window_unit,
    #                             'window_step_vertical': window_step_vertical,
    #                             'window_step_horizontal':
    #                                 window_step_horizontal,
    #                             'window_step_unit': window_step_unit,
    #                             'padding': padding,
    #
    #                             'original_image_height':
    #                                 self._image.pixels.shape[0],
    #                             'original_image_width':
    #                                 self._image.pixels.shape[1],
    #                             'original_image_channels':
    #                                 self._image.pixels.shape[2]}


@ndfeature
def igo(pixels, double_angles=False, verbose=False):
    r"""
    Represents a 2-dimensional IGO features image with N*C number of
    channels, where N is the number of channels of the original image and
    C=[2,4] depending on whether double angles are used.

    Parameters
    ----------
    pixels :  ndarray
        The pixel data for the image, where the last axis represents the
        number of channels.

    double_angles : bool
        Assume that phi represents the gradient orientations. If this flag
        is disabled, the features image is the concatenation of cos(phi)
        and sin(phi), thus 2 channels. If it is enabled, the features image
        is the concatenation of cos(phi), sin(phi), cos(2*phi), sin(2*phi),
        thus 4 channels.

    verbose : bool
        Flag to print IGO related information.

    Raises
    -------
    ValueError
        Image has to be 2D in order to extract IGOs.

    """
    # check number of dimensions
    if len(pixels.shape) != 3:
        raise ValueError('IGOs only work on 2D images. Expects image data '
                         'to be 3D, shape + channels.')
    # feature channels per image channel
    feat_channels = 2
    if double_angles:
        feat_channels = 4
    # compute gradients
    grad = gradient(pixels)
    # compute angles
    grad_orient = np.angle(grad[..., ::2] + 1j * grad[..., 1::2])
    # compute igo image
    igo_pixels = np.empty((pixels.shape[0], pixels.shape[1],
                           pixels.shape[-1] * feat_channels))
    igo_pixels[..., ::feat_channels] = np.cos(grad_orient)
    igo_pixels[..., 1::feat_channels] = np.sin(grad_orient)
    if double_angles:
        igo_pixels[..., 2::feat_channels] = np.cos(2 * grad_orient)
        igo_pixels[..., 3::feat_channels] = np.sin(2 * grad_orient)

    # print information
    if verbose:
        info_str = "IGO Features:\n"
        info_str = "{}  - Input image is {}W x {}H with {} channels.\n".format(
            info_str, pixels.shape[1], pixels.shape[0],
            pixels.shape[2])
        if double_angles:
            info_str = "{}  - Double angles are enabled.\n".format(info_str)
        else:
            info_str = "{}  - Double angles are disabled.\n".format(info_str)
        info_str = "{}Output image size {}W x {}H x {}.".format(
            info_str, igo_pixels.shape[1], igo_pixels.shape[0],
            igo_pixels.shape[2])
        print(info_str)
    return igo_pixels

    # store parameters
    # igo_image.igo_parameters = {'double_angles': double_angles,
    #
    #                             'original_image_height':
    #                                 self._image.pixels.shape[0],
    #                             'original_image_width':
    #                                 self._image.pixels.shape[1],
    #                             'original_image_channels':
    #                                 self._image.pixels.shape[2]}

@ndfeature
def es(image_data, verbose=False):
    r"""
    Represents a 2-dimensional Edge Structure (ES) features image with N*C
    number of channels, where N is the number of channels of the original
    image and C=2. The output object's class is either MaskedImage or Image
    depending on the original image.

    Parameters
    ----------
    image_data :  ndarray
        The pixel data for the image, where the last axis represents the
        number of channels.
    verbose : bool
        Flag to print ES related information.

        Default: False

    Raises
    -------
    ValueError
        Image has to be 2D in order to extract ES features.
    """
    # check number of dimensions
    if len(image_data.shape) != 3:
        raise ValueError('ES features only work on 2D images. Expects '
                         'image data to be 3D, shape + channels.')
    # feature channels per image channel
    feat_channels = 2
    # compute gradients
    grad = gradient(image_data)
    # compute magnitude
    grad_abs = np.abs(grad[..., ::2] + 1j * grad[..., 1::2])
    # compute es image
    grad_abs = grad_abs + np.median(grad_abs)
    es_pixels = np.empty((image_data.shape[0], image_data.shape[1],
                        image_data.shape[-1] * feat_channels))
    es_pixels[..., ::feat_channels] = grad[..., ::2] / grad_abs
    es_pixels[..., 1::feat_channels] = grad[..., 1::2] / grad_abs
    # print information
    if verbose:
        info_str = "ES Features:\n"
        info_str = "{}  - Input image is {}W x {}H with {} channels.\n".format(
            info_str, image_data.shape[1], image_data.shape[0],
            image_data.shape[2])
        info_str = "{}Output image size {}W x {}H x {}.".format(
            info_str, es_pixels.shape[1], es_pixels.shape[0],
            es_pixels.shape[2])
        print(info_str)
    return es_pixels

    # store parameters
    # es_image.es_parameters = {'original_image_height':
    #                               self._image.pixels.shape[0],
    #                           'original_image_width':
    #                               self._image.pixels.shape[1],
    #                           'original_image_channels':
    #                               self._image.pixels.shape[2]}

@winitfeature
def gh(pixels, mode='dense', num_bins=9, cell_size=8, block_size=2,
       signed_gradient=True, l2_norm_clip=0.2, window_height=1, window_width=1,
       window_unit='blocks', window_step_vertical=1, window_step_horizontal=1,
       window_step_unit='pixels', padding=True, verbose=False):
    r"""
    Computes a 2-dimensional Generalized Histogram Binning features image with
    k number of channels, of size `(M, N, C)` and data type `np.float`.

    Parameters
    ----------
    mode : 'dense' or 'sparse'
        The 'sparse' case refers to the traditional usage of GH, so default
        parameters values are passed to the ImageWindowIterator.
        It sets the window height and width equal to block size and the window
        step horizontal and vertical equal to cell size. In the 'dense' case,
        the user can change the window_height, window_width, window_unit,
        window_step_vertical, window_step_horizontal, window_step_unit and
        padding to completely customize the GH calculation.

    window_height : float
        Defines the height of the window for the ImageWindowIterator
        object. The metric unit is defined by window_unit.

    window_width : float
        Defines the width of the window for the ImageWindowIterator object.
        The metric unit is defined by window_unit.

    window_unit : 'blocks' or 'pixels'
        Defines the metric unit of the window_height and window_width
        parameters for the ImageWindowIterator object.

    window_step_vertical : float
        Defines the vertical step by which the window is moved, thus it
        controls the features density. The metric unit is defined by
        window_step_unit.

    window_step_horizontal : float
        Defines the horizontal step by which the window is moved, thus it
        controls the features density. The metric unit is defined by
        window_step_unit.

    window_step_unit : 'pixels' or 'cells'
        Defines the metric unit of the window_step_vertical and
        window_step_horizontal parameters.

    padding : bool
        Enables/disables padding for the close-to-boundary windows in the
        ImageWindowIterator object. When padding is enabled,
        the out-of-boundary pixels are set to zero.

    cell_size : float
        Defines the cell size in pixels. This value is set to both the width
        and height of the cell. This option is valid for both algorithms.

    block_size : float
        Defines the block size in cells. This value is set to both the width
        and height of the block. This option is valid only for the
        'dalaltriggs' algorithm.

    num_bins : float
        Defines the number of orientation histogram bins. This option is
        valid only for the 'dalaltriggs' algorithm.

    signed_gradient : bool
        Flag that defines whether we use signed or unsigned gradient angles.
        This option is valid only for the 'dalaltriggs' algorithm.

    l2_norm_clip : float
        Defines the clipping value of the gradients' L2-norm. This option is
        valid only for the 'dalaltriggs' algorithm.

    constrain_landmarks : bool
        Flag that if enabled, it constrains landmarks that ended up outside of
        the features image bounds.

    verbose : bool
        Flag to print HOG related information.

    Raises
    -------
    ValueError
        GH features mode must be either dense or sparse
    ValueError
        Number of orientation bins must be > 0
    ValueError
        Cell size (in pixels) must be > 0
    ValueError
        Block size (in cells) must be > 0
    ValueError
        Value for L2-norm clipping must be > 0.0
    ValueError
        Window height must be >= block size and <= image height
    ValueError
        Window width must be >= block size and <= image width
    ValueError
        Window unit must be either pixels or blocks
    ValueError
        Horizontal window step must be > 0
    ValueError
        Vertical window step must be > 0
    ValueError
        Window step unit must be either pixels or cells

    """
    # Parse options
    if pixels.shape[2] > 2:
        raise ValueError("GH features only support images with either 1 or 2 "
                         "channels")
    if mode not in ['dense', 'sparse']:
        raise ValueError("GH features mode must be either dense or sparse")
    if num_bins <= 0:
        raise ValueError("Number of orientation bins must be > 0")
    if cell_size <= 0:
        raise ValueError("Cell size (in pixels) must be > 0")
    if block_size <= 0:
        raise ValueError("Block size (in cells) must be > 0")
    if l2_norm_clip <= 0.0:
        raise ValueError("Value for L2-norm clipping must be > 0.0")
    if mode == 'dense':
        if window_unit not in ['pixels', 'blocks']:
            raise ValueError("Window unit must be either pixels or blocks")
        window_height_temp = window_height
        window_width_temp = window_width
        if window_unit == 'blocks':
            window_height_temp = window_height * block_size * cell_size
            window_width_temp = window_width * block_size * cell_size
        if (window_height_temp < block_size * cell_size or
            window_height_temp > pixels.shape[0]):
            raise ValueError("Window height must be >= block size and <= "
                             "image height")
        if (window_width_temp < block_size*cell_size or
            window_width_temp > pixels.shape[1]):
            raise ValueError("Window width must be >= block size and <= "
                             "image width")
        if window_step_horizontal <= 0:
            raise ValueError("Horizontal window step must be > 0")
        if window_step_vertical <= 0:
            raise ValueError("Vertical window step must be > 0")
        if window_step_unit not in ['pixels', 'cells']:
            raise ValueError("Window step unit must be either pixels or cells")

    # Correct input image_data
    pixels = np.asfortranarray(pixels)
    pixels *= 255.

    # Dense case
    if mode == 'dense':
        # Iterator parameters
        if window_unit == 'blocks':
            block_in_pixels = cell_size * block_size
            window_height = np.uint32(window_height * block_in_pixels)
            window_width = np.uint32(window_width * block_in_pixels)
        if window_step_unit == 'cells':
            window_step_vertical = np.uint32(window_step_vertical *
                                             cell_size)
            window_step_horizontal = np.uint32(window_step_horizontal *
                                               cell_size)
        iterator = WindowIterator(pixels, window_height, window_width,
                                  window_step_horizontal,
                                  window_step_vertical, padding)
    # Sparse case
    else:
        # Create iterator
        window_size = cell_size * block_size
        step = cell_size
        iterator = WindowIterator(pixels, window_size, window_size, step,
                                  step, False)
    # Print iterator's info
    if verbose:
        print(iterator)
    # Compute GH
    return iterator.GeneralizedHistogramBinning(num_bins, cell_size, block_size,
                                                signed_gradient, l2_norm_clip,
                                                verbose)

    # store parameters
    # gh_image.gh_parameters = {'mode': mode,
    #                           'num_bins': num_bins,
    #                           'cell_size': cell_size,
    #                           'block_size': block_size,
    #                           'signed_gradient': signed_gradient,
    #                           'l2_norm_clip': l2_norm_clip,
    #
    #                           'window_height': window_height,
    #                           'window_width': window_width,
    #                           'window_unit': window_unit,
    #                           'window_step_vertical': window_step_vertical,
    #                           'window_step_horizontal':
    #                               window_step_horizontal,
    #                           'window_step_unit': window_step_unit,
    #                           'padding': padding,
    #
    #                           'original_image_height':
    #                               self._image.pixels.shape[0],
    #                           'original_image_width':
    #                               self._image.pixels.shape[1],
    #                           'original_image_channels':
    #                               self._image.pixels.shape[2]}

@winitfeature
def lbp(pixels, radius=None, samples=None, mapping_type='riu2',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding=True, verbose=False,
        skip_checks=False):
    r"""
    Computes a 2-dimensional LBP features image with N*C number of channels,
    where N is the number of channels of the original image and C is the number
    of radius/samples values combinations that are used in the LBP computation.

    Parameters
    ----------
    pixels :  `ndarray`
        The pixel data for the image, where the last axis represents the
        number of channels.

    radius : `int` or `list` of `int`
        It defines the radius of the circle (or circles) at which the sampling
        points will be extracted. The radius (or radii) values must be greater
        than zero. There must be a radius value for each samples value, thus
        they both need to have the same length.

        Default: None = [1, 2, 3, 4]

    samples : `int` or `list` of `int`
        It defines the number of sampling points that will be extracted at each
        circle. The samples value (or values) must be greater than zero. There
        must be a samples value for each radius value, thus they both need to
        have the same length.

        Default: None = [8, 8, 8, 8]

    mapping_type : ``'u2'`` or ``'ri'`` or ``'riu2'`` or ``'none'``
        It defines the mapping type of the LBP codes. Select 'u2' for uniform-2
        mapping, 'ri' for rotation-invariant mapping, 'riu2' for uniform-2 and
        rotation-invariant mapping and 'none' to use no mapping nd only the
        decimal values instead.

    window_step_vertical : float
        Defines the vertical step by which the window in the
        ImageWindowIterator is moved, thus it controls the features density.
        The metric unit is defined by window_step_unit.

    window_step_horizontal : float
        Defines the horizontal step by which the window in the
        ImageWindowIterator is moved, thus it controls the features density.
        The metric unit is defined by window_step_unit.

    window_step_unit : ``'pixels'`` or ``'window'``
        Defines the metric unit of the window_step_vertical and
        window_step_horizontal parameters for the ImageWindowIterator object.

    padding : bool
        Enables/disables padding for the close-to-boundary windows in the
        ImageWindowIterator object. When padding is enabled, the
        out-of-boundary pixels are set to zero.

    verbose : `bool`, optional
        Flag to print LBP related information.

    skip_checks : `bool`, optional
        If True
    Raises
    -------
    ValueError
        Radius and samples must both be either integers or lists
    ValueError
        Radius and samples must have the same length
    ValueError
        Radius must be > 0
    ValueError
        Radii must be > 0
    ValueError
        Samples must be > 0
    ValueError
        Mapping type must be u2, ri, riu2 or none
    ValueError
        Horizontal window step must be > 0
    ValueError
        Vertical window step must be > 0
    ValueError
        Window step unit must be either pixels or window
    """
    if radius is None:
        radius = range(1, 5)
    if samples is None:
        samples = [8]*4

    if not skip_checks:
        # Check parameters
        if ((isinstance(radius, int) and isinstance(samples, list)) or
                (isinstance(radius, list) and isinstance(samples, int))):
            raise ValueError("Radius and samples must both be either integers "
                             "or lists")
        elif isinstance(radius, list) and isinstance(samples, list):
            if len(radius) != len(samples):
                raise ValueError("Radius and samples must have the same "
                                 "length")

        if isinstance(radius, int) and radius < 1:
            raise ValueError("Radius must be > 0")
        elif isinstance(radius, list) and sum(r < 1 for r in radius) > 0:
            raise ValueError("Radii must be > 0")

        if isinstance(samples, int) and samples < 1:
            raise ValueError("Samples must be > 0")
        elif isinstance(samples, list) and sum(s < 1 for s in samples) > 0:
            raise ValueError("Samples must be > 0")

        if mapping_type not in ['u2', 'ri', 'riu2', 'none']:
            raise ValueError("Mapping type must be u2, ri, riu2 or "
                             "none")

        if window_step_horizontal <= 0:
            raise ValueError("Horizontal window step must be > 0")

        if window_step_vertical <= 0:
            raise ValueError("Vertical window step must be > 0")

        if window_step_unit not in ['pixels', 'window']:
            raise ValueError("Window step unit must be either pixels or "
                             "window")

    # Correct input image_data
    pixels = np.asfortranarray(pixels)

    # Parse options
    radius = np.asfortranarray(radius)
    samples = np.asfortranarray(samples)
    window_height = np.uint32(2 * radius.max() + 1)
    window_width = window_height
    if window_step_unit == 'window':
        window_step_vertical = np.uint32(window_step_vertical * window_height)
        window_step_horizontal = np.uint32(window_step_horizontal *
                                           window_width)
    if mapping_type == 'u2':
        mapping_type = 1
    elif mapping_type == 'ri':
        mapping_type = 2
    elif mapping_type == 'riu2':
        mapping_type = 3
    else:
        mapping_type = 0

    # Create iterator object
    iterator = WindowIterator(pixels, window_height, window_width,
                              window_step_horizontal, window_step_vertical,
                              padding)

    # Print iterator's info
    if verbose:
        print(iterator)

    # Compute LBP
    return iterator.LBP(radius, samples, mapping_type, verbose)

    # # store parameters
    # lbp_image.lbp_parameters = {'radius': radius, 'samples': samples,
    #                             'mapping_type': mapping_type,
    #
    #                             'window_step_vertical':
    #                                 window_step_vertical,
    #                             'window_step_horizontal':
    #                                 window_step_horizontal,
    #                             'window_step_unit': window_step_unit,
    #                             'padding': padding,
    #
    #                             'original_image_height':
    #                                 self._image.pixels.shape[0],
    #                             'original_image_width':
    #                                 self._image.pixels.shape[1],
    #                             'original_image_channels':
    #                                 self._image.pixels.shape[2]}

@winitfeature
def gbp(pixels, radius=None, samples=None, window_step_vertical=1,
        window_step_horizontal=1, window_step_unit='pixels', padding=True,
        verbose=False, skip_checks=False):
    r"""
    Computes a 2-dimensional Generalized Binary Pattern features image with N*C
    number of channels, where N is the number of channels of the original image
    and C is the number of radius/samples values combinations that are used in
    the GBP computation. The binary codes are converted to decimal.

    Parameters
    ----------
    pixels :  `ndarray`
        The pixel data for the image, where the last axis represents the
        number of channels.

    radius : `int` or `list` of `int`
        It defines the radius of the circle (or circles) at which the sampling
        points will be extracted. The radius (or radii) values must be greater
        than zero. There must be a radius value for each samples value, thus
        they both need to have the same length.

        Default: None = [1, 2, 3, 4]

    samples : `int` or `list` of `int`
        It defines the number of sampling points that will be extracted at each
        circle. The samples value (or values) must be greater than zero. There
        must be a samples value for each radius value, thus they both need to
        have the same length.

        Default: None = [8, 8, 8, 8]

    window_step_vertical : float
        Defines the vertical step by which the window in the
        ImageWindowIterator is moved, thus it controls the features density.
        The metric unit is defined by window_step_unit.

    window_step_horizontal : float
        Defines the horizontal step by which the window in the
        ImageWindowIterator is moved, thus it controls the features density.
        The metric unit is defined by window_step_unit.

    window_step_unit : ``'pixels'`` or ``'window'``
        Defines the metric unit of the window_step_vertical and
        window_step_horizontal parameters for the ImageWindowIterator object.

    padding : bool
        Enables/disables padding for the close-to-boundary windows in the
        ImageWindowIterator object. When padding is enabled, the
        out-of-boundary pixels are set to zero.

    verbose : `bool`, optional
        Flag to print LBP related information.

    skip_checks : `bool`, optional
        If True
    Raises
    -------
    ValueError
        Radius and samples must both be either integers or lists
    ValueError
        Radius and samples must have the same length
    ValueError
        Radius must be > 0
    ValueError
        Radii must be > 0
    ValueError
        Samples must be > 0
    ValueError
        Horizontal window step must be > 0
    ValueError
        Vertical window step must be > 0
    ValueError
        Window step unit must be either pixels or window
    """
    if radius is None:
        radius = range(1, 5)
    if samples is None:
        samples = [8]*4

    if not skip_checks:
        # Check parameters
        if ((isinstance(radius, int) and isinstance(samples, list)) or
                (isinstance(radius, list) and isinstance(samples, int))):
            raise ValueError("Radius and samples must both be either integers "
                             "or lists")
        elif isinstance(radius, list) and isinstance(samples, list):
            if len(radius) != len(samples):
                raise ValueError("Radius and samples must have the same "
                                 "length")

        if isinstance(radius, int) and radius < 1:
            raise ValueError("Radius must be > 0")
        elif isinstance(radius, list) and sum(r < 1 for r in radius) > 0:
            raise ValueError("Radii must be > 0")

        if isinstance(samples, int) and samples < 1:
            raise ValueError("Samples must be > 0")
        elif isinstance(samples, list) and sum(s < 1 for s in samples) > 0:
            raise ValueError("Samples must be > 0")

        if window_step_horizontal <= 0:
            raise ValueError("Horizontal window step must be > 0")

        if window_step_vertical <= 0:
            raise ValueError("Vertical window step must be > 0")

        if window_step_unit not in ['pixels', 'window']:
            raise ValueError("Window step unit must be either pixels or "
                             "window")

    # Correct input image_data
    pixels = np.asfortranarray(pixels)

    # Parse options
    radius = np.asfortranarray(radius)
    samples = np.asfortranarray(samples)
    window_height = np.uint32(2 * radius.max() + 1)
    window_width = window_height
    if window_step_unit == 'window':
        window_step_vertical = np.uint32(window_step_vertical * window_height)
        window_step_horizontal = np.uint32(window_step_horizontal *
                                           window_width)

    # Create iterator object
    iterator = WindowIterator(pixels, window_height, window_width,
                              window_step_horizontal, window_step_vertical,
                              padding)

    # Print iterator's info
    if verbose:
        print(iterator)

    # Compute LBP
    return iterator.GeneralizedBinaryPattern(radius, samples, verbose)

    # # store parameters
    # gbp_image.gbp_parameters = {'radius': radius, 'samples': samples,
    #
    #                             'window_step_vertical':
    #                                 window_step_vertical,
    #                             'window_step_horizontal':
    #                                 window_step_horizontal,
    #                             'window_step_unit': window_step_unit,
    #                             'padding': padding,
    #
    #                             'original_image_height':
    #                                 self._image.pixels.shape[0],
    #                             'original_image_width':
    #                                 self._image.pixels.shape[1],
    #                             'original_image_channels':
    #                                 self._image.pixels.shape[2]}

@ndfeature
def no_op(image_data):
    r"""
    A no operation feature - does nothing but return a copy of the pixels
    passed in.
    """
    return image_data.copy()


@ndfeature
def approximate_normals(pixels):
    import numpy as np
    g = gradient(pixels)

    normals = np.concatenate([g, -np.ones_like(g[..., 0])[..., None]], axis=-1)
    mag = np.sqrt(np.sum(normals ** 2, axis=-1))
    return np.clip(normals / mag[..., None], -1., 1.)


@ndfeature
def depth(pixels):
    return pixels


@ndfeature
def E_first_coeff(depth_pixels):
    g = gradient(depth_pixels)
    # 1 + Zx ^ 2
    return (1.0 + g[..., 1] ** 2)[..., None]


@ndfeature
def F_first_coeff(depth_pixels):
    # Zx * Zy
    g = gradient(depth_pixels)
    return (g[..., 0] * g[..., 1])[..., None]


@ndfeature
def G_first_coeff(depth_pixels):
    # 1 + Zy ^ 2
    g = gradient(depth_pixels)
    return (1.0 + g[..., 0] ** 2)[..., None]


@ndfeature
def e_second_coeff(depth_pixels):
    import numpy as np
    g = gradient(depth_pixels)
    g2 = gradient(g)
    norm = np.sqrt(np.sum(1.0 + g ** 2, axis=-1))
    # Zxx / sqrt(1 + Zx ^ 2 + Zy ^ 2)
    return np.nan_to_num((g2[..., 2] / norm)[..., None])


@ndfeature
def f_second_coeff(depth_pixels):
    import numpy as np
    g = gradient(depth_pixels)
    g2 = gradient(g)
    norm = np.sqrt(np.sum(1.0 + g ** 2, axis=-1))
    # Zxy / sqrt(1 + Zx ^ 2 + Zy ^ 2)
    return np.nan_to_num((g2[..., 3] / norm)[..., None])


@ndfeature
def g_second_coeff(depth_pixels):
    import numpy as np
    g = gradient(depth_pixels)
    g2 = gradient(g)
    norm = np.sqrt(np.sum(1.0 + g ** 2, axis=-1))
    # Zyy / sqrt(1 + Zx ^ 2 + Zy ^ 2)
    return np.nan_to_num((g2[..., 1] / norm)[..., None])


@ndfeature
def K_gaussian_curvature(depth_pixels):
    import numpy as np
    g = gradient(depth_pixels)
    g2 = gradient(g)
    denom = np.sum(1.0 + g ** 2, axis=-1)
    # (Zxx * Zyy - Zxy^2) / (1.0 + Zx^2 + Zy^2)^2
    K = (g2[..., 2] * g2[..., 1] - g2[..., 3] ** 2) / (denom ** 2)
    return np.nan_to_num(K[..., None])


@ndfeature
def H_mean_curvature(depth_pixels):
    import numpy as np
    g = gradient(depth_pixels)
    g2 = gradient(g)
    denom = np.sum(1.0 + g ** 2, axis=-1)
    y_comp = (1.0 + g[..., 0]**2) * g2[..., 2]
    xy_comp = 2.0 * g[..., 1] * g[..., 0] * g2[..., 3]
    x_comp = (1.0 + g[..., 1]**2) * g2[..., 1]
    # ((1 + Zy^2)Zxx - 2*Zx*Zy*Zxy + (1 + Zx^2)Zyy) / 2(1 + Zx^2 + Zy^2)^(3/2)
    H = (y_comp - xy_comp + x_comp) / (2.0 * (denom ** (3.0 / 2.0)))
    return np.nan_to_num(H[..., None])


@ndfeature
def k1_principal_curvature(depth_pixels):
    import numpy as np
    # H + sqrt(H^2 - K)
    H = H_mean_curvature(depth_pixels)
    K = K_gaussian_curvature(depth_pixels)
    return np.nan_to_num(H + np.sqrt(H ** 2 - K))


@ndfeature
def k2_principal_curvature(depth_pixels):
    import numpy as np
    # H - sqrt(H^2 - K)
    H = H_mean_curvature(depth_pixels)
    K = K_gaussian_curvature(depth_pixels)
    return np.nan_to_num(H - np.sqrt(H ** 2 - K))


@ndfeature
def S_shape_index(depth_pixels):
    import numpy as np
    k1 = k1_principal_curvature(depth_pixels)
    k2 = k2_principal_curvature(depth_pixels)
    # (1/2) - (1/pi)arctan(k1+k2 / k1-k2)
    return np.nan_to_num(0.5 - (1.0 / np.pi) * np.arctan2(k1 + k2, k1 - k2))


@ndfeature
def C_curvedness_index(depth_pixels):
    import numpy as np
    k1 = k1_principal_curvature(depth_pixels)
    k2 = k2_principal_curvature(depth_pixels)
    # sqrt((k1^2 + k2^2) / 2.0)
    return np.nan_to_num(np.sqrt((k1**2 + k2**2) / 2.0))
