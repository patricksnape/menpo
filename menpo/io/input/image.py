import numpy as np
import PIL.Image as PILImage

from .base import Importer
from menpo.image import MaskedImage


class PILImporter(Importer):
    r"""
    Imports an image using PIL

    Parameters
    ----------
    filepath : string
        Absolute filepath of image
    """

    def __init__(self, filepath):
        super(PILImporter, self).__init__(filepath)

    def build(self):
        r"""
        Read the image using PIL and then use the
        :class:`menpo.image.base.MaskedImage` constructor to create a class.
        Normalise between 0 and 1.0
        """
        self._pil_image = PILImage.open(self.filepath)
        image_pixels = np.array(self._pil_image, dtype=np.float) / 255.0
        return MaskedImage(image_pixels)


class ABSImporter(Importer):
    r"""
    Allows importing the ABS file format from the FRGC dataset.

    The z-min value is stripped from the mesh to make it renderable.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    def __init__(self, filepath):
        # Setup class before super class call
        super(ABSImporter, self).__init__(filepath)

    def build(self):
        import re

        with open(self.filepath, 'r') as f:
            # Currently these are unused, but they are in the format
            # Could possibly store as metadata?
            # Assume first result for regexes
            re_rows = re.compile(u'([0-9]+) rows')
            n_rows = int(re_rows.findall(f.readline())[0])
            re_cols = re.compile(u'([0-9]+) columns')
            n_cols = int(re_cols.findall(f.readline())[0])

        # This also loads the mask
        #   >>> image_data[:, 0]
        image_data = np.loadtxt(self.filepath, skiprows=3, unpack=True)

        # Replace the lowest value with nan so that we can render properly
        data_view = image_data[:, 1:]
        corrupt_value = np.min(data_view)
        data_view[np.any(np.isclose(data_view, corrupt_value), axis=1)] = np.nan

        return MaskedImage(
            np.reshape(data_view, [n_rows, n_cols, 3]),
            np.reshape(image_data[:, 0], [n_rows, n_cols]).astype(np.bool),
            copy=False)
