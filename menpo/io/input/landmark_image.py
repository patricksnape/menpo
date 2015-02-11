from collections import OrderedDict
import numpy as np

from menpo.shape import PointDirectedGraph
from .landmark import ASFImporter, PTSImporter, LandmarkImporter


class ImageASFImporter(ASFImporter):
    r"""
    Implements the :meth:`_build_points` method for images. Here, `y` is the
    first axis.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the landmarks
    """

    def __init__(self, filepath):
        super(ImageASFImporter, self).__init__(filepath)

    def _build_points(self, xs, ys):
        """
        For images, `axis 0 = ys` and `axis 1 = xs`. Therefore, return the
        appropriate points array ordering.

        Parameters
        ----------
        xs : (N,) ndarray
            Row vector of `x` coordinates
        ys : (N,) ndarray
            Row vector of `y` coordinates

        Returns
        -------
        points : (N, 2) ndarray
            Array with `ys` as the first axis: `[ys; xs]`
        """
        return np.hstack([ys, xs])


class ImagePTSImporter(PTSImporter):
    r"""
    Implements the :meth:`_build_points` method for images. Here, `y` is the
    first axis.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the landmarks
    """

    def __init__(self, filepath):
        super(ImagePTSImporter, self).__init__(filepath)

    def _build_points(self, xs, ys):
        """
        For images, `axis 0 = ys` and `axis 1 = xs`. Therefore, return the
        appropriate points array ordering.

        Parameters
        ----------
        xs : (N,) ndarray
            Row vector of `x` coordinates
        ys : (N,) ndarray
            Row vector of `y` coordinates

        Returns
        -------
        points : (N, 2) ndarray
            Array with `ys` as the first axis: `[ys; xs]`
        """
        return np.hstack([ys, xs])


class VOC2007XMLImporter(LandmarkImporter):
    r"""
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the landmark data. No value is returned.

    Writes out the VOC 2007 XML format.

    Parameters
    ----------
    landmark_image_tuple : (:map:`Image`, map:`LandmarkGroup`) `tuple`
        The landmark group and image pair to write out.
    file_handle : `file`-like object
        The file to write in to.
    """

    def _parse_format(self, asset=None):
        from lxml import etree

        xml = etree.parse(self.filepath)

        # We only support formats with a single landmark group in at the moment
        bndbox = xml.find('object').find('bndbox')

        min_p = [float(bndbox.find('ymin').text),
                 float(bndbox.find('xmin').text)]
        max_p = [float(bndbox.find('ymax').text),
                 float(bndbox.find('xmax').text)]

        self.pointcloud = PointDirectedGraph(
            np.array([min_p, [max_p[0], min_p[1]],
                      max_p, [min_p[0], max_p[1]]]),
            np.array([[0, 1], [1, 2], [2, 3], [3, 0]]), copy=False)

        self.labels_to_masks = OrderedDict([('all', np.ones(4, dtype=np.bool))])
