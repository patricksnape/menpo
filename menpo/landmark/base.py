import fnmatch
import keyword
import re
from collections import OrderedDict, MutableMapping

import numpy as np

from menpo.base import Copyable
from menpo.compatibility import basestring
from menpo.transform.base import Transformable

identifier_regex = re.compile('[_A-Za-z][_a-zA-Z0-9]*$')


def is_valid_identifier(key):
    if hasattr(key, 'isidentifier'):
        isidentifier = key.isidentifier()
    else:
        isidentifier = identifier_regex.match(key)
    return isidentifier and not keyword.iskeyword(key)


# The 'Bunch' like (or attribute dictionary) behaviour is inspired by the
# bunch project - whose license is MIT and can be found here:
#    https://github.com/dsc/bunch/blob/master/LICENSE.txt
class Landmarkable(Copyable):
    r"""
    Abstract interface for object that can have landmarks attached to them.
    Landmarkable objects have a public dictionary of landmarks which are
    managed by a :map:`LandmarkManager`. This means that
    different sets of landmarks can be attached to the same object.
    Landmarks can be N-dimensional and are expected to be some
    subclass of :map:`PointCloud` or :map:`LabelledPointUndirectedGraph`.
    """

    def __init__(self):
        self._landmarks = None

    def n_dims(self):
        """
        The total number of dimensions.

        :type: `int`
        """
        raise NotImplementedError()

    @property
    def landmarks(self):
        """
        The landmarks object.

        :type: :map:`LandmarkManager`
        """
        if self._landmarks is None:
            self._landmarks = LandmarkManager()
        return self._landmarks

    @property
    def has_landmarks(self):
        """
        Whether the object has landmarks.

        :type: `bool`
        """
        return self._landmarks is not None and self.landmarks.n_groups != 0

    @landmarks.setter
    def landmarks(self, value):
        """
        Landmarks setter.

        Parameters
        ----------
        value : :map:`LandmarkManager`
            The landmarks to set.
        """
        # firstly, make sure the dim is correct. Note that the dim can be None
        lm_n_dims = value.n_dims
        if lm_n_dims is not None and lm_n_dims != self.n_dims:
            raise ValueError(
                "Trying to set {}D landmarks on a "
                "{}D object".format(value.n_dims, self.n_dims))
        self._landmarks = value.copy()

    @property
    def n_landmark_groups(self):
        r"""
        The number of landmark groups on this object.

        :type: `int`
        """
        return self.landmarks.n_groups


class LandmarkManager(MutableMapping, Transformable):
    """Store for :map:`PointCloud` or ::map:`LabelledPointUndirectedGraph`
    instances associated with an object.

    Every :map:`Landmarkable` instance has an instance of this class available
    at the ``.landmarks`` property.  It is through this class that all access
    to landmarks attached to instances is handled. In general the
    :map:`LandmarkManager` provides a dictionary-like interface for storing
    landmarks. The LandmarkManager will contain instances of :map:`PointCloud`
    or :map:`LabelledPointUndirectedGraph` or subclasses thereof.
    :map:`LabelledPointUndirectedGraph` is unique in it's ability to
    include labels that refer to subsets of the underlying points that represent
    interesting semantic *labels*. These :map:`PointCloud` or
    :map:`LabelledPointUndirectedGraph` (or subclasses) are stored under
    string keys - these keys are refereed to as the **group name**. A special
    case is where there is a single unambiguous group attached to a
    :map:`LandmarkManager` - in this case ``None`` can be used as a key to
    access this sole group.

    Note that all groups stored on a :map:`Landmarkable` in it's attached
    :map:`LandmarkManager` are automatically transformed and copied with their
    parent object.
    """
    def __init__(self):
        super(LandmarkManager, self).__init__()
        self._landmark_groups = OrderedDict()

    @property
    def n_dims(self):
        """
        The total number of dimensions.

        :type: `int`
        """
        if self.n_groups != 0:
            # Python version independent way of getting the first value
            for v in self._landmark_groups.values():
                return v.n_dims
        else:
            return None

    def copy(self):
        r"""
        Generate an efficient copy of this :map:`LandmarkManager`.

        Returns
        -------
        ``type(self)``
            A copy of this object
        """
        # The dict will be shallow copied - rectify that here
        new = Copyable.copy(self)
        for k, v in new._landmark_groups.items():
            new._landmark_groups[k] = v.copy()
        return new

    def __iter__(self):
        """
        Iterate over the internal landmark group dictionary.
        """
        return iter(self._landmark_groups)

    def __setattr__(self, key, value):
        """
        Sets a new attribute for the provided identifier.

        Parameters
        ----------
        key : object
            Identifier of the attribute
        value : object
            The value to set.
        """
        # Hack to have the _landmark_groups attribute on the __dict__
        if key == '_landmark_groups':
            object.__setattr__(self, '_landmark_groups', value)
            return

        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, key)
        except AttributeError:
            try:
                from menpo.shape import PointCloud
                if key is None:
                    raise ValueError('Cannot set using the key `None`. `None` '
                                     'has a reserved meaning for landmark '
                                     'groups.')

                if isinstance(value, np.ndarray):
                    value = PointCloud(value)

                if not isinstance(value, PointCloud):
                    raise ValueError('Valid types are any subclass of '
                                     'PointCloud')

                # firstly, make sure the dim is correct
                n_dims = self.n_dims
                if n_dims is not None and value.n_dims != n_dims:
                    raise ValueError(
                        'Trying to set {}D landmarks on a '
                        '{}D LandmarkManager'.format(value.n_dims, self.n_dims))

                # Copy the landmark key so that we now own it
                lmark_group = value.copy()
                self._landmark_groups[key] = lmark_group
            except (AttributeError, KeyError):
                raise AttributeError(key)
        else:
            object.__setattr__(self, key, value)

    def __setitem__(self, key, value):
        """
        Sets a new landmark group for the given label. This can be set using
        an any :map`PointCloud` subclass. Existing landmark groups will be
        replaced. Numpy arrays can also set directly and will be coerced to
        :map`PointCloud` instances.

        Parameters
        ----------
        key : `string`
            Label of new group.
        value : :map:`PointCloud` or subclass
            The new landmark group to set.

        Raises
        ------
        DimensionalityError
            If the landmarks and the shape are not of the same dimensionality.
        """
        if not (isinstance(key, basestring) and is_valid_identifier(key)):
            id_url = 'https://docs.python.org/3.6/reference/lexical_analysis.html#identifiers'
            raise ValueError('Keys must be valid Python identifiers (see '
                             '{} for more information)'.format(id_url))
        self.__setattr__(key, value)

    def __getattr__(self, k=None):
        """
        Returns the attribute for the provided key.

        Parameters
        ---------
        k : object
            The attribute identifier.

        Returns
        -------
        attr : object
            The matching attribute.
        """
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return self._landmark_groups[k]
            except KeyError:
                raise AttributeError(k)

    def __getitem__(self, k):
        """
        Returns the group for the provided label.

        Parameters
        ---------
        k : `string`, optional
            The label of the group. If None is provided, and if there is only
            one group, the unambiguous group will be returned.

        Returns
        -------
        lmark_group : :map:`PointCloud` or :map:`LabelledPointUndirectedGraph`
            The matching landmarks.
        """
        if k is None:
            if self.n_groups == 1:
                k = self.group_labels[0]
            else:
                raise ValueError(
                    'Cannot use None as a key as there are {} landmark '
                    'groups'.format(self.n_groups))
        return self.__getattr__(k)

    def __delattr__(self, k):
        """
        Delete the attribute for the provided key.

        Parameters
        ---------
        k : `string`
            The attribute identifier.
        """
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                del self._landmark_groups[k]
            except KeyError:
                raise AttributeError(k)
        else:
            object.__delattr__(self, k)

    def __delitem__(self, k):
        """
        Delete the group for the provided label.

        Parameters
        ---------
        k : `string`
            The label of the group.
        """
        self.__delattr__(k)

    def __len__(self):
        return len(self._landmark_groups)

    def __setstate__(self, state):
        # consistency with older versions imported.
        if not isinstance(state['_landmark_groups'], OrderedDict):
            state['_landmark_groups'] = OrderedDict(state['_landmark_groups'])

        self.__dict__ = state

    def __dir__(self):
        return (super(LandmarkManager, self).__dir__() +
                list(self._landmark_groups.keys()))

    @property
    def n_groups(self):
        """
        Total number of labels.

        :type: `int`
        """
        return len(self._landmark_groups)

    @property
    def has_landmarks(self):
        """
        Whether the object has landmarks or not

        :type: `int`
        """
        return self.n_groups != 0

    @property
    def group_labels(self):
        """
        All the labels for the landmark set sorted by insertion order.

        :type: `list` of `str`
        """
        # Convert to list so that we can index immediately, as keys()
        # is a generator in Python 3
        return list(self._landmark_groups.keys())

    def keys_matching(self, glob_pattern):
        r"""
        Yield only landmark group names (keys) matching a given glob.

        Parameters
        ----------
        glob_pattern : `str`
            A glob pattern e.g. 'frontal_face_*'

        Yields
        ------
        keys: group labels that match the glob pattern
        """
        for key in fnmatch.filter(self.keys(), glob_pattern):
            yield key

    def items_matching(self, glob_pattern):
        r"""
        Yield only items ``(group, PointCloud)`` where the key matches a
        given glob.

        Parameters
        ----------
        glob_pattern : `str`
            A glob pattern e.g. 'frontal_face_*'

        Yields
        ------
        item : ``(group, PointCloud)``
            Tuple of (str, PointCloud) where the group matches the glob.
        """
        for k, v in self.items():
            if fnmatch.fnmatch(k, glob_pattern):
                yield k, v

    def _transform_inplace(self, transform):
        for group in self._landmark_groups.values():
            group._transform_inplace(transform)
        return self

    def view_widget(self, browser_style='buttons', figure_size=(10, 8),
                    style='coloured'):
        r"""
        Visualizes the landmark manager object using an interactive widget.

        Parameters
        ----------
        browser_style : {``'buttons'``, ``'slider'``}, optional
            It defines whether the selector of the landmark managers will have
            the form of plus/minus buttons or a slider.
        figure_size : (`int`, `int`), optional
            The initial size of the rendered figure.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        try:
            from menpowidgets import visualize_landmarks
            visualize_landmarks(self, figure_size=figure_size, style=style,
                                browser_style=browser_style)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def __str__(self):
        out_string = '{}: n_groups: {}'.format(type(self).__name__,
                                               self.n_groups)
        if self.has_landmarks:
            for label in self:
                out_string += '\n'
                out_string += '({}): {}'.format(label, self[label].__str__())

        return out_string


# TODO: Deprecate this - this handles importing old-style LandmarkGroup
class LandmarkGroup(object):

    def __new__(cls, *args, **kwargs):
        # This is a crazy hack that means when old style LandmarkGroup
        # objects are imported it is actually new-style
        # LabelledPointUndirectedGraph objects that are created.
        from menpo.shape import LabelledPointUndirectedGraph
        return LabelledPointUndirectedGraph.__new__(
            LabelledPointUndirectedGraph, *args, **kwargs)
