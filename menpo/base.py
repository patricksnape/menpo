import collections.abc as collections_abc
import os
import textwrap
import warnings
from functools import partial, wraps
from itertools import chain
from pathlib import Path
from pprint import pformat
from typing import (
    Any,
    Callable,
    Iterable,
    NoReturn,
    Optional,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import numpy as np
from typing_extensions import SupportsIndex

if TYPE_CHECKING:
    from menpo.shape import PointCloud

CopyT = TypeVar("CopyT", bound="Copyable")


class Copyable:
    """
    Efficient copying of classes containing numpy arrays.

    Interface that provides a single method for copying classes very
    efficiently.
    """

    def copy(self) -> CopyT:
        r"""
        Generate an efficient copy of this object.

        Note that Numpy arrays and other :map:`Copyable` objects on ``self``
        will be deeply copied. Dictionaries and sets will be shallow copied,
        and everything else will be assigned (no copy will be made).

        Classes that store state other than numpy arrays and immutable types
        should overwrite this method to ensure all state is copied.

        Returns
        -------
        ``type(self)``
            A copy of this object
        """
        new = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            try:
                new.__dict__[k] = v.copy()
            except AttributeError:
                new.__dict__[k] = v
        return new

    def __str__(self) -> str:
        # We have to be sure that we implement __str__ otherwise the __repr__
        # implementation below will lead to an infinite recursion.
        return f"Copyable Menpo Object with keys:\n{pformat(self.__dict__)}"

    def __repr__(self) -> str:
        # Most classes in Menpo derive from Copyable, so it's a handy place
        # to implement Menpo-wide behavior. For use in the notebook, we find
        # __repr__ representations not of very much use, so we default to
        # showing the string representation for this case. See
        # https://github.com/menpo/menpo/issues/752 for discussion.
        return self.__str__()


class Vectorizable(Copyable):
    """
    Flattening of rich objects to vectors and rebuilding them back.

    Interface that provides methods for 'flattening' an object into a
    vector, and restoring from the same vectorized form. Useful for
    statistical analysis of objects, which commonly requires the data
    to be provided as a single vector.
    """

    @property
    def n_parameters(self) -> int:
        r"""The length of the vector that this object produces.

        :type: `int`
        """
        return (self.as_vector()).shape[0]

    def as_vector(self, **kwargs: Any) -> np.ndarray:
        """
        Returns a flattened representation of the object as a single
        vector.

        Returns
        -------
        vector : (N,) ndarray
            The core representation of the object, flattened into a
            single vector. Note that this is always a view back on to the
            original object, but is not writable.
        """
        v = self._as_vector(**kwargs)
        v.flags.writeable = False
        return v

    def _as_vector(self, **kwargs: Any) -> np.ndarray:
        """
        Returns a flattened representation of the object as a single
        vector.

        Returns
        -------
        vector : ``(n_parameters,)`` `ndarray`
            The core representation of the object, flattened into a
            single vector.
        """
        raise NotImplementedError()

    def from_vector_inplace(self, vector: np.ndarray) -> None:
        """
        Deprecated. Use the non-mutating API, :map:`from_vector`.

        For internal usage in performance-sensitive spots,
        see `_from_vector_inplace()`

        Parameters
        ----------
        vector : ``(n_parameters,)`` `ndarray`
            Flattened representation of this object
        """
        warnings.warn(
            "the public API for inplace operations is deprecated "
            "and will be removed in a future version of Menpo. "
            "Use .from_vector() instead.",
            MenpoDeprecationWarning,
        )
        self._from_vector_inplace(vector)

    def _from_vector_inplace(self, vector: np.ndarray) -> None:
        """
        Update the state of this object from a vector form.

        Parameters
        ----------
        vector : ``(n_parameters,)`` `ndarray`
            Flattened representation of this object
        """
        raise NotImplementedError()

    def from_vector(self, vector: np.ndarray) -> "Vectorizable":
        """
        Build a new instance of the object from it's vectorized state.

        ``self`` is used to fill out the missing state required to
        rebuild a full object from it's standardized flattened state. This
        is the default implementation, which is which is a ``deepcopy`` of the
        object followed by a call to :meth:`from_vector_inplace()`. This method
        can be overridden for a performance benefit if desired.

        Parameters
        ----------
        vector : ``(n_parameters,)`` `ndarray`
            Flattened representation of the object.

        Returns
        -------
        object : ``type(self)``
            An new instance of this class.
        """
        new = self.copy()
        new._from_vector_inplace(vector)
        return new

    def has_nan_values(self) -> bool:
        """
        Tests if the vectorized form of the object contains ``nan`` values or
        not. This is particularly useful for objects with unknown values that
        have been mapped to ``nan`` values.

        Returns
        -------
        has_nan_values : `bool`
            If the vectorized object contains ``nan`` values.
        """
        return np.any(np.isnan(self.as_vector()))


class Targetable(Copyable):
    """Interface for objects that can produce a target :map:`PointCloud`.

    This could for instance be the result of an alignment or a generation of a
    :map:`PointCloud` instance from a shape model.

    Implementations must define sensible behavior for:

     - what a target is: see :attr:`target`
     - how to set a target: see :meth:`set_target`
     - how to update the object after a target is set:
       see :meth:`_sync_state_from_target`
     - how to produce a new target after the changes:
       see :meth:`_new_target_from_state`

    Note that :meth:`_sync_target_from_state` needs to be triggered as
    appropriate by subclasses e.g. when :map:`from_vector_inplace` is
    called. This will in turn trigger :meth:`_new_target_from_state`, which each
    subclass must implement.
    """

    @property
    def n_dims(self) -> int:
        r"""The number of dimensions of the :attr:`target`.

        :type: `int`
        """
        return self.target.n_dims

    @property
    def n_points(self) -> int:
        r"""The number of points on the :attr:`target`.

        :type: `int`
        """
        return self.target.n_points

    @property
    def target(self) -> "PointCloud":
        r"""The current :map:`PointCloud` that this object produces.

        :type: :map:`PointCloud`
        """
        raise NotImplementedError()

    def set_target(self, new_target: "PointCloud") -> None:
        r"""
        Update this object so that it attempts to recreate the ``new_target``.

        Parameters
        ----------
        new_target : :map:`PointCloud`
            The new target that this object should try and regenerate.
        """
        self._target_setter_with_verification(new_target)  # trigger the update
        self._sync_state_from_target()  # and a sync

    def _target_setter_with_verification(self, new_target: "PointCloud") -> None:
        r"""Updates the target, checking it is sensible, without triggering a
        sync.

        Should be called by :meth:`_sync_target_from_state` once it has
        generated a suitable target representation.

        Parameters
        ----------
        new_target : :map:`PointCloud`
            The new target that should be set.
        """
        self._verify_target(new_target)
        self._target_setter(new_target)

    def _verify_target(self, new_target: "PointCloud") -> None:
        r"""Performs sanity checks to ensure that the new target is valid.

        This includes checking the dimensionality matches and the number of
        points matches the current target's values.

        Parameters
        ----------
        new_target : :map:`PointCloud`
            The target that needs to be verified.

        Raises
        ------
        ValueError
            If the ``new_target`` has differing ``n_points`` or ``n_dims`` to
            ``self``.
        """
        # If the target is None (i.e. on construction) then dodge the
        # verification
        if self.target is None:
            return
        if new_target.n_dims != self.target.n_dims:
            raise ValueError(
                "The current target is {}D, the new target is {}D - new "
                "target has to have the same dimensionality as the "
                "old".format(self.target.n_dims, new_target.n_dims)
            )
        elif new_target.n_points != self.target.n_points:
            raise ValueError(
                "The current target has {} points, the new target has {} "
                "- new target has to have the same number of points as the"
                " old".format(self.target.n_points, new_target.n_points)
            )

    def _target_setter(self, new_target: "PointCloud") -> None:
        r"""Sets the target to the new value.

        Does no synchronization. Note that it is advisable that
        :meth:`_target_setter_with_verification` is called from
        subclasses instead of this.

        Parameters
        ----------
        new_target : :map:`PointCloud`
            The new target that will be set.
        """
        raise NotImplementedError()

    def _sync_target_from_state(self) -> None:
        new_target = self._new_target_from_state()
        self._target_setter_with_verification(new_target)

    def _new_target_from_state(self) -> "PointCloud":
        r"""Generate a new target that is correct after changes to the object.

        Returns
        -------
        new_target : :map:`PointCloud`
        """
        raise NotImplementedError()

    def _sync_state_from_target(self) -> None:
        r"""Synchronizes the object state to be correct after changes to the
        target.

        Called automatically from the target setter. This is called after the
        target is updated - only handle synchronization here.
        """
        raise NotImplementedError()


def menpo_src_dir_path() -> Path:
    r"""The path to the top of the menpo Python package.

    Useful for locating where the data folder is stored.

    Returns
    -------
    path : ``pathlib.Path``
        The full path to the top of the Menpo package
    """
    return Path(os.path.abspath(__file__)).parent


class MenpoDeprecationWarning(Warning):
    r"""
    A warning that functionality in Menpo will be deprecated in a future major
    release.
    """
    pass


class MenpoMissingDependencyError(ImportError):
    r"""
    An exception that a dependency required for the requested functionality
    was not detected.
    """

    def __init__(self, package_name: Union[str, ImportError]) -> None:
        super(MenpoMissingDependencyError, self).__init__()
        if isinstance(package_name, ImportError):
            package_name = self._handle_importerror(package_name)

        self.message = textwrap.dedent(
            """
            You need to install the '{pname}' package in order to use this
            functionality. We recommend that you use conda to achieve this -
            try the command

                conda install {pname}

            in your terminal. Note that this package may be provided by another
            channel such as the "conda-forge" channel.
            Failing that, try installing use pip:

                pip install {pname}
                
            Note that some packages (e.g. scikit-image) may have a different
            name on pypi/conda than their import (skimage) and thus the above 
            commands may fail.
        """.format(
                pname=package_name
            )
        )

        self.missing_name = package_name

    def _handle_importerror(self, error: ImportError) -> str:
        return error.name or str(error)

    def __str__(self) -> str:
        return self.message


def name_of_callable(c: Callable) -> str:
    r"""
    Return the name of a callable (function or callable class) as a string.
    Recurses on partial function to attempt to find the wrapped
    methods actual name.

    Parameters
    ----------
    c : `callable`
        A callable class or function, or any valid Python object that can
        be wrapped with partial.

    Returns
    -------
    name : `str`
        The name of the passed object.
    """
    try:
        if isinstance(c, partial):  # partial
            # Recursively call as partial may be wrapping either a callable
            # or a function (or another partial for some reason!)
            return name_of_callable(c.func)
        else:
            return c.__name__  # function
    except AttributeError:
        return c.__class__.__name__  # callable class


class LazyList(collections_abc.Sequence, Copyable):
    r"""
    An immutable sequence that provides the ability to lazily access objects.
    In truth, this sequence simply wraps a list of callables which are then
    indexed and invoked. However, if the callable represents a function that
    lazily access memory, then this list simply implements a lazy list
    paradigm.

    When slicing, another `LazyList` is returned, containing the subset
    of callables.

    Parameters
    ----------
    callables : list of `callable`
        A list of `callable` objects that will be invoked if directly indexed.
    """

    def __init__(self, callables: Sequence[Callable]):
        self._callables = callables

    def __getitem__(
        self, slice_: Union[Iterable, int, SupportsIndex, slice]
    ) -> "LazyList":
        # note that we have to check for iterable *before* __index__ as ndarray
        # has both (but we expect the iteration behavior when slicing)
        if isinstance(slice_, Iterable):
            # An iterable object is passed - return a new LazyList
            return LazyList([self._callables[s] for s in slice_])
        elif isinstance(slice_, int) or hasattr(slice_, "__index__"):
            # PEP 357 and single integer index access - returns element
            return self._callables[slice_]()
        else:
            # A slice or unknown type is passed - let List handle it
            return LazyList(self._callables[slice_])

    def __len__(self) -> int:
        return len(self._callables)

    @classmethod
    def init_from_iterable(
        cls, iterable: Iterable, f: Optional[Callable] = None
    ) -> "LazyList":
        r"""
        Create a lazy list from an existing iterable (think Python `list`) and
        optionally a `callable` that expects a single parameter which will be
        applied to each element of the list. This allows for simply
        creating a `LazyList` from an existing list and if no `callable` is
        provided the identity function is assumed.

        Parameters
        ----------
        iterable : `collections.Iterable`
            An iterable object such as a `list`.
        f : `callable`, optional
            Callable expecting a single parameter.

        Returns
        -------
        lazy : `LazyList`
            A LazyList where each element returns each item of the provided
            iterable, optionally with `f` applied to it.
        """
        if f is None:
            # The identity function
            def f(i):
                return i

        return cls([partial(f, x) for x in iterable])

    @classmethod
    def init_from_index_callable(cls, f, n_elements):
        r"""
        Create a lazy list from a `callable` that expects a single parameter,
        the index into an underlying sequence. This allows for simply
        creating a `LazyList` from a `callable` that likely wraps
        another list in a closure.

        Parameters
        ----------
        f : `callable`
            Callable expecting a single integer parameter, index. This is an
            index into (presumably) an underlying sequence.
        n_elements : `int`
            The number of elements in the underlying sequence.

        Returns
        -------
        lazy : `LazyList`
            A LazyList where each element returns the underlying indexable
            object wrapped by ``f``.
        """
        return cls([partial(f, i) for i in range(n_elements)])

    def map(self, f):
        r"""
        Create a new LazyList where the passed callable ``f`` wraps
        each element.

        ``f`` should take a single parameter, ``x``, that is the result
        of the underlying callable -  it must also return a value. Note that
        mapping is lazy and thus calling this function should return
        immediately.

        Alternatively, ``f`` may be a list of `callable`, one per entry
        in the underlying list, with the same specification as above.

        Parameters
        ----------
        f : `callable` or `iterable` of `callable`
            Callable to wrap each element with. If an iterable of callables
            (think list) is passed then it **must** by the same length as
            this LazyList.

        Returns
        -------
        lazy : `LazyList`
            A new LazyList where each element is wrapped by (each) ``f``.
        """

        # We need this delayed helper function in order to ensure that f
        # is passed the actual instantiated object and not the callable itself.
        def delayed(delay_f, delay_x):
            return delay_f(delay_x())

        if isinstance(f, Iterable) and callable(f):
            raise ValueError(
                "It is ambiguous whether the provided argument "
                "is an iterable object or a callable."
            )

        new = self.copy()
        if isinstance(f, Iterable):
            if len(f) != len(new):
                raise ValueError(
                    "A callable per element of the LazyList must " "be passed."
                )
            new._callables = [
                partial(delayed, one_f, x) for one_f, x in zip(f, new._callables)
            ]
        else:
            new._callables = [partial(delayed, f, x) for x in new._callables]
        return new

    def repeat(self, n):
        r"""
        Repeat each item of the underlying LazyList ``n`` times. Therefore,
        if a list currently has ``D`` items, the returned list will contain
        ``D * n`` items and will return immediately (method is lazy).

        Parameters
        ----------
        n : `int`
            The number of times to repeat each item.

        Returns
        -------
        lazy : `LazyList`
            A LazyList where each element returns each item of the provided
            iterable, optionally with `f` applied to it.

        Examples
        --------
        >>> from menpo.base import LazyList
        >>> ll = LazyList.init_from_list([0, 1])
        >>> repeated_ll = ll.repeat(2)  # Returns immediately
        >>> items = list(repeated_ll)   # [0, 0, 1, 1]
        """
        new = self.copy()
        new._callables = list(chain(*zip(*[new._callables] * n)))
        return new

    def copy(self) -> "LazyList":
        r"""
        Generate an efficient copy of this LazyList - copying the underlying
        callables will be lazy and shallow (each callable will **not** be
        called nor copied) but they will reside within in a new `list`.

        Returns
        -------
        ``type(self)``
            A copy of this LazyList.
        """
        new = Copyable.copy(self)
        new._callables = list(self._callables)
        return new

    def __add__(self, other: Union["LazyList", Iterable]) -> "LazyList":
        r"""
        Create a new LazyList from this list and the given list. The passed list
        items will be concatenated to the end of this list to give a new
        LazyList that contains the concatenation of the two lists.

        If a Python list is passed then the elements are wrapped in a function
        that just returns their values to maintain the callable nature of
        LazyList elements.

        Parameters
        ----------
        other : `collections.Sequence`
            Sequence to concatenate with this list.

        Returns
        -------
        lazy : `LazyList`
            A new LazyList formed of the concatenation of this list and
            the ``other`` list.

        Raises
        ------
        ValueError
            If other is not a LazyList or an Iterable
        """
        if isinstance(other, LazyList):
            return LazyList(self._callables + other._callables)
        elif isinstance(other, Iterable):
            return self + LazyList.init_from_iterable(other)
        else:
            raise ValueError(
                "Can only add another LazyList or an Iterable to a LazyList "
                "- {} is neither".format(type(other))
            )

    def view_widget(self):
        r"""
        Visualize this lazy collection of items using menpowidgets.

        The type of the first item will be used to determine an appropriate
        visualization for the list of items.

        Returns
        -------
        widget
            The appropriate menpowidget to view these items


        Raises
        ------
        MenpowidgetsMissingError
            If menpowidgets is not installed
        ValueError
            If menpowidgets cannot locate an appropriate items-visualization
            for the type of items in this :map:`LazyList`
        """
        try:
            from menpowidgets import view_widget
        except ImportError as e:
            from menpo.visualize.base import MenpowidgetsMissingError

            raise MenpowidgetsMissingError(e)
        else:
            return view_widget(self)

    def __str__(self):
        return "LazyList containing {} items".format(len(self))


def partial_doc(func, *args, **kwargs):
    r"""
    Return a partial function but the __doc__ attached to the returned
    partial. Note that no effort is made to correct the docstring for
    any parameters that are covered by the partial.

    Parameters
    ----------
    func : `callable`
        The func to partial and whose docs should be copied.
    args : ...
        Any arguments to partial.
    kwargs : `dict`
        Any keyword arguments to partial.

    Returns
    -------
    p_func : `callable`
        The partially wrapped func with __doc__ attached.
    """
    p = partial(func, *args, **kwargs)
    p.__doc__ = func.__doc__
    return p
