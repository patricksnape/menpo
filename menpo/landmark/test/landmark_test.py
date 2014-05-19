import numpy as np
from nose.tools import assert_equal
from numpy.testing import assert_allclose
from menpo.landmark import LandmarkGroup, LandmarkManager
from menpo.shape import PointCloud
from menpo.testing import is_same_array


def test_LandmarkGroup_copy_true():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    lgroup = LandmarkGroup(None, 'label', pcloud, mask_dict)
    assert (not is_same_array(lgroup._pointcloud.points, points))
    assert (lgroup._labels_to_masks is not mask_dict)
    assert (lgroup._pointcloud is not pcloud)


def test_LandmarkGroup_copy_false():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    lgroup = LandmarkGroup(None, 'label', pcloud, mask_dict, copy=False)
    assert (is_same_array(lgroup._pointcloud.points, points))
    assert (lgroup._labels_to_masks is mask_dict)
    assert (lgroup._pointcloud is pcloud)


def test_LandmarkManager_set_LandmarkGroup_not_copy_target():
    points = np.ones((10, 3))
    mask_dict = {'all': np.ones(10, dtype=np.bool)}
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)
    lgroup = LandmarkGroup(None, 'label', pcloud, mask_dict, copy=False)

    man = LandmarkManager(target)
    man['test_set'] = lgroup
    assert (not is_same_array(man['test_set'].lms.points,
                              lgroup.lms.points))
    assert_allclose(man['test_set']['all'].lms.points, np.ones([10, 3]))
    assert (man['test_set']._labels_to_masks is not lgroup._labels_to_masks)
    assert (man['test_set']._target is target)
    assert_equal(man['test_set'].group_label, 'test_set')


def test_LandmarkManager_set_PointCloud_not_copy_target():
    points = np.ones((10, 3))
    pcloud = PointCloud(points, copy=False)
    target = PointCloud(points)

    man = LandmarkManager(target)
    man['test_set'] = pcloud
    print(man['test_set'])
    assert (not is_same_array(man['test_set'].lms.points,
                              pcloud.points))
    assert_allclose(man['test_set']['all'].lms.points, np.ones([10, 3]))
    assert_equal(man['test_set'].group_label, 'test_set')
    assert (man['test_set']._target is target)