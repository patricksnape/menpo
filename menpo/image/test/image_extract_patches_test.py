from nose.tools import assert_equals, raises
from menpo.shape import PointCloud

import menpo.io as mio
from menpo.landmark import labeller, ibug_face_68


bb = mio.import_builtin_asset('breakingbad.jpg')


@raises(ValueError)
def test_negative_patch_size():
    patch_shape = (-16, -16)
    bb.extract_patches(bb.landmarks['PTS'].lms,
                       patch_size=patch_shape)


def test_squared_even_patches():
    patch_shape = (16, 16)
    patches = bb.extract_patches(bb.landmarks['PTS'].lms,
                                 patch_size=patch_shape)
    assert_equals(len(patches), 68)


def test_squared_odd_patches():
    patch_shape = (15, 15)
    patches = bb.extract_patches(bb.landmarks['PTS'].lms,
                                 patch_size=patch_shape)
    assert_equals(len(patches), 68)


def test_nonsquared_even_patches():
    patch_shape = (16, 18)
    patches = bb.extract_patches(bb.landmarks['PTS'].lms,
                                 patch_size=patch_shape)
    assert_equals(len(patches), 68)


def test_nonsquared_odd_patches():
    patch_shape = (15, 17)
    patches = bb.extract_patches(bb.landmarks['PTS'].lms,
                                 patch_size=patch_shape)
    assert_equals(len(patches), 68)


def test_nonsquared_even_odd_patches():
    patch_shape = (15, 16)
    patches = bb.extract_patches(bb.landmarks['PTS'].lms,
                                 patch_size=patch_shape)
    assert_equals(len(patches), 68)


def test_squared_even_patches_landmarks():
    patch_shape = (16, 16)
    patches = bb.extract_patches_around_landmarks('PTS',
                                                  patch_size=patch_shape)
    assert_equals(len(patches), 68)


def test_squared_even_patches_landmarks_label():
    labeller(bb, 'PTS', ibug_face_68)
    patch_shape = (16, 16)
    patches = bb.extract_patches_around_landmarks('ibug_face_68',
                                                  label='nose',
                                                  patch_size=patch_shape)
    assert_equals(len(patches), 9)


def test_squared_even_patches_single_array():
    labeller(bb, 'PTS', ibug_face_68)
    patch_shape = (16, 16)
    patches = bb.extract_patches(bb.landmarks['PTS'].lms,
                                 as_single_array=True,
                                 patch_size=patch_shape)
    assert_equals(patches.shape, ((68, 1, 3) + patch_shape))


def test_squared_even_patches_sample_offsets():
    labeller(bb, 'PTS', ibug_face_68)
    sample_offsets = PointCloud([[0, 0], [1, 0]])
    patches = bb.extract_patches(bb.landmarks['PTS'].lms,
                                 sample_offsets=sample_offsets)
    assert_equals(len(patches), 136)
