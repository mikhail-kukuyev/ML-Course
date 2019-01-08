import numpy as np


def transform(a):
    a = np.reshape(a, (28, 28))
    return a


def vertical_symmetry_feature(a):
    """Subract right part of image from the left part and find mean of absolute differencies.
    More symmetrical digits will have lower value."""
    a = transform(a)
    b = np.abs(a[:, 14:] - a[:, :13:-1])
    return np.mean(b)


def horizontal_symmetry_feature(a):
    """Subract down part of image from the upper part and find mean of absolute differencies.
    More symmetrical digits will have lower value."""
    a = transform(a)
    b = np.abs(a[14:, :] - a[:13:-1, :])
    return np.mean(b)


def vertical_center_feature(a):
    """Mean of intensities of pixels in vertical center of image:
         digits with more used pixels there will have higher value. Useful for digits 0, 1, 3, 7."""
    a = transform(a)
    return np.mean(a[8:20, 10:18])


def horizontal_center_feature(a):
    """Mean of intensities of pixels in horizontal center of image:
             digits with more used pixels there will have higher value. Useful for digits 0, 4, 7, 8, 9."""
    a = transform(a)
    return np.mean(a[10:18, 8:20])


def uppper_part_feature(a):
    """Mean of intensities of pixels in upper part of image:
     digits with more used pixels there will have higher value. Useful for digits 2, 4, 7, 6, 8, 9."""
    a = transform(a)
    return np.mean(a[:14, :])


def down_part_feature(a):
    """Mean of intensities of pixels in down part of image:
     digits with more used pixels there will have higher value. Useful for digits 2, 6, 8, 9."""
    a = transform(a)
    return np.mean(a[14:, :])


def left_part_feature(a):
    """Mean of intensities of pixels in left part of image:
     digits with more used pixels there will have higher value. Useful for digits 2, 3, 6, 9."""
    a = transform(a)
    return np.mean(a[:, :14])


def right_part_feature(a):
    """Mean of intensities of pixels in right part of image:
     digits with more used pixels there will have higher value. Useful for digits 4, 5, 7, 9."""
    a = transform(a)
    return np.mean(a[:, 14:])


FEATURES = {
    (0, 1): (horizontal_symmetry_feature, horizontal_center_feature),
    (0, 2): (uppper_part_feature, right_part_feature),
    (0, 3): (horizontal_center_feature, left_part_feature),
    (0, 4): (horizontal_center_feature, uppper_part_feature),
    (0, 5): (vertical_center_feature, down_part_feature),
    (0, 6): (uppper_part_feature, down_part_feature),
    (0, 7): (vertical_center_feature, down_part_feature),
    (0, 8): (vertical_center_feature, down_part_feature),
    (0, 9): (horizontal_center_feature, left_part_feature),
    (1, 2): (horizontal_symmetry_feature, vertical_center_feature),
    (1, 3): (horizontal_symmetry_feature, vertical_center_feature),
    (1, 4): (vertical_center_feature, horizontal_center_feature),
    (1, 5): (horizontal_symmetry_feature, horizontal_center_feature),
    (1, 6): (horizontal_symmetry_feature, uppper_part_feature),
    (1, 7): (vertical_center_feature, uppper_part_feature),
    (1, 8): (vertical_symmetry_feature, horizontal_symmetry_feature),
    (1, 9): (vertical_center_feature, horizontal_center_feature),
    (2, 3): (down_part_feature, right_part_feature),
    (2, 4): (horizontal_center_feature, down_part_feature),
    (2, 5): (uppper_part_feature, right_part_feature),
    (2, 6): (horizontal_center_feature, right_part_feature),
    (2, 7): (uppper_part_feature, down_part_feature),
    (2, 8): (horizontal_symmetry_feature, horizontal_center_feature),
    (2, 9): (horizontal_center_feature, down_part_feature),
    (3, 4): (horizontal_center_feature, uppper_part_feature),
    (3, 5): (left_part_feature, right_part_feature),
    (3, 6): (uppper_part_feature, left_part_feature),
    (3, 7): (horizontal_symmetry_feature, left_part_feature),
    (3, 8): (vertical_center_feature, right_part_feature),
    (3, 9): (horizontal_center_feature, down_part_feature),
    (4, 5): (horizontal_center_feature, left_part_feature),
    (4, 6): (horizontal_center_feature, down_part_feature),
    (4, 7): (horizontal_center_feature, uppper_part_feature),
    (4, 8): (horizontal_center_feature, uppper_part_feature),
    (4, 9): (uppper_part_feature, down_part_feature),
    (5, 6): (uppper_part_feature, down_part_feature),
    (5, 7): (down_part_feature, right_part_feature),
    (5, 8): (vertical_symmetry_feature, horizontal_symmetry_feature),
    (5, 9): (horizontal_center_feature, left_part_feature),
    (6, 7): (down_part_feature, right_part_feature),
    (6, 8): (horizontal_symmetry_feature, uppper_part_feature),
    (6, 9): (down_part_feature, right_part_feature),
    (7, 8): (down_part_feature, right_part_feature),
    (7, 9): (horizontal_center_feature, uppper_part_feature),
    (8, 9): (horizontal_center_feature, left_part_feature)
}
