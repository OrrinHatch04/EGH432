from typing import Dict, List, Literal as L
import numpy as np

# --------- Question 2.1 ---------- #


def rotation_create(angle: float, axis: L["x", "y", "z"]) -> np.ndarray:
    """
    Creates an SO(3) rotation matrix about a single axis

    This method takes the argument `angle` and creates a rotation matrix
    about the axis specified by `axis`.

    Parameters
    ----------
    angle
        the angle of rotation in radians
    axis
        the axis of rotation, either `x`, `y`, or `z`

    Returns
    -------
    rotation_matrix
        a (3, 3) numpy array representing the rotation matrix
    """

    # your code goes here
    pass


# --------- Question 2.2 ---------- #


def translation_create(translation: float, axis: L["x", "y", "z"]) -> np.ndarray:
    """
    Creates a 3D translation vector along a single axis

    This method takes the argument `translation` and creates a translation
    vector along the axis specified by `axis`.

    Parameters
    ----------
    translation
        the distance of translation (in metres)
    axis
        the axis of translation, either `x`, `y`, or `z`

    Returns
    -------
    translation_vector
        a (3, 1) numpy array representing the translation vector
    """

    # your code goes here
    pass


# --------- Question 2.3 ---------- #


class TransformCreate:
    """
    Creates a varibale elementary homogeneous transformation matrix

    This class takes the arguments `type` and `axis` and creates a
    homogeneous transformation matrix that is either a rotation or
    translation matrix about the axis specified by `axis`. The type
    of transformation is specified by `type`, either `r` or `t` for
    rotation or translation respectively.

    Parameters
    ----------
    type
        the type of transformation, either `r` or `t` for rotation or
        translation respectively
    axis
        the axis of rotation or translation, either `x`, `y`, or `z

    """

    def __init__(self, type: L["r", "t"], axis: L["x", "y", "z"]):

        # your code goes here
        pass

    def evaluate(self, arg: float) -> np.ndarray:
        """
        Evaluates the transformation matrix at a given value

        This method takes the argument `arg` and evaluates the
        transformation matrix at that value. If the transformation
        type is a 'r', then `arg` is the angle of rotation in radians.
        If the transformation type is 't', then `arg` is the distance
        of translation in metres.

        Parameters
        ----------
        arg
            the value at which to evaluate the transformation matrix

        Returns
        -------
        transform
            a (4, 4) numpy array representing the homogeneous
            transformation matrix

        """

        # your code goes here
        pass


# --------- Question 2.4 ---------- #


def transform_composer(
    aTb: TransformCreate, arg1: float, bTc: TransformCreate, arg2: float
) -> np.ndarray:
    """
    Composes two elementary homogeneous transformation matrices

    This function takes two elementary homogeneous transformation
    matrices and composes them into a single homogeneous transformation
    matrix.

    Parameters
    ----------
    aTb
        represents the transformation from frame `a` to frame `b`
    arg1
        the value at which to evaluate `aTb`
    bTc
        represents the transformation from frame `b` to frame `c`
    arg2
        the value at which to evaluate `bTc`

    Returns
    -------
    aTc
        a (4, 4) numpy array representing the homogeneous
        transformation matrix from frame `a` to frame `c`

    Notes
    -----
    Assuming that we are following the notation convention from the notes, the
    `aTb` argument represents the transformation from frame `a` to frame `b` and
    the `bTc` argument represents the transformation from frame `b` to frame `c`.
    Pretend that the letter before the `T` is a superscript and the letter after
    the `T` is a subscript.

    """

    # your code goes here
    pass


# --------- Question 2.5 ---------- #


class TransformComposer:
    """
    Composes a list of variable elementary homogeneous transformation matrices

    This class takes a list of TransformCreate objects each representing
    a variable elementary homogeneous transformation matrix. The list represents
    a sequence of transformations from frame `a` to frame `b`, frame `b` to frame
    `c`, frame `c` to frame `d`, and so on.

    Parameters
    ----------
    transforms
        a list of TransformCreate objects

    """

    def __init__(self, transforms: List[TransformCreate]):

        # your code goes here
        pass

    def evaluate(self, args: List[float]) -> np.ndarray:
        """
        Evaluates the composition of all transformation matrices at the given values

        This method takes the argument `args` which is a list of rotation angles or
        translation distances where each value corresponds to the value at which to
        evaluate the corresponding TransformCreate from the transforms list passed
        to the constructor. The first value in `args` corresponds to the first
        TransformCreate object, the second value in `args` corresponds to the
        second TransformCreate object, and so on.

        The method returns a `numpy` array of shape `(4, 4)` that represents the
        corresponding homogeneous transformation matrix from frame `a` to frame `n`
        where `n` is the number of transformations in the list.

        Parameters
        ----------
        args
            a list of rotation angles or translation distances of length `n` where `n`
            is the number of transformations in the list passed to the constructor

        Returns
        -------
        transform
            a (4, 4) numpy array representing the commposition of all the
            homogeneous transformation matrices

        """

        # your code goes here
        pass
