#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D transformation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

from so3 import SO3


class SE3:
    """Transformation in 2D that is composed of rotation and translation."""

    def __init__(
        self, translation: ArrayLike | None = None, rotation: SO3 | None = None
    ) -> None:
        """Crete an SE3 transformation. Identity is the default."""
        super().__init__()
        self.translation = (
            np.asarray(translation) if translation is not None else np.zeros(3)
        )
        self.rotation = rotation if rotation is not None else SO3()
        assert self.translation.shape == (3,)

    @classmethod
    def from_homogeneous(cls, H: np.ndarray) -> "SE3":
        R = H[:3, :3]
        t = H[:3, 3]
        return cls(translation=t, rotation=SO3(R))

    def __mul__(self, other: SE3) -> SE3:
        """Compose two transformation, i.e., self * other"""
        result = SE3()
        result.rotation.rot = self.rotation.rot @ other.rotation.rot
        result.translation = self.translation + self.rotation.rot @ other.translation
        return result

    def inverse(self) -> SE3:
        """Compute inverse of the transformation"""
        result = SE3()
        result.rotation.rot = self.rotation.rot.T
        result.translation = -result.rotation.rot @ self.translation
        return result

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given 3D vector by this transformation."""
        result = np.asarray(vector)
        assert result.shape == (3   ,)
        result = self.rotation.rot @ vector + self.translation
        return result
    
    def set_from(self, other: SE3):
        """Copy the properties into current instance."""
        self.translation = other.translation
        self.rotation = other.rotation

    def homogeneous(self) -> np.ndarray:
        """Return homogeneous matrix representation of the transformation."""
        h = np.eye(4)
        h[:3, :3] = self.rotation.rot
        h[:3, 3] = self.translation
        return h

    def __eq__(self, other: SE3) -> bool:
        """Returns true if two transformations are almost equal."""
        return (
            np.allclose(self.translation, other.translation)
            and self.rotation == other.rotation
        )

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return f"(translation={self.translation}, log_rotation={self.rotation.log()})"
