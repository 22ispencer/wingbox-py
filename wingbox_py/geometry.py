from abc import ABC, abstractmethod
from typing import Self
import numpy as np


class Point:
    x: float
    y: float

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def distance_from_point(self, other: Self) -> float:
        """Distance from one point to another"""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class Geometry(ABC):

    @abstractmethod
    def __init__(self, rotation: float, center: Point) -> None:
        self.rotation = rotation
        self.center = center

    @abstractmethod
    def area(self) -> float:
        """Area of the shape"""

    @abstractmethod
    def centroid(self) -> Point:
        """Centroid of the shape"""

    @abstractmethod
    def _smoa_yy_rel(self) -> float:
        """Second moment of the area about the centroidal y-axis"""

    @abstractmethod
    def _smoa_zz_rel(self) -> float:
        """Second moment of the area about the centroidal z-axis"""

    @abstractmethod
    def _pmoa_yz_rel(self) -> float:
        """Product moment of the area about the centroidal y-axis"""

    def smoa(self, phi: float = 0, dist: float = 0) -> float:
        """Second moment of the area about any axis on the yz-plane

        :param phi: The angle of the axis of interest from the y' axis measured counter-clockwise in radians.
        :param dist: The distance of the offset from the axis of interest.
        """
        area = self.area()
        theta = self.rotation - phi
        i_yy = self._smoa_yy_rel()
        i_zz = self._smoa_zz_rel()
        i_yz = self._pmoa_yz_rel()

        rotated = (
            0.5 * (i_yy + i_zz)
            + 0.5 * (i_yy - i_zz) * np.cos(2 * theta)
            - i_yz * np.sin(2 * theta)
        )
        offset = area * dist**2

        return rotated + offset

    def pmoa(self, phi: float = 0, dy: float = 0, dz: float = 0) -> float:
        """Second moment of the area about any axis on the yz-plane

        :param phi: The angle of the axis of interest from the y' axis measured counter-clockwise in radians.
        :param dy: The distance of the centroid from the y-axis of interest.
        :param dz: The distance of the centroid from the z-axis of interest.
        """
        area = self.area()
        theta = self.rotation - phi
        i_yy = self._smoa_yy_rel()
        i_zz = self._smoa_zz_rel()
        i_yz = self._pmoa_yz_rel()

        rotated = 0.5 * (i_yy - i_zz) * np.sin(2 * theta) + i_yz * np.cos(2 * theta)
        offset = area * dy * dz

        return rotated + offset


class Rect(Geometry):
    base: float
    height: float

    def __init__(
        self, rotation: float, center: Point, base: float, height: float
    ) -> None:
        super().__init__(rotation, center)
        self.center = center
        self.base = base
        self.height = height

    def area(self) -> float:
        return self.base * self.height

    def centroid(self) -> Point:
        return self.center

    def _smoa_yy_rel(self) -> float:
        return self.base * self.height**3 / 12

    def _smoa_zz_rel(self) -> float:
        return self.base**3 * self.height / 12

    def _pmoa_yz_rel(self) -> float:
        return 0


class ThinCircularSection(Geometry):
    thickness: float
    beta: float

    def __init__(
        self,
        rotation: float,
        center: Point,
        radius: float,
        thickness: float,
        beta: float,
    ) -> None:
        super().__init__(rotation, center)
        self.radius = radius
        self.thickness = thickness
        self.beta = beta

    def area(self) -> float:
        return 2 * self.beta * self.radius * self.thickness

    def _smoa_yy_rel(self) -> float:
        return (
            0.5
            * self.radius**3
            * self.thickness
            * (2 * self.beta - np.sin(2 * self.beta))
        )

    def _smoa_zz_rel(self) -> float:
        y_bar = -self.radius * np.sin(self.beta) / self.beta
        i_zz_circ_cent = (
            0.5
            * self.radius**3
            * self.thickness
            * (2 * self.beta + np.sin(2 * self.beta))
        )
        return i_zz_circ_cent - self.area() * y_bar**2

    def _pmoa_yz_rel(self) -> float:
        return 0
