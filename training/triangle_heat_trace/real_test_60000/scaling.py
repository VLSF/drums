"""Corner-adapted Dirichlet eigenfrequencies on numerical windows."""

from functools import lru_cache

import numpy as np
from scipy import special


@lru_cache(maxsize=24)
def quadrature(points):
    nodes, weights = special.roots_legendre(points)
    return (nodes + 1.) / 2., weights / 2.


class ScalingSolver:
    def __init__(self, vertices, basis_pad=14., point_factor=.67, gram_tol=2e-13):
        v = np.asarray(vertices, dtype=float)
        a, b = np.roll(v, -1, axis=0) - v, np.roll(v, 1, axis=0) - v
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        angles = np.arctan2(abs(cross), np.sum(a*b, axis=1))
        self.area = abs(cross[0]) / 2.
        vertex = np.argmax(angles)
        u, v = a[vertex], b[vertex]
        norm = np.linalg.norm(u)
        self.left = np.array([norm, 0.]) / np.sqrt(self.area)
        self.right = np.array([u @ v, abs(cross[vertex])]) / norm / np.sqrt(self.area)
        self.radius = max(np.linalg.norm(self.left), np.linalg.norm(self.right))
        self.length = np.linalg.norm(self.right-self.left)
        self.nu = np.pi / angles[vertex]
        self.pad = basis_pad
        self.point_factor = point_factor
        self.gram_tol = gram_tol
        self.table = None

    def basis_size(self, k):
        x = k*self.radius
        return int(np.ceil((x + self.pad*np.cbrt(x)) / self.nu)) + 2

    def evaluation(self, k):
        basis_size = self.basis_size(k)
        points = 8 * int(np.ceil((self.length*k*self.point_factor+25.)/8.))
        frac, weights = quadrature(points)
        xy = self.left + frac[:, None]*(self.right-self.left)
        radii = np.linalg.norm(xy, axis=1)
        angles = np.arctan2(xy[:, 1], xy[:, 0])
        orders = self.nu*np.arange(1., basis_size+1.)
        arguments = k*radii
        angular = np.sin(orders[:, None]*angles) * np.sqrt(weights)
        if self.table is None:
            value = special.jv(orders[:, None], arguments)
            next_order = special.jv(orders[:, None]+1., arguments)
            derivative = (orders[:, None]/k)*value - radii*next_order
        else:
            value, derivative = self.table.evaluate(arguments, basis_size)
            derivative *= radii
        return value*angular, derivative*angular

