"""Tabulate each Bessel order and evaluate its local polynomial interpolant."""

import ctypes
import os
from pathlib import Path

import numpy as np
from scipy import special


class BesselTable:
    def __init__(self, max_order_index, order_step, max_x, step=.35):
        self.step = float(step)
        default_library = Path(__file__).with_name("interpolate.dylib")
        path = os.environ.get("DRUMS_TRIANGLE_INTERPOLATION_LIB", str(default_library))
        self.library = ctypes.CDLL(path)
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.interpolate = self.library.interpolate
        self.interpolate.argtypes = [array, ctypes.c_int, ctypes.c_int,
                                     array, ctypes.c_int, ctypes.c_double, array, array]
        self.interpolate.restype = None
        x = np.arange(np.ceil(max_x/self.step)+14)*self.step
        orders = order_step*np.arange(1., max_order_index+1.)
        self.values = special.jv(orders[:, None], x)

    def evaluate(self, x, rows):
        x = np.ascontiguousarray(x, dtype=float)
        if not 0 < rows <= self.values.shape[0]:
            raise ValueError("Requested Bessel basis size exceeds the table height")
        value = np.empty((rows, len(x)))
        derivative = np.empty_like(value)
        self.interpolate(self.values, self.values.shape[1], rows,
                         x, len(x), self.step, value, derivative)
        return value, derivative
