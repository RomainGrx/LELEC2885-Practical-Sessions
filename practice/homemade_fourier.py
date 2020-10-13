#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2020 Oct 13, 11:58:49
@last modified : 2020 Oct 13, 11:58:49
"""

import numpy as np

def dft(x):
    """
    .. math::
        x[k] = \sum_{n=0}^{N-1} x[n] e^{\dfrac{-2j\pi k n}{N}}
    """
    x = np.array(x)
    N = len(x)
    X = np.zeros(shape=x.shape, dtype=np.complex64)
    for k in range(N):
        fourier_sum = .0 + .0j
        for n in range(N):
            fourier_sum += x[n] * np.exp(-2 * np.pi * k * n * 1j / N)
        X[k] = fourier_sum
    return X


if __name__=='__main__':
    x = np.sin(np.arange(-10, 10, .1))

    assert np.allclose(np.fft.fft(x), dft(x))
