#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2020 Oct 13, 11:58:49
@last modified : 2020 Oct 13, 13:46:46
"""

import numpy as np

def dft(x):
    x = np.array(x)
    if x.ndim == 1:
        return dft1D(x)
    elif x.ndim == 2:
        return dft2D(x)
    else:
        raise NotImplementedError(f"Dimension {x.ndim} not supported.")

def dft1D(x):
    """
    .. math::
        x[k] = \sum_{n=0}^{N-1} x[n] e^{\dfrac{-2j\pi k n}{N}}
    """
    x = np.array(x)
    N = len(x)
    X = np.zeros(shape=x.shape, dtype=np.complex64)

    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2 * np.pi * k * n * 1j / N)
    return X


def dft2D(x):
    """
        .. math::
            x[k,l] = \sum_{n=0}^{N-1} \sum_{m=0}^{N-1} x[n,m] e^{\dfrac{-2j\pi (nk+ml)}{N}}
    """
    x = np.array(x) # Making sure it's an ndarray
    N = len(x)

    X = np.zeros(shape=x.shape, dtype=np.complex64) # Allocate a zero array with same shape as x

    for k in range(N):
        for l in range(N):
            for n in range(N):
                for m in range(N):
                    X[k, l] += x[n, m] * np.exp(-2j * np.pi * (n*k + m*l) / N)
    return X


if __name__=='__main__':
    x = np.sin(np.arange(-10, 10, .1))
    x2d = np.random.normal(size=(10, 10))

    assert np.allclose(np.fft.fft(x), dft(x))
    assert np.allclose(np.fft.fft2(x2d), dft(x2d))

