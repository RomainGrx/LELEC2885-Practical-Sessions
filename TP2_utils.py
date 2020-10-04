import numpy as np
from matplotlib import pyplot as plt
import pywt
from functools import wraps

### TP2


def arrayList2vec(coefs):
    """Takes a list of (tuples of) numpy arrays and concatenate them into a single, one-dimensional array."""
    coefs_flat = None
    for c_tuple in coefs:
        if not isinstance(c_tuple, tuple): # Trick to handle all possible cases: convert to tuple...
            c_tuple = (c_tuple,)
        # ... so that at this point, we can always treat the element of the list as a tuple
        for c in c_tuple:
            if coefs_flat is None: # if this is the first element we add to coef_flat
                coefs_flat = c.reshape(-1)
            else:
                coefs_flat = np.concatenate((coefs_flat,c.reshape(-1)))
    return coefs_flat


def vec2arrayList(coefs_flat,original_coefs_template):
    """Creates a list of arrays containing coefs_flat, in the same size as the list of (tuples of) arrays original_coefs_template."""
    coefs = []
    for i,c_tuple in enumerate(original_coefs_template):
        if not isinstance(c_tuple, tuple): # We can directly add to the list without further packing
            c = c_tuple
            coefs += [coefs_flat[:c.size].reshape(c.shape)]
            coefs_flat = coefs_flat[c.size:] # Remove 
        else:
            list_for_tuple = []
            for c in c_tuple:
                list_for_tuple += [coefs_flat[:c.size].reshape(c.shape)]
                coefs_flat = coefs_flat[c.size:] # Remove
            coefs += [tuple(list_for_tuple)]
    return coefs


def pad(array, length, pos="left"):
    """Returns an array of length *length* with the provided *array* at the
    specified position"""
    l = array.shape[0]
    if pos == "center":
        l1 = (length - l)//2
        l2 = length - l - l1
        return np.concatenate([np.zeros(l1), array, np.zeros(l2)])
    elif pos == "right":
        return np.concatenate([np.zeros(length-l), array])
    elif pos == "left":
        return np.concatenate([array, np.zeros(length-l)])
    raise NotImplementedError("Position {!r} not valid.".format(pos))


def plot_wavelet(coeffs):
    """Plot wavelet coefficients, making non-zero value salient"""
    approx, *details = coeffs.pywt_format
    eps = 1e-8
    #approx = approx/255-0.5
    approx = approx/np.max(approx)
    scale = 1  #np.abs(approx).max()+eps
    deepest_shape = np.array(details[-1][1].shape)
    coeffs = np.zeros([
        approx.shape[0]+sum(x.shape[0] for x,*_ in details),
        approx.shape[1]+sum(x.shape[1] for x,*_ in details),
    ])
    coeffs[:approx.shape[0],:approx.shape[1]] = approx
    y, x = approx.shape
    for H, V, D in details:
        h, w = H.shape
        coeffs[y:y+h,x:x+w] = D*scale/(D.max()+eps)
        coeffs[ :h  ,x:x+w] = H*scale/(H.max()+eps)
        coeffs[y:y+h, :w  ] = V*scale/(V.max()+eps)
        y, x = y+h, x+w
    plt.imshow(coeffs, cmap="gray")


class WaveletCoeffs:
    def __init__(self, original):
        coeffs = []
        
        self._lengths = []
        self._shapes = []
        self._indices = []
        self._level_indices = []
        
        index = 0
        for element in original:
            is_tuple = isinstance(element, tuple)
            self._lengths.append(0 if not is_tuple else len(element))
            level_start = index
            for array in ((element,) if not is_tuple else element):
                coeffs.append(array.reshape(-1))
                self._shapes.append(array.shape)
                self._indices.append(slice(index, index+array.size))
                index += array.size
            level_end = index
            self._level_indices.append(slice(level_start, level_end))

        self._coeffs = np.concatenate(coeffs)
        self.detail_levels = len(self._lengths)-1
        
    def clone(self):
        ret = WaveletCoeffs(self.pywt_format)
        if hasattr(self, "kwargs"):
            ret.kwargs = self.kwargs
        return ret
    
    @property
    def pywt_format(self):
        ret = []
        i = 0
        for sublevels in self._lengths:
            if sublevels == 0:
                ret.append(self.coeffs[self._indices[i]].reshape(self._shapes[i]))
                i += 1
            else:
                sub = []
                for sublevel in range(sublevels):
                    sub.append(self.coeffs[self._indices[i]].reshape(self._shapes[i]))
                    i += 1
                ret.append(tuple(sub))
        return ret
    
    @property
    def coeffs(self):
        return self._coeffs

    @property
    def approx(self):
        assert self._lengths[0] == 0
        return self.coeffs[self._indices[0]]
        
    def details(self, start=None, end=None):
        """get details at from level start to level end (inclusive)"""
        # end and start are relative
        if start is None and end is None:
            start = 0
            end = ...
        assert start is not None
        if end is None:
            end = start
        if end == ...:
            end = self.detail_levels-1

        if start < 0:
            start = self.detail_levels + start
        if end < 0:
            end = self.detail_levels + end
        start, end = start+1, end+1  # increment because approx is level 0
        return self.coeffs[self._level_indices[start].start:self._level_indices[end].stop]
    
    @staticmethod
    def dec2(data, wavelet, mode='symmetric', level=None, axes=(-2, -1)):
        ret = WaveletCoeffs(pywt.wavedec2(data, wavelet, mode, level, axes))
        ret.kwargs = dict(wavelet=wavelet,mode=mode,axes=axes)
        return ret
    def rec2(self):
        return pywt.waverec2(self.pywt_format, **self.kwargs)
    
    @staticmethod
    def dec(data, wavelet, mode='symmetric', level=None, axis=-1):
        ret = WaveletCoeffs(pywt.wavedec(data, wavelet, mode, level, axis))
        ret.kwargs = dict(wavelet=wavelet,mode=mode,axis=axis)
        return ret
    def rec(self):
        return pywt.waverec(self.pywt_format, **self.kwargs)


### TP1


def plot_pyramid(arrays):
    H, W = arrays[0].shape
    next_shapes = [x.shape for x in arrays[1:]]+[(0,0)]
    W = W + next_shapes[0][1]
    image = np.zeros((H, W))+np.nan
    y, x = 0, 0
    for i, (array, next_shape) in enumerate(zip(arrays, next_shapes)):
        h, w = array.shape
        nh, nw = next_shape
        image[y:y+h,x:x+w] = array
        if i % 4 == 0:    # To the right
            y, x = y, x+w
        elif i % 4 == 1:  # To the bottom
            y, x = y+h, x+nw
        elif i % 4 == 2:  # To the left
            y, x = y+nh, x-nw
        elif i % 4 == 3:  # to the top
            y, x = y-nh, x
    plt.imshow(image, cmap="gray")
