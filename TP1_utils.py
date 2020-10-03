import numpy as np
from matplotlib import pyplot as plt

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