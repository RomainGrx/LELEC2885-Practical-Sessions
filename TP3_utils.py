import numpy as np
import pywt
import matplotlib.pyplot as plt

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

def genSparseInDictionnaries(K,list_of_dictionnaries,noise_level = 0.):
    """Generates a toy example signal, K-sparse in each dictionnary of a given set."""
    (d,_) = list_of_dictionnaries[0].shape
    
    y = np.zeros(d)
    list_of_coefficients = []
    
    for dico in list_of_dictionnaries:
        (_,nbatoms) = dico.shape
        coefficient_indices = np.random.choice(nbatoms, K, replace=False)
        coefficient_values = np.random.randn(K)
        
        coefficients = np.zeros(nbatoms)
        coefficients[coefficient_indices] = coefficient_values
        
        list_of_coefficients += [coefficients]
        y += dico@coefficients
        
    y += noise_level*np.random.randn(d)
    
    return y,list_of_coefficients

def visualizeDictionary(D,figsize=(16,6),colorbar=True,title=None):
    plt.figure(figsize=figsize)
    plt.imshow(D, cmap='seismic', interpolation='nearest')
    plt.clim((-1, 1))
    if colorbar:
        plt.colorbar()
    plt.xticks(np.arange(0,0), [])
    plt.yticks(np.arange(0,0), [])
    if title is not None:
        plt.title(title)
    plt.show()
    return


def makeWaveletBasis(N,wavelet_name):
    """
    Returns the wavelet basis with name wavelet_name and in N dimensions.
    
    Note: to compute the wavelet transform, prefer the pywt functions that have fast implementations.
    This is slower but allows to access the atoms explicitly.
    """
    
    test_signal = np.zeros(N)
    
    wavelet_level = pywt.dwt_max_level(test_signal.size, wavelet_name)
    wavelet_coefs_test_signal = pywt.wavedec(test_signal, wavelet_name, level=wavelet_level)
    wavelet_coefs_test_signal_flat = arrayList2vec(wavelet_coefs_test_signal)
    nb_coefs = wavelet_coefs_test_signal_flat.size
    
    Psi = np.zeros((N,nb_coefs))
    
    for i in range(nb_coefs):
        coef = np.zeros(nb_coefs)
        coef[i] = 1.
        coef_list = vec2arrayList(coef,wavelet_coefs_test_signal)
        Psi[:,i] = pywt.waverec(coef_list,wavelet_name)
    
    return Psi


# old visualization function
'''plt.figure(figsize=(15,4))
plot_indices = np.arange(0,d)    
plt.stem(plot_indices,a_hat,linefmt='green',markerfmt='D')
plt.stem(plot_indices+0.15,a_GT,linefmt='red')
plt.title('Coefficients estimated by MP VS ground truth coefficients')
plt.xlabel('Coefficient index')
plt.legend(['a_hat','a_GT'])
plt.show()'''


'''
plt.figure(figsize=(16,4))
plot_indices = np.arange(0,d)    
(markerLines, stemLines, baseLines) = plt.stem(plot_indices-0.25,a_hat_MP,linefmt='blue',markerfmt='gD')
plt.setp(markerLines, markersize = 5, linewidth = 3)

(markerLines, stemLines, baseLines) = plt.stem(plot_indices,a_hat_OMP,linefmt='green',markerfmt='rD')
plt.setp(markerLines, markersize = 5, linewidth = 3)

(markerLines, stemLines, baseLines) = plt.stem(plot_indices+0.25,a_GT,linefmt='red')
plt.setp(markerLines, markersize = 5, linewidth = 3)

plt.title('Coefficients estimated by MP/OMP VS ground truth coefficients')
plt.xlabel('Coefficient index')
plt.legend(['a_MP','a_OMP','a_GT'])
plt.show()'''