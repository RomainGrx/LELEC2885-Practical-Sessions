import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def plotTheCentroids(X,assignations,centroids,iteration = None,plotCentroids=True):
    """Plots a data in 2-D with the associated centroids.
    
    Argument:
    - X: numpy array, the dataset
    - assignations: numpy array of ints, the assignations to clusters of the elements of X
    - centroids: numpy array, the centroids
    - iteration: int, the kmeans iteration at which the plot function is called
    """
    
    plt.figure(figsize=(8,8))
    (K,_) = centroids.shape
    plotlegend = []
    for i in range(K):
        cluster = X[assignations == i]
        centro = centroids[i]
        p = plt.scatter(cluster[:,0],cluster[:,1],s=100,marker='.')
        if plotCentroids:  plt.scatter(centro[0],centro[1],s=200,marker='*',c=p.get_facecolor())
        plotlegend.append('data class ' + str(i))
        if plotCentroids: plotlegend.append('centroid class ' + str(i))
    if K < 5:
        plt.legend(plotlegend)
    if (iteration==0) : # Useful to visually validate the kmeans++ initialization
        cluster = X[assignations == 0]
        plt.scatter(cluster[:,0],cluster[:,1],s=100,marker='.',c='yellow')
    if iteration is not None:
        plt.title('K-means at iteration ' + str(iteration))
    lim = 1.1*np.max(np.abs(X))
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    
def findEdges(assignationsImage):
    """Takes the cluster assignation image, returns the cluster edges."""

    edges = np.zeros(assignationsImage.shape)
    lapkernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) # Laplacian kernel to obtain the edges
    edges = scipy.signal.fftconvolve(assignationsImage,lapkernel,mode='same')
    edges = edges > 0.01
    
    return(edges)

def visualizeSegmentation(image,centroids,assignations):
    if centroids.shape[1]>3:
        centroids = centroids[:,:3]
    # Orinal image
    fig, axes = plt.subplots(1,2, figsize=(12,12))
    
    axes[0].imshow(image)
    axes[0].set_title('Original image')

    # Code to create our own colormap
    # We use the centroids to define the color of each cluster
    from matplotlib.colors import ListedColormap
    newcmp = ListedColormap(centroids,N=centroids.shape[0])

    # Plot the clusters
    assignationsImage = assignations.reshape((image.shape[0],image.shape[1]))
    sp = axes[1].imshow(assignationsImage, newcmp)
    axes[1].set_title('Representing the clusters (one color = one cluster)')
    plt.colorbar(sp, ax=axes[1])

    # Also plot the edges based on the clusters (take a look at the code, you should be able to do it yourself)
    edges = findEdges(assignationsImage)
    # Colormap for the edges (4th dimension is opacity -> invisible where no edges)
    edgecmp = ListedColormap([[0,0,0,0],[0,0,0,1]],N=2)
    axes[1].imshow(edges,cmap=edgecmp)

    plt.show()
    
def compareSegmentations(image,centroids1,assignations1, centroids2, assignations2):
    if centroids1.shape[1]>3:
        centroids1 = centroids1[:,:3]
    if centroids2.shape[1]>3:
        centroids2 = centroids2[:,:3]
   
    fig, axes = plt.subplots(1,3, figsize=(20,20))
    
    # Orinal image
    axes[0].imshow(image)
    axes[0].set_title("Original image")
    # Code to create our own colormap
    # We use the centroids to define the color of each cluster
    from matplotlib.colors import ListedColormap
    newcmp1 = ListedColormap(centroids1,N=centroids1.shape[0])
    newcmp2 = ListedColormap(centroids2,N=centroids2.shape[0])

    # Plot the clusters
    assignationsImage1 = assignations1.reshape((image.shape[0],image.shape[1]))
    assignationsImage2 = assignations2.reshape((image.shape[0],image.shape[1]))
    sp1 = axes[1].imshow(assignationsImage1, newcmp1)
    axes[1].set_title("RGB based segmentation")
    sp2 = axes[2].imshow(assignationsImage2, newcmp2)
    axes[2].set_title("Current segmentation with 5 features")
    # Also plot the edges based on the clusters (take a look at the code, you should be able to do it yourself)
    #     from TP5_utils import findEdges
    edges1 = findEdges(assignationsImage1)
    edges2 = findEdges(assignationsImage2)
    # Colormap for the edges (4th dimension is opacity -> invisible where no edges)
    edgecmp = ListedColormap([[0,0,0,0],[0,0,0,1]],N=2)
    axes[1].imshow(edges1,cmap=edgecmp)
    axes[2].imshow(edges2,cmap=edgecmp)

    plt.show()