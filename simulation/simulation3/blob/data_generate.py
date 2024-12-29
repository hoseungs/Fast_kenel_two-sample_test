import math as math
import numpy as np


def Blobs(n):
    x = gen_blobs(stretch=1, angle=0, blob_distance=5,
                num_blobs=4, num_samples=n)

    y = gen_blobs(stretch=4, angle=math.pi/4,
                blob_distance=5, num_blobs=4,
                num_samples=n)

    return (x, y)

def Blobsnull(n):
    x = gen_blobs(stretch=1, angle=0, blob_distance=5,
                  num_blobs=4, num_samples=n)
        
    y = gen_blobs(stretch=1, angle=0, blob_distance=5, num_blobs=4,num_samples=n)
                  
    return (x, y)

def gen_blobs(stretch, angle, blob_distance, num_blobs, num_samples):
    """Generate 2d blobs dataset """

    # rotation matrix
    r = np.array( [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]] )
    eigenvalues = np.diag(np.array([np.sqrt(stretch), 1]))
    mod_matix = np.dot(r, eigenvalues)
    mean = (blob_distance * (num_blobs-1))/2
    mu = np.random.randint(0, num_blobs,(num_samples, 2))*blob_distance - mean
    return np.random.randn(num_samples,2).dot(mod_matix) + mu
  





