import numpy as np
import math
from sklearn.datasets import make_moons
from scipy.stats import norm
from sklearn.utils import check_random_state


def create_4lines_dataset(n, p):
    # n=2000, p=20
    y=np.repeat(range(4), 500)
    px=np.linspace(0, 1, 500)
    nx = np.linspace(-1, 0, 500)
    y1=np.asarray([(-1)*x+1 for x in px])
    y2 = np.asarray([x - 1 for x in px])
    y3 = np.asarray([(-1)*x - 1 for x in nx])
    y4 = np.asarray([x + 1 for x in nx])

    xy1 = np.stack((px, y1),axis=1)
    xy2 = np.stack((px, y2), axis=1)
    xy3 = np.stack((nx, y3), axis=1)
    xy4 = np.stack((nx, y4), axis=1)

    X=np.concatenate((xy1,xy2,xy3,xy4),axis=0)
    generator = check_random_state(None)
    noise=0.1
    X += generator.normal(scale=noise, size=X.shape)

    # add z dimension
    z = np.repeat([1,2,3,4], 500)
    z=z[:,np.newaxis]
    noise_vector = norm.rvs(loc=0, scale=1, size=[2000, 20 - 3])
    data = np.concatenate([X, z, noise_vector], axis=1)

    # shuffle
    all = np.concatenate((data, y[:, np.newaxis]), axis=1)
    np.random.shuffle(all)
    data=all[:,:-1]
    y=all[:,-1]
    return data, y


# Create a simple dataset
def create_twomoon_dataset(n, p):
    relevant, y = make_moons(n_samples=n, shuffle=True, noise=0.1, random_state=None)
    print(y.shape)
    noise_vector = norm.rvs(loc=0, scale=1, size=[n,p-2])
    data = np.concatenate([relevant, noise_vector], axis=1)
    print(data.shape)
    return data, y


def create_sin_dataset(n, p):
    '''This dataset was added to provide an example of L1 norm reg failure for presentation.
    '''
    assert p == 2
    x1 = np.random.uniform(-math.pi, math.pi, n).reshape(n ,1)
    x2 = np.random.uniform(-math.pi, math.pi, n).reshape(n, 1)
    y = np.sin(x1)
    data = np.concatenate([x1, x2], axis=1)
    print("data.shape: {}".format(data.shape))
    return data, y
    
    

