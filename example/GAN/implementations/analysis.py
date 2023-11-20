import numpy as np
from pre_all import *
import matplotlib.pyplot as plt
from gesture.config import *

gen_method='WGAN_GP' # converge/stable
sid=10
result_dir = top_root_dir + 'example/GAN/implementations/'+gen_method.lower()+'/'
filename=result_dir +'losses.npy'
losses = np.load(filename,allow_pickle='TRUE').item()
d_loss=losses['d_losses'] # list
g_loss=losses['g_losses']

fig, ax=plt.subplots()
ax.plot(d_loss)
ax.plot(g_loss)
ax.legend(['d_loss','g_loss'])