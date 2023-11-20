"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.
Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.
Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------
visualization_metrics.py
Note: Use PCA or tSNE for generated and original data visualization
"""

# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from common_plot import *
   
def visualization(data_list, analysis,labels):
  """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data; shape: (trial_number,time_steps,channel)
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """  
  # Analysis sample size (for faster computation)
  anal_sample_no = min([len(i) for i in data_list])
  #idx = np.random.permutation(len(ori_data))[:anal_sample_no]
  idx = np.random.permutation(anal_sample_no)
  data_list=[i[idx] for i in data_list]

  no, seq_len, dim = data_list[0].shape

  prep_data=[]
  for j in range(len(data_list)):
    for i in range(anal_sample_no):
      if (i == 0):
        prep_data_tmp = np.reshape(np.mean(data_list[j][0, :, :], 1), [1, seq_len])

      else:
        prep_data_tmp=np.concatenate((prep_data_tmp,
                                    np.reshape(np.mean(data_list[j][i,:,:],1), [1,seq_len])))

    prep_data.append(prep_data_tmp)
  # Visualization parameter        
  colors = ["tab:blue" for i in range(anal_sample_no)] + ["tab:orange" for i in range(anal_sample_no)]    

  if analysis == 'PCA':
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(prep_data[0])
    pca_results=[pca.transform(i) for i in prep_data]
    
    # Plotting
    f, ax = plt.subplots(1)
    for i in range(len(data_list)):
      plt.scatter(pca_results[i][:,0], pca_results[i][:,1],
                c = tab_colors_names[i], alpha = 0.2) # , label = "Original"

  
    ax.legend()
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.show()
    
  elif analysis == 'tSNE':
    
    # Do t-SNE Analysis together       
    prep_data_final = np.concatenate(np.asarray(prep_data,), axis = 0)
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
      
    # Plotting
    f, ax = plt.subplots(1)

    for i in range(len(data_list)):
      plt.scatter(tsne_results[anal_sample_no*i:anal_sample_no*(i+1),0], tsne_results[anal_sample_no*i:anal_sample_no*(i+1),1],
                c = tab_colors_names[i], alpha = 0.2, label = labels[i]) #
  
    ax.legend()
      
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.show()
  return f





