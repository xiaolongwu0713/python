pip install tsaug einops torch-summary cv2


# TTS-GAN: A Transformer-based Time-Series Generative Adversarial Network
---

This repository contains code from the paper "TTS-GAN: A Transformer-based Time-Series Generative Adversarial Network".

The paper has been accepted to publish in the 20th International Conference on Artificial Intelligence in Medicine (AIME 2022).

Please find the paper [here](https://arxiv.org/abs/2202.02691)

---

**Abstract:**
Time-series datasets used in machine learning applications often are small in size, making the training of deep neural network architectures ineffective. For time series, the suite of data augmentation tricks we can use to expand the size of the dataset is limited by the need to maintain the basic properties of the signal. Data generated by a Generative Adversarial Network (GAN) can be utilized as another data augmentation tool. RNN-based GANs suffer from the fact that they cannot effectively model long sequences of data points with irregular temporal relations. To tackle these problems, we introduce TTS-GAN, a transformer-based GAN which can successfully generate realistic synthetic time series data sequences of arbitrary length, similar to the original ones. Both the generator and discriminator networks of the GAN model are built using a pure transformer encoder architecture. We use visualizations to demonstrate the similarity of real and generated time series and a simple classification task that shows how we can use synthetically generated data to augment real data and improve classification accuracy.

---

**Key Idea:**

Transformer GAN generate synthetic time-series data

**The TTS-GAN Architecture** 

![The TTS-GAN Architecture](./images/TTS-GAN.png)

The TTS-GAN model architecture is shown in the upper figure. It contains two main parts, a generator, and a discriminator. Both of them are built based on the transformer encoder architecture. An encoder is a composition of two compound blocks. A multi-head self-attention module constructs the first block and the second block is a feed-forward MLP with GELU activation function. The normalization layer is applied before both of the two blocks and the dropout layer is added after each block. Both blocks employ residual connections. 


**The time series data processing step**

![The time series data processing step](./images/PositionalEncoding.png)

We view a time-series data sequence like an image with a height equal to 1. The number of time-steps is the width of an image, *W*. A time-series sequence can have a single channel or multiple channels, and those can be viewed as the number of channels (RGB) of an image, *C*. So an input sequence can be represented with the matrix of size *(Batch Size, C, 1, W)*. Then we choose a patch size *N* to divide a sequence into *W / N* patches. We then add a soft positional encoding value by the end of each patch, the positional value is learned during model training. Each patch will then have the data shape *(Batch Size, C, 1, (W/N) + 1)* This process is shown in the upper figure.

---

**Repository structures:**

> ./images

Several images of the TTS-GAN project


> ./pre-trained-models

Saved pre-trained GAN model checkpoints


> dataLoader.py

The UniMiB dataset dataLoader used for loading GAN model training/testing data


> LoadRealRunningJumping.py

Load real running and jumping data from UniMiB dataset


> LoadSyntheticRunningJumping.py

Load Synthetic running and jumping data from the pre-trained GAN models


> functions.py

The GAN model training and evaluation functions


> train_GAN.py

The major GAN model training file


> visualizationMetrics.py

The help functions to draw T-SNE and PCA plots


> adamw.py 

The adamw function file


> cfg.py

The parse function used for reading parameters to train_GAN.py file


> JumpingGAN_Train.py

Run this file to start training the Jumping GAN model


> RunningGAN_Train.py

Run this file to start training the Running GAN model


---

**Code Instructions:**


To train the Running data GAN model:
```
python RunningGAN_Train.py
```

To train the Jumping data GAN model:
```
python JumpingGAN_Train.py
```

A simple example of visualizing the similarity between the synthetic running&jumping data and the real running&jumping data:
```
Running&JumpingVisualization.ipynb
```
---