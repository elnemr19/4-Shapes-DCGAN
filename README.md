# GANs Project: Shape Generation

This project implements a Generative Adversarial Network (GAN) to generate grayscale images of geometric shapes (circle, square, star, and triangle). The dataset used is from Kaggle's Four Shapes Dataset.




## Table of Contents

1. [Project Overview](https://github.com/elnemr19/4-Shapes-DCGAN/tree/main?tab=readme-ov-file#project-overview)

2. [Dataset](https://github.com/elnemr19/4-Shapes-DCGAN/tree/main?tab=readme-ov-file#dataset)

3. [Model Architecture](https://github.com/elnemr19/4-Shapes-DCGAN/tree/main?tab=readme-ov-file#model-architecture)

4. [Loss Functions](https://github.com/elnemr19/4-Shapes-DCGAN/tree/main?tab=readme-ov-file#loss-functions)

5. [Training Details](https://github.com/elnemr19/4-Shapes-DCGAN/tree/main?tab=readme-ov-file#training-details)

6. [Results](https://github.com/elnemr19/4-Shapes-DCGAN/tree/main?tab=readme-ov-file#results)

## Project Overview

Generative Adversarial Networks (GANs) consist of two models:

1. **Generator:** Creates synthetic images to mimic the dataset.

2. **Discriminator:** Differentiates between real and synthetic images.

This project uses Wasserstein GAN with Gradient Penalty (WGAN-GP) to stabilize training and improve the quality of generated shapes.


## Dataset

The dataset contains grayscale images of four shapes:

* Circle

* Square

* Star

* Triangle

Each image is resized to a uniform size of 56x56 pixels and normalized to the range [-1, 1] for training.

**Preprocessing Steps:**

1. Images are read in grayscale using OpenCV.

2. Normalized using the formula: img = (img - 127.5) / 127.5.

3. Resized to the desired image dimensions.

## Model Architecture

**Generator**

The generator creates synthetic images from random noise. It uses:

* **Fully connected layers** to process input noise and reshape it into an initial feature map.

* **Transposed convolutions** for upsampling the feature map and increasing resolution.

* **Batch normalization** to stabilize training and improve convergence.

* **LeakyReLU activations** for non-linearity.

* **Tanh activation** for the output layer, scaling pixel values to the range [-1, 1].

**Discriminator**

The discriminator classifies images as real or fake. It uses:

* Convolutional layers for feature extraction.

* LeakyReLU activations.

* Dropout for regularization.

* Fully connected layer for binary output.

## Loss Functions

**Generator Loss**

Encourages the generator to produce images that the discriminator classifies as real:


**Discriminator Loss**

Penalizes incorrect classifications and incorporates gradient penalty:


**Gradient Penalty (GP)**

Enforces Lipschitz continuity to stabilize training:



## Training Details

* **Noise Dimension:** 100

* **Epochs:** 160

* **Batch Size:** 128

* **Optimizers:** Adam with learning rate 1e-4, beta_1=0.5, beta_2=0.9 .



Training loop includes:

1. Generating synthetic images.

2. Calculating discriminator and generator losses.

3. Backpropagating and updating weights.

## Results

Generated images show clear distinctions between shapes after training for 160 epochs. Below are some sample outputs from the generator:

![image](https://github.com/user-attachments/assets/8f7a0a19-d4ba-47c2-a0b3-727743e60c3f)

