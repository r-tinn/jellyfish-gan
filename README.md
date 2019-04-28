#jellyfish-gan

## Description

This project seeks to replicate the results of DC-GAN from 2012.
There are several different reasons for me undertaking this project:
- Firstly, I was keen to implement from scratch an academic paper that I enjoyed reading.
- I wanted to create my own toy dataset in a few simple steps, using minimal pre-processing.
- Additionally, it interests me to see how machine learning models perform on more messy real-world datasets to see if their performance holds up.
- This project gives me an excuse to learn a new macine learning framework (PyTorch) which I have little previous experience of.

## Getting started:

- Ensure python is installed (>=3.6.7)
- `pip install -r requirements.txt`
- `conda install lightgbm>=2.2.1`

## Dataset:

My quick and simple idea for creating a dataset was to sample the frames of a video of a specific collection of objects. If most of the video frames contain the objects of interest then 

Initially I was inspired by the many diver coral reef videos which are hours long of tropical fish. however then I found this excellent video of Jellyfish and decided this would be interesting. Could a GAN capture the messy tentacles and shape of Jellyfish.

## Pre-processing

sampled video to get ~ 10,000 datapoints.

<img src="./readme_images/scene_1.jpg" width="200">
<img src="./readme_images/scene_2.jpg" width="200">
<img src="./readme_images/scene_3.jpg" width="200">
<img src="./readme_images/scene_4.jpg" width="200">

Augment the dataset by applying rotation and reflections to each image, end up with 40,000 datapoints.

crop image 

Normalise each image for mean and standard deviation

<img src="./readme_images/conversion.png" width="300">

Sub-sample


## Results

<img src="./readme_images/real_jfish.png" width="100">

<img src="./readme_images/initial_samples.png" width="100">

<img src="./readme_images/gen_jfish.png" width="100">

## Conclusions

- Structure of image
- GANs scale very well with computational power and datasize. It is a challenge on small datasets with limited computational complexity.
- Interesting project for a few hours here and there :)