Speaker_identification_project
==============================

This is a project for speaker identification using siamese neural network. Siamense neural network is a neural network that takes two inputs and outputs a similarity score. The network is trained to output a high similarity score for two inputs that are from the same class and a low similarity score for two inputs that are from different classes. Speaker embeddings are extracted from the network and used for speaker identification. Pytorch is used for the implementation of the network.

Requirements
------------

To install requirements create a virtual environment (preferably conda) and run the following command:

```pip install -r requirements.txt```

Dataset
------------

In this project VoxCeleb1 dataset is used. VoxCeleb1 is a large scale speaker identification dataset. It contains 1251 speakers and 148642 utterances. The dataset can be downloaded from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html).
