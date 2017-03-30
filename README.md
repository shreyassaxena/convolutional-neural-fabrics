## convolutional-neural-fabrics

### Introduction
In our work, we propose a “fabric” that embeds an exponentially large number of CNN architectures. 
The fabrics sidesteps the tedious process of specifying, training and testing individual networks in order to find good architectures. The fabric circumvents 8 out of 10 hyperparameters of the CNN architecture and has only 2 hyperparameters. 

Detailed description of the system is provided in our arXiv technical report: https://arxiv.org/abs/1606.02492 [To appear at NIPS16]. Collected resources are available at our [Project Page][1].
[1]: http://lear.inrialpes.fr/~verbeek/fabrics

### Citation

If you're using this code in a publication, please cite our paper.

    @InProceedings{saxena2016convolutional,
      title={Convolutional Neural Fabrics},
      author={Saxena, Shreyas and Verbeek, Jakob},
      BookTitle={NIPS},
      year={2016}
    }
    

### System Requirements

  0. This software is tested on Fedora release 21 (64bit).
  0. MATLAB (tested with 2013b on 64-bit Linux)
  0. Prerequisites for caffe (http://caffe.berkeleyvision.org/installation.html#prequequisites). 
   

### Getting started
  0. Code: Our caffe based implementation is a modified version of - https://github.com/HyeonwooNoh/caffe.git. You need to recompile caffe for other platforms.
  

  
### Training CNF

** Demo on MNIST
  0. Data: You need to download and post-process MNIST for the demo code (See ./MNIST/demo_mnist.txt for further instructions)
  0. Follow the steps in demo_mnist.txt
  
### PyTorch Implementation
[https://github.com/vabh/convolutional-neural-fabrics](https://github.com/vabh/convolutional-neural-fabrics)

 
