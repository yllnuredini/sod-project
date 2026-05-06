This project implements an End-to-End system for salient object segmentation using an Encoder-Decoder (CNN) architecture built from scratch in PyTorch. The model features Conv2D and ConvTranspose2D layers, optimized with Batch Normalization, Dropout, and Early Stopping (at Epoch 18). Training was conducted on the ECSSD dataset using a BCE + IoU loss function, achieving a Validation Loss of 0.7561. The structure includes dedicated scripts for training (train.py), evaluation (evaluate.py), and a final notebook with a Gradio demo. To use it, clone the repository and install the dependencies:

git clone https://github.com/yllnuredini/sod-project.git
pip install torch torchvision gradio numpy matplotlib
