# Salient Object Detection (SOD) using CNN

This project implements an End-to-End Salient Object Detection system using a Convolutional Neural Network built from scratch in PyTorch.
The model takes an input image and generates a saliency mask that highlights the most visually important object or region in the image.

Project Contents
The repository includes:

- Source code for data loading, model training, and evaluation
- CNN Encoder-Decoder model implementation
- Trained model checkpoint
- Evaluation results and visualizations
- Gradio demo app for testing custom images
- Project notebook

Project Structure

sod-project/
│
├── data_loader.py
├── sod_model.py
├── train.py
├── evaluate.py
├── app.py
├── checkpoints/
├── results/
├── data/ECSSD/
├── SOD_Project (1).ipynb
└── README.md

GitHub Repository
https://github.com/yllnuredini/sod-project

Run the Demo in Google Colab
To run the Gradio demo app in Google Colab, use these commands:
!git clone https://github.com/yllnuredini/sod-project.git
%cd sod-project
!pip install torch torchvision opencv-python matplotlib numpy scikit-learn tqdm gradio
!python app.py
