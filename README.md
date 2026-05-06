## Project Structure

```text
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
```

## GitHub Repository

https://github.com/yllnuredini/sod-project

## Run the Demo in Google Colab

To run the Gradio demo app in Google Colab, use the following commands:

```python
!git clone https://github.com/yllnuredini/sod-project.git
%cd sod-project
```

```python
!pip install torch torchvision opencv-python matplotlib numpy scikit-learn tqdm gradio
```

```python
!python app.py
```
