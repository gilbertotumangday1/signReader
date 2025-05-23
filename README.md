# ASL Project

## Introduction

This project allows users to use their (right) hand to sign letters in the alphabet through ASL. I plan to add typing or audio functionality soon, however for now it is mainly just translating ASL to alphabetic on the screen. Mediapipe was used for hand reading, and the model was trained using Kaggle’s ASL Alphabet database, by Akash Nagaraj:  

**“ASL Alphabet,” Kaggle, Apr. 22, 2018**  
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

---

## Instructions

The following libraries are required:

- mediapipe  
- cv2  
- os  
- csv  
- tqdm  
- torch  
- scikit-learn  
- pandas  
- numpy  
- pickle  

Once you have installed all libraries, you can either choose to simply run `main.py` (which uses a model with pretrained weights and biases, as well as a label encoder), or you can choose to train the model on your own dataset. For the latter, follow the instructions below:

---

### 1. Select a dataset

For this project I used **ASL Alphabet from Kaggle**:  
https://www.kaggle.com/datasets/grassknoted/asl-alphabet

> Note: The format of the dataset must be the same as the ASL Alphabet dataset, which means the folder of the dataset will contain a folder for every letter within it, with each letter folder containing images of the letter being signed.

---

### 2. Put your dataset folder into the root directory

Open `traincsvgen.py` and change:

```python
dataset_path = "asl_alphabet_test"
```
to the path to your dataset.

### 3. run `traincsvgen.py` 

Once it is completed there should be a csv file in the dataProcessing folder. Move this file into the training folder

### 4. Open modeltrainer.py inside the training folder and run it. 

There should be two new files resulting called asl_mlp_modell.pth and label_encoder.pkl. Move these files into resources to replace the old ones.

### 5. run `main.py`


## For reference
To find the list of symbols used in the dataset go to https://www.kaggle.com/datasets/grassknoted/asl-alphabet and open the testing folder to see examples of the sign letters used. The images were all right handed so the model works better if you use your right hand. 

