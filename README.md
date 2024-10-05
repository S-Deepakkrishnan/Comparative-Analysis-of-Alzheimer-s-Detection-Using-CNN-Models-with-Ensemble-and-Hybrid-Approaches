# Comparative Analysis of Alzheimer's Detection Using CNN Models with Ensemble and Hybrid Approaches

This project is a comprehensive exploration of deep learning models, specifically Convolutional Neural Networks (CNNs), to detect various stages of Alzheimer's disease. The goal of this work is to compare the performance of multiple CNN architectures, utilizing both ensemble and hybrid models, to determine the most effective approach for classifying Alzheimer's disease stages from medical imaging data.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Model Architectures](#model-architectures)
4. [Training Strategy](#training-strategy)
5. [Model Evaluation](#model-evaluation)
6. [Results](#results)
7. [Requirements](#requirements)
8. [Usage](#usage)
9. [Conclusion](#conclusion)
10. [References](#references)

## Project Overview
This project focuses on the **comparative analysis** of various CNN models for Alzheimer's disease detection using image data. It employs six different models: 
- **DenseNet**
- **VGG19**
- **EfficientNetB7**
- **Custom CNN model**
- **ResNet**
- **InceptionV3**

Two approaches were used to improve model performance:
1. **Ensemble Model**: Combining DenseNet, VGG19, EfficientNetB7, custom CNN, and ResNet.
2. **Hybrid Model**: A hybrid of ResNet and InceptionV3.

Both ensemble and hybrid models were compared to find the optimal approach in terms of accuracy and class-wise performance.

## Dataset Description
The dataset used in this project contains labeled images for various stages of Alzheimer's disease:
- **Final AD JPEG**: Alzheimer's Disease
- **Final CN JPEG**: Cognitively Normal
- **Final EMCI JPEG**: Early Mild Cognitive Impairment
- **Final LMCI JPEG**: Late Mild Cognitive Impairment
- **Final MCI JPEG**: Mild Cognitive Impairment

The images were processed and split into training, validation, and test sets for model evaluation.

## Model Architectures
### 1. Pre-trained Models
The following models were initialized with pre-trained weights (`h5 files`) and fine-tuned for the task:
- **DenseNet**
- **VGG19**
- **EfficientNetB7**
- **Custom CNN model**

### 2. Scratch Trained Models
The following models were trained from scratch using regular epoch methods:
- **ResNet**
- **InceptionV3**

### 3. Ensemble Model
The ensemble model combined the predictions of:
- **DenseNet**, **VGG19**, **EfficientNetB7**, **Custom CNN**, and **ResNet**  
The final prediction was made using a voting mechanism that aggregates the outputs of these models.

### 4. Hybrid Model
The hybrid model integrated:
- **ResNet** and **InceptionV3**, with shared layers and merged feature maps to create a hybrid output for classification.

## Training Strategy
The models were trained using a variety of techniques, including:
- **Data Augmentation**: Rotations, flips, and scaling to make the models more robust.
- **Early Stopping**: To prevent overfitting, training stopped once the validation accuracy plateaued.
- **Learning Rate Scheduling**: Reduced learning rate on plateau to fine-tune the models during training.
- **Checkpoints**: Saved the best model weights during the training process.

Training was performed over 25 epochs with a validation split of 0.2. Each model was evaluated based on its accuracy, loss, and classification performance across the classes.

## Model Evaluation
Each model's performance was evaluated using:
- **Confusion Matrix**: Visualized the classification performance for each class.
- **Accuracy and Loss Graphs**: Tracked model accuracy and loss over epochs to assess convergence.
- **Class-wise Bar Graphs**: Compared how well each model classified the five classes (`AD`, `CN`, `EMCI`, `LMCI`, `MCI`).

## Results
The following results were recorded for each model:
- **Confusion Matrix**: Displayed class-wise accuracy and misclassification.
- **Accuracy/Loss Graphs**: Provided insight into the model's learning behavior.
- **Class-wise Bar Graphs**: Illustrated how each model performed across different Alzheimer's disease stages.

### Comparison of Ensemble vs Hybrid Models
Both models were compared based on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
  
A final bar graph compared the ensemble model's performance with the hybrid model to highlight strengths and weaknesses.

## Requirements
To run this project, the following dependencies are required:
```bash
pip install tensorflow
pip install keras
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-learn
```

### Hardware Requirements
For efficient training, a system with the following specifications is recommended:
- **GPU**: NVIDIA RTX 3070 or higher
- **RAM**: 16 GB or more

## Usage
To run the models and reproduce the results:
1. Clone the repository:
   ```bash
   git clone https://github.com/S-Deepakkrishnan/Comparative-Analysis-of-Alzheimer-s-Detection-Using-CNN-Models-with-Ensemble-and-Hybrid-Approaches.git
    cd Comparative-Analysis-of-Alzheimer-s-Detection-Using-CNN-Models-with-Ensemble-and-Hybrid-Approaches

   ```
2. Set up the environment:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset in the appropriate directory:
   ```bash
   ./dataset/Final_AD_JPEG/
   ./dataset/Final_CN_JPEG/
   ./dataset/Final_EMCI_JPEG/
   ./dataset/Final_LMCI_JPEG/
   ./dataset/Final_MCI_JPEG/
   ```
4. Train and evaluate the models by running the scripts:
   ```bash
   python train_densenet.py
   python train_vgg19.py
   python train_resnet.py
   python train_inceptionv3.py
   python ensemble_model.py
   python hybrid_model.py
   ```

## Conclusion
This project presents a comparative analysis of CNN architectures for Alzheimer's detection using ensemble and hybrid techniques. The results show that combining multiple models can improve classification performance, especially for complex, multi-class medical imaging datasets. The hybrid approach also demonstrates how the strengths of different architectures can be leveraged to produce more robust predictions.

