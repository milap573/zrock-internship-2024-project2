# Project-2: Disease Diagnosis

This repository contains code for a machine learning project focused on binary classification of skin lesions to distinguish between cancerous and non-cancerous cases.

# Overview

This project aims to develop a machine learning model to classify skin lesions as either cancerous or non-cancerous based on image data. Early detection of skin cancer can significantly improve prognosis, making this project valuable for healthcare applications.

# Dataset
Download dataset used in this project from here: https://www.kaggle.com/datasets/kylegraupe/skin-cancer-binary-classification-dataset

# Steps involved

1. Data Collection: Gathered a diverse dataset of skin lesion images, categorizing them into cancerous and non-cancerous classes. The dataset includes training and validation sets for model development and testing.
2. Data Preprocessing: Applied preprocessing techniques such as resizing, normalization, and augmentation to the images. This step enhances model generalization and performance.
3. Model Development:
   i. Developed a convolutional neural network (CNN) using TensorFlow/Keras.
   ii. Utilized transfer learning with MobileNetV2 as the base model, pre-trained on ImageNet.
   iii. Fine-tuned specific layers of MobileNetV2 to adapt to the skin cancer classification task.
   
4. Training and Evaluation:
   i. Trained the model using the training dataset, monitoring key metrics such as accuracy and loss.
   ii. Validated the model using the validation dataset to assess performance and prevent overfitting.
   iii. Evaluated the model on separate test data to measure its effectiveness in real-world scenarios.
   
5. Results and Analysis:
   i. Analyzed model performance metrics, including accuracy, precision, recall, and F1-score.
   ii. Visualized results with graphs and charts to illustrate training progress and validation performance.
   iii. Discussed insights gained from the project and potential applications in clinical settings.

# Files Included

1. Z_Rock_ML_Internship_Project_2.ipynb: Jupyter notebook containing the entire project code and detailed explanations.
2. best_model.h5: Optimized skin cancer classification model saved during training, based on highest validation performance.
3. skin_cancer_classifier.h5: Final trained model for skin cancer detection from images.
4. requirements.txt: List of Python dependencies required to run this project.

# Installation
Use the following commands to clone the repository
1. git clone https://github.com/milap573/zrock-internship-2024-project2.git
2. cd zrock-internship-2024-project2

# Install dependencies
pip install -r requirements.txt
