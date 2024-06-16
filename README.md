# Project-2: Disease Diagnosis

This repository contains code for a machine learning project focused on binary classification of skin lesions to distinguish between cancerous and non-cancerous cases.

# Project Overview

This project aims to develop a machine learning model to classify skin lesions as either cancerous or non-cancerous based on image data. Early detection of skin cancer can significantly improve prognosis, making this project valuable for healthcare applications.

# Steps invloved

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
