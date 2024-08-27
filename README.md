# Identifying-Sexist-Memes-in-English-and-Spanish-Digital-Culture

# Project Overview
Internet memes have become a significant form of online communication, often perpetuating negative stereotypes and subtle or overt forms of sexism. This project aims to enhance digital safety by employing both advanced multimodal deep learning and traditional machine learning strategies to identify sexist content in memes across English and Spanish digital cultures.

The approach integrates multimodal learning, which leverages both textual and visual data, and machine learning models for robust content classification. This project addresses the limitations of unimodal detection methodologies by considering ethical implications, bias mitigation, sentiment analysis, and cultural context.

# Features
Multimodal Deep Learning: Combines image and text processing using advanced models like BERT, GPT-2, CLIP, and Swin Transformer.
Traditional Machine Learning Models: Includes various classifiers such as Logistic Regression, Random Forest, SVM, and more for meme classification.
Multilingual Support: Supports both English and Spanish meme classification.
Bias Mitigation: Actively evaluates and mitigates bias in model decisions.
Sentiment & Emotion Analysis: Considers sentiment trends to better assess classification justifications.
Contextual Understanding: Leverages both local and global contexts for improved hazardous content identification.

# Installation
1. Clone the Repository
git clone https://github.com/yourusername/identifying-sexist-memes.git
cd identifying-sexist-memes

2. Install the Required Dependencies
pip install -r requirements.txt
Ensure that your environment has the necessary packages such as torch, transformers, sklearn, and others required by the deep learning and machine learning modules.

3. Dataset Preparation
Prepare a dataset of English and Spanish memes and their corresponding annotations. Organize the dataset into a folder structure for easy access by both the deep learning and machine learning modules.

# Usage
Multimodal Deep Learning
All deep learning-related tasks are performed in the Multimodal_Deep_Learning.ipynb notebook.
Open the Notebook: Open Multimodal_Deep_Learning.ipynb in Jupyter Notebook or JupyterLab.
Run the Notebook: Execute the cells to train and validate the multimodal deep learning model.
Multimodal Machine Learning
All machine learning-related tasks are performed in the Multimodal_Machine_Learning.ipynb notebook.

Open the Notebook: Open Multimodal_Machine_Learning.ipynb in Jupyter Notebook or JupyterLab.
Run the Notebook: Execute the cells to extract features and train various machine learning models.
Prediction
For prediction, follow the respective sections in each notebook (Multimodal_Deep_Learning.ipynb and Multimodal_Machine_Learning.ipynb) to generate predictions for new memes.

# Model Architecture

# Machine Learning Models
The machine learning models are applied to the extracted CLIP image features. The following models are trained and evaluated:
Logistic Regression
Random Forest
SVM
Decision Tree
XGBoost
AdaBoost
SGD Classifier
MLP
CatBoost

# Deep Learning Architecture
The deep learning architecture consists of multiple components:
BERT (Multilingual): For encoding textual content.
GPT-2: For enhanced text contextualization.
CLIP: For aligning text and image features.
Swin Transformer: For robust image classification.

# Ablation Study
These code files are experimentations of hybrid approaches consisting of:
1. GPT2, CLIP, SWIN
2. BERT, CLIP, SWIN
3. BERT, GPT2, SWIN

# Contributing
We welcome contributions from the community. Please fork the repository and submit a pull request.

# Contact Information
Name: Umera Pasha
Email: umerapasha.786@gmail.com
GitHub: @umerapasha
