# Sentiment Analysis with Logistic Regression and XAI (SHAP)

This project implements sentiment analysis using a Logistic Regression model. The project incorporates Explainable AI (XAI) to interpret the model’s predictions using SHAP (SHapley Additive exPlanations). Additionally, the model uses TF-IDF (Term Frequency-Inverse Document Frequency) and n-grams for text feature extraction to improve prediction accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data](#data)
- [Usage](#usage)
- [Model Training](#model-training)
- [Explainability with SHAP](#explainability-with-shap)


## Introduction

This project demonstrates a sentiment analysis pipeline that classifies text data into positive, negative, or neutral sentiment categories. The key components of this project are:

1. **Logistic Regression**: A simple yet effective classification algorithm used to predict sentiment.
2. **TF-IDF & N-grams**: Used to convert raw text data into numerical features for machine learning models.
3. **XAI with SHAP**: SHAP values provide a transparent interpretation of how the model makes decisions, showing the impact of individual words on the sentiment classification.

## Prerequisites

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `nltk`
  - `matplotlib`
  - `shap`
  - `seaborn`

## Installation
pip install -r requirements.txt

sentiment-analysis-logistic-regression/
│
├── data/                # Raw data and preprocessed data
│   ├── amzz.csv   # Training dataset
│   └
│
├── notebooks/            # Jupyter notebooks for exploration
│   ├── 
│   └── logistic_regression.ipynb
│

├── requirements.txt     # List of required Python packages
└── README.md            # This README file

Model Training:

The Logistic Regression model is trained using the logistic_regression file. It uses the preprocessed data from the previous step to fit the model.
To train the model:

from src.train_model import train_logistic_regression
model, vectorizer = train_logistic_regression(data)

Explainability with SHAP:

SHAP values are computed to explain the model’s predictions and provide insights into how individual words affect sentiment predictions.
To generate SHAP explanations:


from src.shap_explain import explain_model_with_shap
explain_model_with_shap(model, vectorizer, 'amzz.csv')


