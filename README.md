# **Financial Sentiment Analysis**

## **Overview**
This repository contains an ongoing project for fine-tuning a BERT-based model to perform sentiment analysis on financial data. The model is designed to classify financial texts into three sentiment categories: **positive**, **neutral**, and **negative**. In addition, the project aims to enhance the model to handle numerical comparisons without explicit descriptive comparators, a common challenge in financial text analysis.

## **Dataset**
The dataset used for the initial phase is sourced from [Kaggle's Financial Sentiment Analysis Dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis). It contains labeled financial sentences categorized into **positive**, **neutral**, and **negative** sentiments.

## **Model Training Workflow**

### **1. Data Preparation**
- The dataset is loaded and preprocessed using **BERT tokenizer**.
- **Synonym augmentation** is applied to expand the training set.
- Data is split into **train**, **validation**, and **test** sets.

### **2. Custom Training Loop**
- A custom training loop is implemented using Hugging Face's `Trainer`, with:
  - **Focal Loss** to focus on misclassified examples to address the class imbalance.
  - **Layer-wise learning rate decay** to stabilize fine-tuning of deeper BERT layers.
  - **Gradual unfreezing** to progressively unlock layers during training.

### **3. Optimizer and Scheduler**
- **AdamW optimizer** with weight decay.
- A **linear learning rate scheduler** with warmup steps to optimize learning stability.

### **4. Evaluation Metrics**
- **Accuracy** and **Weighted F1-score** are used to evaluate model performance on the test set.

## **Future Work**
- **Numerical Comparisons**: Improve the model to classify sentiments based on implicit numerical comparisons.
