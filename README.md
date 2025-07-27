# **Financial Sentiment Analysis**
<br>

## **Overview**

This repository contains an advanced implementation for fine-tuning a BERT-based model to perform sentiment analysis on financial texts. The model classifies inputs into **positive**, **neutral**, and **negative** sentiment classes. Key features include robust handling of class imbalance, a discriminative fine-tuning strategy, and data augmentation techniques specifically tailored for financial language. The project also aims to extend support for detecting implicit numerical comparisons, a critical challenge in financial NLP.

<br>

## **Dataset**

The dataset is sourced from [Kaggle's Financial Sentiment Analysis Dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis), which provides labeled sentences annotated with sentiment polarity. Preprocessing involves mapping sentiment strings to integer labels and stratified splitting into training, validation, and test sets.

<br>

## **Implementation**

### **1. Data Preparation**

* **Tokenization**: Sentences are tokenized using `BertTokenizer` from the Hugging Face Transformers library (`bert-base-uncased`).
* **Augmentation**: Synonym replacement is applied using `nlpaug` with WordNet, introducing lexical diversity to improve generalization. Approximately 20% of training samples are augmented.
* **Tensor Conversion**: Data is converted to PyTorch tensors and wrapped in `DataLoader` for efficient mini-batching.

### **2. Model Architecture**

* **Backbone**: The base model is `BertForSequenceClassification` with 3 output labels.
* **Device Support**: Training utilizes GPU acceleration when available.

### **3. Training Strategy**

#### Custom Trainer

A subclassed `Trainer` named `WeightedTrainer` is used to override the loss computation with a **custom Focal Loss** function:

* Focal Loss helps concentrate learning on hard-to-classify samples.
* Integrated **class weights** mitigate the effects of class imbalance.

#### Layer-wise Discriminative Fine-Tuning

* Different learning rates are assigned per BERT encoder layer using exponential decay from output to input layers.
* This preserves general knowledge in earlier layers while allowing deeper adaptation in later ones.

#### Gradual Unfreezing

* Initially, all BERT layers are frozen except the classification head.
* One encoder layer is unfrozen at the start of each epoch, allowing for stable, gradual learning.

#### Optimization and Scheduling

* **Optimizer**: `AdamW` with weight decay.
* **Learning Rate Scheduler**: Linear warmup followed by linear decay using `get_linear_schedule_with_warmup`.

#### Early Stopping

* Training is halted if validation performance stops improving for two consecutive epochs.
<br>

### **4. Training Configuration**

* **Epochs**: 10
* **Batch Size**: 32
* **Initial Learning Rate**: 5e-5
* **Evaluation Strategy**: Per epoch
* **Model Checkpointing**: Best model based on validation accuracy is retained
<br>

### **5. Evaluation Metrics**

* **Accuracy**: Measures overall correctness.
* **Weighted F1-Score**: Balances precision and recall across imbalanced classes.

## **Model Saving and Inference**

* The final model is saved using `model.save_pretrained()` and can be reloaded for inference.
* A Hugging Face `pipeline` is used for quick sentiment predictions.
<br>

## **Future Work**

* **Numerical Sentiment Interpretation**: Incorporate mechanisms to interpret sentiment from numerical data (e.g., revenue up from \$10M to \$15M) without explicit sentiment markers.
