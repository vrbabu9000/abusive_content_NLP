# Abusive Content Detection & Classification
An advanced NLP project that checks for abuse in content. A strong contrast is drawn between baseline approaches and deep learning frameworks like BERT, LSTM.

## Introduction
The project basically classifies text input based on context as abusive or non abusive. Email Classification includes a lot of preprocessing mostly and the modelling and training part in this problem is relatively simple. Our primary goal is to make a more generalised text classifier that is trained on an email dataset which is hand-labelled as abusive/non-abusive. However, while going through the EDA, we realise the significant flaws within this dataset. But we have used advanced feature extraction techniques to squeeze the maximum out our dataset. Our deep learning model; however merges two dataset and balances them which puts it at a huge advantage compared to our baseline machine learning models. Finally we will discuss on our deployment of the project. 

## Business Objective: 
- Inappropriate emails would demotivates and spoil the positive environment that would lead to more attrition rate and low productivity and Inappropriate emails could be on form of bullying, racism, sexual favoritism and hate in the gender or culture, in today's world so dominated by email no organization is immune to these hate emails.
- The goal of the project is to identify such emails on the given day based on the above inappropriate content.

## Data Set Details:
- The dataset contains more than 2 lakh emails generated by employees of an organization.
- Data set details sent in csv file. 
- Highly Biased 
- Huge number of Duplicates

#### Team members:
- Ch V Subramaniyam
- Nikitha S
- Bommu Reddy
- Rakesh Potti
- Anoop Alexander
- Vignesh R Babu

## How to Navigate through this project:

#### Project Notebook
- The Project Notebook file clearly lays out our approach towards this problem.
- Every step in the project is well documented in the Notebook.
- All Baselines models and One deep Learning training was done within this dataset.
- Pytorch model was saved in bert folder and it's saved weights were imported to our notebook.

#### Streamlit deploymnet
- The project was deployed using streamlit and heroku.
- link: 
