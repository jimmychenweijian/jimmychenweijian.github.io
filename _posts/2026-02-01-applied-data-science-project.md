---
layout: post
author: Chen Weijian Jimmy  
title: "Applied Data Science Project Documentation"
categories: ITD214
---
## Project Background
### 1. Business Background

Amazon is one of the largest e-commerce platforms and it collects a huge amount of customer review data every day. These reviews contain both structured information (such as rating score from 1 to 5) and unstructured information (text feedback written by customers).

Although rating scores can show whether customers are generally satisfied or not, the company may not fully understand the overall satisfaction trend or the reasons behind positive and negative reviews. Simply looking at raw data is not enough to generate meaningful business insights.

Therefore, this project aims to use data analytics and text mining techniques to better understand customer satisfaction patterns from Amazon reviews.

### 2. Business Problem Statement

Amazon needs to better leverage its customer review data to answer important business questions, such as:

- What is the overall level of customer satisfaction?
- Are most customers satisfied or dissatisfied?
- What kind of feedback is commonly given in positive vs negative reviews?
- How can review data help improve decision-making?
- How much the customers’ (segments) are spending?

Without systematic analysis, the business may overlook important dissatisfaction signals or fail to identify improvement opportunities. This may affect customer experience and long-term revenue performance.

### 3. Business Goal

The main goal of this project is:

*To understand the overall customer satisfaction level on Amazon by analysing the distribution of review ratings and examining textual feedback.*

By doing so, the business can obtain a clearer picture of how customers perceive products sold on the platform.

### 4. Individual Objective (Objective 2 – Overall Customer Satisfaction Distribution)

For my individual component, I focus on:

#### Objective 2 – To analyse the overall sentiment distribution of Amazon customer reviews.

In this project, customer sentiment is derived directly from rating scores:

- Ratings 4–5 → Positive
- Rating 3 → Neutral
- Ratings 1–2 → Negative

This approach is practical because rating scores serve as a direct proxy for customer satisfaction.

The analysis will:

- Examine the distribution of ratings
- Categorise reviews into sentiment classes
- Identify whether the dataset is imbalanced
- Provide insights into overall satisfaction levels

### 5. Project Plan

To achieve the objective, the project is carried out in several structured steps:

#### Step 1: Dataset Understanding

- Load and explore the Amazon review dataset
- Understand key variables (Score, Text, ProductID, etc.)
- Check dataset size and data quality

#### Step 2: Data Preprocessing

- Clean and prepare textual data
- Convert rating scores into sentiment labels
- Handle missing values
- Apply tokenisation and stopword removal

#### Step 3: Exploratory Data Analysis (EDA)

- Visualise rating distribution (1–5)
- Analyse sentiment distribution
- Identify class imbalance
- Generate summary statistics

#### Step 4: Text Analytics

- Apply TF-IDF feature extraction
- Identify common keywords in positive and negative reviews
- Interpret word patterns

#### Step 5: Model Development (Evaluation Purpose)

Although sentiment is directly derived from rating scores, classification models such as:
- Logistic Regression
- Naive Bayes
are implemented to evaluate whether textual features align with rating-based sentiment.

#### Step 6: Model Assessment

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The performance of Logistic Regression and Naive Bayes is compared to determine which model performs better.

### 6. Expected Outcome

This project is expected to:

- Provide a clear view of overall customer satisfaction levels
- Identify patterns in review sentiments
- Demonstrate how text analytics can support business understanding

Support data-driven decision making

## Work Accomplished
Document your work done to accomplish the outcome

### Step 1: Dataset Understanding & Selection

#### 1. Dataset Overview

The dataset used in this project consists of Amazon customer reviews collected from a public source. It contains textual review content together with rating information, which allows us to analyse customer sentiment.

From review.shape, the dataset contains:

- 568,454 rows
- 10 columns

This large dataset size provides sufficient coverage to analyse overall sentiment distribution and customer behaviour patterns.

<img width="763" height="113" alt="image" src="https://github.com/user-attachments/assets/c3fdd096-276b-4eda-9fa4-761f7a93004b" />

### Modelling
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

### Evaluation
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Recommendation and Analysis
Explain the analysis and recommendations

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. 
