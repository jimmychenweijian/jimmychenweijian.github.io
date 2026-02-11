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

#step-1-dataset-understanding--selection

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

[Jump to Dataset Understanding & Selection](#step-1-dataset-understanding--selection)

#### 1. Dataset Overview

The dataset used in this project consists of Amazon customer reviews collected from a public source. It contains textual review content together with rating information, which allows us to analyse customer sentiment.

From review.shape, the dataset contains:
- 568,454 rows
- 10 columns
This large dataset size provides sufficient coverage to analyse overall sentiment distribution and customer behaviour patterns.

<img width="763" height="113" alt="image" src="https://github.com/user-attachments/assets/c3fdd096-276b-4eda-9fa4-761f7a93004b" />

#### 2. Key Variables

The dataset includes the following important variables:
- Score – Rating from 1 to 5
- Text – Full review text
- Summary – Short review summary
- ProductId – Product identifier
- UserId – Reviewer identifier
- HelpfulnessNumerator & HelpfulnessDenominator – Helpfulness votes
- Time – Timestamp of review

For sentiment analysis, the most relevant features are:
- Score (used to derive sentiment label)
- Text (main input for sentiment modelling)
Columns such as Id, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, and Time do not directly contribute to identifying sentiment and were later removed during preprocessing.

<img width="334" height="34" alt="image" src="https://github.com/user-attachments/assets/5573997f-cd8c-4ea0-87c4-a0cef20bdf59" />

#### 3. Data Types and Structure

Using review.info():
- 5 numeric columns (int64)
- 5 object columns (text-based)
- Memory usage: ~43.4 MB
The dataset structure is clean and well-defined, suitable for text analysis.

<img width="242" height="154" alt="image" src="https://github.com/user-attachments/assets/7868e286-3a35-4cd9-9a7c-769b28ceb415" />

#### 4. Missing Values

From the missing value check:
- Summary: 27 missing
- ProfileName: 26 missing
- All other columns: 0 missing
Since the number of missing values is very small relative to 568k rows, they do not significantly affect analysis.

<img width="141" height="199" alt="image" src="https://github.com/user-attachments/assets/0f103e85-4d4f-4e69-8223-68ceb5adc072" />

#### 5. Duplicate Review Analysis

Using duplicate check on:
subset = ["ProductId", "UserId", "Text"]

We identified:
- 2,122 duplicated records
- These duplicates represent repeated submissions of the same review and may bias sentiment distribution.
- Duplicates were removed during data cleaning.
  
<img width="917" height="112" alt="image" src="https://github.com/user-attachments/assets/8bab9fc7-a53b-41dc-a69e-4c99918e60ac" />

### Step 2: Data Preprocessing (Data Cleaning & Transformation)

To ensure high-quality data for modelling, the following preprocessing steps were performed:

#### 1. Remove Duplicate Reviews

Duplicates were removed using:

review.drop_duplicates(subset=["ProductId", "UserId", "Text"])
- Removed duplicates: 1,309
- Final dataset size: 567,145 rows
This ensures that repeated opinions do not bias the sentiment model.

<img width="267" height="234" alt="image" src="https://github.com/user-attachments/assets/5eebebef-e3dc-43b1-9033-e580b141dfed" />

#### 2. Convert Rating Score to Sentiment Label

We transformed Score into categorical sentiment:
- Score ≥ 4 → Positive
- Score = 3 → Neutral
- Score ≤ 2 → Negative
This transformation aligns numeric rating with sentiment categories, making it suitable for classification modelling.

<img width="836" height="106" alt="image" src="https://github.com/user-attachments/assets/bcb475b9-7a18-4886-adb1-20ee2eb21500" />

#### 3. Lowercase Text Standardisation

All review text was converted to lowercase:

review["Text"] = review["Text"].str.lower()

This ensures consistency and prevents the same word (e.g., "Good" vs "good") from being treated as different tokens during feature extraction.

#### 4. Remove Non-Sentiment Related Features

##### Columns removed:
- Id
- UserId
- ProfileName
- HelpfulnessNumerator
- HelpfulnessDenominator
- Time

##### Remaining columns:
- ProductId
- Score
- Summary
- Text
- Sentiment
- review_length
This reduces noise and ensures only relevant features are retained for modelling.

<img width="395" height="94" alt="image" src="https://github.com/user-attachments/assets/96f0a34f-f966-4850-bdaa-9596a164a722" />

### Step 3: Exploratory Data Analysis (EDA)

#### 1. Rating Distribution (1–5)

From value_counts:
- Score 5: ~63.9%
- Score 4: ~14.2%
- Score 1: ~9.2%
- Score 3: ~7.5%
- Score 2: ~5.2%
This shows a strong skew towards higher ratings.

<img width="106" height="139" alt="image" src="https://github.com/user-attachments/assets/936d8208-f2e3-4288-89e3-f1b134dbd7e9" />

#### 2. Sentiment Distribution

After converting to sentiment:
- Positive: ~78%
- Negative: ~14%
- Neutral: ~7%
This indicates clear class imbalance, with positive reviews dominating the dataset.

Implication:
The model may become biased towards predicting positive sentiment due to majority class dominance.

<img width="358" height="303" alt="image" src="https://github.com/user-attachments/assets/29712ca6-0197-4c50-8059-d7823b5fd205" />

#### 3. Review Length Analysis

We created a new feature:

review_length = number of words in Text

Summary statistics:
- Mean: ~80 words
- Median: 56 words
- Max: 3432 words
Most reviews fall within 0–500 words

This shows that:
- The dataset is dominated by short to medium-length reviews.
- A small number of extremely long reviews exist (outliers).

 <img width="126" height="171" alt="image" src="https://github.com/user-attachments/assets/22f3ebd5-920f-411a-b83b-8e18d3dd4147" />
 <img width="601" height="455" alt="image" src="https://github.com/user-attachments/assets/f087b2e6-10ef-4d27-b2e4-f304b8cbb3c2" />


#### 4. Review Length by Sentiment

From boxplot:

- Review lengths are broadly similar across Positive, Neutral, and Negative categories.
- All categories contain long outliers.
- This suggests that sentiment is not determined purely by review length.

<img width="596" height="445" alt="image" src="https://github.com/user-attachments/assets/826f868c-a96e-4159-bea3-911b0c8b53a7" />

#### 5. Top Reviewed Products

The top 10 products have significantly higher review counts than others.

This indicates:
- Review distribution across products is uneven.
- Popular products may influence overall sentiment distribution more heavily.

<img width="571" height="538" alt="image" src="https://github.com/user-attachments/assets/74867662-7c5d-4b72-aeef-b471c3d64e2c" />


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
