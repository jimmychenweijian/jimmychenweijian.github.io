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


[Step 1: Dataset Understanding & Selection](#step-1-dataset-understanding--selection)

- Load and explore the Amazon review dataset
- Understand key variables (Score, Text, ProductID, etc.)
- Check dataset size and data quality

[Step 2: Exploratory Data Analysis (EDA)](#step-2-exploratory-data-analysis-eda)

- Visualise rating distribution (1–5)
- Analyse sentiment distribution
- Identify class imbalance
- Generate summary statistics
- Exploratory Tokenization (Understanding Text Structure)

[Step 3: Data Preprocessing (Data Cleaning & Transformation)](#step-3-data-preprocessing-data-cleaning--transformation)

- Clean and prepare textual data
- Convert rating scores into sentiment labels
- Handle missing values
- Apply tokenisation and stopword removal
  
[Step 4: Text Analytics](#step-4-text-analytics)

- Identify common keywords in positive and negative reviews
- Interpret word patterns
- Visualising using word cloud
- Apply TF-IDF feature extraction

[Step 5: Model Development](#step-5-model-development)

- Select Modelling Technique
- Generate Test Design
- Although sentiment is directly derived from rating scores, classification models such as:
   - Logistic Regression + TF-IDF
   - Naive Bayes + TF-IDF
   - Naive Bayes + CountVectorizer
   - LSTM + Embedding


[Step 6: Model Assessment](#step-6-model-assessment)

- Models are evaluated using:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion Matrix
- Model Development and Comparison Rationale

[Step 7: Evaluation & Recommendation](#step-7-evaluation--recommendation)

- Evaluation & Recommendation
- Conclusion
- Future Challenges


## Work Accomplished

#### Step 1: Dataset Understanding & Selection

##### 1.1 Dataset Overview

The dataset used in this project consists of Amazon customer reviews collected from a public source. It contains textual review content together with rating information, which allows us to analyse customer sentiment.

From review.shape, the dataset contains:
- 568,454 rows
- 10 columns
This large dataset size provides sufficient coverage to analyse overall sentiment distribution and customer behaviour patterns.

<img width="763" height="113" alt="image" src="https://github.com/user-attachments/assets/c3fdd096-276b-4eda-9fa4-761f7a93004b" />

##### 1.2 Key Variables

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

##### 1.3 Data Types and Structure

Using review.info():
- 5 numeric columns (int64)
- 5 object columns (text-based)
- Memory usage: ~43.4 MB
The dataset structure is clean and well-defined, suitable for text analysis.

<img width="242" height="154" alt="image" src="https://github.com/user-attachments/assets/7868e286-3a35-4cd9-9a7c-769b28ceb415" />

##### 1.4 Missing Values

From the missing value check:
- Summary: 27 missing
- ProfileName: 26 missing
- All other columns: 0 missing
Since the number of missing values is very small relative to 568k rows, they do not significantly affect analysis.

<img width="141" height="199" alt="image" src="https://github.com/user-attachments/assets/0f103e85-4d4f-4e69-8223-68ceb5adc072" />

##### 1.5 Duplicate Review Analysis

Using duplicate check on:
subset = ["ProductId", "UserId", "Text"]

We identified:
- 2,122 duplicated records
- These duplicates represent repeated submissions of the same review and may bias sentiment distribution.
- Duplicates were removed during data cleaning.
  
<img width="917" height="112" alt="image" src="https://github.com/user-attachments/assets/8bab9fc7-a53b-41dc-a69e-4c99918e60ac" />


#### Step 2: Exploratory Data Analysis (EDA)

##### 2.1 Rating Distribution (1–5)

From value_counts:
- Score 5: ~63.9%
- Score 4: ~14.2%
- Score 1: ~9.2%
- Score 3: ~7.5%
- Score 2: ~5.2%
This shows a strong skew towards higher ratings.

<img width="106" height="139" alt="image" src="https://github.com/user-attachments/assets/936d8208-f2e3-4288-89e3-f1b134dbd7e9" />

##### 2.2 Sentiment Distribution

After converting to sentiment:
- Positive: ~78%
- Negative: ~14%
- Neutral: ~7%
This indicates clear class imbalance, with positive reviews dominating the dataset.

Implication:
The model may become biased towards predicting positive sentiment due to majority class dominance.

<img width="358" height="303" alt="image" src="https://github.com/user-attachments/assets/29712ca6-0197-4c50-8059-d7823b5fd205" />

##### 2.3 Review Length Analysis

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


##### 2.4 Review Length by Sentiment

From boxplot:

- Review lengths are broadly similar across Positive, Neutral, and Negative categories.
- All categories contain long outliers.
- This suggests that sentiment is not determined purely by review length.

<img width="596" height="445" alt="image" src="https://github.com/user-attachments/assets/826f868c-a96e-4159-bea3-911b0c8b53a7" />

##### 2.5 Top Reviewed Products

The top 10 products have significantly higher review counts than others.

This indicates:
- Review distribution across products is uneven.
- Popular products may influence overall sentiment distribution more heavily.

<img width="571" height="538" alt="image" src="https://github.com/user-attachments/assets/74867662-7c5d-4b72-aeef-b471c3d64e2c" />

#### Step 3: Data Preprocessing (Data Cleaning & Transformation)

To ensure high-quality data for modelling, the following preprocessing steps were performed:

##### 3.1 Remove Duplicate Reviews

Duplicates were removed using:

review.drop_duplicates(subset=["ProductId", "UserId", "Text"])
- Removed duplicates: 1,309
- Final dataset size: 567,145 rows
This ensures that repeated opinions do not bias the sentiment model.

<img width="267" height="234" alt="image" src="https://github.com/user-attachments/assets/5eebebef-e3dc-43b1-9033-e580b141dfed" />

##### 3.2 Convert Rating Score to Sentiment Label

We transformed Score into categorical sentiment:
- Score ≥ 4 → Positive
- Score = 3 → Neutral
- Score ≤ 2 → Negative
This transformation aligns numeric rating with sentiment categories, making it suitable for classification modelling.

<img width="836" height="106" alt="image" src="https://github.com/user-attachments/assets/bcb475b9-7a18-4886-adb1-20ee2eb21500" />

##### 3.3 Lowercase Text Standardisation

All review text was converted to lowercase:

review["Text"] = review["Text"].str.lower()

This ensures consistency and prevents the same word (e.g., "Good" vs "good") from being treated as different tokens during feature extraction.

##### 3.4 Remove Non-Sentiment Related Features

###### Columns removed:
- Id
- UserId
- ProfileName
- HelpfulnessNumerator
- HelpfulnessDenominator
- Time

###### Remaining columns:
- ProductId
- Score
- Summary
- Text
- Sentiment
- review_length
This reduces noise and ensures only relevant features are retained for modelling.

<img width="395" height="94" alt="image" src="https://github.com/user-attachments/assets/96f0a34f-f966-4850-bdaa-9596a164a722" />

##### 3.5 Text Preprocessing for Feature Engineering

The following preprocessing steps are applied to each review:

- Convert text to lowercase
- Remove punctuation and numbers
- Tokenize text into individual words
- Remove standard English stopwords
- Remove custom domain-specific stopwords (e.g. platform-specific or low-sentiment words)
- Apply stemming to reduce vocabulary size
This step standardises all customer reviews into a clean and consistent token format that can be used for TF-IDF feature extraction and downstream machine learning models.
Earlier tokenization and stopword removal steps were performed for exploratory analysis only.
Here, preprocessing is consolidated into a single pipeline to avoid duplication and ensure consistency.

<img width="526" height="280" alt="image" src="https://github.com/user-attachments/assets/d2906dc9-0ef3-43ad-8620-0ac72c515f5a" />

#### Step 4: Text Analytics
This section performs feature engineering and sentiment-oriented text exploration using cleaned review text. The objective is to extract meaningful word patterns and convert text into structured numerical features for modelling.

##### 4.1 Word Frequency Analysis (Exploratory – Before Cleaning)

Purpose: To understand raw vocabulary distribution and identify noise (e.g., stopwords).

- FreqDist(word_list)
- Plot frequency curve
The most frequent terms include “in”, “that”, and “with”, which are stopwords and do not contribute meaningful sentiment information. This confirms the necessity of further preprocessing.

<img width="945" height="55" alt="image" src="https://github.com/user-attachments/assets/a6509dbc-32d7-44a4-b0a0-6b7eb7539712" />

<img width="1010" height="491" alt="image" src="https://github.com/user-attachments/assets/225144f4-1b58-4dd9-b013-16949ca770c3" />

##### 4.2 Common Keywords in Positive vs Negative Reviews

To compare word patterns between sentiment classes.

Common words (e.g. “taste”, “product”) appear in both clouds, their context differs, showing that the same aspects can contribute to either positive or negative sentiment depending on customer experience.

<img width="591" height="151" alt="image" src="https://github.com/user-attachments/assets/c8cfb8fe-076a-47a3-8602-e811fd509cec" />

##### 4.3 Cleaned Word Frequency & Word Cloud (After Preprocessing)

To identify meaningful high-frequency sentiment words after stopword removal and stemming.

Custom stopwords were applied during the word filtering stage together with punctuation and number removal. This ensured that common and domain-specific words such as “br” and “amazon” were excluded before generating the frequency distribution and WordCloud.

<img width="944" height="504" alt="image" src="https://github.com/user-attachments/assets/d53bea92-db6c-4295-a510-10504cc840b0" />

##### 4.4 TF-IDF Feature Extraction

After cleaning and tokenising the text, each review is stored as a list of processed words in the text_tokens column. However, machine learning models cannot directly understand text. Therefore, we need to convert the cleaned tokens into numerical features.

In this step, TF-IDF (Term Frequency – Inverse Document Frequency) is applied to transform text into a weighted numerical representation. TF-IDF assigns higher weights to words that are important within a review, while reducing the weight of very common words that appear in many reviews and do not provide strong distinguishing power.

###### 4.4.1 Building the Dictionary and Corpus
First, a dictionary is created from all cleaned tokens. This dictionary maps each unique word to an integer ID.

To prevent memory issues and reduce noise (COLAB crashed when using full list), extreme words are filtered:

- Words appearing in fewer than 5 documents are removed (too rare).
- Words appearing in more than 80% of documents are removed (too common).
This helps reduce the vocabulary size and improves computational efficiency. From the output, the vocabulary size was reduced to 27,048 words, which is more manageable for modelling.

<img width="352" height="200" alt="image" src="https://github.com/user-attachments/assets/25022141-3939-474d-87c0-15e41dd8538a" />

###### 4.4.2 Training the TF-IDF Model

Next, the TF-IDF model is trained using the Bag-of-Words corpus. Each review is converted into a TF-IDF weighted vector.

The sparse matrix representation is used to avoid storing a large dense matrix in memory.

This means:
- 567,145 reviews (rows)
- 27,048 unique vocabulary terms (columns)

Each cell in the matrix represents the TF-IDF weight of a word in a particular review.

<img width="810" height="129" alt="image" src="https://github.com/user-attachments/assets/1d4b28a7-55dc-45eb-8c45-f473faec4632" />

### Step 5: Model Development

##### 5.1 Select Modelling Technique

In this project, two supervised machine learning models were selected for comparison:
- Logistic Regression
- Multinomial Naive Bayes

These models were chosen because:
- Both are widely used baseline models for text classification.
- Logistic Regression performs well with high-dimensional sparse features such as TF-IDF.
- Multinomial Naive Bayes is computationally efficient and commonly used for Bag-of-Words and Count-based features.
The objective is to evaluate which model better captures sentiment patterns from textual features.

#### 5.2 Generate Test Design

To ensure fair and consistent evaluation:
- Train/Test split: 80% training / 20% testing
- random_state = 42 for reproducibility
- stratify = y to preserve class distribution across splits

This ensures that:
- The dataset remains balanced between sentiment classes.
- Model comparison is fair.
- Results are reproducible.

#### 5.3 Build Model

##### Model 1: TF-IDF + Logistic Regression

- Feature Engineering: TF-IDF sparse matrix
- Classifier: Logistic Regression
- class_weight="balanced" to handle class imbalance
<img width="349" height="442" alt="image" src="https://github.com/user-attachments/assets/1df2c235-088b-48bb-8d7c-b6189396ca5f" />


##### Model 2: TF-IDF + Multinomial Naive Bayes

- Feature Engineering: TF-IDF sparse matrix
- Same train/test split
- Classifier: MultinomialNB
<img width="370" height="388" alt="image" src="https://github.com/user-attachments/assets/24d5149c-8f3b-4af4-a196-806263d62284" />


##### Model 3: CountVectorizer + Multinomial Naive Bayes

- Bag-of-Words using CountVectorizer
- Stop words removed (stop_words="english")
- Unigrams + Bigrams (ngram_range=(1,2))
- min_df=5 to remove rare words
- max_features=30000 to limit vocabulary size
<img width="417" height="502" alt="image" src="https://github.com/user-attachments/assets/b9dc39bd-cc7e-461e-9b24-8f3a7440d4db" />

##### Model 4: Embedding + LSTM

- Text converted into sequences using Keras Tokenizer
- Vocabulary size limited to max_words = 30000
- All sequences padded to fixed length (max_len = 200)
- Embedding layer (128 dimensions) to learn word representations
- LSTM (128 units) to capture sequential word patterns
- Dropout (0.3) used to reduce overfitting
- Output layer uses Softmax activation for 3-class classification
  
<img width="619" height="447" alt="image" src="https://github.com/user-attachments/assets/938ed4e3-2c8c-4452-b9a0-8200d9e5ec65" />

We selected four models to systematically compare the impact of different classifiers and feature engineering techniques on sentiment classification performance. Logistic Regression with TF-IDF was chosen as a strong baseline model because it performs well on sparse and high-dimensional text data. Naive Bayes was tested with both TF-IDF and CountVectorizer to evaluate whether the classifier performs better using weighted term importance or raw word frequency. In addition, an Embedding + LSTM model was implemented to capture sequential patterns and contextual relationships between words, which traditional bag-of-words models cannot fully represent. This structured comparison allows us to identify which model provides the most balanced and reliable performance across all sentiment classes, rather than focusing only on overall accuracy

### Step 6: Model Assessment

To evaluate the performance of the models, we compared:

- Accuracy
- Neutral Recall
- Positive Recall
- Negative Recall

Although accuracy gives an overall performance measure, it is not sufficient when the dataset is imbalanced. In our dataset, the Positive class has significantly more samples than Neutral and Negative. Therefore, a model can achieve high accuracy simply by predicting most reviews as Positive.

For this reason, we compare recall for each class, especially Neutral recall. Recall measures how many actual samples of a class are correctly identified. This is important because misclassifying Neutral reviews as Positive or Negative can distort the overall sentiment understanding.

#### 6.1 Model Development and Comparison Rationale

<img width="362" height="89" alt="image" src="https://github.com/user-attachments/assets/20230bf7-455a-42b9-be0e-dba0a4d22013" />

We first created TF-IDF + Logistic Regression as our base model. Logistic Regression is widely used for text classification and performs well on high-dimensional sparse data generated by TF-IDF. In our results, this model achieved an accuracy of 0.788 and, more importantly, produced the best Neutral recall (0.64). The performance across Negative, Neutral, and Positive classes was relatively balanced. The model handled class imbalance better than expected. Therefore, this model served as our benchmark for comparison.

Next, we implemented TF-IDF + Naive Bayes to compare how a probabilistic classifier performs using the same TF-IDF features. Although this model achieved a slightly higher accuracy (0.816), we discovered that the Neutral recall was extremely poor (0.01). This means the model almost failed to correctly identify Neutral reviews. The confusion matrix showed that many Neutral samples were misclassified as Positive. This indicates that Naive Bayes with TF-IDF may not handle ambiguous sentiment well and tends to bias toward the dominant Positive class.

Due to the very poor Neutral performance, we created a third model: CountVectorizer + Naive Bayes. The reason is that Multinomial Naive Bayes theoretically works better with raw word counts rather than TF-IDF weighted values. After switching to CountVectorizer, the Neutral recall improved significantly to 0.45, while accuracy remained similar (0.819). This confirms that Naive Bayes performs more naturally with count-based features. However, the model still shows bias toward the Positive class.

Finally, we implemented a deep learning model using Embedding + LSTM to determine whether sequential modelling can outperform traditional machine learning methods. Unlike bag-of-words approaches, LSTM captures word order and contextual relationships between words. This model achieved the highest overall accuracy (0.902) and strong performance for Positive and Negative classes. However, Neutral recall (0.31) remains relatively low compared to Logistic Regression. This suggests that Neutral sentiment is inherently more difficult to detect, possibly due to ambiguous wording and dataset imbalance toward Positive reviews.

##### Observation: All models struggle with Neutral Class

We can see:

- TF-IDF + Naive Bayes performs extremely poorly on Neutral (0.01).
- Even LSTM only achieves 0.31 Neutral recall.
- Logistic Regression performs the best for Neutral (0.64).

This likely happens because:
- Dataset imbalance – The Positive class dominates the dataset.
- Neutral reviews often contain mixed or less strong sentiment words.
- Many Neutral reviews may contain slight positive words, causing the model to classify them as Positive.

#### Step 7: Evaluation & Recommendation

Based on our analysis, we recommend:

Logistic Regression (TF-IDF) as the final selected model

Reason:
- It provides the most balanced performance across all three sentiment classes.
- It achieves the highest Neutral recall.
- It is computationally efficient and easier to interpret.
- It handles class imbalance better

Although LSTM achieves higher overall accuracy, it still struggles with Neutral detection and requires significantly more computational resources. Therefore, for practical business deployment where balanced sentiment classification is important, Logistic Regression is the more suitable choice.

#### 7.1 Conclusion

In this project, we compared four models: TF-IDF + Logistic Regression, TF-IDF + Naive Bayes, CountVectorizer + Naive Bayes, and Embedding + LSTM. Although the LSTM model achieved the highest overall accuracy, we decided to select TF-IDF + Logistic Regression as our final model for the Amazon sentiment analysis task.

The main reason is that Logistic Regression gives a more balanced performance across all three sentiment classes, especially for the Neutral class. Even though LSTM achieved higher accuracy, its Neutral recall was still relatively low. Since Amazon reviews include many mixed or moderate opinions, correctly identifying Neutral sentiment is important for fair analysis.

Another reason is interpretability. Logistic Regression is easier to explain because we can look at the feature coefficients and understand which words influence Positive or Negative predictions. In contrast, LSTM is considered more of a “black box” model because its internal learning process is not easily interpretable. For business use, having a model that we can explain clearly is very important.

It is also important to highlight that the Amazon dataset is highly imbalanced, with a strong dominance of Positive reviews. This imbalance affects model learning and contributes to lower Neutral detection performance across all models. Therefore, evaluation metrics beyond accuracy, such as recall for individual classes, are necessary to ensure fair comparison.

Overall, considering balance, interpretability, computational efficiency, and practical deployment, Logistic Regression is selected as the most suitable final model.


#### 7.2 Future Challenges

Although Logistic Regression performs well overall, there are still some limitations.

First, it does not consider word order. Since TF-IDF treats text as a bag-of-words, it ignores sequence information. For example, the phrase “not good” and “good” may contain similar important words, but their meanings are different. Logistic Regression may not fully capture this contextual difference.

Second, it may struggle with more complex language patterns such as sarcasm, irony, or subtle sentiment expressions. Deep learning models like LSTM can better model sequential relationships and context, so they may handle these cases more effectively.

Third, class imbalance remains a challenge. The dataset is heavily biased toward Positive reviews. Even though we used class_weight="balanced", predicting Neutral sentiment is still difficult. This suggests that more balanced training data or advanced resampling techniques may be needed in the future.

Fourth, real-world deployment would require continuous monitoring and retraining. If new vocabulary, slang, or trending phrases appear in Amazon reviews, the model performance may degrade over time. A mechanism for periodic retraining and performance evaluation would be necessary.

Lastly, improving dataset quality could significantly enhance results. Collecting more balanced samples, refining Neutral labels, and reducing ambiguous annotations would allow the model to learn clearer decision boundaries.


#### Step 8: AI Ethics Considerations
##### 1. Dataset Bias and Class Imbalance

One major ethical issue is data imbalance. The Amazon dataset contains significantly more Positive reviews compared to Neutral and Negative reviews. This creates a bias in the model because it learns patterns mainly from the dominant Positive class.

As a result:
- Models tend to predict Positive more frequently.
- Neutral reviews are often misclassified.
- Accuracy may appear high, but performance is unfair across classes.

If deployed in a real business environment, this bias could:
- Overestimate customer satisfaction.
- Underrepresent neutral or mildly dissatisfied customers.
- Lead to misleading business insights.

##### 2. Fairness Across Sentiment Classes

From our evaluation:
- All models struggle with predicting Neutral reviews.
- Naive Bayes (TF-IDF) almost completely failed to detect Neutral sentiment.
This raises fairness concerns. If certain sentiment categories are consistently misclassified, the system does not treat all classes equally. A fair AI system should provide balanced performance across all groups or categories.

##### 3. Misclassification Risk

Sentiment analysis models are not perfect. Misclassification may occur, especially for ambiguous or sarcastic reviews.

For example:
A sarcastic positive sentence may be misclassified.
A neutral statement with slightly positive words may be predicted as Positive.

If used for automated decision-making (e.g., filtering reviews or prioritising complaints), errors could:
- Misrepresent customer opinions.
- Ignore genuine dissatisfaction.
- Impact business decisions unfairly.

##### 4. Model Transparency and Explainability

Another ethical consideration is model transparency.
- Logistic Regression is interpretable. We can examine feature coefficients to understand which words contribute to predictions.
- LSTM is more of a “black box” model. It is difficult to interpret how the model makes decisions internally.

##### Conclusion on AI Ethics

- In conclusion, the main ethical challenges in this assignment are dataset bias, class imbalance, fairness across sentiment classes, and model transparency. The dominance of Positive reviews affects model learning and leads to lower Neutral detection. To address these issues, future improvements could include balancing the dataset, applying resampling techniques, and carefully monitoring model performance across all classes.

- Selecting Logistic Regression also supports ethical AI principles because it provides more balanced performance and better explainability compared to black-box deep learning models

## Source Codes and Datasets
 
Source Code: [7275454R.ipynb](https://github.com/user-attachments/files/25454483/7275454R.ipynb)

Dataset: https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews
