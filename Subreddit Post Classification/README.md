# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Subreddit posts web scraping & classification

The project can be broken down into 4 main sections
1) Data Collection and Exploration
2) Data Cleaning and Preparation
3) Data Visualisation
4) Model Fitting & Gridsearch Hyperparameter Tuning


### Introduction
The aim of the project was to select 2 subreddit topics and train a model capable of classifying and predicting which subreddit a given post should belong in. Our group chose the Oculus Rift (https://www.reddit.com/r/oculus/) and HTC Vive (https://www.reddit.com/r/Vive/) to train our model with.


## 1) Data Collection and Exploration
To gather the data required to train our model Requests, an Apache2 Licensed HTTP library, was used to request the JSON files of the subreddits and a function was used to repeatedly scrape the subreddit URLs in order to gather a dataset of 1000 posts per subreddit. These 2 JSON files were subsequently saved offline by writing them to a local directory using the Pandas library. This ensured that the dataset used to train the model would be secured and would not be accidentally overwritten by the function or lost upon kernel reset.

Exploring the dataset revealed that the there were 2 main bodies of text which could be helpful to train our model with, these were:
 - subreddit post 'title'
 - subreddit post 'self_text'.


these would later form the X value of the dataset.
The target, or y value was the subreddit topic, being a binary outcome of either 'Oculus' or 'Vive'.


## 2) Data Cleaning and Preparation
The locally saved JSON files were imported as a Pandas DataFrame to enable the data cleaning process to begin.

A combination of libraries were used to process and clean the data, these included BeautifulSoup to isolate the raw text, Regular Expression to remove all non alphabets, stopwords to remove common words which would be likely unhelpful with training the model, and lastly Snowball stemmer and WordNet lemmatizer to reduce the subreddit words to their base form.

The text for the title and body of each respective rows were then merged to form a single feature column named fin_text. This would now be ready to be processed by Tfidf Vectorizer, which would convert a collection of raw documents to a matrix of TF-IDF features.


## 3) Data Visualisation
Upon vectorization, the top 20 features of each subreddit DataFrame were plotted on a bar plot (using Matplotlib), generating a visual representation on the most frequently mentioned words for each DataFrame (Oculus and Vive). The Oculus and Vive datasets were then merged into a single DataFrame, and split with Train_Test_Split.


## 4) Model Fitting & Gridsearch Hyperparameter Tuning
3 classification models were trained with the data, these were Bernoulli, Logistic Regression and Random Forest.

These models were individually inserted into a pipeline with a Tfidf Vectorizer and GridsearchCV to be tuned for the optimal paramaters. The X and y training data was then fit into the pipeline to generate the y_hat or prediction data.


## Results & conclusion
The Logistic Regression model showed the best performance, with an accuracy of 88%, sensitivity of 93% and specificity of 83%. An ROC Curve was plotted and an area under curve of 0.88 was achieved by the Logistic Regression model. Random forest also displayed a good accuracy for predicting subreddit posts accurately, only barely falling short of the Logistic Regression model. However both these models did display a overfit with the training data set. While Bernoulli performed the most poor out of all 3 models in terms of prediction accuracy, it had the least over fit with the training data.

To conclude, Logistic Regression was the ideal classification model for this project, however limitations with computing power severly limited the finetuning of the all 3 models, and this factor could have resulted in their poorer performance. Further testing would have to be conducted to determine if Logistic Regression would indeed be the best model for the task of classifying reddit posts.
