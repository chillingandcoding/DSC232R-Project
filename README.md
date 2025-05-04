# DSC232R-Project

## Part 1 - Abstract
The project will explore the Kaggle dataset 100 Million+ Steam Reviews (https://www.kaggle.com/datasets/kieranpoc/steam-reviews), which contains data such as steamid, language, review sentiment, helpfulness score, review body, among many others. 
The goal of this project is to combine two different machine learning models into an intelligence pipeline to create a recommendation system based on review sentiment and predicted playtime, 
which will all be implemented in Apache Spark. First, we will first apply NLP techniques such as tokenization, stopword removal, and IDF feature processing to clean and prepare our data. Using 
these processed features, we will build a logistic regression model that classifies user sentiment (positive or negative) and if they would recommend the game (0 or 1). Following that, 
we will build a linear regression model that predicts playtime based on inputs such as the full review text, past playtime, as well as the predicted sentiment from the logistic regression model. 
Lastly, we will integrate the outputs from the previous two models into a single intelligence pipeline to create a recommendation system. This system recommends games by predicting how much time 
a user might spend on different titles and selecting the title with the highest projected playtime. This recommendation system aims to simulate a real-world machine learning pipeline that could 
be the framework of much more powerful recommendation systems that platforms like Steam might use. 

## Part 2 - Data Exploration

### Notebook
DSC232R Project - Milestone 2

### Dataset 
100 Million+ Steam Reviews
https://www.kaggle.com/datasets/kieranpoc/steam-reviews

### Packages Needed
All pip packages have been included to run in the notebook - kagglehub and nltk

### Environment
Native Python

### Summary
In part 2, data will be processed by filtering the dataset to determine what reviews are from legitimate users and contain useful keywords. Then we will perform standard sanitary checks
by dropping null values, casting column types, and downsampling to prevent skewing in the label counts. This section will also conduct preliminary graphing explorations to explore data
relationships and correlations, all of which will lay the foundation to build the first model in this project. We will pre-process our review data by utilizing NLTK and its libraries 
such as stopwords, tokenization, and lemmitzation. This step will be performed in Part 3 - building the first model. 
