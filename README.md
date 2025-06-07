# DSC232R-Project

## Final Submission
[DSC232R Project - Steam Review](https://github.com/chillingandcoding/DSC232R-Project/blob/main/DSC232R%20Project%20-%20Steam%20Review.ipynb)

## Written Report
[DSC232R Project - Written Report](https://github.com/chillingandcoding/DSC232R-Project/blob/main/DSC%20232R%20Project%20-%20Written%20Report.pdf)

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
[DSC232R Project - Milestone 2](https://github.com/chillingandcoding/DSC232R-Project/blob/main/DSC232R%20Project%20-%20Milestone%202.ipynb)

### Dataset 
100 Million+ Steam Reviews
https://www.kaggle.com/datasets/kieranpoc/steam-reviews

### Packages Needed
All pip packages have been included to run in the notebook - kagglehub and nltk

### Environment
singularitypro

### Summary
In part 2, data will be processed by filtering the dataset to determine what reviews are from legitimate users and contain useful keywords. Then we will perform standard sanitary checks
by dropping null values, casting column types, and downsampling to prevent skewing in the label counts. This section will also conduct preliminary graphing explorations to explore data
relationships and correlations, all of which will lay the foundation to build the first model in this project. We will pre-process our review data by utilizing NLTK and its libraries 
such as stopwords, tokenization, and lemmitzation. This step will be performed in Part 3 - building the first model. 

## Part 3 - Building the First Model

### Notebook
[DSC232R Project - Milestone 3](https://github.com/chillingandcoding/DSC232R-Project/blob/main/DSC232R%20Project%20-%20Milestone%203.ipynb)

### Summary/Conclusion
In Part 3, we defined a function to tokenize and process our review text as well as downsampling our dataset to counter the skewness of the labels. We then uesd this function to make our dataset compatible to feed into the TF-IDF pipeline that we have also created. 
Through this pipeline, we were able to create 3 sets of data - training, validation, and test and fitted our logistic classifier to it. 

Our first model can be viewed as successful as we were able to produce a model that had relatively good accuracy despite the fact 
that it had skewed proportions for its labels (to which we fixed with downsampling). We tried different variations of token
processing by choosing to keep specific punctuations and optimized how the tokenizer was treating words like 'very-fun'; where before,
the tokenizer would remove the whole word instead of just removing the the dash in the middle. We also tried to utilize N-Gram models to 
fine tune our model but since a 2-gram model and above drasticially increase our number of tokens and we could not load enough memory 
in the Spark enviornment for it to successfully run - so we had to keep our unigram model. However, we were still able to produce a model 
that has a relatiecly high accuracy score through continuous experimentation with our validation set. Something that we could have done better was to set up our Spark enviornment better as we had to 
make a couple of compromises - such as reducing our number of features for our model from 100000 to 8000 and not being able to experiment with 
2-Gram models like mentioned earlier. We could have also create a more robust downsampling or upsampling method as we are risking losing good 
information from the 30 million rows of positive labels we discarded. 

Our next model will be a linear regression that predicts playtime (a float) based on feature inputs such as the review text/sentiment and past playtime. 
Eventually, we will combine this Linear Regression model with our Logistic Regression model where the sentiment prediction will be fed into our linear 
regression model which outputs a predicted playtime. Then we will wrap all of these into a pipeline at the end and create a recommendation system by 
choosing the game with the highest projected playtime by the user. 

## Part 4 - Building the Second Model and the Recommendation Pipeline 
To be continued.
