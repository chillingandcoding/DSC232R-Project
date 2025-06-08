# DSC232R-Project

## Final Submission
[DSC232R Project - Steam Review](https://github.com/chillingandcoding/DSC232R-Project/blob/main/DSC232R%20Project%20-%20Steam%20Review.ipynb)

## Written Report
### Introduction 

The project will explore the Kaggle dataset 100 Million+ Steam Reviews, which contains data such as steamid, language, review sentiment, helpfulness score, review body, among many others. The goal of this project is to combine two different machine learning models into an intelligence pipeline to create a recommendation system based on review sentiment and predicted playtime, which will all be implemented in Apache Spark. First, we will apply NLP techniques such as tokenization, stopword removal, and IDF feature processing to clean and prepare our data. After, using these processed features, we will build a logistic regression model that classifies user sentiment and predicts if they would recommend the game or not (0 or 1). Following that, we will build a linear regression model that predicts playtime based on inputs of the full review text and predicted sentiment from the logistic regression model. Lastly, we will integrate the outputs from the previous two models into a single intelligence pipeline to create a recommendation system. This system recommends games by predicting how much time a user might spend on different titles and select the title with the highest projected playtime. This recommendation system aims to simulate a real-world machine learning pipeline that could
be the framework of much more powerful recommendation systems that platforms like Steam might use.

### Methods

For this project’s methodology, it is broken down into 4 sections - Data Exploration, Processing, First Model, and Second Model and Recommendation Pipeline.

Data Exploration - In this section, our primary methodology is to employ data frame transformations to clean and explore the relationships between the data. First, we dropped some columns that were not used in our project - such as geo locations. Next, we casted all of the columns to types that we can use, i.e. the playtime of certain games from string to float so we can apply transformations to it at later stages. Then, we filtered out data that is useful for our project, such as only reviews in English and are not null. Lastly, to clean our data, we employed a filter to try and only record reviews from real accounts. We did this by first checking to see if the account bought the game and if not, they at least owned another game. This filter is effective because most spam accounts would not have spent money and by checking user inventory, we can effectively remove accounts that were most likely not made by real humans. Furthermore, we filtered again to check for users that have a certain playtime to make sure that most reviews are fair and are made from actual gameplay. After our data had been optimized, we did preliminary exploration of the relationships between features. 

Processing - In this stage of the project, we utilized the NLTK library and tokenized our review text. We built a function - process word that takes strings as input and output tokenized words with stop words removed. For example, with a string ‘This is a sample review! Make sure to upvote this.’, the output would be ['sample', 'review', '!', 'Make', 'sure', 'upvote']. In this function, we also kept specific punctuations because we believe that these carry some element of sentiment, such as ‘!’ and ‘?’. Lastly, we also kept ‘-’ because some words such as the game title ‘Counter-Strike’ are being removed due to ‘-’. After we have successfully built this function, we moved on to addressing the problem of skewness of the dataset where positive sentiment is over 86% while negative is only 14%. In order for the model to not overfit, we will downsample positive sentiment to roughly the same as negative sentiment. This will also help our models later on because processing 40 million rows would run into memory issues in the current Spark environment. After we tokenized the reviews, we did some data exploration such as looking at what the top 20 most used tokens are.

First Model - Logistic Regression was the model we chose to be our first model. It is utilized to predict the sentiment of the review text and predict if the user would recommend (binary 1) or not recommend (binary 0) the game. Before we can initialize the model, we have to first convert our tokenized text into features. To do this, we created a pipeline and introduced both a hashing conversion and an IDF conversion. First, we hash our tokenized text into 8000 features, then we initialize IDF vectors after the hashing has been completed and have our models fit to the finished pipeline (Figure 8). After that, we splitted our model by 60-20-20, which represented the training set, validation set, and test set respectively. Following that, we also initialized an evaluator and analyzed our results, which will be discussed in a later section. 

Second Model and Recommendation Pipeline - Due to issues with SDSC and time constraints, the second model and the recommendation pipeline was not able to be completed on time. However, since the majority of the framework has been completed, this can easily be revisited in the future to continue to be built to completion. 

### Results
From our data exploration stage, we generated a top 20 reviews distribution of the number of reviews per game (Figure 1), as well as the distribution of playtime per game (Figure 2). We also explored the average playtime for all users across the data (Figure 3), where we had to log the results as the data was skewed (Figure 3). Lastly, we explored the relationship between games owned and total playtime, which had around a -13 correlation, meaning that playtime per title went down as the number of titles increased in an user’s inventory (Figure 4). 

In the processing phase, an example of the stop-word removed result can be seen in (Figure 5). An example of how we downsampled our data to remove skewness and prevent memory issues can be seen in (Figure 6) and a visual representation of the top 20 most used tokens can be seen in (Figure 7).

Lastly, for our logistic regression model, our initiation of the model can be seen in (Figure 8), where we created a pipeline for the hashing and IDF functions. The accuracy of the model was 81.8% for the training set, 81.75% for the validation set, and 81.71% for the test set. The confusion matrix of the model can be seen in (Figure 9). In the confusion matrix, we can see that the model is making more mistakes predicting a bad sentiment compared to positive. This is to be expected since there were not a lot of negative labels to learn from - so much so that we had to downsample our positive labels to prevent overfitting.

### Discussion
From the results of our model, it is interesting to note the accuracy of the test set and validation set is very similar, it would be a red flag if the test set has a higher accuracy but it is not in this case. This could indicate a couple things, such as the model having a fairly good generalization and not overfitting to the data. However, to play the devil’s advocate and explore what potential shortcomings that might have caused this, there are several candidates. The first being that the TF-IDF vectors are too sparse for the model to learn and the results of ~80% might be generous due to the specific seed chosen and the actual variations of the results might be worse than portrayed. Another candidate is that we lost a lot of information when we downsampled our positive sentiment data by around 86% to match with the number of negative sentiment data. Although this was partly due to the fact that the project environment would not be able to train such a big cluster of a dataset and this also brings up another point - which was the environment setup. A major bottleneck in this project was the amount of memory available, which might have been due to setup. This bottleneck placed some restrictions on how much of the data we can pull to work with and prevented us from utilizing caching and deploying more memory intensive models such as Random Forest. Overall, the accuracy of the model was satisfactory and the shortcomings that were noted were a valuable experience to add on to future models. 

### Conclusion
Our project, overall, is headed in a solid direction. The next step for this project is further implementation and work on the second model and the recommendation pipeline. We learned a lot from the setbacks during this project, and in the future when working with Spark again, we feel that we have the experience equipped to set up a more robust and efficient environment. This is an area of improvement we want to emphasize because we felt, as iterated earlier, our environment setup was a pretty big bottleneck on the models and transformations we originally planned to do. Running out of memory was a big issue and we cut back a lot on the size of the data and opted for simpler models when models such as Random Forest might have performed better. This also applies to our hashing process, as we wanted to try different methods such as word2vec but we were running into memory issues. 

### Statement of Collaboration 
Franky Liu, solo team member


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
