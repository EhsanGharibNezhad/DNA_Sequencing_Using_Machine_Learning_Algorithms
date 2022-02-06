 # <b>  DNA Sequencing Using Machine Learning Algorithms
  
###  Investigator: Ehsan Gharib-Nezhad

  <p>
  <a href="https://www.linkedin.com/in/ehsan-gharib-nezhad/" rel="nofollow noreferrer">
    <img src="https://i.stack.imgur.com/gVE0j.png" alt="linkedin"> LinkedIn
  </a> &nbsp; 
  <a href="https://github.com/EhsanGharibNezhad/" rel="nofollow noreferrer">
    <img src="https://i.stack.imgur.com/tskMh.png" alt="github"> Github
  </a>
</p>
---

<br></br>
# <a id = 'ProblemStatement'>Problem Statement</b></a>
High-throughput Next Generation Sequencing has played a crucial role in broadening our understanding of biology. Understanding the limitation and accuracy of the recorded data, however, is challening step as the interpretation of human gunume relies on its accuracy. Traditional Machine learning-based tools as well as the deep learning neural network could provide an important means to rigriously vet the sequencing results and leverage the accuracy of the sequencing genomes and
transcriptomes. Critically speaking, .....
In this project, ~XXX DNA sequences from ---, ---, and --- were utilized from the XXXX data [Covid19Positive](https://www.reddit.com/r/COVID19positive/) and ~2,300 posts from the subreddit [PandemicPreps](https://www.reddit.com/r/PandemicPreps/) ......


---

<br></br>
# <a id = 'Content'> Content </b></a>

- [Problem Statement](#ProblemStatement)
- [Content](#Content)    
- [Repo Structure](#RepoStructure)    

    - [Data Dictionary](#ddict)
    - [Background](#Background)
    - [1. Data Scarpping: Application Programming Interface](#api)
   	- [2. Text Normalization](#Text_Normalization)
    	- [2.1. Tokenization](#Tokenization)
    	- [2.2. Lemmatization](#Lemmatization)
    	- [2.3. Stemming](#Stemming)
    - [Methodology](#Methodology)    
    	- [Sentiment Analysis](#Sentiment)	
    - [Exploratory Data Analysis](#eda)    
    - [Results](#Results)    
    - [Conclusion](#Conclusion)
    - [Recommendations](#Recommendations)
    - [References](#references)



---
# <a id = 'RepoStructure'> Repo Structure </b></a>
## notebooks/ <br />

*Setp 1: Exploratory Data Analysis:*\
&nbsp; &nbsp; &nbsp; __ [3__ExploratoryDataAnalysis_EDA.ipynb](notebooks/3__ExploratoryDataAnalysis_EDA.ipynb)<br />

*Setp 2: Traditional Machine Learning Models: Classifiers*\
&nbsp; &nbsp; &nbsp; __ [4-1__model_Logestic_Regression.ipynb](notebooks/4-1__model_Logestic_Regression.ipynb)<br />
&nbsp; &nbsp; &nbsp; __ [4-2__model_Logestic_Regression-Imbalanced.ipynb](notebooks/4-2__model_Logestic_Regression-Imbalanced.ipynb)<br />
&nbsp; &nbsp; &nbsp; __ [4-3__model_Decision_Trees.ipynb](notebooks/4-3__model_Decision_Trees.ipynb)<br />
&nbsp; &nbsp; &nbsp; __ [4-4__model_Bootstrapping_Bagging.ipynb](notebooks/4-4__model_Bootstrapping_Bagging.ipynb)<br />
&nbsp; &nbsp; &nbsp; __ [4-5__model_Random_Forests.ipynb](notebooks/4-5__model_Random_Forests.ipynb)<br />
&nbsp; &nbsp; &nbsp; __ [4-6__model_Extremely_Randomized_Trees__ExtraTrees.ipynb](notebooks/4-6__model_Extremely_Randomized_Trees__ExtraTrees.ipynb)<br />
&nbsp; &nbsp; &nbsp; __ [4-7__model_Adaboost.ipynb](notebooks/4-7__model_Adaboost.ipynb)<br />
&nbsp; &nbsp; &nbsp; __ [4-8__model_Gradient_Boosting.ipynb](notebooks/4-8__model_Gradient_Boosting.ipynb)<br />
&nbsp; &nbsp; &nbsp; __ [4-9__model_xgbooster.ipynb](notebooks/4-9__model_xgbooster.ipynb)<br />
&nbsp; &nbsp; &nbsp; __ [4-10__model_Naive_Bayes.ipynb](notebooks/4-10__model_Naive_Bayes.ipynb)<br />
&nbsp; &nbsp; &nbsp; __ [4-11__model_Support_Vector_Machines_SVMs.ipynb](notebooks/4-11__model_Support_Vector_Machines_SVMs.ipynb)<br />


*Setp 3: Modern Machine Learning tools: Classifiers*\
&nbsp; &nbsp; &nbsp; __ [5-1__sentiment_analysis_VADER.ipynb](notebooks/5-1__sentiment_analysis_VADER.ipynb)<br />
/>



## datasets/<br />
*Unprocessed data collected from sub Reddits:*\
&nbsp; &nbsp; &nbsp; __ [preprocessed_covid19positive_reddit_LAST.csv](datasets/preprocessed_covid19positive_reddit_LAST.csv)<br />
&nbsp; &nbsp; &nbsp; __ [preprocessed_df_PandemicPreps_reddit_LAST.csv](datasets/preprocessed_df_PandemicPreps_reddit_LAST.csv)<br />


*Modeling results: Accuracy, Precision, Recall, Confusion Matrix:*\
&nbsp; &nbsp; &nbsp; __ [models_metrics_report_confusionMatrix.csv](datasets/models_metrics_report_confusionMatrix.csv)<br />
&nbsp; &nbsp; &nbsp; __ [models_metrics_report_precision_recall.csv](datasets/models_metrics_report_precision_recall.csv)<br />
&nbsp; &nbsp; &nbsp; __ [models_metrics_report_accuracy.csv](datasets/models_metrics_report_accuracy.csv)<br />


[presentation.pdf](presentation.pdf)<br />

[ReadMe.md](ReadMe.md)<br />

---
---
# <a id = 'ddict'>Data <b>Dictionary</b></a>


|feature name|data type|Description|
|---|---|---|
| selftext |*object*|Original Reddit posts with no text processing|
| subreddit|*object*|Subreddit category: r\Covid19Positive and r\PandemicPreps|
| created_utc|*int64*|Reddit posting date|
| author|*object*|Author ID|
| num_comments|*int64*|Number of comments/reply to that post|
| post|*object*| Reddit post after text precessing with normal/unstemmed words|
| token|*object*| Reddit post after text precessing with word stemming|

---
---
# <a id = 'Background'>Background</a> 
## 1. <a id = 'api'> Data Scarpping: Application Programming Interface</a> 
The pushshift.io Reddit API was designed and created by the /r/datasets mod team to help provide enhanced functionality and search capabilities for searching Reddit comments and submissions [[ref]](https://github.com/pushshift/api).

## 2. <a id = 'Text_Normalization'>Text Normalization</a> 
Text preprocessing and normalization is a crucial step in Natural Language Processing (NLP) and it means converting the text into the standard form. Examples of text preprossessing include [[ref]](https://towardsdatascience.com/text-preprocessing-for-data-scientist-3d2419c8199d):

- *Lower casing*
- *Removing puctuations*
- *Revoving special characters*
- *Handeling emojis and emoticons (removing them or replacing them with words)* 
- *Stop word removals*
- *Common word removal*
- *Rare word removal*
- *Spelling correction* 
- *Removing URLs and HTML tags* 
- *Tokenization* 
- *Stemming* 
- *Lemmatization* 

Some of these methods and their python tools will be discussed in detail in the following.

### 2.1. <a id = 'Tokenization'> Tokenization  </a> 
This is the first step in text processing and cleaning which consists of separating or tokenizing words from the text [[ref](https://web.stanford.edu/~jurafsky/slp3/)]. Although English words are separated by whitespaces, there are many cases like "San Francisco" which should be treated as a single word. On the other hand, contractions such as "I'm" are another example that needs to be treated separately. Tokenization can be divided into two steps: Paragraph to a sentence or *sentence tokenization*, and sentence to words or *Word Tokenization*. In this project, we implemented the following tools from `NLTK` for this purpose:

`from nltk.tokenize import sent_tokenize, word_tokenize`  

As a part of tokenization, the following methods might be utilized as well [3]: 
- Bigrams: Tokens consist of two consecutive words known as bigrams.
- Trigrams: Tokens consist of three consecutive words known as trigrams.
- Ngrams: Tokens consist of ’N’ number of consecutive words known as n-grams
In this project, processed word tokens are embedded into a list and N-gram assessment is done using the following scripts:

`(pd.Series(nltk.ngrams(words, ngram_value)).value_counts())`  

in which, `ngram_value` is 2, 3, 4, 5, etc. For detailed discussion, check [[ref](https://towardsdatascience.com/from-dataframe-to-n-grams-e34e29df3460)]. Note that in the tokenization, *stopwords* such as "the", "an", "at" will be eliminated as well using the following script:

`from nltk.corpus import stopwords`\
`stopwords.words('english')`




### 2.2. <a id = 'Lemmatization'>  Lemmatization </a> 
Another phase for processing the text is lemmatization which includes the simplification of words based on their root. For example, *spoke*, and *spoken* are all forms of the verb *speak*.

`from nltk.stem import WordNetLemmatizer`\
`lemmatizer = WordNetLemmatizer()`


### 2.3. <a id = 'Stemming'> Stemming </a> 
Stemming is the in-depth version of lemmatization in which all words are converted to their stem root. For instance, *computes*, *computing*, *computer*, *computational* are all from the same root "compute"

`from nltk.stem.porter import PorterStemmer`\
`p_stemmer = PorterStemmer()`


A good overview the the text processing/standarilizing can be found at [[ref](https://medium.com/@jeevanchavan143/nlp-tokenization-stemming-lemmatization-bag-of-words-tf-idf-pos-7650f83c60be) and [ref](https://towardsdatascience.com/text-preprocessing-for-data-scientist-3d2419c8199d)].



# <a id = 'modeling_methodology'>3. Modeling Methodology to Label Subreddit Posts</a>


## 3.1 Logestic Regression
In ordinary least squares linear regression (often just referred to as OLS), we try to predict some response variable (y) from at least one independent variable (x). In contrast, 
Logistic regression deals with categories and gives us the probabilities of being in each class.
For logistic regression, that specific link function that transforms ("bends") our line is known as the logit link.


## 3.2 Classification and Regression Trees (CART) 
Decision Tree can be used to predict the class (discrete) (AKA. *Classification Tree*) or to infer continuous features such as house price which is called *Regression Tree*. The decision tree has the same analogy as the 20-question game to make decisions similar to how humans make decisions. It is a supervised machine learning algorithm that uses a set of rules for classification. The following library from **sklearn** is used to implement this model:

`from sklearn.tree import DecisionTreeClassifier`
 

<!--- ### 3.2.1 Terminology
- **Root Node**: What we see on top is known as the "root node," through which all of our observations are passed.
- **Leaf Nodes**: At each of the "leaf nodes" (colored orange), we contain a subset of records that are as pure as possible.
- A "parent" node is split into two or more "child" nodes. --->


### 3.2.1 Gini impurity
Gini impurity is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. The Gini impurity can be computed by summing the probability of a mistake in categorizing that item [[ref](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)]. In other words, **Gini impurity** is the probability that a randomly-chosen class will be labeled inaccurately, and is calculated by the following equation:


<img src="https://latex.codecogs.com/svg.image?Gini&space;\&space;Index&space;=&space;1&space;-&space;\sum_{i=1}^{n}&space;(p_i)^2" title="Gini \ Index = 1 - \sum_{i=1}^{n} (p_i)^2" />

where *p* is the probability of item *i*. *Gini impurity* is between 0 and 1, where 0 shows the lowest impurity and 1 is the highest one. Both *Classification* and *Regression Trees* implement *Gini coefficient* to select the *root node* as well as the *node splision*. More details can be found in [[ref](https://medium.com/analytics-steps/understanding-the-gini-index-and-information-gain-in-decision-trees-ab4720518ba8)].     






## 3.3 Classification and Regression Trees including Bootstrap Aggregating (Bagging)  

Although *Decision trees* are powerful machine learning models, they tend to learn highly irregular patterns (a.k.a. they overfit their training sets). *Bagging* (*bootstrap aggregating*) mitigates this problem by exposing different trees to different sub-samples of the training set. In this method, a set of bootstrapped samples (or ensemble, see the following flowchart) is generated with the inclusion of replacement. Consequently, a set of decision trees is generated in order to account for the variability and reduce the overfitting issue (or to improve the variance). The following library is used from **sklearn**to implement this approach:

`from sklearn.ensemble import BaggingClassifier`\
`BaggingClassifier(random_state = 42)`



## 3.4 Random Forest

With bagged decision trees, we generate many different trees on pretty similar data. These trees are strongly correlated with one another. Because these trees are correlated with one another, they will have a high variance. Looking at the variance of two random variables X and Y, if X and Y are highly correlated, then the variance will about as high as we'd see with individual decision trees. By "de-correlating" our trees from one another, we can drastically reduce the variance of our model. That's the difference between bagged decision trees and random forests! 

The following equations articulate this correlation and variance in simple math [[ref](https://www.probabilitycourse.com/chapter5/5_3_1_covariance_correlation.php)]:

<img src="https://latex.codecogs.com/svg.image?Var(X&plus;Y)=Var(X)&plus;Var(Y)&plus;2Cov(X,Y)" title="Var(X+Y)=Var(X)+Var(Y)+2Cov(X,Y)" />

where the the covariance between X and Y is defined as:

<img src="https://latex.codecogs.com/svg.image?Cov(X,Y)=&space;E[(X-EX)(Y-EY)]=E[XY]-(EX)(EY)" title="Cov(X,Y)= E[(X-EX)(Y-EY)]=E[XY]-(EX)(EY)" />

where E[X] and E[Y] is the expected (or mean) value of X and Y.

Random forests differ from bagging decision trees in only one way: they use a modified tree learning algorithm that selects, at each split in the learning process, a random subset of the features. This process is sometimes called the random subspace method. The following library is implemented in this model:

`from sklearn.ensemble import RandomForestClassifier`








## 3.5 Extremely Randomized Trees (ExtraTrees)


*ExtraTrees* has an extra feature to even more randomize (and thus de-correlation) decision tress compared to the conventional *Random Forest* method. In the *ExtraTrees* method, the features at the *root node* and *leaf nodes* are selected randomly without utilizing *Gini impurity* or *information gain*. In addition, the split in each node is determined by selecting a subset of randomly selected features within that subset size (i.e., usually the square root of the total features). The following library is used for this method:

`from sklearn.ensemble import ExtraTreesClassifie`



## 3.6 Naïve Bayes

Naive Bayes classifiers rely on the Bayes theorem which is a conditional theory. 
 [Conditional theory](https://www.statisticshowto.com/probability-and-statistics/probability-main-index/bayes-theorem-problems/) is the probability of an event happening, given that it has some relationship to one or more other events. For example, your probability of the temperature given that is connected to the sunlight, humidity, and altitude. 
Another example of Bayes theorem is in medical research cases such as the probability of having kidney disease if they are alcoholic [[ref]](https://www.statisticshowto.com/probability-and-statistics/probability-main-index/bayes-theorem-problems/). 

In conditional theory, P(A|B) is the probability of event A given that event B has occurred. Since we are assuming that A and B are independent, we can write their joint probability of A and B as P(A and B) = P(B and A). Note that the joint probability of X and Y or P(X ∩ Y) can be written as P(X). P(Y|X). Therefore,  P(A and B) = P(B and A) will be written as:

<img src="https://latex.codecogs.com/svg.image?P(A)&space;P(B|A)&space;=&space;P(B)&space;P(A|B)" title="P(A) P(B|A) = P(B) P(A|B)" /> 

Rearranging it leads to Bayes' theorem.  


<img src="https://latex.codecogs.com/svg.image?P(A|B)&space;=&space;\frac{P(B|A)P(A)}{P(B)}" title="P(A|B) = \frac{P(B|A)P(A)}{P(B)}" /> 




**Naïve Bayes** is a supervised machine learning algorithm that could be used for classification problems. However, in comparison with another classifier, this model assumes that features are independent of each other, and there is no correlation between them [[ref]](https://towardsdatascience.com/naive-bayes-classifier-explained-50f9723571ed). This is why it is called *naïve*.  **Naïve Bayes Classifier** needs to store probability distribution for each feature. The type of distributions depends on the feature properties, including:

1. **Bernoulli distribution**: for binary features (e.g., rainy day: Yes or No?)  
2. **Multinomial distribution**: for discrete features (e.g., word counts)  
3. **Gaussian distribution**: for continuous features (e.g., House price)  

Naive Bayes classifiers are relativity fast comparing to *Random Forest*, but the assumption of independent features imposes some inaccuracy to the results. The following library is implemented to use this model:

` from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB `




# 4. <a id = 'sentiment'> Sentiment Analysis: Reddit Posts </a> 
Sentiment analysis is the study of the text sentiment, the positive or negative orientation that a writer expresses toward some topics. For instance, all product review systems such as IMDB have numerous amount of words for each movie, some positive, some negatives, and some neutral. The main goal is to employ some words such as "great", "disappointing", and many others as indicators to conclude and rate the text's sentiment. In this project, *VADER* ( Valence Aware Dictionary for Sentiment Reasoning) is a model used for text sentiment analysis that is sensitive to both polarities (positive/negative) and intensity (strength) of emotion [[ref](https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664)].
The following library is employed to load *VADER*:

`from nltk.sentiment.Vader import SentimentIntensityAnalyzer`

<br></br>
<img src="./plots/sentiment_analysis_vader.png" alt="drawing" width="800"/>
<br></br>





# <a id = 'eda'> <b>5. Exploratory Data Analysis</b>
Posts in Covid12Positive have more words as the following figure shows.

<br></br>
<img src="./plots/eda.png" alt="drawing" width="800"/>
<br></br>
    
# <a id = 'Results'>6. Results</b>

In the entire project, all posts are split into train and testing sets with the proportion of 75% and 25%, respectively. For each dataset, accuracy, precision, recall, F1 score as well as their false positive, false negative, and true positive and negative values are reported. The following equations are showing their definitions [[ref](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)]:

<br></br>
<img src="./plots/defination_equations_accuracy_precision.png" alt="drawing" width="800"/>
<br></br>

## <a id = '6_1'> 6.1 Compare Metric Scores of Different Models</b> 

In the following figures, the accuracy, recall, precision, and F1-score for different models are represented. 
 
<br></br>
<img src="./plots/results_compare_accuracy_all_models.png" alt="drawing" width="800"/>
<br></br>

<br></br>
<img src="./plots/results_compare_precision_recall_all_models.png" alt="drawing" width="800"/>
<br></br>


<br></br>
<img src="./plots/results_compare_accuracy_all_models.png" alt="drawing" width="800"/>
<br></br>

<br></br>
<img src="./plots/results_compare_confusionMatrix_all_models.png" alt="drawing" width="800"/>
<br></br>

---
---

An overview of the final results are presented in the following infograph:


<br></br>
<img src="./plots/results_modeling_Logistic_regresssion.png" alt="drawing" width="800"/>
<br></br>
     
# <a id = 'Conclusion'>Conclusion</b>

Logistic Regression is found to be the best model for classification because of the following reasons: 
- Provides the highest accuracy scores, ~99% and ~96% for training and testing datasets 
- Works great with ultra-imbalanced samples (~93% vs. ~7%)
- High rates for true positive (91.06% out of 93%) and true negative (5.5% out of 7%)
- Low scores for false positive (1.57%) and false-negative (1.87%)
- High scores for precision and recall (~98%) 

In addition, this model is….. 
- Interpretable 
- Optimizable coefficients to reduce variance and bias
- Capable to use different generalization methods i.e., Lasso, Ridge, ElasticNet
- Tunable parameters, solvers, and penalty functions for multiple cases 
- Works best with both large and small datasets




