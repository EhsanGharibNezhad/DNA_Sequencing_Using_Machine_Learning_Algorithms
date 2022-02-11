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


<br></br>
# <a id = 'ProblemStatement'>Problem Statement</b></a>
High-throughput Next Generation Sequencing has played a crucial role in broadening our understanding of biology. Understanding the limitation and accuracy of the recorded data, however, is challening step as the interpretation of human genome relies on its accuracy. Traditional Machine learning-based tools as well as the deep learning neural network could provide an important means to rigriously vet the sequencing results and leverage the accuracy of the sequencing genomes and
transcriptomes. Critically speaking, machine learning provides powerful statistical tools to better understand the accuracy of the DNA sequencing. In this project, 4380 human DNA sequences and 7 family genes were employed to train and test a set of models using multiclass classification technique with traditional machien learning algorithiums including KNN, Randon Forest, SVM. Then, deep neural networks were generated with opimized architacture in order to boost the classification accuracy. The main objective of this project is to accuratly classify the human family genes given a set of DNA sequences.

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
# <a id = 'ddict'>Dataset <b>Dictionary</b></a>


|feature name|data type| possible values | represents| description | reference|
|---|---|---|---|---|---|
| Sequence |*object*| A, T, G, C | DNA sequence|    |   |
| Class|*integer*|0 |  G protein-coupled receptors (GPCRs)| G-protein-coupled receptors (GPCRs) are the largest and most diverse group of membrane receptors in eukaryotes. These cell surface receptors act like an inbox for messages in the form of light energy, peptides, lipids, sugars, and proteins| [[link]](https://www.nature.com/scitable/topicpage/gpcr-14047471/) |
|  |*integer*|1 |  Tyrosine kinase| a large multigene family with particular relevance to many human diseases, including cancer|[[link]](https://www.nature.com/articles/1203957) |
|  |*integer*|2 |  Protein tyrosine phosphatases| Protein tyrosine phosphatases are a group of enzymes that remove phosphate groups from phosphorylated tyrosine residues on proteins| [[link]](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiI9omSsfP1AhVeJ0QIHbQbAF8QFnoECAcQAw&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FProtein_tyrosine_phosphatase&usg=AOvVaw26Gc_GqosG5hJnZu1uf4cy)|
|  |*integer*|3 |  Protein tyrosine phosphatases (PTPs)| to control signalling pathways that underlie a broad spectrum of fundamental physiological processes | [[link]](https://pubmed.ncbi.nlm.nih.gov/17057753/)|
|  |*integer*|4 |  Aminoacyl-tRNA synthetases (AARSs)| responsible for attaching amino acid residues to their cognate tRNA molecules, which is the first step in the protein synthesis | [[link]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC29805/)|
|  |*integer*|5 |  Ion channels| Ion channels are the pathways for the passive transport of various inorganic ions across a membrane| [[ref]](https://www.frontiersin.org/articles/10.3389/fgene.2019.00399/full) |
|  |*integer*|6 |  Transcription Factor| Transcription factors are proteins involved in the process of converting, or transcribing, DNA into RNA | [[link]](https://www.nature.com/scitable/definition/transcription-factor-167/)|

---
---
# <a id = 'Background'>Background</a> 
## 1. <a id = 'api'> DNA sequencing </a> 
DNA (Deoxyribonucleic acid) sequencing is the process of determining the nucleic acid sequence or the order of nucleotides in DNA molecule. In fact, it includes experimental techniques to determine the order of the four bases: Adenine (A), guanine (G), cytosine (C), and thymine (T). DNA sequencing is a crucial technique in biology because it is fundamental step in better understading the root of many genetic deseaise. The following figure illustrate the main concept of DNA sequecing as well as some of its importnat applications [[ref]](https://www.nist.gov/patents/nucleic-acid-sequencer-electrically-determining-sequence-nitrogenous-bases-single-stranded). 


<br></br>
<img src="./plots/DNA_Sequencing_Concept.png" alt="drawing" width="800"/>
<br></br>


---
# <a id = 'ModelingMethodology'>Statistical Models: Methodology and Concepts</b></a>

## <a id = '1Multiclassclassification'>1. Multiclass classification</b></a>

Multiclass classification is a common problem in machine learning and includes targets with more than two classes. In case that the target has two classes, then it is named binary classification. In the following, I will provide background information about the classification methodology:
1. One-Vs-Rest (OvR)
2. One-Vs-One (i.e., OvO)

![image](/plots/multiclassification_concept.png)


## <a id = 'RandomForest'>2. Random Forest</b></a>
A random forest is a supervised machine learning algorithm and is a part of ensemble methods. It is known for having lower overfitting issues than common decision trees because it creates a large ensemble of bootstrap trees and aggregates them. In this blog, I will discuss the fundamental backgrounds to better understand both Random Forest Classifiers and regressors.

In the bagging technique, all features get selected but the number of observations (or rows in the dataset) is different. Therefore, there is still some correlation between the bootstrap trees which results in high variance. Random Forest breaks this correlation by randomly selecting the features and not having all of them in all decision trees. Hence, Random Forest can be represented as a supervised machine learning algorithm that uses an enhanced version of the Bagging technique to create an ensemble of decision trees with low correlation.


![image](/plots/RandomForest_concept.png)


## <a id = 'NeuralNetworks'>3. Neural Networks</b></a>

Dendrites, axons, cell body might not be that familiar terms for everyone; however, the complexity of the neural networks in the brain could be a reasonable naive start to understanding the complexity of teaching a computer to solve problems. Here in this figure 4, the analogy between the signal transferring process and the deep neural network is depicted. 

*"The idea is that the synaptic strengths (the weights w) are learnable and control the strength of influence ... dendrites carry the signal to the cell body where they all get summed. If the final sum is above a certain threshold, the neuron can fire, sending a spike along its axon. In the computational model, we assume that the precise timings of the spikes do not matter and that only the frequency of the firing communicates information. Based on this rate code interpretation, we model the firing rate of the neuron with an activation function f."* - CS231-Stanford

![image](/plots/fnn_concept.png)


---

# <a id = 'modeling_methodology'>3. Modeling Methodology to Classify Gene Family Classes</a>


## 3.1 Classification and Regression Trees (CART) 
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




# <a id = 'Results'>6. Results</b>

In the entire project, all posts are split into train and testing sets with the proportion of 75% and 25%, respectively. For each dataset, accuracy, precision, recall, F1 score as well as their false positive, false negative, and true positive and negative values are reported. The following equations are showing their definitions [[ref](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall)]:

<br></br>
<img src="/plots/compare_knn_RandomForest_results.png" alt="drawing" width="800"/>
<br></br>


<br></br>
<img src="/plots/FNNs_results.png" alt="drawing" width="800"/>
<br></br>


# ============================================================
# ============================================================
# ============================================================
# ============================================================













# <a id = 'eda'> <b>5. Exploratory Data Analysis</b>
Posts in Covid12Positive have more words as the following figure shows.

<br></br>
<img src="./plots/eda.png" alt="drawing" width="800"/>
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




