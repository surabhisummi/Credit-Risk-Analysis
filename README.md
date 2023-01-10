# Credit-Risk-Analysis
Predicting the ability of a borrower to pay back the loan through Traditional Machine Learning Models and comparing to Ensembling Methods

## Traditional ML Model and Ensembling techniques:

Data Overview
encoding	category	count of rows in the dataset
1	Defaulted	5634
0	Paid Back the amount in full	33136

# Project
This project was aimed at exploring different traditional Machine Learning algorithms and comparing them against powerful models like ensembling methods and artificial neural networks in Keras to identify the credit risk and whether the customer will default or pay back the loan in full based on different indicators.

# Sampling
This dataset is an imbalanced dataset and so sampling was a must to get any good results otherwise the model will not be effective in figuring out False Negatives as they are a minority class and end up giving more bad loans.

# Model Evaluation Criteria
False positive rate is the number of false positives divided by the number of false positives plus the number of true negatives. This divides all the cases where we thought a loan would be paid off but it wasn't by all the loans that weren't paid off:

fpr = fp / (fp + tn)

True positive rate is the number of true positives divided by the number of true positives plus the number of false negatives. This divides all the cases where we thought a loan would be paid off and it was by all the loans that were paid off:

tpr = tp / (tp + fn)

# Model Performance
XGBOOST outperformed all the other algorithms and also was great in capturing False negatives with only 6 in a dataset of 20000 samples used for validation while also controling the False positives which were 2071. This model is great in detecting potential bad loans.

# Model Comparison

Model Description	Sampling Method	AUC
LOGISTIC REGRESSSION VANILA	No Sampling	0.50
LOGISTIC REGRESSION: CLASS WEIGHT	Using sklearn class_weight="BALANCED"	0.59
LOGISTIC REGRESSION: WITH SAMPLING	SMOTE Over Sampling (minority class)	0.59
XGBOOST: BOOSTING	SMOTE-Over Sampling Method	0.88
