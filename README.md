
# Student Academic Risk Prediction and Educational Resource Planning

## Project Overview

This project investigates how machine learning can be used to predict students' final mathematical performance and support educational resource planning.
The objective is to identify students who may receive a final grade below 10 using academic information available at different points in the school year.

## Dataset
This project uses the UCI Student Performance mathematics dataset, which contains 395 student records from Portuguese secondary schools. 

The target variable was defined as:
Academic risk (1): Final mathematics grade (G3) below 10.
No academic risk (0): Final mathematical grade (G3) of 10 or above.

The dataset was divided into 316 training records and 79 held-out records.

## ML Models
Two prediction models were developed:
**Model A**:First-period academic risk prediction using Logistic Regression.
**Model B**: Updated academic risk prediction using Random Forest and additional second-period academic information.

This project also examines how classification thresholds affect the number of students flagged for academic-risk review and compares these counts with hypothetical tutoring capacity.

## Methodology
This project includes:
- Exploratory data analysis of academic performance and student characteristics 
- Feature engineering, including changes between first and second period grades
- Data processing using scikit-learn pipelines
- Model development using Logistic Regression and Random Forest
- Five-fold stratified cross-validation and held-out test evaluation
- Error analysis of students incorrectly classified by the models
- Hypothetical tutoring-capacity analysis using different classification thresholds

## Model Performance
|Model	             | Accuracy	| Precision |	Recall | F1 Score |
|---|---|---|---|---|
|Logistic Regression| 0.848	  |  0.733	  | 0.846	 | 0.786|
|Random Forest	     | 0.924	  |  0.955	  | 0.808	 | 0.875|

Model B achieved higher accuracy, precision, and F1-score on the held-out test set. However, Model A achieved a higher recall, identifying 22 of the 26 students who eventually received final grades below 10, compared with 21 students identified by Model B. These results highlight the importance of considering false negatives when evaluating models intended to support early academic-risk identification.

## Education Resource-Planning Analysis
A hypothetical tutoring-capacity scenario was developed for the 79-student test cohort, assuming 20 available tutoring places.

At a classification threshold of 0.5:

Model A flagged 30 students, exceeding the hypothetical capacity by 10 places. Model B flagged 22 students, exceeding the hypothetical capacity by two places.

On the held-out test set, Model B achieved higher accuracy, precision, and F1-score, while Model A achieved slightly higher recall at the default classification threshold of 0.50. Model A correctly identified 22 of the 26 students who eventually received final grades below 10, compared with 21 identified by Model B. Although Model B identified three at-risk students missed by Model A, it also missed four at-risk students whom Model A had correctly identified. These results demonstrate that updated predictions can identify additional students but should not automatically replace earlier academic-support assessments.
<p align='center'>
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/cb89aaf5-905c-463a-9db9-2b386582ebca" />
</p>

Additional threshold analysis showed that increasing the classification threshold reduced the number of students flagged but could increase the number of students with low final grades who were missed.

The analysis illustrates how predictive analytics can inform academic-support planning while highlighting the importance of considering false negatives, available capacity, and educator review.

## Changes in Academic-Risk Predictions

The first-period and second-period models were compared at the student level using the same 79-student held-out test set.

| Prediction group         |   Number of students |   Students with G3 < 10 |
|:-------------------------|---------------------:|------------------------:|
| Flagged by both models   |                   18 |                      18 |
| Flagged only by Model A  |                   12 |                       4 |
| Newly flagged by Model B |                    4 |                       3 |
| Not flagged by either    |                   45 |                       1 |

The second-period model newly flagged four students, three of whom actually received final mathematical grades below 10. All four newly flagged students had lower second-period grades than first-period grades.
<p align="center">
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/4903b54d-de15-4c56-b220-eaed7b654ebc" />

</p>

However, the second-period model also stopped flagging 12 students, including four who eventually received final grades below 10.

These results show that updated model predictions can identify additional students while also missing students who were correctly flagged earlier. For educational resource planning, updated predictions should be considered alongside previous assessments rather than automatically replacing them.


## Limitations 

This project is a proof os concept using a small public dataset from Portuguese secondary schools. The results should not be assumed to generalize to US school districts.

A final grade below 10 is an academic outcome, not a validated measure of individual tutoring need. The dataset also does not establish when every predictor would be available in a real school workflow.

The tutoring-capacity scenario is hypothetical. Different classification thresholds were evaluated on the held-out test set to illustrate how they affect the number of students flagged and the number of at-risk students missed. These thresholds were not validated on an independent dataset and should not be used as established rules for making tutoring decisions.

The models are intended to demonstrate an analytical workflow, not to make automated decisions about individual students.

## Tools and Technologies

Python, Pandas, Numpy, scikit-learn, Matplotlib, Seaborn, and Google. Colab.

## Data Source
Uci Machine Learning Repository: student Performance dataset.

https://archive.ics.uci.edu/dataset/320/student+performance




