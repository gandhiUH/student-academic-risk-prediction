
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

Model B achieved higher accuracy, precision, and F1-score on the held-out test set. However, Model A achieved higher recall, identifying 22 of the 26 students who eventually received final grades below 10, compared with 21 identified by Model B. These results highlight the importance of considering false negatives when evaluating models intended to support early academic-risk identification.

