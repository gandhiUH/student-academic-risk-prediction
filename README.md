
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

## Methodology
This project includes:
- Exploratory data analysis of academic performance and student characteristics 
- Feature engineering, including changes between first and second period grades
- Data processing using scikit-learn pipelines
- Model development using Logistic Regression and Random Forest
- Five-fold stratified cross-validation and held-out test evaluation
- Error analysis of students incorrectly classified by the models
- Hypothetical tutoring-capacity analysis using different classification thresholds


## Exploratory Data Analysis

### Previous Failures
Students with more previous class failures generally had lower final mathematics grades. The median final grade decreased from approximately 11 among students with no previous failures to approximately 7 among students with three previous failures. This pattern suggests that previous failures may be a useful feature for academic-risk prediction. However, low final grades were also observed among students with no previous failures, indicating that this feature alone is insufficient to identify all at-risk students.
|   failures |   count |   mean |   median |   std |
|-----------:|--------:|-------:|---------:|------:|
|          0 |     312 |  11.25 |       11 |  4.17 |
|          1 |      50 |   8.12 |        9 |  4.71 |
|          2 |      17 |   6.24 |        8 |  4.84 |
|          3 |      16 |   5.69 |        7 |  4.19 |
<p align='left'>
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/9f98a92c-4bd5-4bc9-a478-bf7d7b5448d8" />
</p>

### Study Time Vs. Final Grades
|   studytime |   academic_risk |
|------------:|----------------:|
|           1 |           35.24 |
|           2 |           35.35 |
|           3 |           24.62 |
|           4 |           25.93 |
### Correlation Between First-period grade (G1) , Second-period grade (G2), Final grade (G3)
|    |       G1 |       G2 |       G3 |
|:---|---------:|---------:|---------:|
| G1 | 1        | 0.852118 | 0.801468 |
| G2 | 0.852118 | 1        | 0.904868 |
| G3 | 0.801468 | 0.904868 | 1        |

### Relationship between First-period grades and Final grade
<p align='left'>
<img width="500" height="370" alt="image" src="https://github.com/user-attachments/assets/a50a832a-186c-4bd8-b207-6d14729e53b2" />
</p>
### Relationship between Second-period and Final grade
<p align='left'>
  <img width="500" height="370" alt="image" src="https://github.com/user-attachments/assets/7994f519-902e-46c9-a88a-5279ead1249a" />
</p>

### Distribution of Grade Change
<p align ='left'>
<img width="500" height="370" alt="image" src="https://github.com/user-attachments/assets/9c02dead-4f32-4e39-b4cc-c2368c771393" />
</p>p>

### Grade Change by Academic Risk
<p align='left'>
<img width="500" height="370" alt="image" src="https://github.com/user-attachments/assets/e9b00aec-aac4-44f6-ab28-e5db5f5982a2" />
</p>

### Relationship between Study Time and At-Risk Students
|   studytime |   total_students |   at_risk_students |   risk_percentage |
|------------:|-----------------:|-------------------:|------------------:|
|           1 |              105 |                 37 |           35.2381 |
|           2 |              198 |                 70 |           35.3535 |
|           3 |               65 |                 16 |           24.6154 |
|           4 |               27 |                  7 |           25.9259 |

### Relationship between Zero Final Grade and Studytime
   studytime |   False |   True |
|------------:|--------:|-------:|
|           1 |    87.6 |   12.4 |
|           2 |    91.9 |    8.1 |
|           3 |    90.8 |    9.2 |
|           4 |    88.9 |   11.1 |

## Feature Engineering

###  Class Distribution by Stratified Split

```text
				Training  Testing
academic_risk 0    212		53
academic_risk 1    104		26
```


Feature-cols= df\['G1', 'failures', 'studytime', 'schoolsup', 'famsup', 'higher', 'health', 'traveltime'\]
y= df\['academic_risk'\]
## ML Models
### 1. Initial Model Random Forest with First-period grades

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| 0            |       0.839 |    0.887 |      0.862 |     53    |
| 1            |       0.739 |    0.654 |      0.694 |     26    |
| accuracy     |       0.81  |    0.81  |      0.81  |      0.81 |
| macro avg    |       0.789 |    0.77  |      0.778 |     79    |
| weighted avg |       0.806 |    0.81  |      0.807 |     79    |
        

### Feature Importance in Random Forest Pipeline
<p align='left'>
<img width="400" height="370" alt="image" src="https://github.com/user-attachments/assets/6eb8ff51-e9c6-4191-b9f5-b9d4bd8623eb" />
</p>

### 2. 5-Fold Cross-Validation RF 
#### Performance

|    | Metric    |   Mean |   Std |
|---:|:----------|-------:|------:|
|  0 | accuracy  |  0.794 | 0.026 |
|  1 | precision |  0.693 | 0.064 |
|  2 | recall    |  0.704 | 0.12  |
|  3 | f1        |  0.689 | 0.05  |

### 3. Logistic Regression 

|    | Metric    |   Mean |   Std |
|---:|:----------|-------:|------:|
|  0 | accuracy  |  0.817 | 0.058 |
|  1 | precision |  0.691 | 0.096 |
|  2 | recall    |  0.847 | 0.069 |
|  3 | f1        |  0.756 | 0.063 |
	
Two prediction models were developed:
## Model A: First-period academic risk prediction using Logistic Regression.
df\['grade_change\] = df\['G2'\]-df\['G1'\]
features_cols2 = \['G1', 'failures', 'studytime', 'schoolsup', 'famsup', 'higher', 'health', 'traveltime', 'G2', 'grade_change']
Training set shape - Model2: (316, 10)
Testing set shape - Model2: (79, 10)
## Model B: Updated academic risk prediction using Random Forest and additional second-period academic information.

|    | Metric    |   Mean |   Std |
|---:|:----------|-------:|------:|
|  0 | accuracy  |  0.902 | 0.025 |
|  1 | precision |  0.832 | 0.047 |
|  2 | recall    |  0.886 | 0.077 |
|  3 | f1        |  0.855 | 0.04  |


This project also examines how classification thresholds affect the number of students flagged for academic-risk review and compares these counts with hypothetical tutoring capacity.
### Logistic Regressing with Feature_cols2 
|    | Metric    |   Mean |   Std |
|---:|:----------|-------:|------:|
|  0 | accuracy  |  0.889 | 0.029 |
|  1 | precision |  0.795 | 0.065 |
|  2 | recall    |  0.905 | 0.052 |
|  3 | f1        |  0.844 | 0.039 |

### Random Forest with Feature_cols3
Feature_cols3 = df\[featurecols+ 'G2'\], Feature_cols3 does not have 'grade_change' feature
|    | Metric    |   Mean |   Std |
|---:|:----------|-------:|------:|
|  0 | accuracy  |  0.896 | 0.027 |
|  1 | precision |  0.829 | 0.046 |
|  2 | recall    |  0.867 | 0.092 |
|  3 | f1        |  0.844 | 0.045 |

## Selected Models for Academic-Risk Prediction:
1. Model A - Logistic Regression with Feature_cols1 
2. Model B - Random Forest with Feature_cols2 

## Model Performance
|Model	             | Accuracy	| Precision |	Recall | F1 Score |
|---|---|---|---|---|
|Logistic Regression| 0.848	  |  0.733	  | 0.846	 | 0.786|
|Random Forest	     | 0.924	  |  0.955	  | 0.808	 | 0.875|

Model B achieved higher accuracy, precision, and F1-score on the held-out test set. However, Model A achieved a higher recall, identifying 22 of the 26 students who eventually received final grades below 10, compared with 21 students identified by Model B. These results highlight the importance of considering false negatives when evaluating models intended to support early academic-risk identification.

<p align='left'>
<img width="500" height="370" alt="image" src="https://github.com/user-attachments/assets/52f85625-3973-45eb-a045-be274669bc49" />
</p>
## Education Resource-Planning Analysis
A hypothetical tutoring-capacity scenario was developed for the 79-student test cohort, assuming 20 available tutoring places.

At a classification threshold of 0.5:

Model A flagged 30 students, exceeding the hypothetical capacity by 10 places. Model B flagged 22 students, exceeding the hypothetical capacity by two places.

On the held-out test set, Model B achieved higher accuracy, precision, and F1-score, while Model A achieved slightly higher recall at the default classification threshold of 0.50. Model A correctly identified 22 of the 26 students who eventually received final grades below 10, compared with 21 identified by Model B. Although Model B identified three at-risk students missed by Model A, it also missed four at-risk students whom Model A had correctly identified. These results demonstrate that updated predictions can identify additional students but should not automatically replace earlier academic-support assessments.
<p align='left'>
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
<p align="left">
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




