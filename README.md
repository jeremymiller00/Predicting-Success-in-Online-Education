# Predicting Success in Online Education

==============================

## Outline
* Problem Statement
* Results
* Data
* Modeling process
* Evaluation
* Next Steps

## Problem Statement

Non-completion rates are higher and more varied for online college courses than for traditional “classroom” courses. Identifying students at risk for failing or dropping is the first step towards intervention.

The goal of this project is to use behavior and demographics to predict if students will successfully complete the course and flag students for intervention.

## Results

A random forest classifier provides a true positive rate of ~0.75. This means that about 75% of the students who will actually fail are predicted to do so by the mdoel. This is significantly better than the baseline true positive rate of ~0.58.

## Data


## Modeling process


## Evaluation
Models were evaluated prirmarily using the ROC AUC score and the true positive rate (recall). ROC AUC was chosen because ot provides a clear general of sense of how a binary classification perform thoughout the range of prediction thresholds. True prositive rate was chosen because it directly accounts for the proportion of the false negative predictions.  

plots

Though the strength of the predictions relies on the use of many feautres, the following feature were determined to contribute most to predicitions of non-completion:


## Next Steps