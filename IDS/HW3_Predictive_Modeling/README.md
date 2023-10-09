# Project 3: Predictive modeling

## Introduction

This project is mainly centred around predicting the time of the policy cancellation on the data provided by Zurich Insurance Company. Predicting the cancellation time of the policy is of extreme importance for the company. Knowing which policies have a high probability of being cancelled soon allows the company to devote more attention to customers of those policies and potentially take some measures to prevent the cancellation. This problem was approached in two ways, by formulating two subproblems: predicting the policy’s duration in months, and predicting whether the policy will be cancelled prematurely, at the moment of its expiration, or be extended. The first one is of a regression nature, while the second is of a classification nature. For those approaches different regression and classification models were tested, for which the performance was evaluated and compared. 

## Repository overview

Reproducable code for project 3 was written in Python programming languange and could be found in the predictive_modeling.ipynb jupyter notebook. To run this code please set up the environment as instructed in the "Environment setup" below. This code is accompanied by a PDF report project3_report.pdf, where the work methodology is explained in detail and the obtained results are presented and discussed.

## Environment setup

- Libraries needed to run this notebook are: pandas, numpy, matplotlib, datetime, itertools and sklearn.

- Run the following commands to set up the environment:
 
  conda create -n project3 python=3.9 pandas matplotlib
  conda activate project3
  conda install -c conda-forge scikit-learn

- Or use project3_env.yml file from the same GitHub repository to setup the enviroment with command:
  conda create -f project3_env.yml
