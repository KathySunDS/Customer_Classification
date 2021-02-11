# Fraud Mitigation Customer Classification
*(The description is masked purposefully following the employer's guideline.)*


#### **Objective**
This fraud mitigation customer classification project is to help a fraud mitigation team of a tech company to evaluate all 1400+ customers based on their network behavior and thus summarize all customers into several sub groups to design targeted fraud mitigation procedures. In this repo I will show the process of how I clean and wrangle the data, build and compare several K-Means cluster models and finally chose one to further deploy on a BI tool for the team to utilize.
#### **Input Description**
1. General Customer Information
   Dataset is generated on a monthly per customer level. The fields included in this analysis are Global Accout Number (GAN), customer name, average outbound traffic score, as well as average monthly invoice.
2. Historic Fraud Data
   Dataset is generated daily to summarize the fraud traffic cost per customer.
   
#### **Libraries Prerequisite**
```
import pandas as pd 
import os
from re import sub                        #Regular Expresion
from sklearn.cluster import KMeans        #Package for building KMeans model
import scipy.stats as st
import matplotlib.pyplot as plt           #Visualization
import collections
```



