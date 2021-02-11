# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 11:18:38 2021
This Script is for the Customer Classification Analysis
@author: ksun
"""

# In[IMPORT LIBRARIES]:
import pandas as pd
import os
from re import sub

from sklearn.cluster import KMeans
import scipy.stats as st
import matplotlib.pyplot as plt 


# In[IMPORT DATA]:
os.chdir('C:\\Users\ksun\Documents\Projects\Customer_Classification_Analysis')
custm_rep = pd.read_csv('customer_reputation_score.csv')
fraud_cost_dmstc = pd.read_csv('zendesk_fraud_cost_dmstc.csv')
fraud_cost_intl = pd.read_csv('zendesk_fraud_cost_intl.csv')
fraud_credit = pd.read_csv('billing_credit.csv')
    
    
# In[DATA CLEANING]:
   
# Adding Fraud Cost group by GAN and Month of the event to calculate average monthly fraud 
custm_dictionary = custm_rep[['gan','customer']].drop_duplicates()
custm_dictionary.columns = ['GAN', 'customer_name']
fraud_cost_dmstc = fraud_cost_dmstc[['f','Fraud Cost','gan']]\
    .drop_duplicates()
fraud_cost_dmstc.columns = ['Month', 'Fraud_Cost', 'GAN']
fraud_cost_dmstc['Month'] = fraud_cost_dmstc['Month'].str.slice(0,7)
fraud_cost_dmstc = fraud_cost_dmstc.groupby(['GAN','Month']).sum().reset_index()

fraud_cost_intl = fraud_cost_intl[['Date','Fraud Cost','gan']]\
    .drop_duplicates()
fraud_cost_intl.columns = ['Month', 'Fraud_Cost', 'GAN']
fraud_cost_intl['Month'] = fraud_cost_intl['Month'].str.slice(0,7)
fraud_cost_intl = fraud_cost_intl.groupby(['GAN','Month']).sum().reset_index()

fraud_cost = pd.concat([fraud_cost_dmstc,fraud_cost_intl])
fraud_cost = fraud_cost.groupby(['GAN','Month']).sum().reset_index()
fraud_cost = fraud_cost.groupby(['GAN']).mean().reset_index()

# Convert Datatype of Date and calculate average monthly Credit Amount
fraud_credit = fraud_credit[['Date', 'Global Account Number', 'Amount Credited']].dropna()
fraud_credit.columns = ['Month','GAN','Credit_Amount']
fraud_credit['Month'] = pd.to_datetime(fraud_credit['Month']).apply(lambda x:x.strftime("%Y-%m"))
fraud_credit['Credit_Amount'] = fraud_credit['Credit_Amount']\
    .apply(lambda x: float(sub(r'[^\d.]', '', x)))
fraud_credit = fraud_credit.groupby(['GAN','Month']).sum().reset_index()
fraud_credit = fraud_credit.groupby(['GAN']).mean().reset_index()


# Calculating monthly average for fields in reputation score
custm_rep = custm_rep[['gan', 'stat_month','outbound_calling_spend', 
                       'outbound_calling_mou', 'percent_asr', 'percent_calls_short',
                       'percent_calls_good_ani', 'outbound_score',
                       'count_zendesk_fraud_tickets_rolling_3_months','last_invoice_amount']].dropna()
custm_rep.columns=['GAN', 'Month','outbound_calling_spend', 
                       'outbound_calling_mou', 'percent_asr', 'percent_calls_short',
                       'percent_calls_good_ani', 'outbound_score',
                       'count_zendesk_fraud_tickets_rolling_3_months','last_invoice_amount']
custm_rep['Month'] = custm_rep['Month'].str.slice(0,7)
custm_rep = custm_rep.groupby(['GAN']).mean().reset_index()


# Joining all dataframes to customer reputation score table with the complete list of customer
custm_rep['GAN'] = custm_rep['GAN'].astype(str)
fraud_cost['GAN'] = fraud_cost['GAN'].astype(str)
fraud_credit['GAN'] = fraud_credit['GAN'].astype(str)
custm_data = custm_rep.merge(fraud_cost, on = 'GAN', how = 'left')
custm_data = custm_data.merge(fraud_credit, on = 'GAN', how = 'left')

# replace null value with 0
custm_data = custm_data.fillna(0)


# In[Attemtpt1: DATA MODELING - KMeans Cluster 1]:
"""
wcss = []
#custm_data_value_only = custm_data.iloc[:,1:]

for i in range (1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(custm_data.iloc[:,1:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.xlabel('n-cluster')
plt.ylabel('wcss')
plt.title('elbow method')
plt.show()


# according to Elbow Method the best number of cluster is 3
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
group_result = kmeans.fit_predict(custm_data.iloc[:,1:])

import collections
collections.Counter(group_result)
"""
# Got really imbalanced grouping. Attempt 1 Failed.

# In[Attemtpt2: DATA MODELING - KMeans Cluster 2]:
# Use different inputs
# Step 1: use feature scaling to standardize the monthly average fraud cost (not nomrally distributed)
custm_data['Fraud_Cost_z']=(custm_data['Fraud_Cost']-custm_data['Fraud_Cost'].mean())/custm_data['Fraud_Cost'].std() 
custm_data['fraud_score'] = custm_data['Fraud_Cost_z'].apply(lambda x: (1-st.norm.cdf(x)))
#!!! Need to finalize a good score calculation
#Attempt 1 is to use outbound score * fraud score. Inaccuracy led to high fraud cost but okay traffic custm like Cisco
#Attempt 2 is to use weighted average 60% traffic score and 40% fraud score
custm_data['score'] = custm_data['fraud_score'] * 0.3 + custm_data['outbound_score'] * 0.7

custm_data['score_z'] = (custm_data['score']-custm_data['score'].mean())/custm_data['score'].std()
custm_data['score_rescaled'] = custm_data['score_z'].apply(lambda x: st.norm.cdf(x)*100)
custm_data['invoice_z'] =(custm_data['last_invoice_amount']-custm_data['last_invoice_amount'].mean())/custm_data['last_invoice_amount'].std()
custm_data['invoice_rescaled'] = custm_data['invoice_z'].apply(lambda x: st.norm.cdf(x)*100)                        
custm_invoice_score = custm_data[['GAN','score_rescaled','invoice_rescaled']]

wcss = []
#custm_data_value_only = custm_data.iloc[:,1:]

for i in range (1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(custm_invoice_score.iloc[:,1:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.xlabel('n-cluster')
plt.ylabel('wcss')
plt.title('elbow method')
plt.show()

# according to Elbow Method the best number of cluster is 3
kmeans = KMeans(n_clusters = 6, init = 'k-means++')
group_result = kmeans.fit_predict(custm_invoice_score.iloc[:,1:])
custm_invoice_score['cluster_result'] = pd.Series(group_result, index=custm_invoice_score.index)
plt.scatter(custm_invoice_score[custm_invoice_score['cluster_result']==0]['score_rescaled'],
            custm_invoice_score[custm_invoice_score['cluster_result']==0]['invoice_rescaled'],
            s = 50, c = 'red', label = 'cluster1')
plt.scatter(custm_invoice_score[custm_invoice_score['cluster_result']==1]['score_rescaled'],
            custm_invoice_score[custm_invoice_score['cluster_result']==1]['invoice_rescaled'],
            s = 50, c = 'blue', label = 'cluster2')
plt.scatter(custm_invoice_score[custm_invoice_score['cluster_result']==2]['score_rescaled'],
            custm_invoice_score[custm_invoice_score['cluster_result']==2]['invoice_rescaled'],
            s = 50, c = 'green', label = 'cluster3')
plt.scatter(custm_invoice_score[custm_invoice_score['cluster_result']==3]['score_rescaled'],
            custm_invoice_score[custm_invoice_score['cluster_result']==3]['invoice_rescaled'],
            s = 50, c = 'cyan', label = 'cluster4')
plt.scatter(custm_invoice_score[custm_invoice_score['cluster_result']==4]['score_rescaled'],
            custm_invoice_score[custm_invoice_score['cluster_result']==4]['invoice_rescaled'],
            s = 50, c = 'darkorange', label = 'cluster5')
plt.scatter(custm_invoice_score[custm_invoice_score['cluster_result']==5]['score_rescaled'],
            custm_invoice_score[custm_invoice_score['cluster_result']==5]['invoice_rescaled'],
            s = 50, c = 'violet', label = 'cluster6')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s = 100, 
            c ='yellow', label = 'Centroid')

plt.xlabel('score_rescaled')
plt.ylabel('invoice_rescaled')
plt.title('Customer Grouping')
plt.legend()


custm_data = custm_data.merge(custm_invoice_score[['GAN', 'cluster_result']], on = 'GAN', how = 'left')
custm_data = custm_data.merge(custm_dictionary, on ='GAN', how='left')

