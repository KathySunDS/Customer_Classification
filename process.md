*(This markdown serves the purpose of a jupyter notebook since the raw data is protected and couldn't be uploaded)*


### Data Processing
#### IMPORT DATA:
```
os.chdir('\Projects\Customer_Classification_Analysis')
custm_rep = pd.read_csv('customer_reputation_score.csv')
fraud_cost_dmstc = pd.read_csv('zendesk_fraud_cost_dmstc.csv')
fraud_cost_intl = pd.read_csv('zendesk_fraud_cost_intl.csv')
``` 
    
#### DATA CLEANING:
- Most data is calculated to get a monthly average
- In conversation with fraud analysts in the team, I came to conclusion that the average fraud cost per event is a huge identifier in defining how strict the fraud mitigation process should be. So instead of calculating a monthly average fraud cost per customer, I calculated the average fraud cost per event per cutomer.
```
# Adding Fraud Cost group by GAN and Month of the event to calculate average monthly fraud 
custm_dictionary = custm_rep[['gan','customer']].drop_duplicates()
custm_dictionary.columns = ['GAN', 'customer_name']
fraud_cost_dmstc = fraud_cost_dmstc[['f','Fraud Cost','gan']].drop_duplicates()
fraud_cost_dmstc.columns = ['Month', 'Fraud_Cost', 'GAN']
fraud_cost_dmstc['Month'] = fraud_cost_dmstc['Month'].str.slice(0,7)
fraud_cost_dmstc = fraud_cost_dmstc.groupby(['GAN','Month']).sum().reset_index()

fraud_cost_intl = fraud_cost_intl[['Date','Fraud Cost','gan']].drop_duplicates()
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
fraud_credit['Credit_Amount'] = fraud_credit['Credit_Amount'].apply(lambda x: float(sub(r'[^\d.]', '', x)))
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
custm_data = custm_data.fillna(0)           #In this dataset all the null cells are the customers without any fraud events. Replace with 0.
```

#### Attemtpt1: DATA MODELING - KMeans Cluster 1:
Selected all the variables, include count of short calls, count of bad ani calls, minutes of use etc. into this clustering model.
Results: the clusters are highly skewed and too complex to visualize and interpret to the decision makers to utilize.
```
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
```

Results: Counter({0: 1571, 2: 11, 1: 4})
Got really imbalanced grouping. Attempt 1 Failed.

#### Attemtpt2: DATA MODELING - KMeans Cluster 2:
The very imbalanced grouping makes me realize the Euclidean distances are highly effected by the outlier of the dataset.
If we look at the density graph of the invoice varaible, we could see the data is highly skewed to the left with a long tail to the right.

Moreover, to better communicate and represent the idea to the non-technical background decision makers. I decided to finalize a model with 2D data. 
From conversations with stakeholders, I concluded the top two factors defining the fraud mitigation procedure will be the customer's business size as well as their traffic quality.
In order to achieve this, for the first dimension, I picked outbound traffic score: ```outbound traffic score = Long% x Good ANI% x Answered%```and combined it with average fraud ticket cost distirbution cdr to create a new traffic score.
For the other dimension I chose the average invoice amount. 
```
import seaborn as sns
sns.kdeplot(custm_data['last_invoice_amount'])
```
<img src="https://github.com/KathySunDS/Customer_Classification/blob/main/Density.PNG" width="500"> 

*(cut to protect the data privacy)*

Rescale the variables:
Step 1: use feature scaling to standardize the monthly average fraud cost (not nomrally distributed)
```
custm_data['Fraud_Cost_z']=(custm_data['Fraud_Cost']-custm_data['Fraud_Cost'].mean())/custm_data['Fraud_Cost'].std() 
custm_data['fraud_score'] = custm_data['Fraud_Cost_z'].apply(lambda x: (1-st.norm.cdf(x)))

```
One of the key challenge for this project is actually to create a scoring that is both reasonable with the data given and acceptable given the experiences that stakeholders have with each customers in the persepective of fraud mitigation process.
*Method 1*  use outbound score * fraud score. This causing a lot of major customers with huge traffic size have very skewed score due to a huge fraud event that happened once the the past.
*Method 2*  weighted average 30% fraud score plus 70% traffic score. More exceptable during the test run.

```
custm_data['score'] = custm_data['fraud_score'] * 0.3 + custm_data['outbound_score'] * 0.7
custm_data['score_z'] = (custm_data['score']-custm_data['score'].mean())/custm_data['score'].std()
custm_data['score_rescaled'] = custm_data['score_z'].apply(lambda x: st.norm.cdf(x)*100)
custm_data['invoice_z'] =(custm_data['last_invoice_amount']-custm_data['last_invoice_amount'].mean())/custm_data['last_invoice_amount'].std()
custm_data['invoice_rescaled'] = custm_data['invoice_z'].apply(lambda x: st.norm.cdf(x)*100)                        
custm_invoice_score = custm_data[['GAN','score_rescaled','invoice_rescaled']]

wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(custm_invoice_score.iloc[:,1:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.xlabel('n-cluster')
plt.ylabel('wcss')
plt.title('elbow method')
plt.show()
```
<img src="<img src="https://github.com/KathySunDS/Customer_Classification/blob/main/elbow_mthd" width="500"> 

According to the elbow method, it could be debatable whether the optimal number of cluster can be 2 or 6. However, from the stakeholders I understand there are 3 different strategies premade. In order to fit the cluster into those 3 strategies. I deicided to take 6 as the optimal number of clusters.
```
# Using 6 as the optimal number of clusters
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
```

### Summary
<img src="https://github.com/KathySunDS/Customer_Classification/blob/main/Cluster_Plot.png" width="500"> 
Like the above cluster image shows, the customers are now seperated into 6 different groups based on their similarity of the monthly average invoice as well as their position within the company in regard to its traffic score. I would suggest to put the least aggressive fraud mitigation procedure to the group(s) with customers who have high invoice and high traffic score, and on the contrary, put the most aggressive fraud mitigation procedure to group(s) with customers who have the low invoice and low traffic score.
