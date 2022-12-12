##########################################################
# FLO Customer Segmentation with RFM and KMeans Analysis
##########################################################

# business problem
########################################################
"""
FLO, an online shoe store, wants to segment its customers 
and determine marketing strategies according to these 
segments.

To this end, the behaviors of the customers will be 
defined and groups will be formed according to the 
clusters in these behaviors.
"""

# dataset story
########################################################
"""
The dataset consists of the information obtained from the past shopping behavior 
of customers who made their last purchases from Flo as OmniChannel (both online and offline shopper) 
between 2020 and 2021.

master_id: Unique client number
order_channel: Which channel of the shopping platform is used (Android, ios, Desktop, Mobile)
last_order_channel: The channel where the last purchase was made
first_order_date: The date of the first purchase made by the customer
last_order_date: The date of the customer's last purchase
last_order_date_online: The date of the last purchase made by the customer on the online platform
last_order_date_offline: The date of the last purchase made by the customer on the offline platform
order_num_total_ever_online: The total number of purchases made by the customer on the online platform
order_num_total_ever_offline: Total number of purchases made by the customer offline
customer_value_total_ever_offline: The total price paid by the customer for offline purchases
customer_value_total_ever_online: The total price paid by the customer for their online shopping
interested_in_categories_12: List of categories the customer has purchased from in the last 12 months

"""

# Task 1: Understanding and Preparing the Data

# import libraries
import datetime as dt
from operator import index
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer

# making some adjsutments
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# step 1: reading the dataset
df_ = pd.read_csv('datasets/flo_rfm_analizi_dataset.csv')
df_.columns = [col.lower() for col in df_.columns]
df = df_.copy()
df.head()


# step 2: on the dataset, make the following:
# first 10 observations
# variable names
# descriptive statistics
# null values
# variable types

def check_df(dataframe, head=10):
    print('\n', '#' * 15, 'head'.upper(), '#' * 15)
    print(dataframe.head(head))
    print('\n', '#' * 15, 'tail'.upper(), '#' * 15)
    print(dataframe.tail(head))
    print('\n', '#' * 15, 'variable_names'.upper(), '#' * 15)
    print(dataframe.columns)
    print('\n', '#' * 15, 'nul_values'.upper(), '#' * 15)
    print(dataframe.isnull().sum())
    print('\n', '#' * 15, 'info'.upper(), '#' * 15)
    print(dataframe.info())


check_df(df)

# step 3: omnichannel means that customers shop from both online and offline platforms. 
# Create new variables for each customer's total shopping numbers and spendings.

df['total_shoping_number'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['total_shoping_price'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']
df.head()

# step 4: examine the variable types. Change the type of variables that express date to "date".
df.dtypes  # gives variable types

# first method
# variable_with_date = [col for col in df.columns if 'date' in col]
# for col in variable_with_date:
#     df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')

# second method
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)


# step 5: examine the distribution of the number of customers in the shopping channels,
# the total number of products purchased and the total expenditures.
df.groupby('order_channel').agg({'master_id': len,
                                 'total_shoping_number': sum,
                                 'total_shoping_price': sum})

# step 6: rank the top 10 customers with the highest revenue
df.sort_values(by='total_shoping_price', ascending=False).head(10)

# step 7: list the top 10 customers with the most orders
df.sort_values(by='total_shoping_number', ascending=False).head(10)


# step 8: define a function that shows data preparation process
def data_preparation(dataframe, head=10):
    # check the dataframe
    print('\n', '#' * 15, 'head'.upper(), '#' * 15)
    print(dataframe.head(head))
    print('\n', '#' * 15, 'tail'.upper(), '#' * 15)
    print(dataframe.tail(head))
    print('\n', '#' * 15, 'variable_names'.upper(), '#' * 15)
    print(dataframe.columns)
    print('\n', '#' * 15, 'nul_values'.upper(), '#' * 15)
    print(dataframe.isnull().sum())
    print('\n', '#' * 15, 'info'.upper(), '#' * 15)
    print(dataframe.info())

    # Create new variables for each customer's total shopping numbers and spendings
    dataframe['total_shoping_number'] = dataframe['order_num_total_ever_online'] + dataframe[
        'order_num_total_ever_offline']
    dataframe['total_shoping_price'] = dataframe['customer_value_total_ever_offline'] + dataframe[
        'customer_value_total_ever_online']

    # examine the variable types. change the type of variables that express date to "date"
    print(dataframe.dtypes)  # gives variable types
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)
    print(dataframe.dtypes)

    # rank the top 10 customers with the highest revenue
    print(dataframe.sort_values(by='total_shoping_price', ascending=False).head(head))

    # list the top 10 customers with the most orders
    print(dataframe.sort_values(by='total_shoping_number', ascending=False).head(head))

    # examine the distribution of the number of customers in the shopping channels,
    # the total number of products purchased and the total expenditures.
    print(dataframe.groupby('order_channel').agg({'master_id': len,
                                 'total_shoping_number': sum,
                                 'total_shoping_price': sum}))

df = df_.copy()
new_df = data_preparation(df)

# Task 2: Calculating RFM Metrics

# Step 1: Define Recency, Frequency and Monetary
"""
Recency: This value represents the last time the customer shopped from the company.
Frequency: This value represents the total number of purchases made by the customer, that is, the number of transactions.
Monetary: This value represents the monetary value that the customer leaves to the company.
"""

# Step 2: Calculate Recency, Frequency and Monetary metrics for the customer
df['last_order_date'].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)

(today_date - df['last_order_date']).dt.days    # recency
df['total_shoping_number']                      # frequency
df['total_shoping_price']                       # monetary

# Step 3: Assign your calculated metrics to a variable named rfm
rfm = pd.DataFrame()
rfm['master_id'] = df['master_id']
rfm['recency'] = (today_date - df['last_order_date']).dt.days
rfm['frequency'] = df['total_shoping_number']
rfm['monetary'] = df['total_shoping_price']
rfm.shape
rfm.head()

# Step 4: Change the names of the metrics you created to recency, frequency and monetary
rfm[rfm['monetary'] < 0]  # it returns a null list. therefore, there is no value lower than zero
rfm.describe().T
rfm.head()

# Task 3: Calculating RFM Scores

# Step 1: Convert the Recency, Frequency and Monetary metrics to scores between 1-5 with the help of qcut
pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Step 2: Record these scores as recency_score, frequency_score and monetary_score
rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# Step 3: Express recency_score and frequency_score as a single variable and save it as RF_SCORE
rfm['rf_score'] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
rfm.head()

rfm[rfm['rf_score'] == '55'].head()  # it returns champions class
rfm[rfm['rf_score'] == '11'].head()  # it returns hibernatings
rfm['rf_score'].value_counts().sum()
# Task 4: Defining rfm score as segment

# Step 1: Make segment definitions for the generated RF scores
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'cant_loose_them',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

# Step 2: Convert the scores into segments with the help of the seg_map below
rfm['segment'] = rfm['rf_score'].replace(seg_map, regex=True)
rfm.head(10)

# Task 5: Action time

# Step 1: Examine the recency, frequency and monetary averages of the segments
rfm[['segment', 'recency', 'frequency', 'monetary']].groupby('segment').agg(['mean', 'count'])

# Step 2: With the help of RFM analysis, find the customers in the relevant profile for the 2 cases given below 
# and save the customer IDs as csv.
# Step 2a. FLO includes a new women's shoe brand.
# The product prices of the brand it includes are above the general customer preferences.
# For this reason, it is desired to contact the customers in the profile that will be interested
# in the promotion of the brand and product sales. Those who shop from their loyal customers
# (champions, loyal_customers) and women category are the customers to be contacted specifically.
# Save the id numbers of these customers to the csv file.
rfm = rfm.reset_index()
df1 = df[['master_id', 'interested_in_categories_12']]
df_new = pd.merge(rfm, df1, on='master_id')

loyal_woman_df = df_new.loc[((df_new['segment'] == 'champions') | (df_new['segment'] == 'loyal_customers')) &
                            (df_new['interested_in_categories_12'].str.contains('KADIN')), 'master_id']

# df_new[df_new['interested_in_categories_12'].str.contains('KADIN')]
# df_new[((df_new['segment'] == 'champions') | (df_new['segment'] == 'loyal_customers'))]

loyal_woman_df.to_csv('loyal_woman_df.csv', index=False)

# Step 2b. Nearly 40% discount is planned for Men's and Children's products.
# It is aimed to specifically target customers who are good customers in the past,
# but who have not shopped for a long time, who are interested in the categories related to this discount,
# who should not be lost, who are asleep and new customers. Save the ids of the customers in the appropriate
# profile to the csv file.
men_child_segment = df_new.loc[
    ((df_new['segment'] == 'cant_loose_them') |
     (df_new['segment'] == 'hibernating') |
     (df_new['segment'] == 'new_customers')) &
    ((df_new['interested_in_categories_12'].str.contains('ERKEK')) |
     (df_new['interested_in_categories_12'].str.contains('COCUK'))), 'master_id']

men_child_segment.to_csv('men_child_segment.csv', index=False)

################################################
# Customer segmentation with Kmeans algorithm using flo dataframe
################################################

# defining a new dataframe to cluster with K-Means
rfm_cluster = df_new[['monetary', 'frequency', 'recency']]
rfm_cluster.head()

# showing the distribution of rfm metrics
plt.figure(figsize=(30, 4))
plt.subplot(1, 3, 1)
sns.histplot(rfm_cluster['monetary'])
plt.subplot(1, 3, 2)
sns.histplot(rfm_cluster['frequency'])
plt.subplot(1, 3, 3)
sns.histplot(rfm_cluster['recency'])
plt.show(block=True)

# standardization of the dataframe
mms = MinMaxScaler((0, 1)).fit_transform(rfm_cluster)
rfm_cluster_scaled = pd.DataFrame(mms)
rfm_cluster_scaled.head()


# get the descriprive statistics of new dataframe
rfm_cluster_scaled.describe().T

# determination of the optimum cluster number using elbow method
kmeans = KMeans()
ssd = []
K = range(1, 30)
for k in K:
    kmeans = KMeans(n_clusters=k).fit(rfm_cluster_scaled)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, 'bx-')
plt.xlabel('SSE/SSR/SSD values for different K values')
plt.title('Elbow method for the optimum cluster number')
plt.show(block=True)

# # determining the optimal cluster number
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2,20))
elbow.fit(rfm_cluster_scaled)
elbow.poof()

silhouette_score_list = []
for i in range(2, 10):
    kmeans.fit(rfm_cluster_scaled)
    silhouette_score_list.append(silhouette_score(rfm_cluster_scaled, kmeans.labels_))
print(silhouette_score_list)

# kmeans using 4 clusters and k-means++ initialization
kmeans = KMeans(n_clusters=elbow.elbow_value_, init='k-means++', n_init=10, max_iter=300)
kmeans.fit(rfm_cluster_scaled)
pred = kmeans.predict(rfm_cluster_scaled)

# adding clusters within the dataframe
dataframe = pd.DataFrame(rfm_cluster)
dataframe['cluster'] = pred
dataframe['cluster'] = dataframe['cluster'] + 1
dataframe.head()

# get the average values of the variables according to the cluster
dataframe.groupby('cluster').mean()

# reaching the average scores and counts of recency, frequency, monetary in those classes
dataframe[['cluster', 'recency', 'frequency', 'monetary']].groupby('cluster').agg(['mean', 'count'])
