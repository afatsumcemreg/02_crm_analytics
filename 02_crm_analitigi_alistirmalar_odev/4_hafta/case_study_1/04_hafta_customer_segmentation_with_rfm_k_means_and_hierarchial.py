# Import the libraries
import pandas as pd
from sklearn import cluster
from helpers import eda
import datetime as dt
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# import the dateset
df_ = pd.read_excel('datasets/online_retail_II.xlsx', sheet_name='Year 2009-2010')
df_.columns = [col.lower() for col in df_.columns]
df = df_.copy()
df.head()

# Understanding the data
eda.check_df(df)
cat_cols, num_cols, cat_but_car = eda.grab_col_names(df)
num_cols = [col for col in num_cols if col not in ['InvoiceDate','Customer ID']]

# get the unique number of product
df['description'].nunique()

# get the frequence of each product
df['description'].value_counts()

# get how many each product were sold in total
df.groupby('description').agg({'quantity': 'sum'}).sort_values(by='quantity', ascending=False).head()

# get the unique number of invoices
df['invoice'].nunique()

# create a new variable named 'total_price', hich gives the total price of each product
df['total_price'] = df['quantity'] * df['price']
df.sort_values(by='total_price', ascending=False).head()

# get the total price paid per each invoice
df.groupby('invoice').agg({'total_price': 'sum'}).head().sort_values(by='total_price', ascending=False)

# preparation of the dataset

# get the missing data
df.isnull().sum()

# delete the missing data
df.dropna(inplace=True)

# get the invoices containing the 'C', which refers the returns
df[df['invoice'].str.contains('C', na=False)]

# delete the invoices containing the 'C', which refers the returns and assign it to the dataframe
df = df[~df['invoice'].str.contains('C', na=False)]
df.shape

# calculation of rfm metrics (recency, frequency, and monetary)
    # recency = date of analysis - purchase date of the relevant customer
    # frequency = customer's total number of purchases
    # monetary = total monetary value as a result of the customer's total purchases

# get the last invoice date in the dataset
df['invoicedate'].max() # Timestamp('2010-12-09 20:01:00')

# adding 2 days to the calculated last data
today_date = dt.datetime(2010, 12, 11)

# get the type of the today_data
type(today_date)    # <class 'datetime.datetime'>

# the base of the rfm analysis is a simple pandas operation
# in the 'customer id' breakdown, get a groupby() proces and calculate the r, f, and m values
rfm = df.groupby('customer id').agg({
    'invoicedate': lambda x: (today_date - x.max()).days,
    'invoice': lambda x: x.nunique(),
    'total_price': lambda x: x.sum()})
rfm.head()

# change the column names
rfm.columns = ['recency', 'frequency', 'monetary']

# get the descriptive statistics
rfm.describe().T
rfm.shape

# delete the null values from the monetary values since its value can not be null
rfm = rfm[rfm['monetary'] > 0]
rfm.head()

##########################################
# customer segementation with rfm
##########################################

# calculation the rfm scores

# convert the recency values to scores
rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
# normally, the the best recency value is one, but after converting the best recency score is 5

# convert the monetary values to scores
rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# convert the frequency values to scores
rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
# when rank(method='first') is not used, it gives a ValueError. because, there are many repeated frequence values
# and when ordering small to big, the values in the quantiles were the same. to solve this, rank method was used.
# when this method was used, assign the first seen value to the first class

# after this stage, by using the R and F values, the scores can be formed.
rfm['rfm_score'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)
rfm[rfm['rfm_score'] == '55'].head()   # champions group
rfm[rfm['rfm_score'] == '11'].head()   # hibernating group

# creating the rfm segments

# rfm nomenclatures
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

# to add the nomenclatures in the dataframe, use replace() method
rfm['segments'] = rfm['rfm_score'].replace(seg_map, regex=True)

# reaching the average scores and counts of recency, frequency, monetary in those classes
rfm[['segments', 'recency', 'frequency', 'monetary']].groupby('segments').agg(['mean', 'count'])
rfm.head()

##########################################
# Customer segementation with KMeans
##########################################

# defining a new dataframe to cluster with K-Means
rfm_cluster = rfm[['monetary', 'frequency', 'recency']]
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

silhouette_score_list = []
for i in range(2, 10):
    kmeans.fit(rfm_cluster_scaled)
    silhouette_score_list.append(silhouette_score(rfm_cluster_scaled, kmeans.labels_))
    print(silhouette_score_list)

# kmeans using 4 clusters and k-means++ initialization
kmeans = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300)
kmeans.fit(rfm_cluster_scaled)
pred = kmeans.predict(rfm_cluster_scaled)

dataframe = pd.DataFrame(rfm_cluster)
dataframe['cluster'] = pred
dataframe['cluster'] = dataframe['cluster'] + 1
dataframe.head()

# get the average values of the variables according to the cluster
dataframe.groupby('cluster').mean()

# reaching the average scores and counts of recency, frequency, monetary in those classes
dataframe[['cluster', 'recency', 'frequency', 'monetary']].groupby('cluster').agg(['mean', 'count'])

################################################
# Customer segmentation with hierarchical clustering
################################################

# get the dataset
df_hc = rfm[['monetary', 'frequency', 'recency']]
df_hc.head()

# standardization of the dataset
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df_hc)

# The unifying method divides the observation units into clusters according to the Euclidean distance.
hc_average = linkage(df, 'average')

# creating the dendogram
plt.title('Hierarchical clustering dendogram')
plt.xlabel('Observation units')
plt.ylabel('Distances')
dendrogram(hc_average, leaf_font_size=10)
plt.show(block=True)

# to see less observation, you can write the following codes
plt.title('Hierarchical clustering dendogram')
plt.xlabel('Observation units')
plt.ylabel('Distances')
dendrogram(hc_average, truncate_mode='lastp', p=10, show_contracted=True, leaf_font_size=10)
plt.show(block=True)

# creating the final model

cluster = AgglomerativeClustering(n_clusters=5, linkage='average')
clusters = cluster.fit_predict(df)
df_hc = rfm[['monetary', 'frequency', 'recency']]
df_hc['hc_cluster'] = clusters
df_hc['hc_cluster'] = df_hc['hc_cluster'] + 1
df_hc['hc_cluster'].value_counts()
df_hc.head()

# combining KMeans and Hierarchical scores¨
pd.merge(dataframe, df_hc)

################################################
# Association analysis for the country United Kingdom in online_retail_II dataset
################################################
df = df_.copy()
df.head()
df = df[df['country'] == 'United Kingdom']
df_apriori = df.groupby(['invoice', 'description'])['quantity'].sum().unstack().reset_index().fillna(0).set_index('invoice')
df_apriori.head()
df_apriori.shape

def num(x):
    if x <= 0:
        return 0
    else:
        return 1

new_df = df_apriori.applymap(num)
new_df.head()

rule_fp = fpgrowth(new_df, min_support=0.02, use_colnames=True)
rule_fp.head()
items = apriori(new_df, min_support=0.02, use_colnames=True)
rule = association_rules(items, metric='lift', min_threshold=1)
rule.head()