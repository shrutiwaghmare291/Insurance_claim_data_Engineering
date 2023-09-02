#!/usr/bin/env python
# coding: utf-8

# # HEALTH INSURANCE

# ## Problem Statement
# 
# **A key challenge for the insurance industry is to charge each customer an appropriate premium for the risk they represent. The ability to predict a correct claim amount has a significant impact on insurer's management decisions and financial statements. Predicting the cost of claims in an insurance company is a real-life problem that needs to be solved in a more accurate and automated way. Several factors determine the cost of claims based on health factors like BMI, age, smoker, health conditions and others. Insurance companies apply numerous techniques for analyzing and predicting health insurance costs.**

# ## Data Definition
# 
# **age** : Age of the policyholder (Numeric)
# 
# **sex:** Gender of policyholder (Categoric)
# 
# **weight:** Weight of the policyholder (Numeric)
# 
# **bmi**: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight (Numeric)
# 
# **no_of_dependents:** Number of dependent persons on the policyholder (Numeric)
# 
# **smoker:** Indicates policyholder is a smoker or a non-smoker (non-smoker=0;smoker=1) (Categoric)
# 
# **claim:** The amount claimed by the policyholder (Numeric)
# 
# **bloodpressure:** Bloodpressure reading of policyholder (Numeric)
# 
# **diabetes:** Indicates policyholder suffers from diabetes or not (non-diabetic=0; diabetic=1) (Categoric)
# 
# **regular_ex:** A policyholder regularly excercises or not (no-excercise=0; excercise=1) (Categoric)
# 
# **job_title:** Job profile of the policyholder (Categoric)
# 
# **city:** The city in which the policyholder resides (Categoric)
# 
# **hereditary_diseases:**  A policyholder suffering from a hereditary diseases or not (Categoric)

# <a id='import_lib'></a>
# ## 1. Import Libraries

# In[1]:


# supress warnings 
from warnings import filterwarnings
filterwarnings('ignore')

# 'Os' module provides functions for interacting with the operating system 
import os

# 'Pandas' is used for data manipulation and analysis
import pandas as pd 

# 'Numpy' is used for mathematical operations on large, multi-dimensional arrays and matrices
import numpy as np

# 'Matplotlib' is a data visualization library for 2D and 3D plots, built on numpy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 'Seaborn' is based on matplotlib; used for plotting statistical graphics
import seaborn as sns

# 'Scikit-learn' (sklearn) emphasizes various regression, classification and clustering algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso,ElasticNet
from sklearn.model_selection import GridSearchCV

# 'Statsmodels' is used to build and analyze various statistical models
import statsmodels
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.tools.eval_measures import rmse
from statsmodels.compat import lzip
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF

# 'SciPy' is used to perform scientific computations
from scipy.stats import f_oneway
from scipy.stats import jarque_bera
from scipy import stats


# <a id='set_options'></a>
# ## 2. Set Options

# In[2]:


# display all columns of the dataframe
pd.options.display.max_columns = None

# display all rows of the dataframe
pd.options.display.max_rows = None

# return an output value upto 6 decimals
pd.options.display.float_format = '{:.6f}'.format


# <a id='Read_Data'></a>
# ## 3. Read Data

# In[3]:


# read csv file using pandas
df_insurance = pd.read_csv("healthinsurance.csv")

# display the top 5 rows of the dataframe
df_insurance.head()


# <a id='data_preparation'></a>
# ## 4. Data Analysis and Preparation
# 

# <a id='Data_Understanding'></a>
# ### 4.1 Understand the Dataset

# <a id='Data_Shape'></a>
# ### 4.1.1 Data Dimension

# In[4]:


df_insurance.shape


# ### 4.1.2 Data Types
# 

# **1. Check data types**

# In[5]:


df_insurance.dtypes


# **2. Change the incorrect data types**

# In[6]:


# use .astype() to change the data type
# convert numerical variables to categorical  

# convert numeric variable 'smoker' to object (categorical) variable
df_insurance.smoker = df_insurance.smoker.astype('object')

# convert numeric variable 'diabetes' to object (categorical) variable
df_insurance.diabetes = df_insurance.diabetes.astype('object')

# convert 'regular_ex' variable diabetes to object (categorical) variable
df_insurance.regular_ex = df_insurance.regular_ex.astype('object')


# **3. Recheck the data types after the conversion**

# In[7]:


# recheck the data types using .dtypes
df_insurance.dtypes


# Note the data types are now as per the data definition. Now we can proceed with the analysis.

# <a id='Summary_Statistics'></a>
# ### 4.1.3 Summary Statistics

# In[8]:


# describe the numerical data
df_insurance.describe()


# **2. For categorical features, we use .describe(include=object)**

# In[9]:


# describe the categorical data
# include=object: selects the categorical features
df_insurance.describe(include = object)


# <a id='Missing_Values'></a>
# ### 4.1.4 Missing Values

# In[10]:


# Find total missing values
Total = df_insurance.isnull().sum().sort_values(ascending=False) 
Total


# In[11]:


# Find percent missing values
Percent = (df_insurance.isnull().sum()*100/df_insurance.isnull().count()).sort_values(ascending=False)   

# concat the 'Total' and 'Percent' columns using 'concat' function
missing_data = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])

# print the missing data
missing_data


# The missing values are present in the data for the `age` and `bmi` variables. There are 396 (2.6%) missing values for the variable `age` and 956 (6.4%) missing values for the variable `bmi`

# ### Visualize Missing Values using Heatmap

# In[12]:


plt.figure(figsize=(15, 8))
sns.heatmap(df_insurance.isnull(), cbar=False)
plt.show()


# ### Deal with Missing Values
# 
# Discuss - How to deal with missing data?<br>
# 

# In[13]:


# check the average age for male and female

df_insurance['age'].groupby(df_insurance['sex'], axis=0).mean()


# The average age for the male and female is nearly the same. We will fill in missing values with the mean age of the policyholder.

# In[14]:


# fill the missing values with the mean value of 'age' using 'fillna()'
df_insurance['age'].fillna(df_insurance['age'].mean(), inplace=True)


# Replace missing values by mean for the BMI.

# In[15]:


# fill the missing values with the mean value of 'bmi' using 'fillna()'

df_insurance['bmi'].fillna(df_insurance['bmi'].mean(), inplace=True)


# We have seen that the the minimum bloodpressure is 0, which is absurd. It implies that these are missing values. 
# Let us replace these missing values with the median value.

# In[16]:


# calculate the median of the bloodpressure using 'median()''
median_bloodpressure = df_insurance['bloodpressure'].median()

# replace zero values by median using 'replace()'
df_insurance['bloodpressure'] = df_insurance['bloodpressure'].replace(0,median_bloodpressure) 


# Recheck the summary statistics to confirm the missing value treatment for the variable 'bloodpressure'.

# In[17]:


# obtain the summary statistics of numeric variables using 'describe()'
df_insurance.describe()


# To confirm the data is valid, observe the minimum and maximum value of the variable `bloodpressure` is 40

# Let's view the missing value plot once again to see if the missing values have been imputed.

# In[18]:


# set the figure size
plt.figure(figsize=(15, 8))
sns.heatmap(df_insurance.isnull(), cbar=False)
plt.show()


# Now, we obtain the dataset with no missing values.

# <a id='correlation'></a>
# ### 4.1.5 Correlation

# In[19]:



df_numeric_features = df_insurance.select_dtypes(include=np.number)
df_numeric_features.columns


# In[20]:


# generate the correlation matrix
corr =  df_numeric_features.corr()


# In[21]:


# set the figure size
plt.figure(figsize=(15, 8))

sns.heatmap(corr, cmap='YlGnBu', vmax=1.0, vmin=-1.0, annot = True, annot_kws={"size": 15}, )
plt.title('Correlation between numeric features')
plt.show()


# <a id='categorical'></a>
# ### 4.1.6 Analyze Categorical Variables
# 
# 

# In[22]:


df_insurance.describe(include=object)


# There are 6 categorical variables. From the output we see that the variable cities has most number of categories. There are 91 cities in the data, of which NewOrleans occurs highes number of times.
# 
# Let us visualize the variables. However, we shall exculde the variable `city` from it.

# In[23]:


# create a list of all categorical variables
df_categoric_features = df_insurance.select_dtypes(include='object').drop(['city'], axis=1)

# plot the count distribution for each categorical variable 
# 'figsize' sets the figure size
fig, ax = plt.subplots(3, 2, figsize=(25, 20))

# plot a count plot for all the categorical variables
for variable, subplot in zip(df_categoric_features, ax.flatten()):
    
    countplot = sns.countplot(y=df_insurance[variable], ax=subplot )
       
    countplot.set_ylabel(variable, fontsize = 30)

plt.tight_layout()   
plt.show()


# Now consider the variable `city`.

# In[24]:


# set the figure size
plt.figure(figsize=(15, 15))
countplot = sns.countplot(y=df_insurance['city'], orient="h")
countplot.set_ylabel('City', fontsize = 30)
plt.show()


# ### 4.1.7 Analyze Relationship Between Target and Categorical Variables

# In[25]:


# plot the boxplot for each categorical variable 
# create subplots using subplots()
# 6 subplots in 3 rows and 2 columns
# 'figsize' sets the figure size
fig, ax = plt.subplots(3, 2, figsize=(25, 15))

# plot a boxplot for all the categorical variables 
for variable, subplot in zip(df_categoric_features, ax.flatten()):
    
    # x: variable on x-axis
    # y: variable in y-axis
    # data: dataframe to be used
    # ax: specifies the axes object to draw the plot onto
    boxplt = sns.boxplot(x=variable, y='claim', data=df_insurance, ax=subplot)
    
    # set the x-axis labels 
    # fontsize = 30: sets the font size to 30
    boxplt.set_xlabel(variable, fontsize = 30)

# avoid overlapping of the plots using tight_layout()    
plt.tight_layout()   

# display the plot
plt.show() 


# Since the variable `city` has 91 categories, we shall plot it separately.

# In[26]:


# set the figure size
plt.figure(figsize=(15, 8))

# plot the boxplot for categorical variable 'city'

ax = sns.boxplot(x=df_insurance["city"], y=df_insurance['claim'], data=df_insurance)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize = 10)
plt.show()


# <a id='Feature_Engineering'></a>
# ### 4.1.8 Feature Engineering
# 
# Create a new feature 'region' by combining the cities.

# **There are 91 unique cities. We will divide these cities into North-East, West, Mid-West, and South regions.**

# Let's create a new variable region. We will replace the original variable `city` with it.

# In[27]:


# create a region column and combine the north-east cities
df_insurance['region'] = df_insurance['city'].replace(['NewYork', 'Boston', 'Phildelphia', 'Pittsburg', 'Buffalo',
                                                       'AtlanticCity','Portland', 'Cambridge', 'Hartford', 
                                                       'Springfield', 'Syracuse', 'Baltimore', 'York', 'Trenton',
                                                       'Warwick', 'WashingtonDC', 'Providence', 'Harrisburg',
                                                       'Newport', 'Stamford', 'Worcester'],
                                                      'North-East')


# In[28]:


# combine all the southern cities into the 'region' column
df_insurance['region'] = df_insurance['region'].replace(['Atlanta', 'Brimingham', 'Charleston', 'Charlotte',
                                                         'Louisville', 'Memphis', 'Nashville', 'NewOrleans',
                                                         'Raleigh', 'Houston', 'Georgia', 'Oklahoma', 'Orlando',
                                                         'Macon', 'Huntsville', 'Knoxville', 'Florence', 'Miami',
                                                         'Tampa', 'PanamaCity', 'Kingsport', 'Marshall'],
                                                         'Southern')


# In[29]:


# combine all the mid-west cities into the 'region' column
df_insurance['region'] = df_insurance['region'].replace(['Mandan', 'Waterloo', 'IowaCity', 'Columbia',
                                                         'Indianapolis', 'Cincinnati', 'Bloomington', 'Salina',
                                                         'KanasCity', 'Brookings', 'Minot', 'Chicago', 'Lincoln',
                                                         'FallsCity', 'GrandForks', 'Fargo', 'Cleveland', 
                                                         'Canton', 'Columbus', 'Rochester', 'Minneapolis', 
                                                         'JeffersonCity', 'Escabana','Youngstown'],
                                                         'Mid-West')


# In[30]:


# combine all the western cities into the 'region' column
df_insurance['region'] = df_insurance['region'].replace(['SantaRosa', 'Eureka', 'SanFrancisco', 'SanJose',
                                                         'LosAngeles', 'Oxnard', 'SanDeigo', 'Oceanside', 
                                                         'Carlsbad', 'Montrose', 'Prescott', 'Fresno', 'Reno',
                                                         'LasVegas', 'Tucson', 'SanLuis', 'Denver', 'Kingman',
                                                         'Bakersfield', 'Mexicali', 'SilverCity', 'Pheonix',
                                                         'SantaFe', 'Lovelock'],
                                                         'West')


# In[31]:


# check the unique values of the region using 'unique()'
df_insurance['region'].unique()


# In[32]:


df_insurance['region'].value_counts()


# In[33]:


# drop the 'city' variable from the dataset using drop()
df_insurance = df_insurance.drop(['city'], axis=1)


# Check whether the new variable added into the data frame or not.

# In[34]:


# display the top 5 rows of the dataframe
df_insurance.head()


# #### Analyze relationship between region and claim variable

# In[35]:


# set figure size
plt.figure(figsize=(15,8))
ax = sns.boxplot(x="region", y="claim", data=df_insurance)
ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=90)
plt.show()


# The plot shows that there is not much significant difference in the variance of the insurance claim across the regions.

# <a id='outliers'></a>
# ### 4.1.9 Discover Outliers

# **1. Plot boxplot for numerical data**

# In[36]:


plt.rcParams['figure.figsize']=(15,8)
df_numeric_features.boxplot(column=['bmi', 'weight','no_of_dependents', 'bloodpressure', 'age'])
plt.show()


# **2. Note the variables for which outliers are present**

# From the above plot, we notice that for the variable 'bmi' and 'bloodpressure' contain outliers

# **3. Remove outliers by IQR method**

# In[37]:


# calculate interquartile range 

# compute the first quartile using quantile(0.25)
# use .drop() to drop the target variable 
Q1 = df_numeric_features.drop(['claim'], axis=1).quantile(0.25)

# compute the first quartile using quantile(0.75)
# use .drop() to drop the target variable 
Q3 = df_numeric_features.drop(['claim'], axis=1).quantile(0.75)

# calculate of interquartile range 
IQR = Q3 - Q1
print(IQR)


# In[38]:


# filter out the outlier values
# ~ : selects all rows which do not satisfy the condition
# |: bitwise operator OR in python
# any() : returns whether any element is True over the columns
# axis : "1" indicates columns should be altered (use "0" for 'index')
df_insurance = df_insurance[~((df_insurance < (Q1 - 1.5 * IQR)) | (df_insurance > (Q3 + 1.5 * IQR))).any(axis=1)]


# A simple way to know whether the outliers have been removed or not is to check the dimensions of the data. 

# In[39]:


# check the shape of data using shape
df_insurance.shape


# There is a reduction in the number of rows(from 15000 to 14723).

# **4. Plot boxplot to recheck for outliers**

# In[40]:


plt.rcParams['figure.figsize']=(15,8)

df_insurance.boxplot(column=['bmi', 'weight','no_of_dependents', 'bloodpressure', 'age'])
plt.show()


# Observing the range of the boxplot, we say that the outliers are removed from the original data. The new 'outliers' that you see are moderate outliers that lie within the min/max range before removing the actual outliers

# ### 4.1.10 Recheck the Correlation

# In[41]:


df_numeric_features = df_insurance.select_dtypes(include=np.number)
df_numeric_features.columns


# In[42]:



corr =  df_numeric_features.corr()

# print the correlation matrix
corr


# **3. Pass the correlation matrix to the heatmap() function of the seaborn library to plot the heatmap of the correlation matrix**

# In[43]:


plt.figure(figsize=(15, 8))

sns.heatmap(corr, cmap='YlGnBu', vmax=1.0, vmin=-1.0, annot = True, annot_kws={"size": 15})
plt.title('Correlation between numeric features')
plt.show()


# **4. Check multicollinearity using VIF**

# In[44]:


df_numeric_X =  df_numeric_features.drop('claim',axis=1)
df_numeric_X = sm.add_constant(df_numeric_X)
vif_value = [VIF(df_numeric_X.values, i) for i in range(df_numeric_X.shape[1])]
pd.DataFrame(vif_value, columns=['VIF_Value'], index=df_numeric_X.columns).sort_values('VIF_Value', ascending=False)


# VIF Values indicate that there is no multi collinearity

# <a id='Data_Preparation'></a>
# ## 4.2 Prepare the Data

# <a id='Normality'></a>
# ### 4.2.1 Check for Normality

# As per the assumptions of linear regression, residuals (actual values - predicted values) should be normally distributed. If the target variable is normally distributed then the residuals are also normally distributed, thus we check the normality only for target variable.

# **1. Plot a histogram and also perform the Jarque-Bera test**

# In[45]:


df_insurance.claim.hist()
plt.show()


# Let us perform the Jarque-Bera test to check the normality of the target variable.

# In[46]:


# normality test using jarque_bera()
#The null and alternate hypothesis of Jarque-Bera test are as follows: <br>
#    H0: The data is normally distributed
#    H1: The data is not normally distributed
# the test returns the the test statistics and the p-value of the test
stat, p = jarque_bera(df_insurance["claim"])
print('Statistics=%.3f, p-value=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('The data is normally distributed (fail to reject H0)')
else:
    print('The data is not normally distributed (reject H0)')


# **2. If the data is not normally distributed, use log transformation to get near normally distributed data**

# In[47]:


# log transformation for normality using np.log()
df_insurance['log_claim'] = np.log(df_insurance['claim'])
df_insurance.head()


# **3. Recheck for normality by plotting histogram and performing Jarque-Bera test**
# 

# In[48]:


# recheck for normality 
df_insurance.log_claim.hist()
plt.show()


# The variable claim is near normally distributed. However we again confirm by Jarque Bera test

# Let us perform Jarque Bera test

# In[49]:



statn, pv = jarque_bera(df_insurance['log_claim'])
print('Statistics=%.3f, p-value=%.3f' % (stat, p))
alpha = 0.05
if p > alpha:
    print('The data is normally distributed (fail to reject H0)')
else:
    print('The data is not normally distributed (reject H0)')


# It can be visually seen that the data has near-normal distribution, but Jarque Bera test does not support the claim.
# Note that in reality it might be very tough for your data to adhere to all assumptions your algorithm needs

# 
# ### 4.2.2 One-Way Anova 

# Perform One-Way ANOVA to compare two means from two independent (unrelated) groups. For example, we apply ANOVA to see whether the mean of claim is significantly different across gender
# 
# The null and alternate hypothesis of one-way anova are as follows:
# 
#     H0: Population means all are equal
#     H1: Population means are not all equal

# #### One Way Anova for 'Sex' on 'Claim'

# In[50]:


# perform one way anova for sex on claim using f_oneway()
f_oneway(df_insurance['claim'][df_insurance['sex'] == 'male'], 
             df_insurance['claim'][df_insurance['sex'] == 'female'])


# The F-statistic = 68.99 and the p-value < 0.05, which indicates that there is a significant difference in the mean of the insurance claim across gender. We may consider building separate models for each gender. However, in this example we go ahead and build a single model for both genders.

# <a id='dummy'></a>
# ### 4.2.3 Dummy Encoding of Categorical Variables

# **1. Filter numerical and categorical variables**

# In[51]:


# filter the numerical features in the dataset using select_dtypes()
df_numeric_features = df_insurance.select_dtypes(include=np.number)
df_numeric_features.columns


# In[52]:


# filter the categorical features in the dataset using select_dtypes()
df_categoric_features = df_insurance.select_dtypes(include=[np.object])
df_categoric_features.columns


# **2. Dummy encode the catergorical variables**

# In[53]:


# create data frame with only categorical variables that have been encoded

# for all categoric variables create dummy variables
for col in df_categoric_features.columns.values:
    
    dummy_encoded_variables = pd.get_dummies(df_categoric_features[col], prefix=col, drop_first=True)
    
    # concatenate the categoric features with dummy variables using concat()
    df_categoric_features = pd.concat([df_categoric_features, dummy_encoded_variables],axis=1)
    
    # drop the orginal categorical variable from the dataframe
    df_categoric_features.drop([col], axis=1, inplace=True)


# **3. Concatenate numerical and dummy encoded categorical variables**

# In[54]:


# concatenate the numerical and dummy encoded categorical variables using concat()
# axis=1: specifies that the concatenation is column wise
df_insurance_dummy = pd.concat([df_numeric_features, df_categoric_features], axis=1)

# display data with dummy variables
df_insurance_dummy.head()


# <a id='LinearRegression'></a>
# ## 5. Linear Regression (OLS)

# <a id='withLog'></a>
# ### 5.1 Multiple Linear Regression - Full Model - with Log Transformed Dependent Variable (OLS)

# **1. Split the data into training and test sets**

# In[55]:


# add the intercept column to the dataset
df_insurance_dummy = sm.add_constant(df_insurance_dummy)

# Define X and Y
X = df_insurance_dummy.drop(['claim','log_claim'], axis=1)
y = df_insurance_dummy[['log_claim','claim']]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# check the dimensions of the train & test subset for 
print("The shape of X_train is:",X_train.shape)
print("The shape of X_test is:",X_test.shape)
print("The shape of y_train is:",y_train.shape)
print("The shape of y_test is:",y_test.shape)


# **2. Build model using sm.OLS().fit()**

# In[56]:


# build a full model using OLS()
# consider the log of claim 
linreg_full_model_withlog = sm.OLS(y_train["log_claim"], X_train).fit()

print(linreg_full_model_withlog.summary())


# **3. Predict the values using test set**

# In[57]:


# predict the 'log_claim' using predict()
linreg_full_model_withlog_predictions = linreg_full_model_withlog.predict(X_test)


# In[58]:


# Note that the predicted values are log transformed claim. 
# In order to get claim values, we take the antilog of these predicted values by using the function np.exp()

predicted_claim = np.exp(linreg_full_model_withlog_predictions)


# extract the 'claim' values from the test data
actual_claim = y_test['claim']


# **4. Compute accuracy measures**
# 
# Now we calculate accuray measures like Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[59]:


# calculate rmse using rmse()
linreg_full_model_withlog_rmse = rmse(actual_claim, predicted_claim)

# calculate R-squared using rsquared
linreg_full_model_withlog_rsquared = linreg_full_model_withlog.rsquared
#linreg_full_model_withlog_test_rsquared = r2_score(actual_claim, predicted_claim)  #if we want test-RSqaured


# calculate Adjusted R-Squared using rsquared_adj
linreg_full_model_withlog_rsquared_adj = linreg_full_model_withlog.rsquared_adj 



# **5. Tabulate the results**

# In[60]:


# create dataframe 'score_card'
# columns: specifies the columns to be selected
score_card = pd.DataFrame(columns=['Model_Name', 'R-Squared', 'Adj. R-Squared', 'RMSE'])
#score_card = pd.DataFrame(columns=['Model_Name', 'R-Squared', 'Adj. R-Squared', 'Test_R-Squared', 'RMSE'])  # for Test-RSquared

# print the score card
score_card


# In[61]:


# compile the required information
linreg_full_model_withlog_metrics = pd.Series({
                     'Model_Name': "Linreg full model with log of target variable",
                     'RMSE':linreg_full_model_withlog_rmse,
                     'R-Squared': linreg_full_model_withlog_rsquared,
                     'Adj. R-Squared': linreg_full_model_withlog_rsquared_adj 
                  # 'Test_R-Squared': linreg_full_model_withlog_test_rsquared    # for Test-RSquared
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
score_card = score_card.append(linreg_full_model_withlog_metrics, ignore_index=True)
score_card


# <a id='withoutLog'></a>
# ### 5.2 Multiple Linear Regression - Full Model - without Log Transformed Dependent Variable (OLS)

# **1. Build model using sm.OLS().fit()**

# In[62]:


# build a full model using OLS()
linreg_full_model_withoutlog = sm.OLS(y_train['claim'], X_train).fit()

# print the summary output
print(linreg_full_model_withoutlog.summary())


# #### Calculate the p-values to know the insignificant variables

# In[63]:


# calculate the p-values for all the variables
# create a dataframe using pd.DataFrame()
# columns: specifies the column names
linreg_full_model_withoutlog_pvalues = pd.DataFrame(linreg_full_model_withoutlog.pvalues, columns=["P-Value"])

linreg_full_model_withoutlog_pvalues


# The above table shows the p-values for all the variables to decide the significant variables

# Let's create a list of insignificant variables

# In[ ]:





# In[64]:


# select insignificant variables
insignificant_variables = linreg_full_model_withoutlog_pvalues[
                                                        linreg_full_model_withoutlog_pvalues['P-Value']  > 0.05]

# get the position of a specified value
insigni_var = insignificant_variables.index

# convert the list of variables to 'list' type
insigni_var = insigni_var.to_list()

# get the list of insignificant variables
insigni_var


# **2. Predict the values using test set**

# In[65]:


# predict the claim using predict()
predicted_claim = linreg_full_model_withoutlog.predict(X_test)

# extract the 'claim' values from the test data
actual_claim = y_test['claim']


# **3. Compute model accuracy measures**

# In[66]:


# calculate rmse using rmse()
linreg_full_model_withoutlog_rmse = rmse(actual_claim, predicted_claim)

# calculate R-squared using rsquared
linreg_full_model_withoutlog_rsquared = linreg_full_model_withoutlog.rsquared
# linreg_full_model_withoutlog_test_rsquared = r2_score(actual_claim, predicted_claim)

# calculate Adjusted R-Squared using rsquared_adj
linreg_full_model_withoutlog_rsquared_adj = linreg_full_model_withoutlog.rsquared_adj 


# **4. Tabulate the results**

# In[67]:


# compile the required information
linreg_full_model_withoutlog_metrics = pd.Series({
                     'Model_Name': "Linreg full model without log of target variable",
                     'RMSE':linreg_full_model_withoutlog_rmse,
                     'R-Squared': linreg_full_model_withoutlog_rsquared,
                     'Adj. R-Squared': linreg_full_model_withoutlog_rsquared_adj
                   })

# append our result table using append()
score_card = score_card.append(linreg_full_model_withoutlog_metrics, ignore_index=True)
score_card


# <a id='Finetuning'></a>
# ## 5.3. Fine Tune Linear Regression Model (OLS)

# <a id='RemovingInsignificantVariable'></a>
# ### 5.3.1 Linear Regression after Removing Insignificant Variable (OLS)
# 

# **1. Consider the significant variables**

# In[68]:


X_train.head()


# In[69]:


# drop the insignificant variables
X_significant = df_insurance.drop(["sex","job_title","region","claim","log_claim"], axis=1)


# In[70]:


# filter the categorical features in the dataset using select_dtypes()
df_significant_categoric_features = X_significant.select_dtypes(include=[np.object])
df_significant_categoric_features.columns


# **Dummy encode the catergorical variables**

# In[71]:


# create data frame with only categorical variables that have been encoded

# for all categoric variables create dummy variables
for col in df_significant_categoric_features.columns.values:
    
    dummy_encoded_variables = pd.get_dummies(df_significant_categoric_features[col], prefix=col, drop_first=True)
    df_significant_categoric_features = pd.concat([df_significant_categoric_features, dummy_encoded_variables],axis=1)
    df_significant_categoric_features.drop([col], axis=1, inplace=True)


# **Concatenate numerical and dummy encoded categorical variables**

# In[72]:


# concatenate the numerical and dummy encoded categorical variables using concat()
df_insurance_significant = pd.concat([df_numeric_features, df_significant_categoric_features], axis=1)
df_insurance_significant.head()


# **2. Split the data into training and test sets**

# In[73]:


# add the intercept column to the dataset
df_insurance_significant = sm.add_constant(df_insurance_significant)

# separate the independent and dependent variables
X = df_insurance_significant.drop(['claim','log_claim'], axis=1)
y = df_insurance_significant[['log_claim','claim']]

# split data into train subset and test subset for predictor and target variables
X_train_significant, X_test_significant, y_train, y_test = train_test_split(X, y, random_state=1)

print("The shape of X_train is:",X_train_significant.shape)
print("The shape of X_test is:",X_test_significant.shape)
print("The shape of y_train is:",y_train.shape)
print("The shape of y_test is:",y_test.shape)


# **1. Build model using sm.OLS().fit()**

# In[74]:


# build a full model with significant variables using OLS()
linreg_model_with_significant_var = sm.OLS(y_train['claim'], X_train_significant).fit()

# to print the summary output
print(linreg_model_with_significant_var.summary())


# **2. Predict the values using test set**

# In[75]:


# predict the 'claim' using predict()
predicted_claim = linreg_model_with_significant_var.predict(X_test_significant)

# extract the 'claim' values from the test data
actual_claim = y_test['claim']


# **3. Compute model accuracy measures**
# 
# Now we calculate accuray measures like Root-mean-square-error (RMSE), R-squared and Adjusted R-squared.

# In[76]:


# calculate rmse using rmse()
linreg_model_with_significant_var_rmse = rmse(actual_claim, predicted_claim)

# calculate R-squared using rsquared
linreg_model_with_significant_var_rsquared = linreg_model_with_significant_var.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_model_with_significant_var_rsquared_adj = linreg_model_with_significant_var.rsquared_adj 


# **4. Tabulate the results**

# In[77]:


# compile the required information
linreg_model_with_significant_var_metrics = pd.Series({
                     'Model_Name': "Linreg full model with significant variables",
                     'RMSE': linreg_model_with_significant_var_rmse,
                     'R-Squared': linreg_model_with_significant_var_rsquared,
                     'Adj. R-Squared': linreg_model_with_significant_var_rsquared_adj
                   }) 

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
score_card = score_card.append(linreg_model_with_significant_var_metrics, ignore_index=True)

# print the result table
score_card


# ### 5.3.2 Check the Assumptions of the Linear Regression

# ### 5.3.2.1 Detecting Autocorrelation
# 
# ** Based on Durbin Watson score, we can conclude that there is no autocorrelation**

# ### 5.3.2.2 Detecting Heteroskedasticity
# Breusch-Pagan is the test for detecting heteroskedasticity:
# 
# The null and alternate hypothesis of Breusch-Pagan test is as follows:<BR>
#     
#     H0: The residuals are homoskedastic
#     H1: The residuals are not homoskedastic

# In[78]:


# create vector of result parmeters
name = ['f-value','p-value']           

# perform Breusch-Pagan test using het_breushpagan()
breuschpagan_score = sms.het_breuschpagan(linreg_model_with_significant_var.resid, linreg_model_with_significant_var.model.exog)
breuschpagan_score


# We observe that p-value is less than 0.05 and thus reject the null hypothesis. We conclude that there is heteroskedasticity present in the data.

# ### 5.3.2.3 Linearity of Residuals
# 

# In[79]:


# create subplots of scatter plots

fig, ax = plt.subplots(nrows = 4, ncols= 5, figsize=(25, 20))

# use for loop to create scatter plot for residuals and each independent variable (do not consider the intercept)
# 'ax' assigns axes object to draw the plot onto 
for variable, subplot in zip(X_train_significant.columns[1:], ax.flatten()):
    sns.scatterplot(X_train_significant[variable], linreg_model_with_significant_var.resid , ax=subplot)

plt.show()


# From the plots we see that none of the plots show a specific pattern. Hence, we may conclude that the variables are linearly related to the dependent variable.

# <a id='Normality_of_Residuals'></a>
# ### 5.3.2.4 Normality of Residuals
# 
# The assumption of normality is an important assumption for many statistical tests. The normal Q-Q plot is one way to assess normality. This q-q or quantile-quantile is a scatter plot which helps us validate the assumption of normal distribution in a data set.

# In[80]:


# calculate fitted values
fitted_vals = linreg_model_with_significant_var.predict(X_test_significant)

# calculate residuals
resids = actual_claim - fitted_vals

# fig, ax = plt.subplots(1, 1, figsize=(15, 8))

stats.probplot(resids, plot=plt)


plt.show()


# Using this plot, we can infer that the residuals do not come from a normal distribution. Also Jarqque Bera test suggests residuals do not come from a normal distribution.
# This is possible since our target variable is not normally distributed.

# In[81]:


# check the mean of the residual
linreg_model_with_significant_var.resid.mean()


# ### 5.3.4 Linear Regression with Interaction (OLS)
# 

# **1. Compute the interaction effect**

# In[82]:


# Bsed on domain knowledge - possible interaction between bmi and smoker
# create a copy of the entire dataset to add the interaction effect using copy()
df_insurance_interaction = df_insurance_dummy.copy()

# add the interaction variable
df_insurance_interaction['bmi*smoker'] = df_insurance_interaction['bmi']*df_insurance_interaction['smoker_1'] 
df_insurance_interaction.head()


# **2. Split the data into training and test sets**
# 

# In[83]:


# separate the independent and dependent variables

X = df_insurance_interaction.drop(['claim','log_claim'], axis=1)
y = df_insurance_interaction['claim']

# split data into train subset and test subset for predictor and target variables
X_train_interaction, X_test_interaction, y_train, y_test = train_test_split( X, y, random_state=1)

# check the dimensions of the train & test subset for 
print("The shape of X_train_interaction is:",X_train_interaction.shape)
print("The shape of X_test_interaction is:",X_test_interaction.shape)
print("The shape of y_train is:",y_train.shape)
print("The shape of y_test is:",y_test.shape)


# **3. Build model using sm.OLS().fit()**

# In[84]:


# building a full model with an interaction term using OLS()
linreg_with_interaction = sm.OLS(y_train, X_train_interaction).fit()

# print the summary output
print(linreg_with_interaction.summary())


# **4. Predict the values using test set**

# In[85]:


# predict the 'claim' using predict()
predicted_claim = linreg_with_interaction.predict(X_test_interaction)

# extract the 'claim' values from the test data
actual_claim = y_test


# **5. Compute model accuracy measures**
# 

# In[86]:


# calculate rmse using rmse()
linreg_with_interaction_rmse = rmse(actual_claim, predicted_claim)

# calculate R-squared using rsquared
linreg_with_interaction_rsquared = linreg_with_interaction.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_with_interaction_rsquared_adj = linreg_with_interaction.rsquared_adj 


# **6. Tabulate the results**

# In[87]:


# compile the required information
linreg_with_interaction_metrics = pd.Series({
                     'Model_Name': "linreg_with_interaction",
                     'RMSE': linreg_with_interaction_rmse,
                     'R-Squared': linreg_with_interaction_rsquared,
                     'Adj. R-Squared': linreg_with_interaction_rsquared_adj
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
score_card = score_card.append(linreg_with_interaction_metrics, ignore_index = True)

# print the result table
score_card


# In[ ]:





# ## 6. Regularization (OLS)

# ### 6.1 Ridge Regression (OLS)

# **1. Define train and test sets**

# In[88]:


print("The shape of X_train_interaction is:",X_train_interaction.shape)
print("The shape of X_test_interaction is:",X_test_interaction.shape)
print("The shape of y_train is:",y_train.shape)
print("The shape of y_test is:",y_test.shape)


# In[89]:


X_train_interaction_No_const = X_train_interaction.drop('const', axis=1)
X_test_interaction_No_const = X_test_interaction.drop('const', axis=1)

# check shate after drop of constant
print("The shape of X_train_interaction is:",X_train_interaction_No_const.shape)
print("The shape of X_test_interaction is:",X_test_interaction_No_const.shape)


# **2. Perform Grid Search to identify Best Parameter**

# We first create a list of all the variable names and accuracy metrics whose values we want.

# In[90]:


# Define parameter grid
param = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]}

# Create instance of algorithm
algo_name = Ridge(normalize=True)

grid_cv = GridSearchCV(estimator=algo_name, param_grid= param, cv=5, scoring='r2' )
grid_cv.fit(X_train_interaction_No_const, y_train)
grid_cv.best_params_


# **3. Build model with Best Parameters**

# In[91]:


# build Ridge model with best parameter from grid search

# Use Normalize=True
ridge_regression = Ridge(alpha=0.0001, normalize=True)
ridge_model = ridge_regression.fit(X_train_interaction_No_const, y_train)


# **4. Predict the values using test set**

# In[92]:


# predict the scaled claim using predict()
train_predicted_claim = ridge_model.predict(X_train_interaction_No_const)
predicted_claim = ridge_model.predict(X_test_interaction_No_const)

# extract the 'claim' values from the test data
actual_claim = y_test


# **5. Compute model accuracy measures**
# 
# 

# In[93]:


# calculate rmse using rmse()
ridge_regression_rmse = rmse(actual_claim, predicted_claim)

# calculate R-squared using rsquared
ridge_regression_rsquared = r2_score(y_train, train_predicted_claim)
# ridge_regression_test_rsquared = test_r2 = r2_score(actual_claim, predicted_claim)

# calculate Adjusted R-Squared using rsquared_adj 
# compute number of observations
n = X_train_interaction_No_const.shape[0]
# compute number of independent variables
k = X_train_interaction_No_const.shape[1]
ridge_regression_rsquared_adj = 1 - (1 - ridge_regression_rsquared)*(n-1)/(n-k-1)


# In[94]:


# Discuss the above Adj R2


# **6. Tabulate the results**

# In[95]:


# compile the required information
ridge_regression_metrics = pd.Series({
                     'Model_Name': "Ridge Regression with Interaction",
                     'RMSE': ridge_regression_rmse,
                     'R-Squared': ridge_regression_rsquared,
                     'Adj. R-Squared': ridge_regression_rsquared_adj
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
score_card = score_card.append(ridge_regression_metrics, ignore_index = True)

# print the result table
score_card


# ### 6.2 Lasso Regression (OLS)

# **1. Define train and test sets**

# We already have Train and Test data defined during Ridge
# 

# **2. Perform Grid Search to identify Best Parameter**

# In[96]:


# Define parameter grid
param = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]}

# Create instance of algorithm
algo_name = Lasso(normalize=True)

grid_cv = GridSearchCV(estimator=algo_name, param_grid= param, cv=5, scoring='r2' )
grid_cv.fit(X_train_interaction, y_train)
grid_cv.best_params_


# **3. Build model with Best Parameters**

# In[97]:


# Build Lassso model
lasso_regression = Lasso(alpha=0.01, normalize=True)
lasso_model = lasso_regression.fit(X_train_interaction_No_const, y_train)


# **4. Predict the values using test set**

# In[98]:


# predict the scaled claim using predict()
train_predicted_claim = lasso_model.predict(X_train_interaction_No_const)
predicted_claim = lasso_model.predict(X_test_interaction_No_const)

# extract the 'claim' values from the test data
actual_claim = y_test


# **5. Compute model accuracy measures**
# 
# 

# In[99]:


# calculate rmse using rmse()
lasso_regression_rmse = rmse(actual_claim, predicted_claim)

# calculate R-squared using rsquared
lasso_regression_rsquared = r2_score(y_train, train_predicted_claim)
# lasso_regression_test_rsquared = test_r2 = r2_score(actual_claim, predicted_claim)

# calculate Adjusted R-Squared using rsquared_adj 
# compute number of observations
n = X_train_interaction_No_const.shape[0]
# compute number of independent variables
k = X_train_interaction_No_const.shape[1]
lasso_regression_rsquared_adj = 1 - (1 - lasso_regression_rsquared)*(n-1)/(n-k-1)


# In[100]:


# Discuss Adj R2


# **6. Tabulate the results**

# In[101]:


# compile the required information
lasso_regression_metrics = pd.Series({
                     'Model_Name': "Lasso Regression with Interaction",
                     'RMSE': lasso_regression_rmse,
                     'R-Squared': lasso_regression_rsquared,
                     'Adj. R-Squared': lasso_regression_rsquared_adj
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
score_card = score_card.append(lasso_regression_metrics, ignore_index = True)

# print the result table
score_card


# In[102]:


# Discuss Adj R2 in the contexxt of Ridge, Lasso, ElasticNet


# ### 6.2 ElasticNet Regression

# **1. Define train and test sets**

# We already have Train and Test data defined during Ridge

# **2. Perform Grid Search to identify Best Parameter**

# In[103]:


# Define parameter grid
param = {'alpha':[0.001, 0.01, 0.1, 0.5],
         'l1_ratio':[0.01,0.1,0.3,0.5,0.7]}

# Create instance of algorithm
algo_name = ElasticNet(normalize=True)

grid_cv = GridSearchCV(estimator=algo_name, param_grid= param, cv=5, scoring='r2' )
grid_cv.fit(X_train_interaction, y_train)
grid_cv.best_params_


# In[104]:


from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV


# **3. Build model with Best Parameters**

# In[105]:


# build Gidge model with best parameter from grid search

# Use Normalize=True
elasticnet_regression = ElasticNet(alpha=0.01, l1_ratio=1, normalize=True)
elasticnet_model = elasticnet_regression.fit(X_train_interaction_No_const, y_train)


# **4. Predict the values using test set**

# In[106]:


# predict the scaled claim using predict()
train_predicted_claim = elasticnet_model.predict(X_train_interaction_No_const)
predicted_claim = elasticnet_model.predict(X_test_interaction_No_const)

# extract the 'claim' values from the test data
actual_claim = y_test


# **5. Compute model accuracy measures**
# 
# 

# In[107]:


# calculate rmse using rmse()
elasticnet_regression_rmse = rmse(actual_claim, predicted_claim)

# calculate R-squared using rsquared
elasticnet_regression_rsquared = r2_score(y_train, train_predicted_claim)
# elasticnet_regression_test_rsquared = test_r2 = r2_score(actual_claim, predicted_claim)

# calculate Adjusted R-Squared using rsquared_adj 
# compute number of observations
n = X_train_interaction_No_const.shape[0]
# compute number of independent variables
k = X_train_interaction_No_const.shape[1]
elasticnet_regression_rsquared_adj = 1 - (1 - elasticnet_regression_rsquared)*(n-1)/(n-k-1)


# **6. Tabulate the results**

# In[108]:


# compile the required information
elasticnet_regression_metrics = pd.Series({
                     'Model_Name': "ElasticNet Regression with Interaction",
                     'RMSE': elasticnet_regression_rmse,
                     'R-Squared': elasticnet_regression_rsquared,
                     'Adj. R-Squared': elasticnet_regression_rsquared_adj
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
score_card = score_card.append(elasticnet_regression_metrics, ignore_index = True)

# print the result table
score_card


# In[109]:


# Discuss Adj R2 in the contexxt of Ridge, Lasso, ElasticNet


# <a id='rmse_and_r-squared'></a>
# ## 8. Conclusion and Interpretation

# To take the final conclusion, let us recall the result table again

# In[110]:


# view the result table
score_card


# **Let visualize graphically the above table**

# In[111]:


# plot the accuracy measure for all models
# secondary_y: specify the data on the secondary axis
score_card.plot(secondary_y=['R-Squared','Adj. R-Squared'])

# display just the plot
plt.show()


# ## Business Interpretation
# 1. **From this model we can look at millions of data points and generate smart insights — without direction from a human agent — can be instantly scalable and create a robust user experience.** 
# 2. **The ultimate goal is the best possible user experience, based on intuition drawn from relevant information and that weeds out the irrelevant.**
# 3. **It’s important to note that a majority of the price consumers pay when enrolling in health insurance goes into risk prediction and risk management. By using this model user can get more accurate risk models and predict which individuals need specific types of care, health insurance providers can spend more money on their beneficiaries and less on those processes.**
