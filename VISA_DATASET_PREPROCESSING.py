# Converted from Jupyter Notebook
# Original file: VISA_DATASET_PREPROCESSING.ipynb

# In[]:
import pandas as pd

df = pd.read_csv(r"C:\Users\manid\OneDrive\Desktop\AI Enabled Visa Status Prediction and Processing Time Estimator\dataset\us_perm_visas.csv")

df.shape
df.columns


# In[]:
df.head()

# In[]:
df[['case_received_date','decision_date']].isnull().sum()

# In[]:
df.isnull().sum()

# In[]:
df['case_received_date'] = pd.to_datetime(df['case_received_date'], errors='coerce')
df['decision_date'] = pd.to_datetime(df['decision_date'], errors='coerce')

# In[]:
df[['case_received_date','decision_date']].isnull().sum()

# In[]:
df_clean = df[
    (df['case_received_date'].notna()) &
    (df['decision_date'].notna())
]

df_clean = df_clean[
    df_clean['decision_date'] >= df_clean['case_received_date']
]
df_clean.shape


# In[]:
df_clean['processing_days'] = (
    df_clean['decision_date'] - df_clean['case_received_date']
).dt.days

# In[]:
print(df_clean['processing_days'].describe())

# In[]:
lower = df_clean['processing_days'].quantile(0.01)
upper = df_clean['processing_days'].quantile(0.99)

df_clean = df_clean[
    (df_clean['processing_days'] >= lower) &
    (df_clean['processing_days'] <= upper)
]

print("After Outlier Removal:")
df_clean.shape

# In[]:
important_cols = [
    'country_of_citizenship',
    'class_of_admission',
    'employer_state',
    'us_economic_sector',
    'case_status',
    'foreign_worker_info_education',
    'foreign_worker_info_major',
    'case_received_date',
    'decision_date',
    'processing_days'
]

df_clean = df_clean[important_cols]

# In[]:
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = df_clean[col].fillna("UNKNOWN")

# In[]:
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = df_clean[col].str.strip().str.upper()

# In[]:
df_clean['application_year'] = df_clean['case_received_date'].dt.year
df_clean['application_month'] = df_clean['case_received_date'].dt.month
df_clean['application_quarter'] = df_clean['case_received_date'].dt.quarter
df_clean['application_dayofweek'] = df_clean['case_received_date'].dt.dayofweek

# In[]:
final_status = ['CERTIFIED', 'DENIED']
df_clean = df_clean[df_clean['case_status'].isin(final_status)]

# In[]:
df_clean = df_clean.drop_duplicates()
print("Final Shape:")
df_clean.shape

# In[]:
df_clean.isnull().sum()

# In[]:
df_clean.to_csv("cleaned_us_visa_dataset.csv", index=False)

# In[]:


