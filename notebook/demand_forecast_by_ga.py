# flake8: noqa
# coding: utf-8
# # Setup
# ## Jupyter Shell
# In[1]:


shell = "ZMQInteractiveShell"
IN_JUPYTER = 'get_ipython' in globals() and get_ipython().__class__.__name__ == shell

# Allow modules and files to be loaded with relative paths
from pkg_resources import resource_filename as fpath
import sys
sys.path.append(fpath(__name__, ""))


# ## Theme

# In[2]:


if IN_JUPYTER:
    get_ipython().system('jt -l')
    # toggle toolbar ON and notebook name ON
    get_ipython().system('jt -t grade3 -T -N')


# # Load Packages & Track Versions

# In[3]:


# check the versions of key python librarise
# Python
import sys
import platform
print('python: %s' % platform.python_version())


# In[4]:


pkgs = [
    'numpy', 'matplotlib', 'pandas', 'statsmodels', 'sklearn', 'fbprophet',
    'numba',
]
for pkg in pkgs:
    try:
        globals()['est_module'] = __import__(pkg)
        print(pkg, ': %s' % est_module.__version__)
    except ModuleNotFoundError:
        print(pkg, 'Not Found')


# In[6]:


import os
if IN_JUPYTER:
    workspace_dir = os.path.realpath('..')
else:
    workspace_dir = os.getcwd()
print('Workspace Dir ->', workspace_dir)


# In[7]:


import pandas as pd
import numpy as np
from fbprophet import Prophet

import datetime
from numba import jit
import math

if IN_JUPYTER:
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')


# # Functions

# In[8]:


def group_by_col(df, col):
    group = df.groupby(df[str(col)])
    group_by = pd.DataFrame(group.size().reset_index(name="Count"))
    return group_by


# In[9]:


def delete_columns(df, cols):
    df = df.drop(list(cols), axis=1)
    return df


# In[10]:


def print_cols_type(df):
    # Print Column Type
    for col in df:
        print(str(col), '->', type(df[col][1]))


# In[11]:


def coerce_columns_to_numeric(df, column_list):
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')


# In[12]:


import dateutil
# Convert date from string to date times


def coerce_columns_to_date(df, col):
    df[str(col)] = df[str(col)].apply(dateutil.parser.parse, dayfirst=True)


# In[13]:


# function to create a DataFrame in the format required by Prophet
def create_df_for_prophet(ts):
    ts.columns = ["ds", "y"]
    ts = ts.dropna()
    ts.reset_index(drop=True, inplace=True)
    return ts


# In[14]:


from scipy import stats
import numpy as np
dir_name = workspace_dir + '/data/output/'


def remove_outliers_by_col(df, col):
    file = dir_name + 'outliers_' + str(col).lower() + '.csv'
    z = np.abs(stats.zscore(df[str(col)]))
    threshold = 3
    df[(z > 3)].to_csv(file, index=False)
    print('Removed Outliers Stores In ->', file)
    return df[(z < 3)]


# In[15]:


def visualize_outliers_by_col(df, col):
    if IN_JUPYTER:
        import seaborn as sns
        sns.boxplot(x=df[str(col)])


# In[16]:


# function to remove any negative forecasted values.
def remove_negtives(ts):
    ts['yhat'] = ts['yhat'].clip_lower(0)
    ts['yhat_lower'] = ts['yhat_lower'].clip_lower(0)
    ts['yhat_upper'] = ts['yhat_upper'].clip_lower(0)
    return ts


# In[17]:


import math


def mse(y_actual, y_pred):
    # compute the mean square error
    mse = ((y_actual - y_pred)**2).mean()
    return mse


# In[18]:


# Symmetric Mean Absolute Percent Error (SMAPE)
# function to calculate in sample SMAPE scores
def smape_fast(y_true, y_pred):
    out = 0
    for i in range(y_true.shape[0]):
        if (y_true[i] != None and np.isnan(y_true[i]) == False):
            a = y_true[i]
            b = y_pred[i]
            c = a + b
            if c == 0:
                continue
            out += math.fabs(a - b) / c
    out *= (200.0 / y_true.shape[0])
    return out


# In[19]:


def visualize_user_access(df):
    if IN_JUPYTER:
        df.set_index('ds').plot(style=['+'])
        plt.xlabel('Date')
        plt.ylabel('Users')
        plt.title('User Access By Date')
        plt.show()


# In[20]:


def visualize_forecast(df):
    if IN_JUPYTER:
        mdl.plot(df)
        plt.show()


# In[21]:


def visualize_forecast_details(df):
    if IN_JUPYTER:
        # plot time series components
        mdl.plot_components(df)
        plt.show()


# In[64]:


def convert_notebook_to_python():
    get_ipython().system('jupyter nbconvert --to=python notebook demand_forecast_by_ga.ipynb')
    get_ipython().system('ls')


# # Sanity Check - Input Data

# In[22]:


# import required data
from subprocess import check_output
input_dir = workspace_dir + "/data/input/"
print(check_output(["ls", input_dir]).decode("utf8"))


# # Predict - From Google Analytics Data

# ## Load & Clean Up Data

# In[23]:


max_date_past_data = '2018-08-31'  # str(clean_ga_data.ds.max().date())
data_file = workspace_dir + "/data/input/est_daily_access.csv"

ga_data = pd.read_csv(data_file)
m = ga_data.shape[0]
n = ga_data.shape[1]

print('        Data Set Details')
print('+++++++++++++++++++++++++++++++')
print('# Of Observations', str(m))
print('# Of Features', str(n))


# In[24]:


visualize_outliers_by_col(ga_data, 'Users')


# In[25]:


ga_data = remove_outliers_by_col(ga_data, 'Users')
m = ga_data.shape[0]
print(' Data Set without Outliers')
print('+++++++++++++++++++++++++++++++')
print('# Of Observations', str(m))
ga_data.tail()


# In[26]:


clean_ga_data = create_df_for_prophet(ga_data)
coerce_columns_to_numeric(clean_ga_data, ['y'])
coerce_columns_to_date(clean_ga_data, 'ds')
print_cols_type(clean_ga_data)
clean_ga_data.tail()


# In[27]:


visualize_user_access(clean_ga_data)


# In[28]:


# log transform data
ga_data['y'] = np.log(ga_data['y'])
ga_data.tail()


# In[29]:


visualize_user_access(ga_data)


# ## Prediction

# In[30]:


holidays_csv = workspace_dir + "/data/input/us_holidays.csv"
us_public_holidays = pd.read_csv(holidays_csv)
mdl = Prophet(
    interval_width=0.95,
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    holidays=us_public_holidays,
)
mdl.fit(ga_data)

ga_future = mdl.make_future_dataframe(
    periods=31 + 28, freq='D', include_history=True,
)
ga_forecast = mdl.predict(ga_future)


# In[31]:


ga_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[32]:


np.exp(ga_forecast[['yhat', 'yhat_lower', 'yhat_upper']].tail())


# In[33]:


ga_forecast = remove_negtives(ga_forecast)


# In[34]:


visualize_forecast(ga_forecast)


# In[35]:


visualize_forecast_details(ga_forecast)


# In[ ]:


ga_forecast['yhat'] = np.exp(ga_forecast[['yhat']])
ga_forecast['yhat_lower'] = np.exp(ga_forecast[['yhat_lower']])
ga_forecast['yhat_upper'] = np.exp(ga_forecast[['yhat_upper']])

ga_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


modelling_csv = workspace_dir + "/data/output/prediction_based_ga_modelling.csv"
ga_forecast.to_csv(modelling_csv)


# In[ ]:


# retransform using e
y_hat = ga_forecast['yhat'][:]
y_true = clean_ga_data['y']
mse = mse(y_hat, y_true)
print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(
    mse, math.sqrt(mse),
))


# In[ ]:


y_prediction = ga_forecast['yhat'][:]
y_actual = clean_ga_data['y']
smape = smape_fast(y_actual.values, y_prediction.values)
print('Prediction quality: SMAPE :  {:.2f}  '.format(smape))


# In[ ]:


prediction = ga_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
column_headers = [
    'Date', 'PredictedUser', 'Lower(PredictedUser)', 'Upper(PredictedUser)',
]
prediction.columns = column_headers
forecast_csv = workspace_dir + '/data/output/forecast_for_future.csv'
prediction_future = prediction[prediction.Date > max_date_past_data]
prediction_future.to_csv(forecast_csv, index=False)
prediction_future.tail()


# In[ ]:


ds = ga_forecast[['ds']]
actual = clean_ga_data['y']
forecast = ga_forecast[['yhat', 'yhat_lower', 'yhat_upper']]
frames = [ds, actual, forecast]
column_headers = [
    'Date', 'ActualUser', 'PredictedUser', 'Lower(PredictedUser)',
    'Upper(PredictedUser)',
]
result = pd.concat(frames, axis=1, join='inner')
result.columns = column_headers
forecast_csv = workspace_dir + '/data/output/forecast_for_past.csv'
result.to_csv(forecast_csv, index=False)
result.tail()
