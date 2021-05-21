#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


shampoo = pd.read_csv('F:/Python/Sampoo/shampoo_with_exog.csv')


# In[3]:


shampoo.head()


# In[4]:


type(shampoo)


# In[11]:


#converting the type from Dataframe to Series
shampoo = pd.read_csv('F:/Python/Sampoo/shampoo_with_exog.csv', index_col= [0], parse_dates = True, squeeze = True)


# In[12]:


type(shampoo)


# In[4]:


shampoo.drop('Inflation', inplace=True, axis=1)


# In[5]:


shampoo.head()


# In[6]:


shampoo.columns


# In[21]:


shampoo.plot()


# In[29]:


shampoo.columns


# In[30]:


shampoo.plot(style = 'k.')


# In[8]:


shampoo.size


# In[9]:


shampoo.describe()


# In[25]:


#smoothing the time Series that is Moving Average

shampoo_ma = shampoo.rolling(window = 10).mean()


# In[26]:


shampoo_ma.plot()


# In[13]:


shampoo_base = pd.concat([shampoo,shampoo.shift(1)], axis=1)


# In[14]:


shampoo_base


# In[30]:


shampoo


# In[23]:


shampoo_base.columns = ['Actual_Sales', 'Forecast_Sales']


# In[24]:


shampoo_base


# In[25]:


shampoo_base.dropna(inplace=True)


# In[26]:


shampoo_base


# In[18]:


from sklearn.metrics import mean_squared_error
import numpy as np


# In[27]:


shampoo_error = mean_squared_error(shampoo_base.Actual_Sales, shampoo_base.Forecast_Sales)


# In[28]:


shampoo_error


# In[38]:


np.sqrt(shampoo_error)


# In[16]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[63]:


plot_acf(shampoo)


# In[40]:


plot_pacf(shampoo)


# In[41]:


from statsmodels.tsa.arima_model import ARIMA


# In[42]:


shampoo.size


# In[30]:


shampoo_train = shampoo[0:25]
shampoo_test = shampoo[25:36]


# In[31]:


len(shampoo_train)


# In[32]:


len(shampoo_test)


# In[46]:


shampoo_model = ARIMA(shampoo_train, order = (3,1,2))


# In[47]:


shampoo_model


# In[48]:


shampoo_model_fit = shampoo_model.fit()


# In[49]:


shampoo_model_fit


# In[50]:


shampoo_model_fit.aic


# In[51]:


shampoo_forecast = shampoo_model_fit.forecast(steps=11)[0]


# In[52]:


shampoo_model_fit.plot_predict(1,47)


# In[53]:


shampoo_forecast


# In[54]:


np.sqrt(mean_squared_error(shampoo_test, shampoo_forecast))


# In[55]:


p_values = range(0,5)
d_values = range(0,3)
q_values = range(0,5)


# In[56]:


import warnings
warnings.filterwarnings("ignore")


# In[57]:


for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p,d,q)
            train,test = shampoo[0:25], shampoo[25:36]
            predictions = list()
            for i in range(len(test)):
                try:
                    model = ARIMA(train,order)
                    model_fit = model.fit(disp=0)
                    pred_y = model_fit.forecast()[0]
                    predictions.append(pred_y)
                    error = mean_squared_error(test,predictions)
                    print('ARIMA%s RMSE = %.2f'% (order,error))
                except:
                    continue


# In[ ]:




