# ODD2023-Datascience-Ex06
# EX-06 FEATURE TRANSFORMATION
# Aim:
To read the given data and perform Feature Transformation process and save the data to a file.

# Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# Algorithm:
Step1: Read the given Data.  
Step2: Clean the Data Set using Data Cleaning Process.  
Step3: Apply Feature Transformation techniques to all the features of the data set.   
Step4: Print the transformed features.    

```
Name: Manoj kumar G
reg no: 212222230078
```
# PROGRAM:
```
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Data_to_Transform.csv")
df=pd.DataFrame(data)
df.info()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.title("Highly Positive Skew")
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.title("Highly Negative Skew")
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.title("Moderate Positive Skew")
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.title("Moderate Negative Skew")
plt.show()

np.log(df["Highly Positive Skew"])
np.reciprocal(df["Moderate Positive Skew"])
np.sqrt(df["Highly Positive Skew"])
df["Highly Positive Skew_boxcox"], parameter=stats.boxcox(df["Highly Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm

sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew_1'],line='45')
plt.show()

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
# OUTPUT:
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/c71ef397-c813-4b8c-ab8f-f7a45ea4d231)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/ff072536-a16a-4801-a5da-cc8c8f72cfbe)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/7ef96ce3-fcd7-45b6-a014-ab136ce20197)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/6a47465d-f9e5-4337-9d7d-614b49198a07)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/7105a0d5-b7d7-4c3b-87d8-3dafb5995fb6)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/d1e2e403-6e12-4686-bfb8-2be98ab3dbf3)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/85215274-51d4-4265-aca2-dc8b46983cc3)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/2da583af-0b07-47b0-ad11-ea5518947c2e)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/0b8496d3-ceef-4374-83d5-95cc2b616556)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/b6df3083-2a1e-4776-8784-524cd2e71035)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/654bf356-7524-417c-a906-540ece43e0e0)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/e648d702-ff57-4c89-9f2f-650e76688276)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/b5a49244-3fb2-4a07-b98d-866fe8ce8930)
![image](https://github.com/ASHWINKUMAR2903/ODD2023-Datascience-Ex06/assets/119407186/1186bfcc-cef2-4e52-b82b-125722bf0257)
# Result:
Thus feature transformation is done for the given dataset.
