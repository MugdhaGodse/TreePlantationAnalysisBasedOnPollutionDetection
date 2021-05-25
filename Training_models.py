import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn import linear_model
import math
from sklearn.preprocessing import LabelEncoder
import pickle


df=pd.read_csv('dataset.csv')
#print(df)
#print(df.columns)
inputs=df.drop(['Type of Industries','Area in sq. m','area for planting','no_of_trees','tree name'],axis=1)
target=df['tree name']
median_CO2 = math.floor(df.CO2.median())
print(median_CO2)
df.CO2.fillna(median_CO2)
reg = linear_model.LinearRegression()
reg.fit(df[['area for planting']],df.no_of_trees)
model = tree.DecisionTreeClassifier()
model.fit(inputs, target)

with open('model_decisiontree','wb')as f:
    pickle.dump(model,f)


with open('model_regression','wb')as f:
    pickle.dump(reg,f)


