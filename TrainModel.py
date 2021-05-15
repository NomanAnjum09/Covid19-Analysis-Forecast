import pandas as pd
import numpy as np
from io import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


def clean_dataset(data):
    
    data = data.dropna()
    data = data[data['corona_result']!="אחר"]
    data['corona_result'] = data['corona_result'].replace({"שלילי":"N","חיובי":"Positive"})
    data['gender'] = data['gender'].replace({"זכר":"Male","נקבה":"Female"})

    
    data['corona_result'] = LabelEncoder().fit_transform(data['corona_result'])
    data['age_60_and_above'] = LabelEncoder().fit_transform(data['age_60_and_above'])
    data['gender'] = LabelEncoder().fit_transform(data['gender'])

    data = data.drop(['test_indication','test_date'],axis=1) 

    
    
    return data


file = open('/home/noman/B-Drive/University-Docs/Semester8/Assignments/DS Assg/Project/Data/corona_tested_individuals_ver_00139.csv','rb')
data = file.read()
print(data)
data = str(data,'utf-8')

data = StringIO(data) 

data = pd.read_csv(data)    

data = clean_dataset(data)
print(np.unique(data['corona_result']))
Y = data['corona_result']
X = data.drop(["corona_result"],axis=1)
DT = DecisionTreeClassifier()
DT.fit(X,Y)
