import os, re,sys

import pickle
import pandas as pd

from io import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


DIR_PATH = os.path.dirname(os.path.abspath("Covid19-Analysis-Forecast"))
PICKLES_PATH = os.path.join(DIR_PATH, "Pickles")
# CODE_PATH = os.path.join(DIR_PATH, "Code")



from fastapi import BackgroundTasks ,  FastAPI ,  File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(debug=False)

origins = [
    "*",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def welcome():
    return {"message": "Welcome, Sarcasm Detector!"}

def makeDataFrame(cough:int , fever:int,sore_throat:int,shortness_of_breath:int,head_ache:int,age_60_and_above:int,	gender:int):
    input = {
        'cough':cough,
        'fever':fever,
        'sore_throat':sore_throat,
        'shortness_of_breath':shortness_of_breath,
        'head_ache':head_ache,
        'age_60_and_above':age_60_and_above,
        'gender':gender
    }
    df = pd.DataFrame(input,index=[0])
    return df
@app.get("/is_covid/")
def is_covid(model:str,cough:int , fever:int,	sore_throat:int,shortness_of_breath:int,head_ache:int,age_60_and_above:int,	gender:int):
    is_covid = False
    
    df = makeDataFrame(cough,fever,sore_throat,shortness_of_breath,head_ache,age_60_and_above,gender)

    in_file = open(os.path.join(PICKLES_PATH, model), "rb")
    model = pickle.load(in_file)
    in_file.close()
    is_covid =  model.predict(df)
    is_covid = is_covid.tolist()
    return {"is_covid": is_covid[0]}

@app.get("/models/")
def models():
    _,_,_models = next(os.walk(PICKLES_PATH))
    return {"models": _models}

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
@app.post("/train/")
def train_model(file: UploadFile = File(...)):
    data = file.read()

    data = str(data,'utf-8')

    data = StringIO(data) 

    data = pd.read_csv(data)    
    
    data = clean_dataset(data)
    print(data.head())
    Y = data['corona_result']
    X = data.drop(["corona_result"],axis=1)
    DT = DecisionTreeClassifier()
    DT.fit(X,Y)
    pickle.dump(DT, open("DTTrained"+str(time.now), 'wb'))
    
    return {"filename": file.filename}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)