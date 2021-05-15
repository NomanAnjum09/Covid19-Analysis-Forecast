import os, re,sys

import pickle
import pandas as pd

DIR_PATH = os.path.dirname(os.path.abspath("Covid19-Analysis-Forecast"))
PICKLES_PATH = os.path.join(DIR_PATH, "Pickles")
# CODE_PATH = os.path.join(DIR_PATH, "Code")



from fastapi import FastAPI
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



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)