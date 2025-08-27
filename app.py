from fastapi import FastAPI,Request
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd 
from fastapi.responses import JSONResponse,HTMLResponse

from pydantic import BaseModel


app=FastAPI()

templates=Jinja2Templates(directory="templates")


with open("30-diamond_model_complete.pkl","rb") as f :
    saved_data=pickle.load(f)
    model=saved_data["model"]
    encoders=saved_data["encoders"]
    scaler=saved_data["scaler"]


# GET , POST , PUT , DELETE

class DiamondML(BaseModel):
    carat:float
    cut:str
    color:str
    clarity:str
    depth:float
    table:float
    x:float
    y:float
    z:float

@app.get("/" , response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})






@app.post("/predict")

async def predict(features: DiamondML):
    input_data=pd.DataFrame([features.model_dump()])

    for col in ['cut','color','clarity']:
        input_data[col]=encoders[col].transform(input_data[col])

    input_scaled=scaler.transform(input_data)

    prediction=model.predict(input_scaled)

    return {"predicted_price" :prediction[0]}
    
    























































