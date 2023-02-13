import tensorflow as tf
load_model = tf.keras.models.load_model
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from typing import List



app = FastAPI()

@app.get("/")
def hello():
    return {"hello":"FastAPI"}

class Data(BaseModel):
    dataList:List=[[[]]]


def loadModel():
    global predict_model

    predict_model = load_model('crack_model03.h5')

loadModel()


async def predict(data):
    classNameCat = {0: 'NG(0)', 1: 'NG(1)', 2:'OK(2)', 3:'OK(3)', 4:'OK(4)', 5:'OK(5)'}
    img = np.array(data.dataList)
    
    X=img.reshape(img.shape[0],img.shape[1], img.shape[2],1)
    X=X / 255.0
    classification = predict_model.predict(X)

    result = np.argmax(classification, axis=-1)

    return result.tolist(), classification.tolist()


@app.post('/getclass')
async def get_class(data:Data):
    result, classification = await predict(data)
    res = {'class':result, 'classification':classification}
    return {'results': res}