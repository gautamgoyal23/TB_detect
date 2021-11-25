from fastapi import FastAPI, File, UploadFile, HTTPException,Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from typing import List
import io
import numpy as np
import sys
import secrets
filepath = './tb_detector.h5'
model = load_model(filepath, compile = True)

input_shape = model.layers[0].input_shape
security = HTTPBasic()
app = FastAPI()

class Prediction(BaseModel):
  filename: str
  contenttype: str
  prediction: str
  likely_class: int
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "tbdetect")
    correct_password = secrets.compare_digest(credentials.password, "gautam")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
@app.get('/')
def root_route():
  return { 'error': 'Use GET /prediction instead of the root route!' }

@app.post('/prediction/', response_model=Prediction)
async def prediction_route(username: str = Depends(get_current_username),file: UploadFile = File(...)):

  if file.content_type.startswith('image/') is False:
    raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

  try:
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))

    pil_image = pil_image.resize((input_shape[1], input_shape[2]))

    if pil_image.mode == 'RGBA':
      pil_image = pil_image.convert('RGB')

    if input_shape[3] and input_shape[3] == 1:
      pil_image = pil_image.convert('L')

    numpy_image = np.array(pil_image).reshape((input_shape[1], input_shape[2], input_shape[3]))

    numpy_image = numpy_image / 255
    
    
    prediction_array = np.array([numpy_image])
    predictions = model.predict(prediction_array)
    prediction = predictions[0]
    if(prediction>0.5):
        likely_class=1
    else:
        likely_class=0
    prediction_string="You have {}% chances of having Tubercolosis".format((1-prediction)*100)
    return {
      'filename': file.filename,
      'contenttype': file.content_type,
      'prediction': prediction_string,
      'likely_class': likely_class          
    }
  except:
    e = sys.exc_info()[1]
    raise HTTPException(status_code=500, detail=str(e))
