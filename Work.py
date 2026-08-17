import base64
import numpy as np
import streamlit as st
from PIL import ImageOps,Image
def classify(image,model,class_names,pokemon_descriptions):

    image_sized=ImageOps.fit(image, (128,128), Image.Resampling.LANCZOS)
    image_array=np.asarray(image_sized)
    image_normal= image_array.astype(np.float32)/255
    data=np.ndarray(shape=(1,128,128,3),dtype=np.float32)
    data[0]=image_normal

    prediction=model.predict(data)
    index=np.argmax(prediction)
    classn=class_names[index]
    confidence_score= prediction[0][index]*100
    description=pokemon_descriptions[index]

    return classn,confidence_score,description



