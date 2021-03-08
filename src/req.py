import requests
from PIL import Image
import numpy as np


resp = requests.post('http://0.0.0.0:5000/predict', files={"file": open('0.png', 'rb')})
image =  resp.json()
image=  image['array'][0]
image = np.asarray(image)
print(image.shape)
im = Image.fromarray(image)
im.save("your_file.jpeg")

