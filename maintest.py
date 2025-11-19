import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load model
model = load_model('BrainTumor10Epochscategorical.h5')

# Load image
image = cv2.imread('C:\\Users\\Akash R M\\Downloads\\Brain_tumor_Detection1\\dataset\\yes\\y7.jpg')
img = Image.fromarray(image)
img = img.resize((64,64))

# Convert to numpy array
img = np.array(img)

# Normalize (important)
img = img / 255.0

# Expand dimensions to match model input
input_img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(input_img)       # correct!
cls = pred.argmax()                   # correct for softmax model

print("Raw prediction:", pred)

# Interpret
if cls == 0:
    print("NO TUMOR detected")
    print("Confidence:", pred[0][0])
else:
    print("TUMOR detected")
    print("Confidence:", pred[0][1])