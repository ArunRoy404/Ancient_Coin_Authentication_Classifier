from flask import Flask, render_template, request


import os
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model



app = Flask(__name__)
model = load_model("vgg16")