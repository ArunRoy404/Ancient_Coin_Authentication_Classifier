from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

app = Flask(__name__)
model = load_model(r'C:\Users\Tamanna\Desktop\ML_Models\Deployment\vgg16.h5')

UPLOAD_FOLDER = r'C:\Users\Tamanna\Desktop\ML_Models\Deployment\uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def page():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], "image_" + imagefile.filename)
    imagefile.save(image_path)

    img = load_img(image_path, target_size=(256, 256))
    img = img_to_array(img)
    img = np.expand_dims(img / 255.0, axis=0)
    pred = model.predict(img)
    
    if pred > 0.5:
        classification = "Real"
    else:
        classification = "Fake"

    return render_template('index.html', prediction=classification, image_path="uploaded_images/image_" + imagefile.filename)

@app.route('/uploaded_images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
