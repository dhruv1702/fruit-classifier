from flask import Flask, request, render_template, jsonify, redirect
import numpy as np
import PIL
from PIL import Image
import os
from keras import backend as K
from werkzeug import secure_filename
from load_model import model, graph

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'upload')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def landing_page():
    #return render_template('index.html')
    return redirect('/predict')


@app.route('/predict', methods=["GET","POST"])
def predict_image():
    if request.method=='POST':
        file = request.files['image']
        if not file:
            return render_template('index.html', label="No file")
        IMG_NAME = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], IMG_NAME))
        # Preprocess the image so that it matches the training input
        image = Image.open(file)
        image = np.asarray(image.resize((224,224)))
        image = image[:,:,:3]
        image = image.reshape(1,224,224,3)

        # Use the loaded model to generate a prediction.
        with graph.as_default():
            pred = model.predict(image)
        # Prepare and send the response.
        digit = np.argmax(pred)
        prediction = {'digit':int(digit)}

        dict = {0:'Apple',1:'Pear',2:'Orange'}
        label = dict[digit]

        fruit_path = os.path.join(app.config['UPLOAD_FOLDER'], IMG_NAME)
        return render_template('result.html', label=label, fruit_image=fruit_path)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
