from flask import Flask, render_template, request
from nst import style_transfer
import numpy as np
import cv2
import base64


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    transformed_image = None

    if request.method == 'POST':
        if 'content_image' in request.files and 'style_image' in request.files:
            content_image = request.files['content_image']
            style_image = request.files['style_image']
            transformed_image = style_transfer(content_image, style_image)
            transformed_image = np.squeeze(transformed_image, axis=0)
            transformed_image = (transformed_image * 255).astype(np.uint8)
            transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            _, transformed_encoded = cv2.imencode('.png', transformed_image)
            transformed_image = base64.b64encode(transformed_encoded).decode()

    return render_template('index.html', transformed_image=transformed_image)

if __name__ == '__main__':
    app.run(debug=True)
