# from flask import Flask, render_template, request, redirect, url_for, send_file
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from datetime import datetime
# plt.use('Agg')
# # Initialize app
# app = Flask(__name__)
# model = load_model('pneumonia_model.h5')

# # Ensure results directory exists
# os.makedirs('static/results', exist_ok=True)

# def predict_and_save(img_path, save_path):
#     img = image.load_img(img_path, target_size=(150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0

#     prediction = model.predict(img_array)
#     confidence = prediction[0][0]

#     plt.imshow(img)
#     plt.axis('off')

#     if confidence > 0.5:
#         label = f"🔴 Pneumonia ({confidence * 100:.2f}%)"
#     else:
#         label = f"🟢 Normal ({(1 - confidence) * 100:.2f}%)"

#     plt.title(label)
#     plt.savefig(save_path)
#     plt.close()

#     return label

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     results = []
#     if request.method == 'POST':
#         files = request.files.getlist('images')
#         for file in files:
#             filename = file.filename
#             file_path = os.path.join('static/results', filename)
#             file.save(file_path)

#             result_img_path = os.path.join('static/results', f"result_{filename}")
#             label = predict_and_save(file_path, result_img_path)

#             results.append({
#                 'original': file_path,
#                 'result': result_img_path,
#                 'label': label
#             })
#     return render_template('index.html', results=results)

# if __name__ == '__main__':
#     app.run(debug=True)
import os
import numpy as np
from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Create results folder if it doesn't exist
RESULTS_FOLDER = os.path.join('static', 'results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load trained model
model = load_model('pneumonia_model.h5')

def predict_and_save(img_path, save_path):
    # Preprocess image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)
    confidence = prediction[0][0]

    # Prepare label
    if confidence > 0.5:
        label = f"Pneumonia ({confidence * 100:.2f}%)"
        color = "🔴"
    else:
        label = f"Normal ({(1 - confidence) * 100:.2f}%)"
        color = "🟢"

    # Plot and save result image
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{color} {label}")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return label, os.path.basename(save_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    result_img = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded.")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file.")

        # Save uploaded image
        upload_path = os.path.join(RESULTS_FOLDER, file.filename)
        file.save(upload_path)

        # Predict + save result
        result_file = f"result_{file.filename}"
        result_img_path = os.path.join(RESULTS_FOLDER, result_file)
        result, result_img_name = predict_and_save(upload_path, result_img_path)

        result_img = url_for('static', filename=f"results/{result_img_name}")

    return render_template('index.html', result=result, result_img=result_img)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    # app.run(debug=True)
