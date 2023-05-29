from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import os
from scipy.io import wavfile
import noisereduce as nr
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing import image
import tempfile


app = Flask(__name__)
CORS(app)
# Load the pre-trained model
SVM_model = tf.keras.models.load_model('best_model_trained150.h5')
model = tf.keras.models.load_model('my_scratch100.h5')
#SVM_model = tf.keras.models.load_model("C:\\Users\\amirl\\Desktop\\phase two\\cnn models\\best_model_trained150.h5")
#model = tf.keras.models.load_model("C:\\Users\\amirl\\Desktop\\phase two\\cnn models\\my_scratch100.h5")

@app.route('/predict', methods=['POST'])
def predict():
    # Handle file upload
    file = request.files['file']
    # check if the file is a .wav file
    if not file.filename.endswith('.wav'):
    #if not file.content_type == 'audio/wav':
        return {'code': 3, 'result': 'File extension not allowed, only .wav files are allowed'}

    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        file.save(temp_file.name)
        file_path = temp_file.name

    # Retrieve form data
    fever_or_chills = request.form.get('fever_or_chills')
    shortness_of_breath = request.form.get('shortness_of_breath')
    fatigue = request.form.get('fatigue')
    muscle_or_body_aches = request.form.get('muscle_or_body_aches')
    headache = request.form.get('headache')
    loss_of_taste_or_smell = request.form.get('loss_of_taste_or_smell')
    congestion_or_runny_nose = request.form.get('congestion_or_runny_nose')
    sore_throat = request.form.get('sore_throat')
    nausea_or_vomiting = request.form.get('nausea_or_vomiting')

    # Print the form data to the console
    print('fever_or_chills:', fever_or_chills)
    print('shortness_of_breath:', shortness_of_breath)
    print('fatigue:', fatigue)
    print('muscle_or_body_aches:', muscle_or_body_aches)
    print('headache:', headache)
    print('loss_of_taste_or_smell:', loss_of_taste_or_smell)
    print('congestion_or_runny_nose:', congestion_or_runny_nose)
    print('sore_throat:', sore_throat)
    print('nausea_or_vomiting:', nausea_or_vomiting)

    booleans = [fever_or_chills, shortness_of_breath, fatigue, muscle_or_body_aches, headache, loss_of_taste_or_smell, congestion_or_runny_nose, sore_throat, nausea_or_vomiting]
    true_count = booleans.count(True)
    false_count = booleans.count(False)
    if true_count > false_count:
        form = True
    else:
        form = False

    #file preprocessing
    # Read input file
    rate, data = wavfile.read(file_path)

    # Reduce noise
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    # Write output file
    wavfile.write(file_path, rate, reduced_noise)
    print("Noise removed successfully.")

    # Read reduced noise file
    orig_sound = AudioSegment.from_file(file_path, format="wav")

    # Remove silence parts from the audio segment
    sound_without_silence = orig_sound.strip_silence(silence_len=1000, silence_thresh=-40)
    # Export the modified audio segment to the output file
    sound_without_silence.export(file_path, format="wav")
    print("Silence removed successfully.")


    # Create Mfcc image
    x, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=x, sr=sr)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.savefig(file_path + ".png")
    print("MFCC image created.")

    #image class prediction
    # Load the image
    img = image.load_img(file_path +".png", target_size=(224, 224))

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Make a prediction
    prediction_scor = model.predict(img_array)
    predictionSVM_scor = SVM_model.predict(img_array)
    print("cnn",prediction_scor)
    print("SVM",predictionSVM_scor)

    # Print the prediction
    #model prediction
    if prediction_scor < 0.5:
        print("The image is in the 'negative' class")
        prediction=0
    else:
        print("The image is in the 'positive' class")
        prediction=1

     #SVM #model prediction   
    if predictionSVM_scor < 1:
        print("The image is in the 'negative' class")
        predictionSVM=0
    else:
        print("The image is in the 'positive' class") 
        predictionSVM=1 

    ##Return


    message = ""
    code=0
    if prediction and predictionSVM:
        code = 0
        if form:
            # print("According to your symptoms and your cough sample, I hate to tell you that you are infected.")
            message = "According to your symptoms and your cough sample, I hate to tell you that you are infected."
        else:
            # print("Even if you don't feel covid-19 symptoms but according to your cough sample, I hate to tell you that you are infected.")
            message = "Even if you don't feel covid-19 symptoms but according to your cough sample, I hate to tell you that you are infected."

    elif not prediction and not predictionSVM:
        code = 1
        if form:
        #    print("Congratulation You are healthy, you have a standard cough sound, and for your symptoms, it's a regular fever.")
            message = "Congratulation You are healthy, you have a standard cough sound, and for your symptoms, it's a regular fever."
        else:
            # print("Congratulation You are healthy, have no symptoms and your cough sounds typical.") 
            message = "Congratulation You are healthy, have no symptoms and your cough sounds typical."

    if (prediction and not predictionSVM) or (not prediction and  predictionSVM):
        # print("I need another cough sample to check, please cough again.")
        message = "I need another cough sample to check, please cough again."
        code = 2

        
    # Return the response (e.g., a JSON response)

    return {'code':code,'result': message}




