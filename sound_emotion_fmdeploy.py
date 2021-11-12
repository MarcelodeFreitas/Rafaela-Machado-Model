import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import logging

def extract_features(filename):
    try:
        features = np.empty((0,193))
        X, sample_rate = librosa.load(filename)
        stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        return features
    except: 
        return logging.exception("extract_features: ")

def predict_sound(model, sound_file_path, output_file_name, output_directory_path):
    try:
        emotions = ["Anger","Excitement","Fear","Joy","Relaxing","Sadness",
                "Surprise"]
        sound = extract_features(sound_file_path)
        prediction = model.predict(sound)
        if len(prediction) == 0: 
            print ("No prediction")
        ind = np.argpartition(prediction[0], -2)[-2:]
        ind[np.argsort(prediction[0][ind])]
        ind = ind[::-1]
        
        emotion_result = ["Top guess: " + emotions[ind[0]], "2nd guess: " + emotions[ind[1]]]
        """ print("emotion_result: ", emotion_result) """
        percentage_result = [round(prediction[0,ind[0]]*100,1), round(prediction[0,ind[1]]*100,1)]
        """ print("percentage_result: ", percentage_result) """
        
        plt.bar(emotion_result, percentage_result)
        plt.title('Sound emotion prediction')
        plt.xlabel('Emotion')
        plt.ylabel('Prediction accuracy (%)')
        plt.text(0, round(prediction[0,ind[0]]*100,1) + 0.25, round(prediction[0,ind[0]]*100,1), fontweight = 'bold')
        plt.text(1, round(prediction[0,ind[1]]*100,1) + 0.25, round(prediction[0,ind[1]]*100,1), fontweight = 'bold')
        plt.savefig(output_directory_path + output_file_name + ".png", format="png", dpi=400)
        """ plt.show() """
        
        print ("Top guess: ", emotions[ind[0]], " (",round(prediction[0,ind[0]],3),")")
        print ("2nd guess: ", emotions[ind[1]], " (",round(prediction[0,ind[1]],3),")")
    except: 
        return logging.exception("predict_sound: ")
    
def input_validation(input_file_path):
    try:
        if not (input_file_path.lower().endswith(('.wav'))):
            raise ValueError('file extension error: file must be .wav')
    except Exception as e:
        logging.error(str(e))
    
def load_models(modelpaths):
    global model
    try:
        for i in modelpaths:
            name = i["name"]
            path = i["path"]
            with tf.device('/cpu:0'):
                if (name == "best_MPLEmotions_model.h5"):
                    model = load_model(path, compile = True)
    except: 
        return logging.exception("load_models: ")
    
def run(input_file_path, output_file_name, output_directory_path):
    try:
        predict_sound(model, input_file_path, output_file_name, output_directory_path)
    except:
        return logging.exception("run: ")