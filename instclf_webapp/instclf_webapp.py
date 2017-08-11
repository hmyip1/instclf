import os
from instclf import classify
import numpy as np
import glob
import pickle
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session
# from wtforms import Form, TextAreaField, validators
from vectorizer import vect
import joblib
from werkzeug.utils import secure_filename
from collections import OrderedDict

TARGET_NAMES = ["piano", "violin", "drum_set", "distorted_electric_guitar", "female_singer", "male_singer", "clarinet", "flute", "trumpet", "tenor_saxophone"]
MFCC_MEANS_PATH = "../instclf/resources/mfcc_means.npy"
MFCC_STD_PATH = "../instclf/resources/mfcc_std.npy"
MFCC_MATRIX_PATH = "../instclf/resources/mfcc_matrix.npy"
LABEL_MATRIX_PATH = "../instclf/resources/label_matrix.npy"
MODEL_SAVE_PATH = "../instclf/resources/instrument_classifier.pkl"
UPLOAD_FOLDER="uploads"
ALLOWED_EXTENSIONS = set(['wav', 'mp3'])




APP = Flask(__name__)
APP.secret_key = "super secret key"

APP.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cur_dir = os.path.dirname(__file__)



def allowed_file(filename):
    file_ext = filename.rsplit('.', 1)[1]
    return '.' in filename and file_ext in ALLOWED_EXTENSIONS


@APP.route('/')
def index():

    return render_template('index.html')

@APP.route('/upload', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':

        if 'file' not in request.files:
            flash('No file part')
            return redirect("/")

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect("/")

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file_save_path = os.path.join(cur_dir, APP.config['UPLOAD_FOLDER'], filename)
            file.save(file_save_path)
            return redirect(url_for('results'))
            

    return render_template('results.html')


@APP.route('/results', methods=['GET', 'POST'])
def results():
    list_of_files = glob.glob("uploads/*")
    file_save_path = max(list_of_files, key=os.path.getctime)
    guess, guess_dict = classify.predict(file_save_path, mfcc_means_path=MFCC_MEANS_PATH, mfcc_std_path=MFCC_STD_PATH, model_save_path=MODEL_SAVE_PATH)
    guess_dict = OrderedDict(sorted(guess_dict.items(), key=lambda item: (item[1], item[0]), reverse=True))

    if request.method == 'POST':
        return redirect("/")

    return render_template('results.html', guess=guess, guess_dict=guess_dict)



if __name__ == '__main__':

    APP.run(debug=True)
