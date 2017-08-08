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
        # if request.form['submit'] == "Upload":
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

        # if request.form['submit'] == "Submit Recorded Audio":
            

    return render_template('results.html')


@APP.route('/results', methods=['GET', 'POST'])
def results():
    list_of_files = glob.glob("uploads/*")
    file_save_path = max(list_of_files, key=os.path.getctime)
    guess, guess_dict = classify.real_data(file_save_path, mfcc_means_path=MFCC_MEANS_PATH, mfcc_std_path=MFCC_STD_PATH, model_save_path=MODEL_SAVE_PATH)
    guess_dict = OrderedDict(sorted(guess_dict.items(), key=lambda item: (item[1], item[0]), reverse=True))

    if request.method == 'POST':
        return redirect("/")

    return render_template('results.html', guess=guess, guess_dict=guess_dict)


if __name__ == '__main__':

    APP.run(debug=True)




"""------------LIVE RECORDING OF AUDIO-----------------"""



import pyaudio
import wave
 
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
WAVE_OUTPUT_FILEPATH = "uploads/file.wav"
 

@APP.route('/record')
def record_audio():

    if request.method == 'POST':
        if request.form['submit'] == 'Do Something':
            pass # do something
        elif request.form['submit'] == 'Do Something Else':
            pass # do something else
        else:
            pass # unknown
    elif request.method == 'GET':
        return render_template('contact.html', form=form)





    if request.method == 'POST':

        audio = pyaudio.PyAudio()
         
        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        print ("recording...")
        frames = []
         
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print ("finished recording")
         
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
         
        waveFile = wave.open(WAVE_OUTPUT_FILEPATH, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        return redirect(url_for('results'))

    return render_template('results.html')



# def is_silent(audio_rec):
#     "Returns 'True' if below the 'silent' threshold"
#     return max(audio_rec) < THRESHOLD

# def normalize(audio_rec):
#     "Average the volume out"
#     MAXIMUM = 16384
#     times = float(MAXIMUM)/max(abs(i) for i in audio_rec)

#     r = array('h')
#     for i in audio_rec:
#         r.append(int(i*times))
#     return r

# def trim(audio_rec):
#     "Trim the blank spots at the start and end"
#     def _trim(audio_rec):
#         snd_started = False
#         r = array('h')

#         for i in audio_rec:
#             if not snd_started and abs(i)>THRESHOLD:
#                 snd_started = True
#                 r.append(i)

#             elif snd_started:
#                 r.append(i)
#         return r

#     # Trim to the left
#     audio_rec = _trim(audio_rec)

#     # Trim to the right
#     audio_rec.reverse()
#     audio_rec = _trim(audio_rec)
#     audio_rec.reverse()
#     return audio_rec

# def add_silence(audio_rec, seconds):
#     "Add silence to the start and end of 'audio_rec' of length 'seconds' (float)"
#     r = array('h', [0 for i in xrange(int(seconds*RATE))])
#     r.extend(audio_rec)
#     r.extend([0 for i in xrange(int(seconds*RATE))])
#     return r

# def record():
#     """
#     Record a word or words from the microphone and 
#     return the data as an array of signed shorts.

#     Normalizes the audio, trims silence from the 
#     start and end, and pads with 0.5 seconds of 
#     blank sound to make sure VLC et al can play 
#     it without getting chopped off.
#     """
#     p = pyaudio.PyAudio()
#     stream = p.open(format=FORMAT, channels=1, rate=RATE,
#         input=True, output=True,
#         frames_per_buffer=CHUNK_SIZE)

#     num_silent = 0
#     snd_started = False

#     r = array('h')

#     while 1:
#         # little endian, signed short
#         audio_rec = array('h', stream.read(CHUNK_SIZE))
#         if byteorder == 'big':
#             audio_rec.byteswap()
#         r.extend(audio_rec)

#         silent = is_silent(audio_rec)

#         if silent and snd_started:
#             num_silent += 1
#         elif not silent and not snd_started:
#             snd_started = True

#         if snd_started and num_silent > 30:
#             break

#     sample_width = p.get_sample_size(FORMAT)
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     r = normalize(r)
#     r = trim(r)
#     r = add_silence(r, 0.5)
#     return sample_width, r

# def record_to_file(path):
#     "Records from the microphone and outputs the resulting data to 'path'"
#     sample_width, data = record()
#     data = pack('<' + ('h'*len(data)), *data)

#     wf = wave.open(path, 'wb')
#     wf.setnchannels(1)
#     wf.setsampwidth(sample_width)
#     wf.setframerate(RATE)
#     wf.writeframes(data)
#     wf.close()

# if __name__ == '__main__':
#     print("listening to audio through the microphone")
#     record_to_file('audio.wav')
#     print("done - result written to audio.wav")

