"""
Main webapp.py function
"""
#!/usr/bin/env python
import os
from time import gmtime, strftime
import argparse
import numpy
from flask import Flask, jsonify, render_template, request, Response, flash, redirect, url_for
from flask_mail import Mail
from utils import connect_db, get_header, send_mail
from utils import fill_table, format_headers, format_comments, allowed_file
from emails import REQUEST_BODY as request_record_body
from emails import ASSIGNEE_BODY as assignee_body
from functools import wraps
import csv
import classify


APP = Flask(__name__)
APP.config.update(dict(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=465,
    MAIL_USE_TLS=False,
    MAIL_USE_SSL=True,
    MAIL_USERNAME='medleydbaccess@gmail.com',
))

APP.config.update(dict(
    DATABASE=os.path.join(APP.root_path, 'static', 'ticketmanager.db'),
    UPLOAD_FOLDER="uploads"
))


"""------------LIVE RECORDING OF AUDIO-----------------"""

from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100

def is_silent(audio_rec):
    "Returns 'True' if below the 'silent' threshold"
    return max(audio_rec) < THRESHOLD

def normalize(audio_rec):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in audio_rec)

    r = array('h')
    for i in audio_rec:
        r.append(int(i*times))
    return r

def trim(audio_rec):
    "Trim the blank spots at the start and end"
    def _trim(audio_rec):
        snd_started = False
        r = array('h')

        for i in audio_rec:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    audio_rec = _trim(audio_rec)

    # Trim to the right
    audio_rec.reverse()
    audio_rec = _trim(audio_rec)
    audio_rec.reverse()
    return audio_rec

def add_silence(audio_rec, seconds):
    "Add silence to the start and end of 'audio_rec' of length 'seconds' (float)"
    r = array('h', [0 for i in xrange(int(seconds*RATE))])
    r.extend(audio_rec)
    r.extend([0 for i in xrange(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        audio_rec = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            audio_rec.byteswap()
        r.extend(audio_rec)

        silent = is_silent(audio_rec)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

if __name__ == '__main__':
    print("listening to audio through the microphone")
    record_to_file('audio.wav')
    print("done - result written to audio.wav")



"""------------------UPLOAD AUDIO--------------"""

@APP.route('/uploadaudio')
def upload_audio():
    """
    Renders upload.html
    """
    return render_template('uploadaudio.html', upload_url='upload')




@APP.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    Uploads audio file to the upload folder from the upload.html page
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('uploadaudio'))

        requested_file = request.files['file']
        if requested_file.filename == '':
            flash('No selected file')
            return redirect(url_for('uploadaudio'))

        if requested_file and allowed_file(requested_file.filename):
            file_ext = requested_file.filename.rsplit('.', 1)[1]
            file_save_name = 'uploaded_audio'
            file_save_path = os.path.join(APP.config['UPLOAD_FOLDER'], file_save_name)
            classify.predict(file_save_path)

        return redirect(url_for('prediction'))


@APP.route('/prediction')
def prediction():

    return render_template('prediction.html')



if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description="instclf_webapp"
        )
    PARSER.add_argument("password", type=str, help="medleydb gmail password")
    PARSER.add_argument(
        "--debug", action="store_const", const=True, default=False
    )
    ARGS = PARSER.parse_args()
    APP.config.update(dict(MAIL_PASSWORD=ARGS.password))

    MAIL = Mail(APP)

    APP.run(port=5080, host='0.0.0.0', debug=ARGS.debug)


