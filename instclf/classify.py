import numpy as np
import librosa 
from os import environ
environ["MEDLEYDB_PATH"] = "/Datasets/MedleyDB"
import medleydb as mdb
import sklearn
from sklearn.externals import joblib
import os
import sox
import tempfile as tmp
import matplotlib.pyplot as plt1
from sklearn.ensemble import RandomForestClassifier


MODEL_PATH = "resources/instrument_classifier.pkl"
TARGET_NAMES = ["piano", "violin", "drum set", "distorted electric guitar", "female singer", "male singer", "clarinet", "flute", "trumpet", "tenor saxophone"]


def create_data(mfcc_means_path, mfcc_std_path, mfcc_matrix_path, label_matrix_path, target_names=TARGET_NAMES):

    #list and label all multitracks without bleed
    loader = mdb.load_all_multitracks()
    no_bleed_mtracks = []
    for mtrack in loader:
        if not mtrack.has_bleed:
            no_bleed_mtracks.append(mtrack)

    valid_labels = mdb.multitrack.get_valid_instrument_labels()

    file_dict = {} #dictionary of multitrack without bleed

    for label in valid_labels:
        label_list = list(mdb.utils.get_files_for_instrument(label, multitrack_list=no_bleed_mtracks))

        if len(label_list) == 0:
            continue
        file_dict[label] = label_list

    #get labels and mfccs
    train_mfcc_list = []
    train_label_list = []

    label_index = 0

    for label in TARGET_NAMES:
        
        instrument_mfcc_list_train = []
        instrument_label_list_train = []


        # loop over files for instruments
        for fpath in file_dict[label]:

            # normalizing volume, removing silence
            temp_fpath = tmp.NamedTemporaryFile(suffix=".wav")
            tfm = sox.Transformer()
            tfm.norm(db_level=-6)
            tfm.silence()
            tfm.build(fpath, temp_fpath.name)

            # load audio
            y, fs = librosa.load(temp_fpath.name)

            # compute MFCCs for individual audio file
            M = librosa.feature.mfcc(y, sr=fs, n_mfcc=40)

            lab = np.zeros((len(M[0]), )) + label_index
            instrument_mfcc_list_train.append(M)
            instrument_label_list_train.append(lab)


        instrument_mfcc_matrix_train = np.hstack(instrument_mfcc_list_train) #stacking matrices for each audio file
        instrument_label_matrix_train = np.hstack(instrument_label_list_train)
        np.save(train_mfcc_file, instrument_mfcc_matrix_train)
        np.save(train_label_file, instrument_label_matrix_train)
        
        train_mfcc_list.append(instrument_mfcc_matrix_train) #master master, all instruments smushed
        train_label_list.append(instrument_label_matrix_train)

        label_index = label_index + 1

    #SAVING TRAINING DATA

    train_mfcc_matrix = np.hstack(train_mfcc_list).T
    train_label_matrix = np.hstack(train_label_list)
    


    #STANDARDIZING MFCC MATRIX

    train_mfcc_means = np.mean(train_mfcc_matrix, axis = 0)
    train_mfcc_std = np.std(train_mfcc_matrix, axis=0)
    np.save(mfcc_means_path, train_mfcc_means)
    np.save(mfcc_std_path, train_mfcc_std)

    train_mfcc_matrix_normal = (train_mfcc_matrix - train_mfcc_means)/train_mfcc_std
    np.save(mfcc_matrix_path, train_mfcc_matrix_normal)
    np.save(label_matrix_path, train_label_matrix)

    label_values = sorted(list(file_dict.keys()))
    np.save("label_values.npy", TARGET_NAMES)



def train(mfcc_matrix_path, label_matrix_path):

    train_mfcc_matrix_normal = np.load(mfcc_matrix_path)
    train_label_matrix = np.load(label_matrix_path)

    x_train, y_train = (train_mfcc_matrix_normal, train_label_matrix)
   
    clf = RandomForestClassifier(n_estimators=400, class_weight=None) #unweighted based on class occurance
    clf.fit(x_train, y_train)

    joblib.dump(clf, MODEL_PATH)

    return clf
            

def predict(audio_file, mfcc_means_path, mfcc_std_path):
    joblib.load(MODEL_PATH)

    train_mfcc_means = np.load(mfcc_means_path)
    train_mfcc_std = np.load(mfcc_std_path)

    # normalizing volume
    temp_fpath = tmp.NamedTemporaryFile(suffix=".wav")
    tfm = sox.Transformer()
    tfm.norm(db_level=-6)
    tfm.build(audio_file, temp_fpath.name)

    # load audio
    y, fs = librosa.load(temp_fpath.name)

    # compute MFCCs for individual audio file
    M = librosa.feature.mfcc(y, sr=fs, n_mfcc=40)
    
    audio_mfcc_matrix_normal = (M - train_mfcc_means)/train_mfcc_std

    predictions = clf.predict(audio_mfcc_matrix_normal)
    max_prediction = np.argmax(predictions)
    return TARGET_NAMES[max_prediction]
 




