import numpy as np
import librosa
import six
from os import environ
environ["MEDLEYDB_PATH"] = "/Datasets/MedleyDB"
import medleydb as mdb
import sklearn
from sklearn.externals import joblib
import os
import sox
import tempfile as tmp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
from collections import namedtuple
import operator
import pandas as pd
from collections import OrderedDict



TARGET_NAMES = ["piano", "violin", "drum set", "distorted electric guitar", "female singer", "male singer", "clarinet", "flute", "trumpet", "tenor saxophone"]


def get_multitracks():
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

    label_values = sorted(list(file_dict.keys()))
    np.save("label_values.npy", label_values)

    print (file_dict)
    return file_dict



# def compute_features():
#     y, fs = librosa.load(temp_fpath.name)

#     mfcc = np.array(librosa.feature.mfcc(y, sr=fs, n_mfcc=40))
#     mfcc_delta = np.array(librosa.feature.delta(mfcc))
#     mfcc_delta_delta = np.array(librosa.feature.delta(mfcc, order=2))

#     M = np.vstack((mfcc, mfcc_delta, mfcc_delta_delta))

#     return M, y, fs

def normalize_MFCC(file):
    temp_fpath = tmp.NamedTemporaryFile(suffix=".wav")
    tfm = sox.Transformer()
    tfm.norm(db_level=-6)
    tfm.silence()
    tfm.build(file, temp_fpath.name)

    # M, y, fs = compute_features()

    y, fs = librosa.load(temp_fpath.name)


    M = librosa.feature.mfcc(y, sr=fs, n_mfcc=40)
    return M, y, fs


def mfcc_and_label(n_instruments=None, file_dict=None):

    if file_dict is None:
        file_dict = get_multitracks()

    train_mfcc_list = []
    train_label_list = []

    label_index = 0

    if n_instruments is None:
        instrument_labels = TARGET_NAMES
    else:
        instrument_labels = TARGET_NAMES[n_instruments:]

    for label in instrument_labels:
        
        instrument_mfcc_list_train = []
        instrument_label_list_train = []

        # loop over files for instruments
        for fpath in file_dict[label]:

            M, y, fs = normalize_MFCC(fpath)

            lab = np.zeros((len(M[0]), )) + label_index
            instrument_mfcc_list_train.append(M)
            instrument_label_list_train.append(lab)


        instrument_mfcc_matrix_train = np.hstack(instrument_mfcc_list_train) #stacking matrices for each audio file
        instrument_label_matrix_train = np.hstack(instrument_label_list_train)
        
        train_mfcc_list.append(instrument_mfcc_matrix_train) #master master, all instruments smushed
        train_label_list.append(instrument_label_matrix_train)

        label_index = label_index + 1


    train_mfcc_matrix = np.hstack(train_mfcc_list).T
    train_label_matrix = np.hstack(train_label_list)

    print ("shape1: " + str(train_mfcc_matrix.shape))
    print ("shape1: " + str(train_label_matrix.shape))

    return (train_mfcc_matrix, train_label_matrix)


def standardize_matrix(matrix, mean, std):

    matrix_normal = (matrix - mean)/std
    return matrix_normal




def create_data(n_instruments=None, train_mfcc_matrix=None, train_label_matrix=None,
    mfcc_means_path="/Users/hmyip/Documents/repositories/instclf/instclf/resources/mfcc_means.npy", 
    mfcc_std_path="/Users/hmyip/Documents/repositories/instclf/instclf/resources/mfcc_std.npy", 
    mfcc_matrix_path="/Users/hmyip/Documents/repositories/instclf/instclf/resources/mfcc_matrix.npy", 
    label_matrix_path="/Users/hmyip/Documents/repositories/instclf/instclf/resources/label_matrix.npy", 
    target_names=TARGET_NAMES):

    if train_mfcc_matrix is None and train_label_matrix is None:
        train_mfcc_matrix, train_label_matrix = mfcc_and_label(n_instruments)

    #get labels and mfccs of all multitracks without bleed
        

    #STANDARDIZING MFCC MATRIX

    train_mfcc_means = np.mean(train_mfcc_matrix, axis = 0)
    train_mfcc_std = np.std(train_mfcc_matrix, axis=0)
    np.save(mfcc_means_path, train_mfcc_means)
    np.save(mfcc_std_path, train_mfcc_std)
    
    train_mfcc_matrix_normal = standardize_matrix(train_mfcc_matrix, train_mfcc_means, train_mfcc_std)

    np.save(mfcc_matrix_path, train_mfcc_matrix_normal)
    np.save(label_matrix_path, train_label_matrix)



def train(n_estimators, mfcc_matrix_path="/Users/hmyip/Documents/repositories/instclf/instclf/resources/mfcc_matrix.npy", 
    label_matrix_path="/Users/hmyip/Documents/repositories/instclf/instclf/resources/label_matrix.npy",
    model_save_path="instclf/resources/instrument_classifier.pkl"):

    train_mfcc_matrix_normal = np.load(mfcc_matrix_path)
    train_label_matrix = np.load(label_matrix_path)

    x_train, y_train = (train_mfcc_matrix_normal, train_label_matrix)
   
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight=None) #unweighted based on class occurance
    clf.fit(x_train, y_train)

    joblib.dump(clf, model_save_path)

    return clf
            

def predict_mode(classifier, matrix):
    predictions1 = classifier.predict(matrix)
    return predictions1

def instrument(predictions):
    
    unique_elements, counts = np.unique(predictions, return_counts=True)
    print (unique_elements)
    frequency_predictions = [0 for i in range(len(TARGET_NAMES))]
    # print frequency_predictions
    # print counts
    
    for i, j in zip(unique_elements, range(len(counts))):
        frequency_predictions[int(i)] = counts[int(j)]/float(len(predictions))

    print (frequency_predictions)
    guess_dict = {}
    instrument_probability = zip(TARGET_NAMES, frequency_predictions)
    for name, probability in instrument_probability:
        guess_dict[name] = round(probability, 3)

    # sorted_guesses = OrderedDict(sorted(guess_dict.items(), key=operator.itemgetter(1), reverse=True))

    mode_predictions = mode(predictions)
    guess = TARGET_NAMES[int(mode_predictions[0])]

    return guess, guess_dict

# def predict_prob(classifier, matrix):
#     predictions2 = classifier.predict_proba(matrix)
#     return predictions2

# def instrument2(predictions):
#     avg_predictions = np.round(predictions.mean(axis=0), 3)
#     max_prediction = np.argmax(avg_predictions)
#     guess2 = TARGET_NAMES[max_prediction]
#     print avg_predictions

#     guess_dict = {}
#     instrument_probability = zip(TARGET_NAMES, avg_predictions)
#     for name, probability in instrument_probability:
#         guess_dict[name] = round(probability, 3)

#     sorted_guesses = OrderedDict(sorted(guess_dict.items(), key=operator.itemgetter(1), reverse=True))

#     return sorted_guesses, guess2



def real_data(audio_file, 
    mfcc_means_path="/Users/hmyip/Documents/repositories/instclf/instclf/resources/mfcc_means.npy",
    mfcc_std_path="/Users/hmyip/Documents/repositories/instclf/instclf/resources/mfcc_std.npy", 
    model_save_path="/Users/hmyip/Documents/repositories/instclf/instclf/resources/instrument_classifier.pkl"):
    
    clf = joblib.load(model_save_path)

    train_mfcc_means = np.load(mfcc_means_path)
    train_mfcc_std = np.load(mfcc_std_path)

    # normalizing volume and compute MFCC

    M, y, fs = normalize_MFCC(audio_file)
    
    audio_mfcc_matrix_normal = standardize_matrix(M.T, train_mfcc_means, train_mfcc_std)
    # np.save("/Users/hmyip/Documents/repositories/instclf/tests/data/piano_matrix.npy", audio_mfcc_matrix_normal)

    #prediction with mode
    predictions1 = predict_mode(clf, audio_mfcc_matrix_normal)
    guess, guess_dict = instrument(predictions1)
    print ("guess1: " + str(guess))

    sorted_guesses = sorted(guess_dict.iteritems(), reverse=True, key=lambda(k,v): (v,k))

    for key, value in sorted_guesses:
        print ("%s: %s" % (key, value))

    # guess_chart = pd.DataFrame(data=sorted_guesses.as_matrix, columns = ["instrument", "percent chance"], copy=False)
    # print (guess_chart)
    return sorted_guesses

    # #prediction with probabilities


    # predictions2 = clf.predict_proba(audio_mfcc_matrix_normal)
    # sorted_guesses, guess2 = instrument2(predictions2)
    # print pd.DataFrame(sorted_guesses.items(), columns = ["instrument", "percent chance"])
    # print ("guess2: " + str(guess2))




    # #method 1
    
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.plot(np.arange(len(TARGET_NAMES)), predictions, "o")
    # plt.xticks(np.arange(len(TARGET_NAMES)), TARGET_NAMES, rotation="vertical")

    # #method 2

    # plt.subplot(1,2,2)
    # plt.plot(np.arange(len(TARGET_NAMES)), avg_predictions, "o")
    # plt.xticks(np.arange(len(TARGET_NAMES)), TARGET_NAMES, rotation="vertical")
    # plt.show()



