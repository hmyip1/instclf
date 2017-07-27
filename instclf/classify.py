import numpy as np
import librosa
import six
from os import environ
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
import json



TARGET_NAMES = ["piano", "violin", "drum_set", "distorted_electric_guitar", "female_singer", "male_singer", "clarinet", "flute", "trumpet", "tenor_saxophone"]

MFCC_MEANS_PATH = "resources/mfcc_means.npy"
MFCC_STD_PATH = "resources/mfcc_std.npy"
MFCC_MATRIX_PATH = "resources/mfcc_matrix.npy"
LABEL_MATRIX_PATH = "resources/label_matrix.npy"
MODEL_SAVE_PATH = "resources/instrument_classifier.pkl"



#STEP 1------------------------------------

def get_data():

    """
    Load the dictionary of data from MedleyDB and Philharmonia.

    Parameters
    ----------
    None

    Returns
    -------
    file_dict: dictionary
        dictionary of instrument and filepaths of examples from MedleyDB and Philharmonia
    """

    with open('file_dict.json', 'r') as fp:
        file_dict = json.load(fp)
    return file_dict


def compute_features(file):

    """
    Compute the features of an audio file and return a stacked matrix

    Parameters
    ----------
    file: str
        file path of audio

    Returns
    -------
    M: array
        Matrix of features vs time
    y: array
        audio time series
    fs: int
        sampling rate
    """

    y, fs = librosa.load(file)

    M = np.array(librosa.feature.mfcc(y, sr=fs, n_mfcc=40))
    # mfcc_delta = np.array(librosa.feature.delta(mfcc))
    # mfcc_delta_delta = np.array(librosa.feature.delta(mfcc, order=2))

    # M = np.vstack((mfcc, mfcc_delta, mfcc_delta_delta))

    return M, y, fs

def normalize_audio(file):

    """
    Normalize volume and remove silence from audio file.

    Parameters
    ----------
    file: str
        file path of audio

    Returns
    -------
    None
    """

    temp_fpath = tmp.NamedTemporaryFile(suffix=".wav")
    tfm = sox.Transformer()
    tfm.norm(db_level=-6)
    tfm.silence()
    tfm.build(file, temp_fpath.name)



def mfcc_and_label(n_instruments=None, file_dict=None):
    """
    Retrieves data, loops over every file for each instrument in TARGET NAMES 
    to normalize audio and compute features. Matrices are concatenated into a 
    master matrix across all audio files. Returns the MFCC and label matrix.


    Parameters
    ----------
    n_instruments: int
        Specifies which instruments of TARGET NAMES to retrieve data - 
        purpose is to shorten data fetching while fixing code.

    file_dict: dictionary
        dictionary of instrument and filepaths of examples from MedleyDB and Philharmonia
 

    Returns
    -------
    train_mfcc_matrix: array
        Matrix of features across all training examples

    train_label_matrix: array
        Matrix of labels across all training examles
    """


    if file_dict is None:
        file_dict = get_data()

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

        # loop over files for each instrument
        for fpath in file_dict[label]:

            normalize_audio(fpath)
            M, y, fs = compute_features(fpath)

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

    return (train_mfcc_matrix, train_label_matrix)



def standardize_matrix(matrix, mean, std):

    """
    Standardizes a matrix

    Paramters
    ---------
    matrix: array
        matrix that is being normalized.

    mean: array
        Row vector containing the mean of each column.
        
    std: array
        standard deviation of of matrix
    """

    matrix_normal = (matrix - mean)/std
    
    return matrix_normal




#STEP 2 ------------------------------------
def create_data(n_instruments=None, train_mfcc_matrix=None, train_label_matrix=None,
    mfcc_means_path=MFCC_MEANS_PATH, 
    mfcc_std_path=MFCC_STD_PATH, 
    mfcc_matrix_path=MFCC_MATRIX_PATH, 
    label_matrix_path=LABEL_MATRIX_PATH, 
    target_names=TARGET_NAMES):
    """
    Retrieves feature and label matrix, standardizes the feature matrix,
    and saves the normalized feature matrix and label matrix.

    Paramters
    ---------
    n_instrumnets: int

    train_mfcc_matrix: array
        feature matrix

    train_label_matrix: array
        label matrix

    mfcc_means_path: str
        file path to mean of feature matrix

    mfcc_std_path: str
        file path to standard deviation of feature matrix

    mfcc_matrix_path: str
        file path to feature matrix

    label_matrix_path: str
        file path to label matrix

    TARGET_NAMES: array
        vector of instruments being dealt with


    Returns
    -------
    None
    """
    

    if train_mfcc_matrix is None and train_label_matrix is None:
        train_mfcc_matrix, train_label_matrix = mfcc_and_label(n_instruments)

    #STANDARDIZING MFCC MATRIX

    train_mfcc_means = np.mean(train_mfcc_matrix, axis = 0)
    train_mfcc_std = np.std(train_mfcc_matrix, axis=0)
    np.save(mfcc_means_path, train_mfcc_means)
    np.save(mfcc_std_path, train_mfcc_std)
    
    train_mfcc_matrix_normal = standardize_matrix(train_mfcc_matrix, train_mfcc_means, train_mfcc_std)

    np.save(mfcc_matrix_path, train_mfcc_matrix_normal)
    np.save(label_matrix_path, train_label_matrix)




#STEP 3------------------------------------

def train(n_estimators, mfcc_matrix_path=MFCC_MATRIX_PATH, 
    label_matrix_path=LABEL_MATRIX_PATH,
    model_save_path=MODEL_SAVE_PATH):

    """
    Trains the RandomForest model using normalized feature and label matrix.

    Parameters
    ----------
    n_estimators: int
        Number of estimators the random forest classifier should use

    mfcc_matrix_path: str
        file path to feature matrix

    label_matrix_path: str
        file path to label matrix

    model_save_path: str
        file path to save location of classifier model

    Returns
    -------
    clf: classifier
        Trained RandomForest Classifier
    """


    train_mfcc_matrix_normal = np.load(mfcc_matrix_path)
    train_label_matrix = np.load(label_matrix_path)

    x_train, y_train = (train_mfcc_matrix_normal, train_label_matrix)
   
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight=None) #unweighted based on class occurance
    clf.fit(x_train, y_train)

    joblib.dump(clf, model_save_path)

    return clf
            



def instrument(predictions):

    """
    Predicts the instrument by calculating the mode of the vector of predictions.

    Parameters
    ----------
    predictions: array
        vector of predictions for each slice of feature

    Returns
    -------
    guess: str
        final prediciton of instrument in audio file

    guess_dict: dictionary
        dictionary of probability of each instrument in TARGET NAMES being the instrument in audio file
    """
    
    unique_elements, counts = np.unique(predictions, return_counts=True)
    frequency_predictions = [0 for i in range(len(TARGET_NAMES))]
    
    for i, j in zip(unique_elements, range(len(counts))):
        frequency_predictions[int(i)] = counts[int(j)]/float(len(predictions))

    guess_dict = {}
    instrument_probability = zip(TARGET_NAMES, frequency_predictions)
    for name, probability in instrument_probability:
        guess_dict[name] = round(probability, 3)

    mode_predictions = mode(predictions)
    guess = TARGET_NAMES[int(mode_predictions[0])]

    return guess, guess_dict



def real_data(audio_file, 
    mfcc_means_path=MFCC_MEANS_PATH,
    mfcc_std_path=MFCC_STD_PATH, 
    model_save_path=MODEL_SAVE_PATH):
    """
    Tests the classifier on an audio file given by user. 
    Prints the guess and a table of probabilities for each instrument.

    Parameters
    ----------

    audio_file: str
        file path to audio file that the classifier will act upon

    mfcc_means_path: str
        file path of feature means

    mfcc_std_path: str
        file path of feature standard deviations

    model_save_path: str
        file path of classification model

    Returns
    -------
    sorted guesses: sorted dictionary
        table of probabilities for each instrument sorted by probabiities
    """



    clf = joblib.load(model_save_path)

    train_mfcc_means = np.load(mfcc_means_path)
    train_mfcc_std = np.load(mfcc_std_path)

    # normalizing volume and compute MFCC

    normalize_audio(audio_file)
    M, y, fs = compute_features(audio_file)
    
    audio_mfcc_matrix_normal = standardize_matrix(M.T, train_mfcc_means, train_mfcc_std)
    # np.save("/Users/hmyip/Documents/repositories/instclf/tests/data/piano_matrix.npy", audio_mfcc_matrix_normal)

    #prediction with mode
    predictions = clf.predict(audio_mfcc_matrix_normal)
    guess, guess_dict = instrument(predictions)
    print ("guess: " + str(guess))

    sorted_guesses = sorted(guess_dict.items(), key=lambda item: (item[1], item[0]), reverse=True)

    for key, value in sorted_guesses:
        print ("%s: %s" % (key, value))

    return guess, guess_dict






# #prediction with probabilities


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



