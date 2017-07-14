import unittest
from instclf import classify
from sklearn.ensemble import forest
from sklearn.ensemble.forest import ForestClassifier
import os
import numpy as np
import medleydb as mdb
import sklearn
from sklearn.externals import joblib
import sox
import tempfile as tmp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
import json
from collections import namedtuple
import operator
import pandas as pd
from collections import OrderedDict


def relpath(f):
	return os.path.join(os.path.dirname(__file__),f)


class Test(unittest.TestCase):
	def test(self):
		pass


# class TestGetMultitracks(unittest.TestCase):
# 	def():


class TestNormalizeAudioComputeMFCC(unittest.TestCase):
	def test_normalize_audio_compute_MFCC(self):
		
		fpath = "/Users/hmyip/Documents/repositories/instclf/tests/data/piano2.wav"
		actual_M, actual_y, actual_fs = classify.normalize_MFCC(fpath)
		self.assertEqual(actual_M.shape[0], 40)
		self.assertTrue(isinstance(actual_y, np.ndarray))
		self.assertTrue(isinstance(actual_fs, int))
		


class TestComputeMFCCAndLabelMatrix(unittest.TestCase):
	def test_compute_MFCC_and_label(self):
		with open("/Users/hmyip/Documents/repositories/instclf/tests/data/test_file_dict.json") as fp:
			file_dict = json.load(fp)

		# {'piano': ["tests/data/piano2.wav"],
		# 'violin': ["tests/data/piano2.wav"],
		# 'drum set': ["tests/data/piano2.wav"],
		# 'distorted electric guitar': ["tests/data/piano2.wav"],
		# 'female singer': ["tests/data/piano2.wav"],
		# 'male singer': ["tests/data/piano2.wav"],
		# 'clarinet': ["tests/data/piano2.wav"],
		# 'flute': ["tests/data/piano2.wav"],
		# 'trumpet': ["tests/data/piano2.wav"],
		# 'tenor saxophone': ["tests/data/piano2.wav"]}

		actual_mfcc, actual_label = classify.mfcc_and_label(file_dict)
		self.assertEqual(actual_mfcc.shape[0], actual_label.shape[0])


class TestStandardizeMatrix(unittest.TestCase):
	def test_standardize_matrix(self):
		matrix = np.load(relpath("data/test_mfcc_matrix.npy"))
		mean = np.load(relpath("data/test_mfcc_mean.npy"))
		std = np.load(relpath("data/test_mfcc_std.npy"))

		actual_standardized_matrix = classify.standardize_matrix(matrix=matrix, mean=mean, std=std)
		self.assertEqual(actual_standardized_matrix.shape, matrix.shape)


class TestTrain(unittest.TestCase):
	def test_random_data(self):
		mfcc_path = relpath("data/test_mfcc_matrix.npy")
		label_path = relpath("data/test_label_matrix.npy")
		model_save_path = relpath("data/model.pkl")

		if os.path.exists(model_save_path):
			os.remove(model_save_path)

		actual_clf = classify.train(mfcc_matrix_path=mfcc_path, label_matrix_path=label_path, model_save_path=model_save_path)
		self.assertTrue(isinstance(actual_clf, ForestClassifier))
		self.assertTrue(os.path.exists(model_save_path))



class TestPredictMode(unittest.TestCase):
	def test_predict_using_modes(self):
		model_save_path = relpath("data/model.pkl")
		clf = joblib.load(model_save_path)
		matrix_normal = np.load(relpath("data/test_mfcc_normal.npy"))

		actual_prediction = classify.predict_mode(clf, matrix_normal)
		self.assertEqual(actual_prediction.shape[0], matrix_normal.shape[0])

class TestInstrumentGuess(unittest.TestCase):
	def test_instrument_using_mode_predictions(self):
		audio = relpath("data/piano2.wav")
		model_save_path = relpath("data/model.pkl")
		clf = joblib.load(model_save_path)
		matrix_normal = np.load(relpath("/Users/hmyip/Documents/repositories/instclf/tests/data/piano_matrix.npy"))
		predictions = classify.predict_mode(clf, matrix_normal)
		sorted_guesses = {}
		actual_instrument, sorted_guesses = classify.instrument(predictions)
		self.assertEqual(actual_instrument, "piano")



# class TestPredictProbability(unittest.TestCase):
# 	def test_predict_using_prob(self):
# 		model_save_path = relpath("data/model.pkl")
# 		clf = joblib.load(model_save_path)
# 		matrix_normal = np.load(relpath("data/test_mfcc_normal.npy"))

# 		actual_prediction = classify.predict_prob(clf, matrix_normal)
# 		self.assertEqual(actual_prediction.shape[0], matrix_normal.shape[0])





# class TestInstrumentGuess2(unittest.TestCase):
# 	def test_instrument_using_probability_predictions(self):
# 		audio = relpath("data/piano2.wav")
# 		model_save_path = relpath("data/model.pkl")
# 		clf = joblib.load(model_save_path)
# 		matrix_normal = np.load(relpath("/Users/hmyip/Documents/repositories/instclf/tests/data/piano_matrix.npy"))
# 		predictions2 = classify.predict_prob(clf, matrix_normal)

# 		sorted_guesses = {}
# 		sorted_guesses, actual_instrument = classify.instrument2(predictions2)
# 		self.assertEqual(actual_instrument, "piano")



