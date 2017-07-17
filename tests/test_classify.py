import unittest
from instclf import classify
from sklearn.ensemble import forest
from sklearn.ensemble.forest import ForestClassifier
import os
import numpy as np
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
import json
from collections import namedtuple
import operator
from collections import OrderedDict
import pandas as pd
from pandas import DataFrame

TARGET_NAMES = ["piano", "violin", "drum set", "distorted electric guitar", "female singer", "male singer", "clarinet", "flute", "trumpet", "tenor saxophone"]


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

		actual_mfcc, actual_label = classify.mfcc_and_label(file_dict)
		self.assertEqual(actual_mfcc.shape[0], actual_label.shape[0])


class TestStandardizeMatrix(unittest.TestCase):
	def test_standardize_matrix(self):
		matrix = np.load(relpath("data/test_mfcc_matrix.npy"))
		mean = np.load(relpath("data/test_mfcc_mean.npy"))
		std = np.load(relpath("data/test_mfcc_std.npy"))

		actual_standardized_matrix = classify.standardize_matrix(matrix=matrix, mean=mean, std=std)
		self.assertEqual(actual_standardized_matrix.shape, matrix.shape)


class TestCreatingData(unittest.TestCase):
	def test_create_data(self):
		mfcc_matrix = np.load(relpath("data/test_mfcc_matrix.npy"))
		label_matrix = np.load(relpath("data/test_label_matrix.npy"))
		mean_path = relpath("data/test_mfcc_mean.npy")
		std_path = relpath("data/test_mfcc_std.npy")
		mfcc_path = relpath("data/test_mfcc_matrix.npy")
		label_path = relpath("data/test_label_matrix.npy")

		create_data = classify.create_data(train_mfcc_matrix=mfcc_matrix, train_label_matrix=label_matrix, 
			mfcc_means_path=mean_path, mfcc_std_path=std_path, mfcc_matrix_path=mfcc_path, label_matrix_path=label_path)

		self.assertTrue(os.path.exists(mfcc_path))
		self.assertTrue(os.path.exists(label_path))


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
		self.assertTrue(isinstance(sorted_guesses, OrderedDict))
		for i in range(len(TARGET_NAMES)-1):
			self.assertTrue(sorted_guesses.values()[i] >= sorted_guesses.values()[i+1])


class TestRealData(unittest.TestCase):
	def test_classifier_on_real_audio_data(self):
		audio = relpath("data/piano2.wav")
		mean_path = relpath("data/test_mfcc_mean.npy")
		std_path = relpath("data/test_mfcc_std.npy")

		guess_chart = classify.real_data(audio_file=audio)
		self.assertTrue(isinstance(guess_chart, DataFrame))
		self.assertEqual(guess_chart.shape[0], len(TARGET_NAMES))
		self.assertEqual(guess_chart.shape[1], 2)
		
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



