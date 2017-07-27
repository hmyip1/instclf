import unittest
from instclf import classify
from sklearn.ensemble import forest
from sklearn.ensemble.forest import ForestClassifier
import os
import numpy as np
import tempfile as tmp
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import mode
import json
from collections import namedtuple
import operator
from collections import OrderedDict


def relpath(f):
	return os.path.join(os.path.dirname(__file__),f)


TARGET_NAMES = ["piano", "violin", "drum_set", "distorted_electric_guitar", "female_singer", "male_singer", "clarinet", "flute", "trumpet", "tenor_saxophone"]
MFCC_PATH = relpath("data/test_mfcc_matrix.npy")
MFCC_MEAN_PATH = relpath("data/test_mfcc_mean.npy")
MFCC_STD_PATH = relpath("data/test_mfcc_std.npy")
LABEL_PATH = relpath("data/test_label_matrix.npy")
AUDIO_PATH = relpath("piano2.wav")
MODEL_PATH = relpath("data/model.pkl")



class Test(unittest.TestCase):
	def test(self):
		pass



class TestGettingFileDict(unittest.TestCase):
	def test_get_data(self):

		fp = relpath("data")
		os.path.join("data")
		file = relpath("data/test_file_dict.json")
		with open(file, 'r') as fp:
			file_dict = json.load(fp)

		self.assertEqual(len(file_dict.keys()), 10)



class TestComputeFeatures(unittest.TestCase):
	def test_compute_features(self):
		
		fpath = relpath("data/piano2.wav")
		temp_fpath = tmp.NamedTemporaryFile(suffix=".wav")
		M, y, fs = classify.compute_features(fpath)
		self.assertEqual(M.shape[0], 40)
		self.assertTrue(isinstance(y, np.ndarray))
		self.assertTrue(isinstance(fs, int))


class TestComputeMFCCAndLabelMatrix(unittest.TestCase):
	def test_compute_MFCC_and_label(self):

		audio_path = relpath("data/piano2.wav")

		file_dict = {"tenor_saxophone": [audio_path], 
		"male_singer": [audio_path], 
		"distorted_electric_guitar": [audio_path], 
		"female_singer": [audio_path], 
		"drum_set": [audio_path], 
		"violin": [audio_path], 
		"piano": [audio_path], 
		"flute": [audio_path], 
		"trumpet": [audio_path], 
		"clarinet": [audio_path]}

		actual_mfcc, actual_label = classify.mfcc_and_label(file_dict=file_dict)
		self.assertEqual(actual_mfcc.shape[0], actual_label.shape[0])


class TestStandardizeMatrix(unittest.TestCase):
	def test_standardize_matrix(self):
		matrix = np.load(MFCC_PATH)
		mean = np.load(MFCC_MEAN_PATH)
		std = np.load(MFCC_STD_PATH)

		actual_standardized_matrix = classify.standardize_matrix(matrix=matrix, mean=mean, std=std)
		self.assertEqual(actual_standardized_matrix.shape, matrix.shape)


class TestCreatingData(unittest.TestCase):
	def test_create_data(self):
		mfcc_matrix = np.load(MFCC_PATH)
		label_matrix = np.load(LABEL_PATH)

		create_data = classify.create_data(train_mfcc_matrix=mfcc_matrix, train_label_matrix=label_matrix, 
			mfcc_means_path=MFCC_MEAN_PATH, mfcc_std_path=MFCC_STD_PATH, mfcc_matrix_path=MFCC_PATH, label_matrix_path=LABEL_PATH)

		self.assertTrue(os.path.exists(MFCC_PATH))
		self.assertTrue(os.path.exists(LABEL_PATH))


class TestTrain(unittest.TestCase):
	def test_random_data(self):

		if os.path.exists(MODEL_PATH):
			os.remove(MODEL_PATH)

		actual_clf = classify.train(n_estimators=10, 
			mfcc_matrix_path=MFCC_PATH, label_matrix_path=LABEL_PATH, model_save_path=MODEL_PATH)
		self.assertTrue(isinstance(actual_clf, ForestClassifier))
		self.assertTrue(os.path.exists(MODEL_PATH))


class TestInstrumentGuess(unittest.TestCase):
	def test_instrument_using_mode_predictions(self):
		audio = AUDIO_PATH
		model_save_path = MODEL_PATH
		clf = joblib.load(MODEL_PATH)
		matrix_normal = np.load(relpath("data/piano_matrix.npy"))
		predictions = clf.predict(matrix_normal)

		actual_instrument, guess_dict = classify.instrument(predictions)
		self.assertTrue(actual_instrument in TARGET_NAMES)
		self.assertEqual(len(guess_dict.keys()), len(TARGET_NAMES))
		

class TestRealData(unittest.TestCase):
	def test_classifier_on_real_audio_data(self):

		guess, guess_dict = classify.real_data(audio_file=AUDIO_PATH, 
			mfcc_means_path=MFCC_MEAN_PATH,
			mfcc_std_path=MFCC_STD_PATH,
			model_save_path=MODEL_PATH)
		self.assertTrue(isinstance(guess_dict, dict))
		self.assertEqual(guess_dict.shape[0], len(TARGET_NAMES))
		# for i in range(len(TARGET_NAMES)-1):
		# 	self.assertTrue(list(sorted_guesses.values())[i] >= list(sorted_guesses.values())[i+1])

		
# class TestPredictProbability(unittest.TestCase):
# 	def test_predict_using_prob(self):
# 		model_save_path = MODEL_PATH
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



