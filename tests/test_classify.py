import unittest
from instclf import classify
from sklearn.ensemble import forest
from sklearn.ensemble.forest import ForestClassifier
import os

def relpath(f):
	return os.path.join(os.path.dirname(__file__),f)


class Test(unittest.TestCase):
	def test(self):
		pass



# class CreateData(unittest.TestCase):
# 	def test_load_multitracks(self):
# 		self.loader = mdb.load_all_multitracks()


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




# class TestPredict(unittest.TestCase):





