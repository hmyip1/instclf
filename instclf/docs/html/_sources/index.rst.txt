.. Instrument Classifier documentation master file, created by
   sphinx-quickstart on Mon Jul 24 15:57:00 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Instrument Classifier's documentation!
=================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



.. function:: get_data()

.. function:: compute_features(file)

.. function:: normalize_audio(file)

.. function:: mfcc_and_label(n_instruments=None, file_dict=None)

.. function:: standardize_matrix(matrix, mean, std)

.. function:: create_data(n_instruments=None, train_mfcc_matrix=None, 
	train_label_matrix=None, mfcc_means_path=MFCC_MEANS_PATH, 
	mfcc_std_path=MFCC_STD_PATH, mfcc_matrix_path=MFCC_MATRIX_PATH, 
	label_matrix_path=LABEL_MATRIX_PATH, target_names=TARGET_NAMES)

.. function:: train(n_estimators, mfcc_matrix_path=MFCC_MATRIX_PATH, 
    label_matrix_path=LABEL_MATRIX_PATH, model_save_path=MODEL_SAVE_PATH)

.. function:: instrument(predictions)

.. function:: real_data(audio_file, mfcc_means_path=MFCC_MEANS_PATH,	mfcc_std_path=MFCC_STD_PATH, model_save_path=MODEL_SAVE_PATH)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
