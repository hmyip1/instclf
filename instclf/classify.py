
# coding: utf-8

# # Imports

# # import numpy as np
# import librosa 
# from os import environ
# environ["MEDLEYDB_PATH"] = "/Datasets/MedleyDB"
# import medleydb as mdb
# import sklearn
# import os
# import sox
# import tempfile as tmp
# import matplotlib.pyplot as plt1
# %matplotlib inline
# from sklearn.ensemble import RandomForestClassifier

# # Getting list of multitracks without bleed

# In[2]:

loader = mdb.load_all_multitracks()
no_bleed_mtracks = []
for mtrack in loader:
    print mtrack.track_id
    if not mtrack.has_bleed:
        no_bleed_mtracks.append(mtrack)


# # Get instrument labels

# In[47]:

#list of all possible instrument labels
valid_labels = mdb.multitrack.get_valid_instrument_labels()
print valid_labels


# # Loop over instrument to get files

# In[4]:

file_dict = {}
#dictionary of list of labels for every multitrack without bleed

for label in valid_labels:
    label_list = list(mdb.utils.get_files_for_instrument(label, multitrack_list = no_bleed_mtracks))
    #list of labels of multitracks with no bleed
    
    if len(label_list) == 0:
        continue
    file_dict[label] = label_list


# # Compute MFCC and splitting data

# In[109]:

train_mfcc_list = []
train_label_list = []

test_mfcc_list = []
test_label_list = []

label_index = 0

# loop over instrument labels
# for label in list(file_dict.keys()):
target_names = ["piano", "violin", "drum set", "distorted electric guitar", "female singer", "male singer", "clarinet", "flute", "trumpet", "tenor saxophone"]

for label in target_names:
    print label
    if label == "fx/processed sound":
        continue
    print len(file_dict[label])
    
    #creating folders and file names for mfcc and labels
    train_folder = "training_data"
    train_mfcc_file = os.path.join(train_folder,"%s-mfcc-train.npy" % label)
    train_label_file = os.path.join(train_folder,"%s-label-train.npy" % label)
    
    test_folder = "testing_data"
    test_mfcc_file = os.path.join(test_folder, "%s-mfcc-test" % label)
    test_label_file = os.path.join(test_folder, "%s-label-test" % label)
    
    #load existing mfcc and label files
    if os.path.exists(train_mfcc_file) and os.path.exists(train_label_file) and os.path.exists(test_mfcc_file) and os.path.exists(test_label_file):
        print "loading existing training file..."
        master_train_instrument_mfcc_matrix = np.load(train_mfcc_file) 
        master_train_instrument_label_matrix = np.load(train_label_file)
        
        if master_train_instrument_mfcc_matrix.shape[0] != 40:
            os.remove(train_mfcc_file)
            os.remove(train_label_file)
            continue
            
        print master_train_instrument_mfcc_matrix.shape
        print master_train_instrument_label_matrix.shape
            
        print "loading existing testing file..."
        master_test_instrument_mfcc_matrix = np.load(test_mfcc_file) 
        master_test_instrument_label_matrix = np.load(test_label_file)
        
        if master_test_instrument_mfcc_matrix.shape[0] != 40:
            os.remove(test_mfcc_file)
            os.remove(test_label_file)
            continue
            
        print master_test_instrument_mfcc_matrix.shape
        print master_test_instrument_label_matrix.shape
    
    #create new mfcc and label files
    else:
        print "creating new file..."
        instrument_mfcc_list = []
        instrument_label_list = []
        

        # loop over files for instruments
        for fpath in file_dict[label]:
            print fpath
            
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
            instrument_mfcc_list.append(M)
            instrument_label_list.append(lab)

        #splitting data into training and testing
        n_examples = len(instrument_mfcc_list)
        n_examples_train = int(np.floor(0.75 * n_examples))
        
        instrument_mfcc_list_train = instrument_mfcc_list[:n_examples_train]
        instrument_label_list_train = instrument_label_list[:n_examples_train]
        
        instrument_mfcc_list_test = instrument_mfcc_list[n_examples_train:]
        instrument_label_list_test = instrument_label_list[n_examples_train:]
        
        #saving training data
        master_train_instrument_mfcc_matrix = np.hstack(instrument_mfcc_list_train) #stacking matrices for each audio file
        master_train_instrument_label_matrix = np.hstack(instrument_label_list_train)
        
        print ("master_train_instrument_mfcc_matrix shape: " + str(master_train_instrument_mfcc_matrix.shape))
        np.save(train_mfcc_file, master_train_instrument_mfcc_matrix)
        print ("master_train_instrument_label_matrix shape: " + str(master_train_instrument_label_matrix.shape))
        np.save(train_label_file, master_train_instrument_label_matrix)
        
        #saving testing data
        master_test_instrument_mfcc_matrix = np.hstack(instrument_mfcc_list_test)
        master_test_instrument_label_matrix = np.hstack(instrument_label_list_test)
        
        print ("master_test_instrument_mfcc_matrix shape: " + str(master_test_instrument_mfcc_matrix.shape))
        np.save(test_mfcc_file, master_test_instrument_mfcc_matrix)
        print ("master_test_instrument_label_matrix shape: " + str(master_test_instrument_label_matrix.shape))
        np.save(test_label_file, master_test_instrument_label_matrix)
          
            
    # combining data set
    train_mfcc_list.append(master_train_instrument_mfcc_matrix) #master master, all instruments smushed
    train_label_list.append(master_train_instrument_label_matrix)
    
    test_mfcc_list.append(master_test_instrument_mfcc_matrix)
    test_label_list.append(master_test_instrument_label_matrix)
        
    print ""
    label_index = label_index + 1


# In[110]:

train_mfcc_list
print "train_mfcc_list shapes"
print (len(train_mfcc_list))
print (train_mfcc_list[0].shape)
print (train_mfcc_list[-1].shape)


# In[111]:

train_mfcc_matrix = np.hstack(train_mfcc_list).T
train_label_matrix = np.hstack(train_label_list)


# In[112]:

np.save("train_mfcc_matrix.npy", train_mfcc_matrix)
np.save("train_label_matrix.npy", train_label_matrix)


# In[113]:

test_mfcc_list
print "test_mfcc_list shapes"
print (len(test_mfcc_list))
print (test_mfcc_list[0].shape)
print (test_mfcc_list[-1].shape)


# In[114]:

test_mfcc_matrix = np.hstack(test_mfcc_list).T
test_label_matrix = np.hstack(test_label_list)


# In[115]:

np.save("test_mfcc_matrix.npy", test_mfcc_matrix)
np.save("test_label_matrix.npy", test_label_matrix)


# # Standardizing MFCC Matrix

# In[116]:

train_mfcc_means = np.mean(train_mfcc_matrix, axis = 0)
train_mfcc_std = np.std(train_mfcc_matrix, axis=0)

np.save("train_mfcc_means.npy", train_mfcc_means)
np.save("train_mfcc_std.npy", train_mfcc_std)

train_mfcc_matrix_normal = (train_mfcc_matrix - train_mfcc_means)/train_mfcc_std
test_mfcc_matrix_normal = (test_mfcc_matrix - train_mfcc_means)/train_mfcc_std

label_values = list(file_dict.keys())
np.save("label_values.npy", label_values)


# # Plot MFCC Data Matrix

# In[15]:

label_index_list = np.arange(len(label_values))
plt.figure(figsize = (30, 50))

start_index=0
end_index=-1

plt.subplot(2,2,1)
# plt.imshow(mfcc_matrix[start_index:end_index, 0:2].T, origin = "lower", interpolation='none')
plt.plot(mfcc_matrix[:, 0])
plt.plot(mfcc_matrix[:, 1])
plt.plot(mfcc_matrix[:, 2])
# plt.colorbar()
plt.axis("auto")
plt.axis("tight")
plt.xlabel("Example Number")
plt.ylabel("MFCC Coefficient")

plt.subplot(2,2,3)
plt.plot(label_matrix[start_index:end_index], "o")
plt.axis("auto")
plt.axis("tight")
plt.xlabel("Example Number")
plt.ylabel("Instrument Label Number")
plt.yticks(label_index_list, label_values, rotation='horizontal')

plt.subplot(2,2,2)
# plt.imshow(mfcc_matrix_normal[start_index:end_index, 0:2].T, origin = "lower", interpolation='none')
# plt.colorbar()
plt.plot(mfcc_matrix_normal[:, 0])
plt.plot(mfcc_matrix_normal[:, 1])
plt.plot(mfcc_matrix_normal[:, 2])
plt.axis("auto")
plt.axis("tight")
plt.xlabel("Example Number")
plt.ylabel("Standardized MFCC Coefficient")

plt.subplot(2,2,4)
plt.plot(label_matrix[start_index:end_index], "o")
plt.axis("auto")
plt.axis("tight")
plt.xlabel("Example Number")
plt.ylabel("Instrument Label Number")
plt.yticks(label_index_list, label_values, rotation='horizontal')

plt.show()


# # Training classifier

# In[117]:

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = (train_mfcc_matrix_normal, test_mfcc_matrix_normal, train_label_matrix, test_label_matrix)


# In[118]:

print x_train.shape
print x_test.shape
print y_train.shape
print y_test.shape


# In[119]:

print "training classifier with unweighted classes"
clf = RandomForestClassifier(n_estimators = 10, class_weight = None) #unweighted based on class occurance
clf.fit(x_train, y_train)

print "training classifier with weighted classes"
clf2 = RandomForestClassifier(n_estimators = 10, class_weight = "balanced") #weighted based on class occurance
clf2.fit(x_train, y_train)

print "saving classifiers"
from sklearn.externals import joblib
joblib.dump(clf, "instrument_classifier.pkl")
joblib.dump(clf2, "instrument_classifier2.pkl")


# # Scores for weighted and unweighted classifiers

# In[120]:

print ("OVERALL SCORES")
print ("")

#unweighted scores
training_score = clf.score(x_train, y_train) #how well classifer makes model to fit data
testing_score = clf.score(x_test, y_test) #how well the model generalizes to new data

print ("Training Score of Unweighted Classifier: " + str(training_score))
print ("Training Score of Unweighted Classifier: " + str(testing_score))

print ("")
#weighted scores
training_score2 = clf2.score(x_train, y_train) #how well classifer makes model to fit data
testing_score2 = clf2.score(x_test, y_test) #how well the model generalizes to new data

print ("Training Score of Weighted Classifier: " + str(training_score2))
print ("Testing Score of Weighted Classifier: " + str(testing_score2))
print ""



# # Classification Report and Confusion Matrix

# In[121]:

from sklearn.metrics import classification_report

y_predicted = clf.predict(x_test) #unweighted
y_predicted2 = clf2.predict(x_test) #weighted


print "CLASSIFICATION REPORT FOR UNWEIGHTED MODEL"
print ""
print(classification_report(y_test, y_predicted, target_names=target_names))
print ""
print "CLASSIFICATION REPORT FOR WEIGHTED MODEL"
print ""
print(classification_report(y_test, y_predicted2, target_names=target_names))


# In[127]:

from sklearn.metrics import confusion_matrix

print ("Classes: " + str(target_names))
print ""

print "CONFUSION MATRIX FOR UNWEIGHTED MODEL"
print ""

c_mat = confusion_matrix(y_test, y_predicted)
support = np.sum(c_mat, axis=1).astype(float)
c_mat_norm = np.round(c_mat/support[:,None], 3)
print c_mat_norm

print ""

print "CONFUSION MATRIX FOR WEIGHTED MODEL"

print ""
c_mat2 = confusion_matrix(y_test, y_predicted2)
c_mat2_norm = np.round(c_mat2/support[:,None], 3)
print c_mat2_norm


# In[128]:

plt.figure(figsize=(24, 7))

plt.subplot(1,2,1)
plt.title("Confusion Matrix of Unweighted Model", size = 22)
print ""
plt.imshow(c_mat_norm, origin="upper", interpolation="none")
plt.colorbar()
target_index = range(len(target_names))
plt.xticks(target_index, target_names, rotation="vertical", size=15)
plt.xlabel("Predicted Instrument", size=20)
plt.yticks(target_index, target_names, size=15)
plt.ylabel("Actual Instrument", size=20)

plt.subplot(1,2,2)
plt.title("Confusion Matrix of Weighted Model", size = 22)
plt.imshow((c_mat2_norm), origin="upper", interpolation="none")
plt.colorbar()
target_index = range(len(target_names))
plt.xticks(target_index, target_names, rotation="vertical", size=15)
plt.xlabel("Predicted Instrument", size=20)
plt.yticks(target_index, target_names, size=15)
plt.ylabel("Actual Instrument", size=20)



# In[27]:

#IRRELEVANT NOW
# def clf_predict(audio_fpath, clf, mfcc_means, mfcc_std):
#     # normalizing volume, removing silence
#     temp_fpath = tmp.NamedTemporaryFile(suffix=".wav")
#     tfm = sox.Transformer()
#     tfm.norm(db_level=-6)
#     tfm.silence()
#     tfm.build(audio_fpath, temp_fpath.name)
        
#     # load audio
#     y, fs = librosa.load(temp_fpath.name)
        
#     # compute MFCCs
#     M = librosa.feature.mfcc(y, sr=fs, n_mfcc=40)
    
#     M_normal = (M - mfcc_means)/mfcc_std
    
#     predicted_label = clf.predict(M_normal.T)
    

