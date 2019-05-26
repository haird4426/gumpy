#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'examples'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# # ConvNet Architecture for Decoding EEG MI Data using Spectrogram Representations
#%% [markdown]
# ## Preparation
# 
# In case that gumpy is not installed as a module, we need to specify the path to ``gumpy``. In addition, we wish to configure jupyter notebooks and any backend properly. Note that it may take some time for ``gumpy`` to load due to the number of dependencies

#%%
from __future__ import print_function
import os; os.environ["THEANO_FLAGS"] = "device=gpu0"
import os.path
from datetime import datetime
import sys
sys.path.append('../../gumpy')

import gumpy
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# 
# To use the models provided by `gumpy-deeplearning`, we have to set the path to the models directory and import it. If you installed `gumpy-deeplearning` as a module, this step may not be required.

#%%
sys.path.append('..')
import models

#%% [markdown]
# ## Utility functions
# 
# The examples for ``gumpy-deeplearning`` ship with a few tiny helper functions. For instance, there's one that tells you the versions of the currently installed keras and kapre. ``keras`` is required in ``gumpy-deeplearning``, while ``kapre`` 
# can be used to compute spectrograms.
# 
# In addition, the utility functions contain a method ``load_preprocess_data`` to load and preprocess data. Its usage will be shown further below

#%%
import utils
utils.print_version_info()

#%% [markdown]
# ## Setup parameters for the model and data
# Before we jump into the processing, we first wish to specify some parameters (e.g. frequencies) that we know from the data.

#%%
DEBUG = True
CLASS_COUNT = 2
DROPOUT = 0.2   # dropout rate in float

# parameters for filtering data
FS = 250
LOWCUT = 2
HIGHCUT = 60
ANTI_DRIFT = 0.5
CUTOFF = 50.0 # freq to be removed from signal (Hz) for notch filter
Q = 30.0  # quality factor for notch filter 
W0 = CUTOFF/(FS/2)
AXIS = 0

#set random seed
SEED = 42
KFOLD = 5

#%% [markdown]
# ## Load raw data
# 
# Before training and testing a model, we need some data. The following code shows how to load a dataset using ``gumpy``.

#%%
# specify the location of the GrazB datasets
data_dir = '../../Data/Graz'
subject = 'B01'

# initialize the data-structure, but do _not_ load the data yet
grazb_data = gumpy.data.GrazB(data_dir, subject)

# now that the dataset is setup, we can load the data. This will be handled from within the utils function, 
# which will first load the data and subsequently filter it using a notch and a bandpass filter.
# the utility function will then return the training data.
x_train, y_train = utils.load_preprocess_data(grazb_data, True, LOWCUT, HIGHCUT, W0, Q, ANTI_DRIFT, CLASS_COUNT, CUTOFF, AXIS, FS)

#%% [markdown]
# ## Augment data

#%%
x_augmented, y_augmented = gumpy.signal.sliding_window(data = x_train[:,:,:],
                                                          labels = y_train[:,:],
                                                          window_sz = 4 * FS,
                                                          n_hop = FS // 10,
                                                          n_start = FS * 1)
x_subject = x_augmented
y_subject = y_augmented
x_subject = np.rollaxis(x_subject, 2, 1)

#%% [markdown]
# ## Run the model

#%%
from sklearn.model_selection import StratifiedKFold
from models import CNN_STFT

# define KFOLD-fold cross validation test harness
kfold = StratifiedKFold(n_splits = KFOLD, shuffle = True, random_state = SEED)
cvscores = []
ii = 1
for train, test in kfold.split(x_subject, y_subject[:, 0]):
    print('Run ' + str(ii) + '...')
    # create callbacks
    model_name_str = 'GRAZ_CNN_STFT_3layer_' +                      '_run_' + str(ii)
    #callbacks_list = model.get_callbacks(model_name_str)

    # initialize and create the model
    model = CNN_STFT(model_name_str)
    model.create_model(x_subject.shape[1:], dropout = DROPOUT, print_summary = False)
    
    # fit model. If you specify monitor=True, then the model will create callbacks
    # and write its state to a HDF5 file
    model.fit(x_subject[train], y_subject[train], monitor=True,
              epochs = 100, 
              batch_size = 256, 
              verbose = 0, 
              validation_split = 0.1, callbacks = callbacks_list)

    # evaluate the model
    print('Evaluating model on test set...')
    scores = model.evaluate(x_subject[test], y_subject[test], verbose = 0)
    print("Result on test set: %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
    ii += 1
    
# print some evaluation statistics and write results to file
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
cv_all_subjects = np.asarray(cvscores)
print('Saving CV values to file....')
np.savetxt('GRAZ_CV_' + 'CNN_STFT_3layer_' + str(DROPOUT) + 'do'+'.csv', 
            cv_all_subjects, delimiter = ',', fmt = '%2.4f')
print('CV values successfully saved!\n')

#%% [markdown]
# # Load the trained model 

#%%
model.save('CNN_STFTmonitoring.h5')  # creates a HDF5 file 'my_model.h5'
model2 = load_model('CNN_STFTmonitoring.h5', 
                                 custom_objects={'Spectrogram': kapre.time_frequency.Spectrogram, 
                                                 'Normalization2D': kapre.utils.Normalization2D})

#%% [markdown]
# # New predictions 

#%%
# Method 1 for predictions using predict 
y_pred = model2.predict(X_test,batch_size=64,verbose=1)
Y_pred = np.argmax(y_pred,axis=1)
Y_test = np.argmax(Y_test,axis=1)
accuracy = (len(Y_test) - np.count_nonzero(Y_pred - Y_test) + 0.0)/len(Y_test)
print(accuracy)


# Method 1 for predictions using evaluate (only print the accuracy on the test data)
score, acc = model2.evaluate(X_test, Y_test, batch_size=64)
print('\nTest score:', score)
print('Test accuracy:', acc)


#%%



