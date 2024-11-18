#!/usr/bin/python3

import sys
import shutil
import os
from os.path import expanduser
HOME = expanduser("~") #if needed; provides home directory of current user
DEFAULT_ORIGIN = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(DEFAULT_ORIGIN + "/Utils/"))


#########################################################
####################### General #########################
#########################################################

import sklearn.model_selection
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import pickle
import time
import xgboost as xgb
import pandas as pd


######### DenseWeight ##########
from denseweight import DenseWeight

########### SMOGN #################
import smogn

############# AXIL ############# 
import lightgbm as lgb
from AXIL.axil import Explainer

############# tcr ################
import pipeline_functions
import dataset_provider


################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################




"""
###
BEGINNING OF SCRIPT
###
"""





################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################








def main():

    ARGS = sys.argv[1:]
    print("\n")
    print("ARGS:", ARGS)
    print("\n")

    
    ######################################################################################################################################################################################################
    ######################################################################################################################################################################################################
    ######################################################################################################################################################################################################

    SET_NAME = ARGS[0] 
    ITERATOR = int(ARGS[1]) 
    METHOD = ARGS[2]
    ADDON_KEY = int(ARGS[3])
    MAX_SAM_SIZE_KEY = int(ARGS[4])
    TCR_PERCENTAGE_KEY = int(ARGS[5])

    if METHOD == "DENSEWEIGHT": 
        ADDON_LIST = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        MAX_SAM_SIZE = -1
        TCR_PERCENTAGE = -1

    elif METHOD == "SMOGN":
        ADDON_LIST = [
        (5, 0.01, 0.8), #these are the original baseline parameters, see also the DenseWeight paper
        ]    
        MAX_SAM_SIZE = -1
        TCR_PERCENTAGE = -1

    elif METHOD == "AXIL": 
        ADDON_LIST = ['X']    
        MAX_SAM_SIZE = -1
        TCR_PERCENTAGE = -1

    elif METHOD == "TCR":
        ADDON_LIST = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        MAX_SAM_SIZE_LIST = [-1, 5000, 10000, 20000, 50000]
        TCR_PERCENTAGE_LIST = [0.05, 0.1, 0.2]

    else:
        ADDON_LIST = [-1]

    #######################################################################
    #####
    ADDON = ADDON_LIST[ADDON_KEY]
    if METHOD == "TCR":
        MAX_SAM_SIZE = MAX_SAM_SIZE_LIST[MAX_SAM_SIZE_KEY]
        TCR_PERCENTAGE = TCR_PERCENTAGE_LIST[TCR_PERCENTAGE_KEY]
    #####
    #######################################################################

    print("Using this addon:", ADDON)    
    print()
    if METHOD == "TCR":
        print("Using these additional parameters (MAX_SAM_SIZE, TCR_PERCENTAGE):", MAX_SAM_SIZE, TCR_PERCENTAGE)
        print()

    VERBOSE = True #Good to trace everything 

    DEFAULT_DIRECTORY = DEFAULT_ORIGIN + "/Results/"
    MODEL_DIRECTORY = DEFAULT_ORIGIN + "/H2O_Processing/" 

    #directory with robustness values
    ROBUSTNESS_DIRECTORY = DEFAULT_ORIGIN + "/Processing/"

    
    # loading, training, predicting, saving
    def load_train_pred_save(modeldirec, 
        savedirec,
        meth,
        add,
        sam_wts=None,
        ordering=None,
        diff_training_data=None,
        verbose=True
        ):

        with open(modeldirec + "_" + "nativeXGBoostParamDict.pickle", 'rb') as f:
            nativeXGBoostParamDict = pickle.load(f)

        if verbose:
            print("nativeXGBoostParamDict", nativeXGBoostParamDict)
            print()

        if diff_training_data is not None:
            nativeXGBoostInput_TRAINING = diff_training_data
            if verbose:
                print("Changed the training data (!)")
        else:
            nativeXGBoostInput_TRAINING = xgb.DMatrix(modeldirec + "_" + "nativeXGBoostInput_TRAINING.buffer")

        nativeXGBoostInput_PREDICTION = xgb.DMatrix(modeldirec + "_" + "nativeXGBoostInput_PREDICTION.buffer")

        if sam_wts is not None:
            if ordering is not None:
                sam_wts = sam_wts[ordering]
            nativeXGBoostInput_TRAINING.set_info(weight=sam_wts)
            if verbose:
                print("New sample weights set as: ", sam_wts)
                print()

        nativeModel = xgb.train(dtrain=nativeXGBoostInput_TRAINING,
                            params=nativeXGBoostParamDict[0],                        
                            num_boost_round=nativeXGBoostParamDict[1]
                            )

        preds = nativeModel.predict(nativeXGBoostInput_PREDICTION)

        if verbose:
            print("test_set_predictions[:10]:", preds[:10])
        with open(savedirec + "_" + meth + "_" + add + "_test_set_predictions.pickle", 'wb') as f:
            pickle.dump(preds, f)

     



    
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################




    """
    ###
    LOADING AND RESCALING DATASET
    ###
    """




################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
    

    
    """
    ***
    DATASET
    ***
    """
    
    
    _, X, y, NR_BINS = dataset_provider.dataset_decision(SET_NAME) #see dedicated script
    
    LEN_ALL = len(X)
    
    print("Set used:", SET_NAME, NR_BINS)
    print()
    print("Random seed for this run:", ITERATOR)
    print()

    if VERBOSE:

        print()
        print("Dataset-Example 0:", X[0])
        print("Dataset-Example 1:", X[1])
        print("Dataset-Example 2:", X[2])
        print()

        print()
        print("Label-Example 0:", y[0])
        print("Label-Example 1:", y[1])
        print("Label-Example 2:", y[2])
        print()
    
    print("Normalising Dataset and Labels:")
    print()

    #Normalising:
    DATA_SCALER = MinMaxScaler()
    X = DATA_SCALER.fit_transform(X)  
    LABEL_SCALER = MinMaxScaler()
    y = LABEL_SCALER.fit_transform(y)  

    # see: https://stackoverflow.com/questions/29438265/stratified-train-test-split-in-scikit-learn

    REG_BINS = np.quantile(y, np.linspace(start=1/NR_BINS, stop=1, num=NR_BINS))
    print("Bin Thresholds:", np.linspace(start=1/NR_BINS, stop=1, num=NR_BINS))
    print()
    print("Quantiles according to NR_BINS:", REG_BINS)
    print()
    y_binned = np.digitize(y.copy(), REG_BINS, right=True)

    ABD = [0]*NR_BINS
    for i in range(len(y_binned)):
        ABD[y_binned[i][0]] += 1
    print("Absolute Bin Distribution:", ABD)
    print()

    ###for plotting purposes:

    BINS = [np.min(y)]
    for i in range(NR_BINS):
        BINS.append(REG_BINS[i])

    print("BINS:", BINS)


        
        

    if VERBOSE:

        print()
        print("Normalised Dataset-Example 0:", X[0])
        print("Normalised Dataset-Example 1:", X[1])
        print("Normalised Dataset-Example 2:", X[2])
        print()

        print()
        print("(Normalised) Label-Example 0:", y[0])
        print("(Normalised) Label-Example 1:", y[1])
        print("(Normalised) Label-Example 2:", y[2])
        print()    


    

    
    
    
    
    




################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################




    """
    ###
    BEGINNING OF DATA PREPROCESSING
    ###
    """




################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################




    print("Starting Timer:")
    print()
    start = time.time()
    

    #from sklearn: Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, stratify=y_binned, random_state=ITERATOR) #starting with a train/test split of 4:1; will later produce 3 parts training set, 1 part proxy-validation set and 1 part test set (see below)

    len_train = len(X_train)
    len_train_half = len_train // 2
    len_test = len(X_test)
    len_test_half = len_test // 2
    
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")

    NR_OF_FEATURES = len(X_train[0])
    print("NR_OF_FEATURES:", NR_OF_FEATURES)
    print()

    y_train = np.array(y_train, dtype="float32")
    y_test = np.array(y_test, dtype="float32")

        
    print()
    print("X_train.shape, y_train.shape (still including validation data):", X_train.shape, y_train.shape)
    print("X_test.shape, y_test.shape:", X_test.shape, y_test.shape)
    print()



      
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################




    """
    ###
    LOADING ROBUSTNESS VALUES 
    ###
    """




################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
           
    if MAX_SAM_SIZE == -1:        

        with open(ROBUSTNESS_DIRECTORY + SET_NAME + '/train_rob_vals_' + str(ITERATOR) + '.pickle', 'rb') as f:
            train_rob_vals = pickle.load(f)
        with open(ROBUSTNESS_DIRECTORY + SET_NAME + '/train_indices_ordered_' + str(ITERATOR) + '.pickle', 'rb') as f:
            train_indices_ordered = pickle.load(f)

    else:
        with open(ROBUSTNESS_DIRECTORY + SET_NAME + '/train_rob_vals_' + str(ITERATOR) + '_' + str(MAX_SAM_SIZE) + '.pickle', 'rb') as f:
            train_rob_vals = pickle.load(f)
        with open(ROBUSTNESS_DIRECTORY + SET_NAME + '/train_indices_ordered_' + str(ITERATOR) + '_' + str(MAX_SAM_SIZE) + '.pickle', 'rb') as f:
            train_indices_ordered = pickle.load(f)
        

    with open(ROBUSTNESS_DIRECTORY + SET_NAME + '/test_rob_vals_' + str(ITERATOR) + '.pickle', 'rb') as f:
        test_rob_vals = pickle.load(f)
    with open(ROBUSTNESS_DIRECTORY + SET_NAME + '/test_indices_ordered_' + str(ITERATOR) + '.pickle', 'rb') as f:
        test_indices_ordered = pickle.load(f)


    X_test = X_test[test_indices_ordered]    
    y_test = y_test[test_indices_ordered]
    
    #order training data w.r.t. robustness values
    X_train = X_train[train_indices_ordered]    
    y_train = y_train[train_indices_ordered]





################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
    

    ### DETERMINE FEATURE IMPORTANCES, TRAIN MODELS, PREDICT AND SAVE RESULTS


################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
    

    #### code taken/adapted from: https://github.com/SteiMi/density-based-weighting-for-imbalanced-regression
    if METHOD == "DENSEWEIGHT":
        y_AB = y_train.copy()
        print("Using DenseWeight with alpha =", ADDON)
        # Define DenseWeight
        dw = DenseWeight(alpha=ADDON)
        SAMPLE_WEIGHTS = dw.fit(y_AB)

        if VERBOSE:
            print("this goes in as SAMPLE_WEIGHTS", SAMPLE_WEIGHTS)

        print()
        print("Starting training now:", round(time.time()-start, 2), "Secs")
        print()

        load_train_pred_save(modeldirec=MODEL_DIRECTORY + SET_NAME + "/" + str(ITERATOR), 
            savedirec=DEFAULT_DIRECTORY + SET_NAME + "/" + str(ITERATOR),
            meth=METHOD,
            add=str(ADDON),
            sam_wts=SAMPLE_WEIGHTS,
            ordering=train_indices_ordered,
            diff_training_data=None,
            verbose=VERBOSE)

        print()
        print("Done.", round(time.time()-start, 2), "Secs")
        print()




    #### code taken/adapted from: https://github.com/nickkunz/smogn
    elif METHOD == "SMOGN":
        #for SMOGN compatibility        
        columns_X = ["c"+str(k) for k in range(NR_OF_FEATURES)]
        columns_y = ["target"] 

        data_df = pd.DataFrame(np.concatenate([X_train, y_train], axis=1), columns=columns_X+columns_y)

        if VERBOSE:
            print("data_df.info()", data_df.info())
            print()
        
        k_in = ADDON[0]
        pert_in = ADDON[1]
        rel_thres_in = ADDON[2]

        ADDON = str(k_in)+'_'+str(pert_in)+'_'+str(rel_thres_in)
        if VERBOSE:
            print("ADDON", ADDON)
            print()

        ## conduct smogn --- see: https://github.com/nickkunz/smogn/blob/master/examples/smogn_example_3_adv.ipynb
        data_smogn = smogn.smoter(
            
            ## main arguments
            data = data_df,           ## pandas dataframe
            y = 'target',             ## string ('header name')
            k = k_in,                    ## positive integer (k < n)
            pert = pert_in,              ## real number (0 < R < 1)
            samp_method = 'balance',  ## string ('balance' or 'extreme')
            drop_na_col = True,       ## boolean (True or False)
            drop_na_row = True,       ## boolean (True or False)
            replace = False,          ## boolean (True or False)

            ## phi relevance arguments
            rel_thres = rel_thres_in,         ## real number (0 < R < 1)
            rel_method = 'auto',    ## string ('auto' or 'manual')
            # rel_xtrm_type = 'both', ## unused (rel_method = 'manual')
            # rel_coef = 1.50,        ## unused (rel_method = 'manual')
            #rel_ctrl_pts_rg = rg_mtrx ## 2d array (format: [x, y])
        )

        if VERBOSE:
            print("Shape of training data changed from", data_df.shape, "to", data_smogn.shape)
            print()


        if data_df.shape[0] < data_smogn.shape[0]:
            if VERBOSE:
                print("Subsampling the larger SMOGN data fopr fairness")    
                print()
            data_smogn = data_smogn.sample(n=data_df.shape[0], axis=0, random_state=ITERATOR)
            if VERBOSE:
                print("New set size of SMOGN data is", data_smogn.shape[0])
                print()

        if VERBOSE:
            print("data_smogn.head()", data_smogn.head())
            print()
            print("data_smogn.info()", data_smogn.info())
            print()

        data_smogn = xgb.DMatrix(data_smogn[columns_X], data_smogn[columns_y])

        print()
        print("Starting training now:", round(time.time()-start, 2), "Secs")
        print()

        load_train_pred_save(modeldirec=MODEL_DIRECTORY + SET_NAME + "/" + str(ITERATOR), 
            savedirec=DEFAULT_DIRECTORY + SET_NAME + "/" + str(ITERATOR),
            meth=METHOD,
            add=str(ADDON),
            sam_wts=None,
            ordering=None,
            diff_training_data=data_smogn,
            verbose=VERBOSE)

        print()
        print("Done.", round(time.time()-start, 2), "Secs")
        print()


    ### code taken/adapted from: https://github.com/pgeertsema/AXIL_paper/tree/main
    elif METHOD == "AXIL":

        # constants
        LR = 0.1 # default
        LEAVES = 4 # to prevent overfitting in small datasets (default = 31)
        TREES = 100 #default

        class BaseModel:
            """A base class for the algorithms."""
            def fit(self, X: np.array, y: np.array):
                raise NotImplementedError("Subclass must implement abstract method")

            def _calculate_K(self) -> np.array:
                raise NotImplementedError("Subclass must implement abstract method")

            def linear(self) -> np.array:
                K = self._calculate_K()  # Returns (N, N)
                
                # store original sum of elements
                orig_sum = K.sum()
                
                # set diagonal elements to 0, so implements leave-one-out prediction
                np.fill_diagonal(K, 0)
                
                # rescale so each row and column sum to same value as previously (since K is symetric)
                K = K * (orig_sum/K.sum())
                return K

        class AXILClass(BaseModel):
            """
            Calculatex AXIL weights
            """
            def __init__(self, model, learning_rate) -> None:
                super().__init__()
                self.model = model
                self.learning_rate = learning_rate

            def fit(self, X: np.array, y: np.array):  # X is (N, M), y is (N, 1)
                self.X = X
                self.explainer = Explainer(self.model, learning_rate=self.learning_rate)
                self.explainer.fit(self.X)


            def _calculate_K(self) -> np.array:  # Returns (N, N)
                K = self.explainer.transform(self.X)        
                # free up memory
                self.explainer.reset()
                return K


        # train LightGBM model
        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": LEAVES,
            "verbose": 1,
            "min_data": 2,
            "learning_rate": LR,    
        }

        # reshape y to (N,1)
        y_train = y_train.reshape(-1, 1)

        # build GBM model
        lgb_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, lgb_data, num_boost_round=TREES-1)

        axil_exp = AXILClass(model, LR)
        axil_exp.fit(X_train, y_train)
        K = axil_exp.linear()

        #y_hat = np.dot(K.T, y) means that we get the weights by transposing K.T (i.e. (K.T).T = K) and setting y = 1 (as a vector)
        SAMPLE_WEIGHTS = np.dot(K, np.ones_like(y_train))

        if VERBOSE:
            print("this goes in as SAMPLE_WEIGHTS", SAMPLE_WEIGHTS)
        
        print()
        print("Starting training now:", round(time.time()-start, 2), "Secs")
        print()

        load_train_pred_save(modeldirec=MODEL_DIRECTORY + SET_NAME + "/" + str(ITERATOR), 
            savedirec=DEFAULT_DIRECTORY + SET_NAME + "/" + str(ITERATOR),
            meth=METHOD,
            add=str(ADDON),
            sam_wts=SAMPLE_WEIGHTS,
            ordering=train_indices_ordered,
            diff_training_data=None,
            verbose=VERBOSE)

        print()
        print("Done.", round(time.time()-start, 2), "Secs")
        print()




    elif METHOD == "TCR":

        tcr_absolute = int(TCR_PERCENTAGE * len_train) 
        SAMPLE_WEIGHTS = np.array([ADDON for k in range(tcr_absolute)] + [1.0 for k in range(len_train - tcr_absolute)]) 

        if VERBOSE:
            print("this goes in as SAMPLE_WEIGHTS", SAMPLE_WEIGHTS)
        
        print()
        print("Starting training now:", round(time.time()-start, 2), "Secs")
        print()

        load_train_pred_save(modeldirec=MODEL_DIRECTORY + SET_NAME + "/" + str(ITERATOR), 
            savedirec=DEFAULT_DIRECTORY + SET_NAME + "/" + str(ITERATOR),
            meth=METHOD+"_"+str(TCR_PERCENTAGE) if MAX_SAM_SIZE == -1 else METHOD+"_"+str(MAX_SAM_SIZE)+"_"+str(TCR_PERCENTAGE),
            add=str(ADDON),
            sam_wts=SAMPLE_WEIGHTS,
            ordering=train_indices_ordered,
            diff_training_data=None,
            verbose=VERBOSE)

        print()
        print("Done.", round(time.time()-start, 2), "Secs")
        print()


################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################


    print()
    print("Finished!!!", "Time Elapsed:", round(time.time()-start, 2), "Secs")
    print()


if __name__ == '__main__':
    print("Starting Pipeline")
    print()
    main()
    print("The Ending")



