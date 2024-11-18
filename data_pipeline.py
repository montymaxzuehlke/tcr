import sys
import os
from os.path import expanduser
HOME = expanduser("~") #if needed; provides home directory of current user
DEFAULT_ORIGIN = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(DEFAULT_ORIGIN + "/Utils/"))
import sklearn.model_selection
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from multiprocessing import Pool
import random
import pickle
import time
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.utils import resample

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
    print("ARGS:", ARGS)
    print("\n")

  

    ######################################################################################################################################################################################################
    ######################################################################################################################################################################################################
    ######################################################################################################################################################################################################
    
    CALCULATE_ROBUSTNESS_LISTS = True
    MAX_SAM_SIZE = -1
    BOOTSTRAP_ITERATIONS = 1
    LOAD_ROBUSTNESS_LISTS = not CALCULATE_ROBUSTNESS_LISTS 

    RANDOMISING_LISTS = False
    
    ADVERSARIAL_VALIDATION = False
    CALCULATE_LABEL_DISTRIBUTIONS = False #(will be computed as backup to investigate any class imbalance dependent biases)
    COMPUTE_COS_SIM_MATRIX = False

    SHOW_GRAPHS = False #Graphs will always be saved -if created- but only displayed at will 
    VERBOSE = True #Good to trace whatever 
             
    DEFAULT_DIRECTORY1 = DEFAULT_ORIGIN + "/Processing/" 

    if RANDOMISING_LISTS:
        DEFAULT_DIRECTORY2 = DEFAULT_ORIGIN + "/r_Processing_DA/" 
    else:
        DEFAULT_DIRECTORY2 = DEFAULT_ORIGIN + "/Processing_DA/" 


    SET_NAME = ARGS[0] 
    ITERATOR = int(ARGS[1]) 


    HOW_MANY_CPU_CORES_ARE_IN_THE_BOX = len(os.sched_getaffinity(0))


    print("HOW_MANY_CPU_CORES_ARE_IN_THE_BOX?", HOW_MANY_CPU_CORES_ARE_IN_THE_BOX) #<- this is what you want on clusters
    print()


    TRAIN_N_JOBS = HOW_MANY_CPU_CORES_ARE_IN_THE_BOX
    TEST_N_JOBS = HOW_MANY_CPU_CORES_ARE_IN_THE_BOX
    N_JOBS = HOW_MANY_CPU_CORES_ARE_IN_THE_BOX

    
    ######################################################################################################################################################################################################
    ######################################################################################################################################################################################################
    ######################################################################################################################################################################################################
   











################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################




    """
    ###
    LOADING AND PREPROCESSING DATASET
    ###
    """




################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
    if not os.path.exists(DEFAULT_DIRECTORY1 + SET_NAME):
        os.mkdir(DEFAULT_DIRECTORY1 + SET_NAME)
        print("Directory " , SET_NAME ,  " Created ")
        print()
    else:
        print("Directory " , SET_NAME ,  " already exists")
        print()
    
    if not os.path.exists(DEFAULT_DIRECTORY2 + SET_NAME):
        os.mkdir(DEFAULT_DIRECTORY2 + SET_NAME)
        print("Directory " , SET_NAME ,  " Created ")
        print()
    else:
        print("Directory " , SET_NAME ,  " already exists")
        print()


    print("Starting Timer:")
    print()
    start = time.time()

    
    
    """
    ***
    DATASET
    ***
    """
    
    
    SET_HANDLE, X, y, NR_BINS = dataset_provider.dataset_decision(SET_NAME) #see dedicated script
    
    print()
    print("Set used:", SET_NAME, NR_BINS)
    print()
    print("Random seed for this run:", ITERATOR)
    print()

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

    """ #just for reference
    trasubsets = ["Removing 50% Upper", "Removing 35% Upper", "Removing 25% Upper", "Removing 10% Upper",
                  "Entire Trainingset",
                  "Removing 10% Lower", "Removing 25% Lower", "Removing 35% Lower", "Removing 50% Lower"]
    valsubsets = ["vali_lower_10", "vali_lower_25", "vali_lower_50", "X_vali", "vali_upper_50", "vali_upper_25", "vali_upper_10"]
    tessubsets = ["test_lower_10", "test_lower_25", "test_lower_50", "X_test", "test_upper_50", "test_upper_25", "test_upper_10"]
    """

    len_tra_subsets = 9 # ==len(trasubsets)

    #from sklearn: Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, stratify=y_binned, random_state=ITERATOR) #starting with a train/test split of 4:1; will later produce 3 parts training set, 1 part proxy-validation set and 1 part test set (see below)

    len_train = len(X_train)
    len_test = len(X_test)
    
    #will be used later on 
    ten_tr = int(len_train * 0.1)
    twenty_five_tr = int(len_train * 0.25)
    thirty_five_tr = int(len_train * 0.35)
    half_tr = int(len_train * 0.5)
    sixty_five_tr = int(len_train * 0.65)
    seventy_five_tr = int(len_train * 0.75)
    ninety_tr = int(len_train * 0.9)

    ten = int(len_test * 0.1)
    twenty_five = int(len_test * 0.25)
    half = int(len_test * 0.5)
    seventy_five = int(len_test * 0.75)
    ninety = int(len_test * 0.9)
    
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")

    NR_OF_FEATURES = len(X_train[0])
    print("NR_OF_FEATURES:", NR_OF_FEATURES)
    print()

    y_train = np.array(y_train, dtype="float32")
    y_test = np.array(y_test, dtype="float32")

        
        
        
    """
    ***
    CREATING COSINE-SIMILARITY LISTS
    ***
    """

    L_eigenvectors = []
    L_eigenvalues = []

    COS_SIM_H_MAP = np.array([[0.0]*len_tra_subsets]*len_tra_subsets, dtype='float32') #array to collect all weighted+averaged cosine similarity values
    
    M_FEAT_1 = np.array([[0.0 for x in range(NR_OF_FEATURES)] for y in range(len_tra_subsets)], dtype="float32")
    M_FEAT_2 = np.array([[0.0 for x in range(NR_OF_FEATURES)] for y in range(len_tra_subsets)], dtype="float32")
    M_FEAT_3 = np.array([[0.0 for x in range(NR_OF_FEATURES)] for y in range(len_tra_subsets)], dtype="float32")


    ### Sanity Check:
    print()
    print("M_FEAT_1.shape:", M_FEAT_1.shape)
    print("M_FEAT_2.shape:", M_FEAT_2.shape)
    print("M_FEAT_3.shape:", M_FEAT_3.shape)    
    print()    

    print()
    print("X_train.shape, y_train.shape (still including validation data):", X_train.shape, y_train.shape)
    print("X_test.shape, y_test.shape:", X_test.shape, y_test.shape)
    print()

    print()
    print("Dataset-Example 0:", X_train[0])
    print("Dataset-Example 1:", X_train[1])
    print("Dataset-Example 2:", X_train[2])
    print()

    print()
    print("Label-Example 0:", y_train[0])
    print("Label-Example 1:", y_train[1])
    print("Label-Example 2:", y_train[2])
    print()


    


    if CALCULATE_LABEL_DISTRIBUTIONS:
        # the greater the width of the bars, the larger the spread among the elements (more sparse parts of the distribution then coprrespond to wider bars or in other words: the more clustered points in a bin, the smaller the bin width)
        fig1, ax = plt.subplots(figsize=(20,8))
        plt.hist([y_train.copy().reshape(-1), y_test.copy().reshape(-1)], bins=BINS, label=["y_train", "y_test"], density=True)
        plt.xticks(BINS)
        plt.title("Label Distribution of y_train and y_test")
        plt.legend()
        plt.savefig(DEFAULT_DIRECTORY2 + SET_NAME + '/Label_Distribution_train_test' + '_' + str(ITERATOR) + '.png', dpi=200)
        if SHOW_GRAPHS:
            plt.show()




    


            
            
            
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################




    """
    ###
    ROBUSTNESS CALCULATIONS 
    ###
    """




################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
           
            
    if LOAD_ROBUSTNESS_LISTS:

        if MAX_SAM_SIZE == -1:        

            with open(DEFAULT_DIRECTORY1 + SET_NAME + '/train_rob_vals_' + str(ITERATOR) + '.pickle', 'rb') as f:
                train_rob_vals = pickle.load(f)
            with open(DEFAULT_DIRECTORY1 + SET_NAME + '/train_indices_ordered_' + str(ITERATOR) + '.pickle', 'rb') as f:
                train_indices_ordered = pickle.load(f)

        else:
            with open(DEFAULT_DIRECTORY1 + SET_NAME + '/train_rob_vals_' + str(ITERATOR) + '_' + str(MAX_SAM_SIZE) + '.pickle', 'rb') as f:
                train_rob_vals = pickle.load(f)
            with open(DEFAULT_DIRECTORY1 + SET_NAME + '/train_indices_ordered_' + str(ITERATOR) + '_' + str(MAX_SAM_SIZE) + '.pickle', 'rb') as f:
                train_indices_ordered = pickle.load(f)
            

        with open(DEFAULT_DIRECTORY1 + SET_NAME + '/test_rob_vals_' + str(ITERATOR) + '.pickle', 'rb') as f:
            test_rob_vals = pickle.load(f)
        with open(DEFAULT_DIRECTORY1 + SET_NAME + '/test_indices_ordered_' + str(ITERATOR) + '.pickle', 'rb') as f:
            test_indices_ordered = pickle.load(f)



    
    else:

        if MAX_SAM_SIZE == -1:
            X_train_sample = X_train
            y_train_sample = y_train

        elif MAX_SAM_SIZE < len_train:
            X_train_sample, y_train_sample = resample(X_train, y_train, replace=False, n_samples=MAX_SAM_SIZE, random_state=ITERATOR)
        
        else:
            X_train_sample = X_train
            y_train_sample = y_train


        stopwatch = time.time()                        

        #can help to produce smoother distributions via bootstrapping samples of the whole training set (BOOTSTRAP_ITERATIONS=1 yields the original setup)
        BOOTSTRAPS_LENGTH_TRAIN = int(len(X_train_sample) / BOOTSTRAP_ITERATIONS)

        train_jump = int(len(X_train_sample) / TRAIN_N_JOBS) + 1
        train_jump_list = [train_jump * k for k in range(TRAIN_N_JOBS)]
        train_jump_list.append(len(X_train_sample))
        #print("train_jump_list:", train_jump_list)

        with Pool(processes=TRAIN_N_JOBS) as pool:
            train_rob_vals_parts = pool.starmap(pipeline_functions.rob_calc, [
                (X_train_sample, y_train_sample, X_train_sample[train_jump_list[k] : train_jump_list[k+1]],
                 y_train_sample[train_jump_list[k] : train_jump_list[k+1]],
                 None,
                 None,
                 SET_HANDLE,
                 BOOTSTRAP_ITERATIONS,
                 BOOTSTRAPS_LENGTH_TRAIN) for k in range(TRAIN_N_JOBS)])

        train_rob_vals_sample = np.array([], dtype="float32")
        for p in range(TRAIN_N_JOBS):
            train_rob_vals_sample = np.concatenate([train_rob_vals_sample, train_rob_vals_parts[p]])



        if MAX_SAM_SIZE == -1:
            train_rob_vals = train_rob_vals_sample

        elif MAX_SAM_SIZE < len_train:
            rob_finder = KNeighborsRegressor(n_neighbors=1, n_jobs=TRAIN_N_JOBS) #by definition KNN uses the Euclidean metric, equivalent to the Frobenius norm
            rob_finder.fit(X_train_sample.copy(), train_rob_vals_sample.copy()) #using "x=training set sample with "y=robustness values"
            train_rob_vals = rob_finder.predict(X_train.copy()) #determining proxy robustness values

        else:
            train_rob_vals = train_rob_vals_sample
        


        train_indices_ordered = sorted(range(len(train_rob_vals)), key=lambda k: train_rob_vals[k])

        #print("train_rob_vals:", train_rob_vals)
        #print("sorted(train_rob_vals=:", sorted(train_rob_vals))
        #print("train_indices_ordered:", train_indices_ordered)



        if MAX_SAM_SIZE == -1:        

            with open(DEFAULT_DIRECTORY1 + SET_NAME + "/" + "train" +"_rob_vals_" + str(ITERATOR) + ".pickle", 'wb') as f:
                pickle.dump(train_rob_vals, f)
            with open(DEFAULT_DIRECTORY1 + SET_NAME + "/" + "train" +"_indices_ordered_" + str(ITERATOR) + ".pickle", 'wb') as f:
                pickle.dump(train_indices_ordered, f)

        else:
            with open(DEFAULT_DIRECTORY1 + SET_NAME + "/" + "train" +"_rob_vals_" + str(ITERATOR) + '_' + str(MAX_SAM_SIZE) + ".pickle", 'wb') as f:
                pickle.dump(train_rob_vals, f)
            with open(DEFAULT_DIRECTORY1 + SET_NAME + "/" + "train" +"_indices_ordered_" + str(ITERATOR) + '_' + str(MAX_SAM_SIZE) + ".pickle", 'wb') as f:
                pickle.dump(train_indices_ordered, f)


        stopwatch_click = time.time() - stopwatch

        if MAX_SAM_SIZE == -1:  
            with open(DEFAULT_DIRECTORY1 + SET_NAME + "/" +"comp_time_" + str(ITERATOR) + ".pickle", 'wb') as f:
                pickle.dump(stopwatch_click, f)

        else:
            with open(DEFAULT_DIRECTORY1 + SET_NAME + "/" +"comp_time_" + str(ITERATOR) + '_' + str(MAX_SAM_SIZE) + ".pickle", 'wb') as f:
                pickle.dump(stopwatch_click, f)







        
        #can help to produce smoother distributions via bootstrapping samples of the whole training set (BOOTSTRAP_ITERATIONS=1 yields the original setup)
        BOOTSTRAPS_LENGTH_TEST = int(len(X_test) / BOOTSTRAP_ITERATIONS)

        #computing and saving "train_rob_vals" and "train_indices_ordered"
        test_jump = int(len(X_test) / TEST_N_JOBS) + 1
        test_jump_list = [test_jump * k for k in range(TEST_N_JOBS)]
        test_jump_list.append(len(X_test))

        with Pool(processes=TEST_N_JOBS) as pool:
            test_rob_vals_parts = pool.starmap(pipeline_functions.rob_calc, [
                (X_train, y_train, X_test[test_jump_list[k] : test_jump_list[k+1]],
                 None,
                 None,
                 None,
                 SET_HANDLE,
                 BOOTSTRAP_ITERATIONS,
                 BOOTSTRAPS_LENGTH_TEST) for k in range(TEST_N_JOBS)])


        test_rob_vals = np.array([], dtype="float32")
        for p in range(TEST_N_JOBS):
            test_rob_vals = np.concatenate([test_rob_vals, test_rob_vals_parts[p]])

        test_indices_ordered = sorted(range(len(test_rob_vals)), key=lambda k: test_rob_vals[k])


        with open(DEFAULT_DIRECTORY1 + SET_NAME + "/" + "test" +"_rob_vals_" + str(ITERATOR) + ".pickle", 'wb') as f:
            pickle.dump(test_rob_vals, f)
        with open(DEFAULT_DIRECTORY1 + SET_NAME + "/" + "test" +"_indices_ordered_" + str(ITERATOR) + ".pickle", 'wb') as f:
            pickle.dump(test_indices_ordered, f)
        
            
        """
        ### WE NEED THIS FOR THE BIG SETS, WHERE WE WILL NOT CALCULATE THE ENTIRE LISTS AT ALL
        with open(DEFAULT_DIRECTORY1 + SET_NAME + "/" + "train" +"_rob_vals_" + str(ITERATOR) + ".pickle", 'wb') as f:
            pickle.dump(np.array([0.0]*len_train, dtype="float32"), f)
        with open(DEFAULT_DIRECTORY1 + SET_NAME + "/" + "train" +"_indices_ordered_" + str(ITERATOR) + ".pickle", 'wb') as f:
            pickle.dump(np.arange(len_train), f)

        with open(DEFAULT_DIRECTORY1 + SET_NAME + "/" + "test" +"_rob_vals_" + str(ITERATOR) + ".pickle", 'wb') as f:
            pickle.dump(np.array([0.0]*len_test, dtype="float32"), f)
        with open(DEFAULT_DIRECTORY1 + SET_NAME + "/" + "test" +"_indices_ordered_" + str(ITERATOR) + ".pickle", 'wb') as f:
            pickle.dump(np.arange(len_test), f)
        """













            
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################




    """
    ###
    DETERMINING PROXY VALIDATION AND TEST SUBSETS (TO COMPARE LABEL DISTRIBUTIONS)
    ###
    """


    if RANDOMISING_LISTS:
        randomlist_tr = random.sample(range(len_train), len_train)
        train_rob_vals = np.array(train_rob_vals)[randomlist_tr]
        train_indices_ordered = np.array(train_indices_ordered)[randomlist_tr]
        
        randomlist_te = random.sample(range(len_test), len_test)
        test_rob_vals = np.array(test_rob_vals)[randomlist_te]
        test_indices_ordered = np.array(test_indices_ordered)[randomlist_te]



################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
            
            

    rob_neigh = KNeighborsClassifier(n_neighbors=1, n_jobs=N_JOBS) #by definition KNN uses the Euclidean metric, equivalent to the Frobenius norm
    rob_neigh.fit(X_train.copy(), list(range(len_train))) #using "x=training set ordered w.r.t. their robustness values" with "y=indices of the training data"
    remover_indices = rob_neigh.predict(X_test.copy()) #determining proxy validation set indices
    remover_indices_unique = list(set(remover_indices.copy()))
    len_vali_unique = len(remover_indices_unique)
    vali_uniqueness = len_vali_unique / len_test #measures the relative amount of distinct elements in the validation set

    X_vali = X_train.copy()[np.array(remover_indices)]
    y_vali = y_train.copy()[np.array(remover_indices)]

    X_vali_unique = X_train.copy()[np.array(remover_indices_unique)]
    y_vali_unique = y_train.copy()[np.array(remover_indices_unique)]

    vali_rob_vals = train_rob_vals.copy()[np.array(remover_indices)]
    vali_indices_ordered = sorted(range(len(vali_rob_vals)), key=lambda k: vali_rob_vals[k]) 


    test_lower_50 = np.delete(X_test, test_indices_ordered[half:], 0)
    test_labels_lower_50 = np.delete(y_test, test_indices_ordered[half:], 0)

    test_upper_50 = np.delete(X_test, test_indices_ordered[:half], 0)
    test_labels_upper_50 = np.delete(y_test, test_indices_ordered[:half], 0)

    test_lower_25 = np.delete(X_test, test_indices_ordered[twenty_five:], 0)
    test_labels_lower_25 = np.delete(y_test, test_indices_ordered[twenty_five:], 0)

    test_upper_25 = np.delete(X_test, test_indices_ordered[:seventy_five], 0)
    test_labels_upper_25 = np.delete(y_test, test_indices_ordered[:seventy_five], 0)

    test_lower_10 = np.delete(X_test, test_indices_ordered[ten:], 0)
    test_labels_lower_10 = np.delete(y_test, test_indices_ordered[ten:], 0)

    test_upper_10 = np.delete(X_test, test_indices_ordered[:ninety], 0)
    test_labels_upper_10 = np.delete(y_test, test_indices_ordered[:ninety], 0)

    print("Built Test(Sub)Sets:", "Time Elapsed:", round(time.time()-start, 2), "Secs")
    print()


    vali_lower_50 = np.delete(X_vali, vali_indices_ordered[half:], 0)
    vali_labels_lower_50 = np.delete(y_vali, vali_indices_ordered[half:], 0)

    vali_upper_50 = np.delete(X_vali, vali_indices_ordered[:half], 0)
    vali_labels_upper_50 = np.delete(y_vali, vali_indices_ordered[:half], 0)

    vali_lower_25 = np.delete(X_vali, vali_indices_ordered[twenty_five:], 0)
    vali_labels_lower_25 = np.delete(y_vali, vali_indices_ordered[twenty_five:], 0)

    vali_upper_25 = np.delete(X_vali, vali_indices_ordered[:seventy_five], 0)
    vali_labels_upper_25 = np.delete(y_vali, vali_indices_ordered[:seventy_five], 0)

    vali_lower_10 = np.delete(X_vali, vali_indices_ordered[ten:], 0)
    vali_labels_lower_10 = np.delete(y_vali, vali_indices_ordered[ten:], 0)

    vali_upper_10 = np.delete(X_vali, vali_indices_ordered[:ninety], 0)
    vali_labels_upper_10 = np.delete(y_vali, vali_indices_ordered[:ninety], 0)

    print("Built Vali(Sub)Sets:", "Time Elapsed:", round(time.time()-start, 2), "Secs")
    print()


    print("X_train.shape (still including validation data):", X_train.shape)
    print("X_vali.shape:", X_vali.shape, "with Uniqueness:", vali_uniqueness)
    print("X_vali_unique.shape:", X_vali_unique.shape, "with Uniqueness:", 1)
    print("X_test.shape:", X_test.shape)
    print()







    for vlset, han_vlset in [
        (vali_lower_10, "Vali_Lower_10"),
        (vali_lower_25, "Vali_Lower_25"),
        (vali_lower_50, "Vali_lower_50"),

        (X_vali, "Vali"),

        (vali_upper_50, "Vali_Upper_50"),
        (vali_upper_25, "Vali_Upper_25"),
        (vali_upper_10, "Vali_Upper_10")
    ]:

        uni = (len(np.unique(vlset.copy(), axis=0))) / len(vlset)

        with open(DEFAULT_DIRECTORY2 + SET_NAME + "/" + han_vlset + "_Uniqueness_" + str(ITERATOR) + ".pickle", 'wb') as f:
            pickle.dump(uni, f)

        print("Ratio of distinct elements in", han_vlset, ":", uni)     
        print()





    if CALCULATE_LABEL_DISTRIBUTIONS:

        for tes_labs, val_labs, hans in [
            (test_labels_lower_10, vali_labels_lower_10, "Test_Vali_Labels_Lower_10"),
            (test_labels_lower_25, vali_labels_lower_25, "Test_Vali_Labels_Lower_25"),
            (test_labels_lower_50, vali_labels_lower_50, "Test_Vali_Labels_Lower_50"),

            (y_test, y_vali, "Test_Vali_Labels"),

            (test_labels_upper_50, vali_labels_upper_50, "Test_Vali_Labels_Upper_50"),
            (test_labels_upper_25, vali_labels_upper_25, "Test_Vali_Labels_Upper_25"),
            (test_labels_upper_10, vali_labels_upper_10, "Test_Vali_Labels_Upper_10")
        ]:

            #displaying label distribution for each training(sub)set
            pipeline_functions.reg_label_distribution_displayer([tes_labs.copy().reshape(-1), val_labs.copy().reshape(-1)], BINS, hans , DEFAULT_DIRECTORY2 + SET_NAME, ITERATOR, ["y_test", "y_vali"], SHOW_GRAPHS)







################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################




    """
    ###
    ADVERSARIAL VALIDATION
    ###
    """




################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################


    if ADVERSARIAL_VALIDATION:

        pipeline_functions.adversarial_validation(np.delete(X_train, pipeline_functions.Union([], train_indices_ordered[half_tr:]), 0),
                                                  np.delete(X_train, pipeline_functions.Union([], train_indices_ordered[:half_tr]), 0), 
                                                  test_lower_50, 
                                                  test_upper_50,  
                                                  DEFAULT_DIRECTORY2 + SET_NAME + '/',
                                                  ITERATOR
                                                 )
        
    

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################




    """
    ###
    PCA-BASED COSINE SIMILARITY MATRIX 
    ###
    """




################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################


        
    """
    ***
    Creating All Training Subsets by Removing Data 
    ***
    """
    
    for tra_han, tra_set, tra_labs in [

        ("Removing_50_Upper", np.delete(X_train, pipeline_functions.Union([], train_indices_ordered[half_tr:]), 0), np.delete(y_train, pipeline_functions.Union([], train_indices_ordered[half_tr:]), 0)),
        ("Removing_35_Upper", np.delete(X_train, pipeline_functions.Union([], train_indices_ordered[sixty_five_tr:]), 0), np.delete(y_train, pipeline_functions.Union([], train_indices_ordered[sixty_five_tr:]), 0)),
        ("Removing_25_Upper", np.delete(X_train, pipeline_functions.Union([], train_indices_ordered[seventy_five_tr:]), 0), np.delete(y_train, pipeline_functions.Union([], train_indices_ordered[seventy_five_tr:]), 0)),
        ("Removing_10_Upper", np.delete(X_train, pipeline_functions.Union([], train_indices_ordered[ninety_tr:]), 0), np.delete(y_train, pipeline_functions.Union([], train_indices_ordered[ninety_tr:]), 0)),

        ("Entire_Trainingset", np.delete(X_train, [], 0), np.delete(y_train, [], 0)),

        ("Removing_10_Lower", np.delete(X_train, pipeline_functions.Union([], train_indices_ordered[:ten_tr]), 0), np.delete(y_train, pipeline_functions.Union([], train_indices_ordered[:ten_tr]), 0)),
        ("Removing_25_Lower", np.delete(X_train, pipeline_functions.Union([], train_indices_ordered[:twenty_five_tr]), 0), np.delete(y_train, pipeline_functions.Union([], train_indices_ordered[:twenty_five_tr]), 0)),
        ("Removing_35_Lower", np.delete(X_train, pipeline_functions.Union([], train_indices_ordered[:thirty_five_tr]), 0), np.delete(y_train, pipeline_functions.Union([], train_indices_ordered[:thirty_five_tr]), 0)),
        ("Removing_50_Lower", np.delete(X_train, pipeline_functions.Union([], train_indices_ordered[:half_tr]), 0), np.delete(y_train, pipeline_functions.Union([], train_indices_ordered[:half_tr]), 0))

        ]:
        
        #Keeping track of which tra_set is being trained on
        print()
        print(tra_han, "Length of Training Set:", len(tra_set), "Time Elapsed:", round(time.time()-start, 2), "Secs")
        print()
        
        
        """
        ***
        Data Analysis Block For Each Individual Training Subset 
        ***
        """
        
        if CALCULATE_LABEL_DISTRIBUTIONS:
            #displaying label distribution for each training(sub)set
            pipeline_functions.reg_label_distribution_displayer(tra_labs.copy(), BINS, tra_han, DEFAULT_DIRECTORY2 + SET_NAME, ITERATOR, "y_train", SHOW_GRAPHS)

        if COMPUTE_COS_SIM_MATRIX:
            pipe = make_pipeline(PCA())
            pipe.fit(tra_set)
            L_eigenvectors.append(pipe[0].components_)
            L_eigenvalues.append(pipe[0].explained_variance_)


    #combining previous results    
    if COMPUTE_COS_SIM_MATRIX:
        #Collecting all average cosine similarities between the different PCA covariance matrix eigenvectors weighted by the corresponding eigenvalues; by construction symmetric and with 0.0 on the diagonal
        for a in range(len_tra_subsets):
            
            for b in range(a+1,len_tra_subsets):

                cos_sim = pipeline_functions.cosine_similarity(L_eigenvectors[a], L_eigenvalues[a], L_eigenvectors[b], L_eigenvalues[b])

                COS_SIM_H_MAP[a][b] = cos_sim
                COS_SIM_H_MAP[b][a] = cos_sim


        with open(DEFAULT_DIRECTORY2 + SET_NAME + '/' + SET_NAME + '_Cosine_Similarity_Matrix_' + str(ITERATOR) + ".pickle", 'wb') as f:
            pickle.dump(COS_SIM_H_MAP, f)
            

            

                
################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################




    """
    ###
    ENDING
    ###
    """




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
    print("The End")



