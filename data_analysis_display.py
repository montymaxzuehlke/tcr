#!/usr/bin/python3

import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from os.path import expanduser
from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection

HOME = expanduser("~") #if needed; provides home directory of current user
DEFAULT_ORIGIN = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(DEFAULT_ORIGIN + "/Utils/"))

import dataset_provider



X_VAL_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
ROBUSTNESS_LISTS = True
COSINE_SIMILARITY_HEATMAPS, SPLIT_IN_HALF = False, False
ADVERSARIAL_VALIDATION = False
UNIQUENESS_OF_LABELS = False
USE_RANDOM_RESULTS = False




FLAG = ""
if USE_RANDOM_RESULTS:
    FLAG = "r_"

ARGS = sys.argv[1:]
print("ARGS:", ARGS)
print("\n")

SET_NAME = ARGS[0]










################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################



"""
###
STARTING COLLECTING AND DISPLAYING
###
"""



################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
     

HOME = expanduser("~")
DEFAULT_DIRECTORY1 = DEFAULT_ORIGIN + "/Processing/"
DEFAULT_DIRECTORY2 = DEFAULT_ORIGIN + "/Processing_DA/"
DEFAULT_DIRECTORY_RANDOM = DEFAULT_ORIGIN + "/r_Processing_DA/"

if USE_RANDOM_RESULTS:
    DEFAULT_DIRECTORY2 = DEFAULT_DIRECTORY_RANDOM

SHOW_GRAPHS = True
NR_X_VAL = len(X_VAL_LIST)



print("###")
print("###")
print("###")
print(SET_NAME)
print("###")
print("###")
print("###")

trasubsets = ["Removing_50_Upper", "Removing_35_Upper", "Removing_25_Upper", "Removing_10_Upper", 
                  "Entire_Trainingset", 
                  "Removing_10_Lower", "Removing_25_Lower", "Removing_35_Lower", "Removing_50_Lower"]    

trasubsets_display = ["-[50^]", "-[35^]", "-[25^]", "-[10^]", 
                  "+/- 0", 
                  "-[10_]", "-[25_]", "-[35_]", "-[50_]"]    

len_tra_subsets = len(trasubsets)


################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################



"""
###
GRAPHIC DISPLAY OF AVERAGE DEFAULT ROBUSTNESS DISTRIBUTIONS (TRAIN AND TEST)
###
"""



################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
                    

if ROBUSTNESS_LISTS:
    #Loading all training and test set robustness values; first we need to establish the right mean format (we do not need info on the datasets for this)
    with open(DEFAULT_DIRECTORY1 + SET_NAME + '/train_rob_vals_' + str(X_VAL_LIST[0]) + '.pickle', 'rb') as f:
        A = pickle.load(f)

    with open(DEFAULT_DIRECTORY1 + SET_NAME + '/test_rob_vals_' + str(X_VAL_LIST[0]) + '.pickle', 'rb') as f:
        B = pickle.load(f)

    # len(A) == len(B) is true
    train_rob_vals_mean = [0.0] * len(A)
    test_rob_vals_mean = [0.0] * len(B)
    label_vals_mean = [0.0] * len(A)


    _, X, y, NR_BINS = dataset_provider.dataset_decision(SET_NAME) #see dedicated script
    

    NR_OF_FEATURES = len(X[0])

    print()
    print("Set used:", SET_NAME, NR_BINS)
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



    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True, figsize=(8,4))
    fig.suptitle(SET_NAME, fontsize=16)
    for ITERATOR in X_VAL_LIST:

        #from sklearn: Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, stratify=y_binned, random_state=ITERATOR) #starting with a train/test split of 4:1; will later produce 3 parts training set, 1 part proxy-validation set and 1 part test set (see below)

        with open(DEFAULT_DIRECTORY1 + SET_NAME + '/train_rob_vals_' + str(ITERATOR) + '.pickle', 'rb') as f:
            train_rob_vals = pickle.load(f)

        with open(DEFAULT_DIRECTORY1 + SET_NAME + '/test_rob_vals_' + str(ITERATOR) + '.pickle', 'rb') as f:
            test_rob_vals = pickle.load(f)
            
            
        sorted_train_rob_vals = sorted(train_rob_vals)  
        sorted_test_rob_vals = sorted(test_rob_vals)
        sorted_label_vals = sorted(y_train)
        
        axs[0].plot(range(len(train_rob_vals)), sorted_train_rob_vals, 'b', alpha=0.1) 
        axs[0].plot([(len(train_rob_vals) / (len(test_rob_vals)-1))*n for n in range(len(test_rob_vals))], sorted_test_rob_vals, 'r', alpha=0.1)
        #axs[0].plot(range(len(y_train)), sorted_label_vals, 'k', alpha=0.1) 

        
        for i in range(len(train_rob_vals)):
            train_rob_vals_mean[i] += sorted_train_rob_vals[i]
            label_vals_mean[i] += sorted_label_vals[i]

        for i in range(len(test_rob_vals)):
            test_rob_vals_mean[i] += sorted_test_rob_vals[i]
        
        

    for i in range(len(train_rob_vals_mean)):
        train_rob_vals_mean[i] /= NR_X_VAL
        label_vals_mean[i] /= NR_X_VAL

    for i in range(len(test_rob_vals_mean)):
        test_rob_vals_mean[i] /= NR_X_VAL   
        
        
        
        

    Q = np.quantile(train_rob_vals_mean, [0.1,0.15,0.25,0.35,0.5,0.65,0.75,0.85,0.9,1])
    print()
    print("Train-Pareto-Ratio", round(((Q[0] - min(train_rob_vals_mean)) + (Q[9] - Q[8])) / (max(train_rob_vals_mean) - min(train_rob_vals_mean)), 4))
    print()
    print("Train-Mean:", round(np.mean(train_rob_vals_mean), 4), "Train-Median:", round(np.median(train_rob_vals_mean), 4), "Train-Quantiles:", [round(q, 4) for q in Q])    



    Q_ = np.quantile(test_rob_vals_mean, [0.1,0.15,0.25,0.35,0.5,0.65,0.75,0.85,0.9,1])
    print()
    print("Test-Pareto-Ratio", round(((Q_[0] - min(test_rob_vals_mean)) + (Q_[9] - Q_[8])) / (max(test_rob_vals_mean) - min(test_rob_vals_mean)), 4))
    print()
    print("Test-Mean:", round(np.mean(test_rob_vals_mean), 4), "Test-Median:", round(np.median(test_rob_vals_mean), 4), "Test-Quantiles:", [round(q_, 4) for q_ in Q_])    
    print()    
        
    axs[0].plot(range(len(train_rob_vals_mean)), train_rob_vals_mean, color="b", label='Train Rob.', linewidth=1, alpha=0.6, markersize=5, marker='o', markevery=0.04) 
    axs[0].plot([(len(train_rob_vals_mean) / (len(test_rob_vals_mean)-1))*n for n in range(len(test_rob_vals_mean))], test_rob_vals_mean, color="r", label='Test Rob.', linewidth=1, alpha=0.6, markersize=20, marker='+', markevery=0.075)    
    #axs[0].plot(range(len(label_vals_mean)), label_vals_mean, color="k", label='Target ('+str(len(np.unique(y)))+')', linewidth=1, alpha=0.6, markersize=5, marker='^', markevery=0.02)
    axs[0].axvline(x=0.05 * len(train_rob_vals_mean), color="k", label="5% Mark", linestyle="dotted")
    axs[0].axvline(x=0.1 * len(train_rob_vals_mean), color="k", label="10% Mark", linestyle="dashed")
    axs[0].axvline(x=0.2 * len(train_rob_vals_mean), color="k", label="20% Mark", linestyle="solid")
    axs[0].legend(prop={'size': 8})

    quant_05 = np.quantile(train_rob_vals_mean, 0.05)
    quant_10 = np.quantile(train_rob_vals_mean, 0.1)
    quant_20 = np.quantile(train_rob_vals_mean, 0.2)
    for r in range(1000):
        axs[1].axvline(x=r/1000 * quant_05, color="grey", alpha=0.01)
        axs[1].axvline(x=r/1000 * quant_10, color="grey", alpha=0.01)
        axs[1].axvline(x=r/1000 * quant_20, color="grey", alpha=0.01)

    axs[1].hist(train_rob_vals_mean, bins=100, density=True, label="Train Rob.")
    axs[1].axvline(x=quant_05, color="k", label="5% Mark", linestyle="dotted")  
    axs[1].axvline(x=quant_10, color="k", label="10% Mark", linestyle="dashed")
    axs[1].axvline(x=quant_20, color="k", label="20% Mark", linestyle="solid")
    axs[1].legend(loc='upper left', prop={'size': 8}) #change this to the line below if the legend is misplaced
    #axs[1].legend(prop={'size': 8})

    #plt.title(str(['%.2f' % round(q,2) for q in Q]).replace("'", "") + '         \n' + str(['%.2f' % round(q_,2) for q_ in Q_]).replace("'", "") + '         ',
    #         fontdict={'fontsize':9})
     
    plt.savefig(DEFAULT_DIRECTORY2 + SET_NAME + '/' + SET_NAME + '_' + "Robustness_Values" + '_' + str(X_VAL_LIST) + '.png', dpi=600)
    if SHOW_GRAPHS:
        plt.show()






################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################



"""
###
DETERMINING AVERAGES 
###
"""



################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################

################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
                    





        
        
        
        
            
            
"""
###
AVERAGING COSINE-SIMILARITY-HEATMAPS AND DISPLAYING
###
"""     


if COSINE_SIMILARITY_HEATMAPS:

    COS_SIM_H_MAP_AVERAGE = np.array([[0.0]*len_tra_subsets]*len_tra_subsets, dtype='float32')

    if SPLIT_IN_HALF:
        for ITERATOR in X_VAL_LIST:

            with open(DEFAULT_DIRECTORY2 + SET_NAME + '/' + 'Cosine_Similarity_Matrix_' + str(ITERATOR) + ".pickle", 'rb') as f:
                COS_SIM_H_MAP = pickle.load(f)      

            for i in range(len_tra_subsets):
                for j in range(len_tra_subsets):
                    if i>j:
                        COS_SIM_H_MAP_AVERAGE[i][j] += COS_SIM_H_MAP[i][j]


            with open(DEFAULT_DIRECTORY_RANDOM + SET_NAME + '/' + 'Cosine_Similarity_Matrix_' + str(ITERATOR) + ".pickle", 'rb') as f:
                COS_SIM_H_MAP = pickle.load(f)      

            for i in range(len_tra_subsets):
                for j in range(len_tra_subsets):
                    if i<j:
                        COS_SIM_H_MAP_AVERAGE[i][j] += COS_SIM_H_MAP[i][j]


    else:
        for ITERATOR in X_VAL_LIST:

            with open(DEFAULT_DIRECTORY2 + SET_NAME + '/' + SET_NAME + '_' + 'Cosine_Similarity_Matrix_' + str(ITERATOR) + ".pickle", 'rb') as f:
                COS_SIM_H_MAP = pickle.load(f)      

            for i in range(len_tra_subsets):
                for j in range(len_tra_subsets):
                    COS_SIM_H_MAP_AVERAGE[i][j] += COS_SIM_H_MAP[i][j]




    ### Averaging        
    for i in range(len_tra_subsets):
        for j in range(len_tra_subsets):
            COS_SIM_H_MAP_AVERAGE[i][j] /= NR_X_VAL 



    #displaying the Heatmap right away
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(COS_SIM_H_MAP_AVERAGE)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len_tra_subsets))
    ax.set_yticks(np.arange(len_tra_subsets))
    # ... and label them with the respective list entries
    ax.set_xticklabels(trasubsets_display)
    ax.set_yticklabels(trasubsets_display)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len_tra_subsets):
        for j in range(len_tra_subsets):
            text = ax.text(j, i, round(COS_SIM_H_MAP_AVERAGE[i, j], 3),
                           ha="center", va="center", color="w")

    ax.set_title(SET_NAME)
    fig.tight_layout()
    plt.savefig(DEFAULT_DIRECTORY2 + SET_NAME + '/' + FLAG + SET_NAME + '_' + 'Cosine_Similarity_Matrix_' + str(X_VAL_LIST) + '.png', dpi=600)
    if SHOW_GRAPHS:
        plt.show()            
                
                
            
                


            
################################################################################################################################################################################################################################################                
################################################################################################################################################################################################################################################               

################################################################################################################################################################################################################################################              
################################################################################################################################################################################################################################################                

           
#VALI UNIQUENESS            
            
            

################################################################################################################################################################################################################################################                
################################################################################################################################################################################################################################################                

################################################################################################################################################################################################################################################                
################################################################################################################################################################################################################################################                


if UNIQUENESS_OF_LABELS:

    VALI_LOWER_10 = np.array([0.0]*NR_X_VAL)
    VALI_LOWER_25 = np.array([0.0]*NR_X_VAL)
    VALI_LOWER_50 = np.array([0.0]*NR_X_VAL)

    X_VALI = np.array([0.0]*NR_X_VAL)

    VALI_UPPER_50 = np.array([0.0]*NR_X_VAL)
    VALI_UPPER_25 = np.array([0.0]*NR_X_VAL)
    VALI_UPPER_10 = np.array([0.0]*NR_X_VAL)




    ctr = -1
    for ITERATOR in X_VAL_LIST:
        ctr += 1
        for VLSET, han_vlset in [
            (VALI_LOWER_10, "Vali_Lower_10"),
            (VALI_LOWER_25, "Vali_Lower_25"),
            (VALI_LOWER_50, "Vali_lower_50"),

            (X_VALI, "Vali"),

            (VALI_UPPER_50, "Vali_Upper_50"),
            (VALI_UPPER_25, "Vali_Upper_25"),
            (VALI_UPPER_10, "Vali_Upper_10")
        ]:
            with open(DEFAULT_DIRECTORY2 + SET_NAME + "/" + han_vlset + "_Uniqueness_" + str(ITERATOR) + ".pickle", 'rb') as f:
                vlset = pickle.load(f)
        
            VLSET[ctr] += vlset
        
    print()
    print("Vali Uniqueness")
    print()    
    if USE_RANDOM_RESULTS:
        print("CAREFUL, RANDOM RESULTS!!!")
        print()

    print(SET_NAME, 
          "&", round(np.mean(VALI_LOWER_10), 2), "("+str(round(np.std(VALI_LOWER_10), 2))+")", 
          "&", round(np.mean(VALI_LOWER_25), 2), "("+str(round(np.std(VALI_LOWER_25), 2))+")", 
          "&", round(np.mean(VALI_LOWER_50), 2), "("+str(round(np.std(VALI_LOWER_50), 2))+")", 
          
          "&", round(np.mean(X_VALI), 2), "("+str(round(np.std(X_VALI), 2))+")", 
          
          "&", round(np.mean(VALI_UPPER_50), 2), "("+str(round(np.std(VALI_UPPER_50), 2))+")", 
          "&", round(np.mean(VALI_UPPER_25), 2), "("+str(round(np.std(VALI_UPPER_25), 2))+")", 
          "&", round(np.mean(VALI_UPPER_10), 2), "("+str(round(np.std(VALI_UPPER_10), 2))+")", 
          "\\\\"
         )
    print("\hline")
        
    

################################################################################################################################################################################################################################################                
################################################################################################################################################################################################################################################               

################################################################################################################################################################################################################################################              
################################################################################################################################################################################################################################################                


            
            
            
            

################################################################################################################################################################################################################################################                
################################################################################################################################################################################################################################################               

################################################################################################################################################################################################################################################              
################################################################################################################################################################################################################################################                


#ADVERSARIAL VALIDATION


################################################################################################################################################################################################################################################                
################################################################################################################################################################################################################################################                

################################################################################################################################################################################################################################################                
################################################################################################################################################################################################################################################                


if ADVERSARIAL_VALIDATION:


    GLOBAL_XGB_TEST_MEAN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    GLOBAL_XGB_TEST_STD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


    """#structure ("s":= determined by symmetry, "-":= not interesting)
    -     xgb_01 xgb_02 xgb_03  
    s     -      xgb_12 xgb_13
    s     s      -      xgb_23
    s     s      s      -

    legend of the indices:
    "0" := X_low
    "1" := T_low
    "2" := X_upp
    "3" := T_upp

    xgb_01 = xgb_runner(X_low, T_low)
    xgb_02 = xgb_runner(X_low, X_upp)  
    xgb_03 = xgb_runner(X_low, T_upp)

    xgb_12 = xgb_runner(T_low, X_upp)  
    xgb_13 = xgb_runner(T_low, T_upp)

    xgb_23 = xgb_runner(X_upp, T_upp)
    """


    for ITERATOR in X_VAL_LIST:
        for han, ix in [("xgb_01", 0), 
                        ("xgb_02", 1),
                        ("xgb_03", 2),

                        ("xgb_12", 3),
                        ("xgb_13", 4),

                        ("xgb_23", 5),
                ]:

                with open(DEFAULT_DIRECTORY2 + SET_NAME + '/' + han + '_' + str(ITERATOR) + ".pickle", 'rb') as f:
                    xgb = pickle.load(f)

                print(han, xgb)    

                GLOBAL_XGB_TEST_MEAN[ix] += xgb[2] #cross_val_results["test-auc-mean"][-1]
                GLOBAL_XGB_TEST_STD[ix] += xgb[3] #cross_val_results["test-auc-std"][-1]
                
                #if ix == 4:
                    #print("xgb[2]:", xgb[2])
        

    for r in range(len(GLOBAL_XGB_TEST_MEAN)):
        GLOBAL_XGB_TEST_MEAN[r] /= NR_X_VAL
        GLOBAL_XGB_TEST_STD[r] /= NR_X_VAL

    ADV_VAL_MTRX_TEST = np.array([

        ["-", (round(GLOBAL_XGB_TEST_MEAN[0], 2), round(GLOBAL_XGB_TEST_STD[0], 2)), (round(GLOBAL_XGB_TEST_MEAN[1], 2), round(GLOBAL_XGB_TEST_STD[1], 2)), (round(GLOBAL_XGB_TEST_MEAN[2], 2), round(GLOBAL_XGB_TEST_STD[2], 2))],
        ["s", "-", (round(GLOBAL_XGB_TEST_MEAN[3], 2), round(GLOBAL_XGB_TEST_STD[3], 2)), (round(GLOBAL_XGB_TEST_MEAN[4], 2), round(GLOBAL_XGB_TEST_STD[4], 2))],
        ["s", "s", "-", (round(GLOBAL_XGB_TEST_MEAN[5], 2), round(GLOBAL_XGB_TEST_STD[5], 2))],
        ["s", "s", "s", "-"]
                               
        ], dtype="object").transpose()


    if USE_RANDOM_RESULTS:
        print("CAREFUL, RANDOM RESULTS!!!")
        print()


    print("ADV_VAL_MTRX_TEST: \n\n", ADV_VAL_MTRX_TEST)

    print()
    print("#############")
    print()

    print()
    print(SET_NAME, "&", "$ \mathbf{x-} $", "&", "$ \mathbf{t-} $", "&", "$ \mathbf{x+} $", "\\\\")
    print("\hline")
    print("$ \mathbf{t-} $", "&", "%s (%s)" % (round(GLOBAL_XGB_TEST_MEAN[0], 2), round(GLOBAL_XGB_TEST_STD[0], 2)), "&", "-", "&", "-", "\\\\")
    print("$ \mathbf{x+} $", "& \\textbf{", "%s (%s)" % (round(GLOBAL_XGB_TEST_MEAN[1], 2), round(GLOBAL_XGB_TEST_STD[1], 2)), "} & \\textbf{", "%s (%s)" % (round(GLOBAL_XGB_TEST_MEAN[3], 2), round(GLOBAL_XGB_TEST_STD[3], 2)), "} &", "-", "\\\\")
    print("$ \mathbf{t+} $", "& \\textbf{", "%s (%s)" % (round(GLOBAL_XGB_TEST_MEAN[2], 2), round(GLOBAL_XGB_TEST_STD[2], 2)), "} & \\textbf{", "%s (%s)" % (round(GLOBAL_XGB_TEST_MEAN[4], 2), round(GLOBAL_XGB_TEST_STD[4], 2)), "} &", "%s (%s)" % (round(GLOBAL_XGB_TEST_MEAN[5], 2), round(GLOBAL_XGB_TEST_STD[5], 2)), "\\\\")
    print()
    print("\hline")
    print()


################################################################################################################################################################################################################################################                
################################################################################################################################################################################################################################################               

################################################################################################################################################################################################################################################              
################################################################################################################################################################################################################################################                






print("The End")

