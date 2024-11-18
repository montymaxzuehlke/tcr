#!/usr/bin/python3

import numpy as np
import pickle
import os
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
from os.path import expanduser
HOME = expanduser("~") #if needed; provides home directory of current user
DEFAULT_ORIGIN = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(DEFAULT_ORIGIN + "/Utils/"))

ORIGIN = DEFAULT_ORIGIN + "/Results"
X_VAL_LIST = [k for k in range(10)] #Default random seeds
MAX_SAM_SIZE = "" # "_5000"


# select the dataset
SET_NAME_LIST = [
#"Diabetes",                  
#"Treasury",                 
#"Wine_Quality_REG",         
#"Topo21",                   
#"Bike_Sharing_Demand",      
#"Mimic_Los",                 
#"California_Housing",       
#"Online_News_Popularity",   
#"Diamonds",                  
#"BNG",                      
#"Superconduct",             
#"House_Sales",              
#"MiamiHousing2016",         
#"MBGM",                     
#"Yprop41",                  
#"Elevators",                
#"Isolet",                   
#"CPU_Act",                  
#"Pol",                      
#"Ailerons"  
]






################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################


################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################


################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################
    



for SET_NAME in SET_NAME_LIST:

    NR_X_VAL = len(X_VAL_LIST)
    print()
    print(SET_NAME)

    MAE_MEANS = []
    RMSE_MEANS = []
    R2_MEANS = []

    matplotlib.rcParams.update({'font.size': 7})    
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    plt.style.use('ggplot')

    for lrp, LEAST_ROB_PERCENTAGE in enumerate(["_0.05", "_0.1", "_0.2"]):

        METHODS_and_ADDONS = [
            ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE, "0.95"),
            ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE, "0.9"),
            ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE, "0.8"),
            ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE, "0.7"),
            ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE, "0.6"),
            ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE, "0.5"),
            ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE, "0.4"),
            ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE, "0.3"),
            ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE, "0.2"),
            ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE, "0.1")
        ]
        T_KEYS = [
        ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+"_0.95"),
        ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+"_0.9"),
        ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+"_0.8"),
        ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+"_0.7"),
        ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+"_0.6"),
        ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+"_0.5"),
        ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+"_0.4"),
        ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+"_0.3"),
        ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+"_0.2"),
        ("TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+"_0.1")
        ]


        OVERVIEW_DICT = {'REFERENCE': np.zeros(NR_X_VAL)}
        
        for METHOD, ADDON in METHODS_and_ADDONS: 
            KEY = METHOD+'_'+ADDON
            OVERVIEW_DICT[KEY] = np.zeros(NR_X_VAL)
          
        OVERVIEW_MAE = pd.DataFrame(OVERVIEW_DICT)
        OVERVIEW_RMSE = pd.DataFrame(OVERVIEW_DICT)
        OVERVIEW_R2 = pd.DataFrame(OVERVIEW_DICT)
           
        ctr = -1
        for ITERATOR in X_VAL_LIST:
            ctr += 1
       
            with open(DEFAULT_ORIGIN + '/Results_REF/'+SET_NAME+'/'+str(ITERATOR)+'_test_set_labels.pickle', 'rb') as f:
                test_set_labels = pickle.load(f).reshape(-1)  
            with open(DEFAULT_ORIGIN + '/Results_REF/'+SET_NAME+'/'+str(ITERATOR)+'_REFERENCE_test_set_predictions.pickle', 'rb') as f:
                test_set_predictions = pickle.load(f).reshape(-1)   
                    
            MAE_REFERENCE = mean_absolute_error(test_set_labels, test_set_predictions)
            OVERVIEW_MAE["REFERENCE"][ctr] = - MAE_REFERENCE #switch sign so that larger means better
            
            RMSE_REFERENCE = mean_squared_error(test_set_labels, test_set_predictions, squared=False)
            OVERVIEW_RMSE["REFERENCE"][ctr] = - RMSE_REFERENCE #switch sign so that larger means better
            
            R2_REFERENCE = r2_score(test_set_labels, test_set_predictions)
            OVERVIEW_R2["REFERENCE"][ctr] = R2_REFERENCE #larger means better by desgin
            
            
            for METHOD, ADDON in METHODS_and_ADDONS:
                KEY = METHOD+'_'+ADDON
                try:
                    with open(ORIGIN+'/'+SET_NAME+'/'+str(ITERATOR)+'_'+METHOD+'_'+ADDON+'_test_set_predictions.pickle', 'rb') as f:
                        test_set_predictions = pickle.load(f).reshape(-1)   

                    MAE = mean_absolute_error(test_set_labels, test_set_predictions)
                    OVERVIEW_MAE[KEY][ctr] = - MAE #switch sign so that larger means better

                    RMSE = mean_squared_error(test_set_labels, test_set_predictions, squared=False)
                    OVERVIEW_RMSE[KEY][ctr] = - RMSE #switch sign so that larger means better

                    R2 = r2_score(test_set_labels, test_set_predictions)
                    OVERVIEW_R2[KEY][ctr] = R2 #larger means better by desgin

                except OSError as err:
                    print("I did not find:", ORIGIN+'/'+SET_NAME+'/'+str(ITERATOR)+'_'+METHOD+'_'+ADDON+'_test_set_predictions.pickle')
                    # this will result in the delta being 0.0
                    OVERVIEW_MAE[KEY][ctr] = OVERVIEW_MAE["REFERENCE"][ctr]
                    OVERVIEW_RMSE[KEY][ctr] = OVERVIEW_RMSE["REFERENCE"][ctr]
                    OVERVIEW_R2[KEY][ctr] = OVERVIEW_R2["REFERENCE"][ctr]
                    #print("OS error:", err)
                    #print(SET_NAME, ctr, KEY)

                    
        #print("mae", OVERVIEW_MAE.head())
        #print("rmse", OVERVIEW_RMSE.head())
        #print("r2", OVERVIEW_R2.head())
        
        
        REFERENCE_VALUES_MAE = OVERVIEW_MAE["REFERENCE"]
        REFERENCE_VALUES_RMSE = OVERVIEW_RMSE["REFERENCE"]
        REFERENCE_VALUES_R2 = OVERVIEW_R2["REFERENCE"]
        
        for col in OVERVIEW_MAE:
            OVERVIEW_MAE[col] = OVERVIEW_MAE[col] - REFERENCE_VALUES_MAE #if > 0 -> improvement; else not useful
            OVERVIEW_RMSE[col] = OVERVIEW_RMSE[col] - REFERENCE_VALUES_RMSE #if > 0 -> improvement; else not useful
            OVERVIEW_R2[col] = OVERVIEW_R2[col] - REFERENCE_VALUES_R2 #if > 0 -> improvement; else not useful
            
        OVERVIEW_MAE = OVERVIEW_MAE.drop(["REFERENCE"], axis=1) #these are just 0 everywhere now
        OVERVIEW_RMSE = OVERVIEW_RMSE.drop(["REFERENCE"], axis=1) #these are just 0 everywhere now
        OVERVIEW_R2 = OVERVIEW_R2.drop(["REFERENCE"], axis=1) #these are just 0 everywhere now
        
        MAE_MEANS.append(np.array(OVERVIEW_MAE.mean(axis=0)).reshape(-1,1))
        RMSE_MEANS.append(np.array(OVERVIEW_RMSE.mean(axis=0)).reshape(-1,1))
        R2_MEANS.append(np.array(OVERVIEW_R2.mean(axis=0)).reshape(-1,1))

    MAE_MEANS = np.concatenate(MAE_MEANS, axis=1)
    RMSE_MEANS = np.concatenate(RMSE_MEANS, axis=1)
    R2_MEANS = np.concatenate(R2_MEANS, axis=1)

    sns.heatmap(MAE_MEANS,  ax=ax[0], annot=True, linewidth=.5, xticklabels=["0.05", "0.1", "0.2"], yticklabels=["0.95", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"], cbar=False, center=0.0, cmap=sns.color_palette("vlag", as_cmap=True)).set(title='MAE ('+SET_NAME+")") 
    sns.heatmap(RMSE_MEANS, ax=ax[1], annot=True, linewidth=.5, xticklabels=["0.05", "0.1", "0.2"], yticklabels=["0.95", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"], cbar=False, center=0.0, cmap=sns.color_palette("vlag", as_cmap=True)).set(title='RMSE ('+SET_NAME+")") 
    sns.heatmap(R2_MEANS,   ax=ax[2], annot=True, linewidth=.5, xticklabels=["0.05", "0.1", "0.2"], yticklabels=["0.95", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"], cbar=False, center=0.0, cmap=sns.color_palette("vlag", as_cmap=True)).set(title='R2 ('+SET_NAME+")") 
        
    ax[0].tick_params(axis='x', rotation=45)
    ax[1].tick_params(axis='x', rotation=45)
    ax[2].tick_params(axis='x', rotation=45)

    ax[0].tick_params(axis='y', rotation=45)
    ax[1].tick_params(axis='y', rotation=45)
    ax[2].tick_params(axis='y', rotation=45)
        
    plt.subplots_adjust(bottom=0.15)


    if MAX_SAM_SIZE != "":
        savename = ORIGIN+"/"+SET_NAME+"_"+str(X_VAL_LIST)+"_Sensitivity_Comparison_for_TCR_at"+MAX_SAM_SIZE+".png"
    else:
        savename = ORIGIN+"/"+SET_NAME+"_"+str(X_VAL_LIST)+"_Sensitivity_Comparison_for_TCR_at_all.png" 


    plt.savefig(savename, dpi=600)
    plt.show()
    
print("The End")