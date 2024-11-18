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
LEAST_ROB_PERCENTAGE = "_0.1"

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


weight_arrangement = ["0.95", "0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"]



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

    ALL_MAE_DF = pd.DataFrame()
    ALL_RMSE_DF = pd.DataFrame()
    ALL_R2_DF = pd.DataFrame()

    matplotlib.rcParams.update({'font.size': 7})    
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    plt.style.use('ggplot')

    for mss, MAX_SAM_SIZE in enumerate(["_5000", "_10000", "_20000", ""]):

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
        

        for col_ix, col in enumerate(OVERVIEW_MAE.columns):
            intermediate = pd.DataFrame()
            intermediate["Performance Delta"] = OVERVIEW_MAE[col].values
            intermediate["Weight"] = weight_arrangement[col_ix]
            intermediate["Sample Size"] = MAX_SAM_SIZE.replace("_", "") if MAX_SAM_SIZE != "" else "all"
            ALL_MAE_DF = pd.concat([ALL_MAE_DF, intermediate], axis=0)

        for col_ix, col in enumerate(OVERVIEW_RMSE.columns):
            intermediate = pd.DataFrame()
            intermediate["Performance Delta"] = OVERVIEW_RMSE[col].values
            intermediate["Weight"] = weight_arrangement[col_ix]
            intermediate["Sample Size"] = MAX_SAM_SIZE.replace("_", "") if MAX_SAM_SIZE != "" else "all"
            ALL_RMSE_DF = pd.concat([ALL_RMSE_DF, intermediate], axis=0)

        for col_ix, col in enumerate(OVERVIEW_R2.columns):
            intermediate = pd.DataFrame()
            intermediate["Performance Delta"] = OVERVIEW_R2[col].values
            intermediate["Weight"] = weight_arrangement[col_ix]
            intermediate["Sample Size"] = MAX_SAM_SIZE.replace("_", "") if MAX_SAM_SIZE != "" else "all"
            ALL_R2_DF = pd.concat([ALL_R2_DF, intermediate], axis=0)

    #print(ALL_MAE_DF)
    #print(ALL_RMSE_DF)
    #print(ALL_R2_DF)

    #print(ALL_MAE_DF.to_string())

    mae_title = SET_NAME +" MAE ("+LEAST_ROB_PERCENTAGE.replace("_", "")+")"
    rmse_title = SET_NAME +" RMSE ("+LEAST_ROB_PERCENTAGE.replace("_", "")+")"
    r2_title = SET_NAME +" R2 ("+LEAST_ROB_PERCENTAGE.replace("_", "")+")"

    sns.lineplot(ALL_MAE_DF, x="Weight", y="Performance Delta",  hue="Sample Size", style="Sample Size", ax=ax[0], palette="flare", legend='brief', markers=True, dashes=False, errorbar="sd", err_kws={"alpha": 0.1}).set(title=mae_title)
    sns.lineplot(ALL_RMSE_DF, x="Weight", y="Performance Delta",  hue="Sample Size", style="Sample Size", ax=ax[1], palette="flare", legend='brief', markers=True, dashes=False, errorbar="sd", err_kws={"alpha": 0.1}).set(title=rmse_title)
    sns.lineplot(ALL_R2_DF, x="Weight", y="Performance Delta",  hue="Sample Size", style="Sample Size", ax=ax[2], palette="flare", legend='brief', markers=True, dashes=False, errorbar="sd", err_kws={"alpha": 0.1}).set(title=r2_title)

    ax[0].set_xticks(np.arange(10), weight_arrangement)
    ax[1].set_xticks(np.arange(10), weight_arrangement)
    ax[2].set_xticks(np.arange(10), weight_arrangement)

    ax[1].set_ylabel("")
    ax[2].set_ylabel("")

    ax[0].axhline(0.0, ls='--', c='black')
    ax[1].axhline(0.0, ls='--', c='black')
    ax[2].axhline(0.0, ls='--', c='black')
   
    ax[0].tick_params(axis='x', rotation=45)
    ax[1].tick_params(axis='x', rotation=45)
    ax[2].tick_params(axis='x', rotation=45)

    plt.subplots_adjust(bottom=0.15)

    savename = ORIGIN+"/"+SET_NAME+"_"+str(X_VAL_LIST)+"_Scaling_Comparison_for_TCR_at"+LEAST_ROB_PERCENTAGE+".png" 

    plt.savefig(savename, dpi=600)
    plt.show()
    
print("The End")