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


ARGS = sys.argv[1:]
print("\n")
print("ARGS:", ARGS)
print("\n")


######################################################################################################################################################################################################
######################################################################################################################################################################################################
######################################################################################################################################################################################################


ORIGIN = DEFAULT_ORIGIN + "/Results"
HEATMAP, INCLUDE_BEST = True, True
BOXPLOTS = False 
DENSEWEIGHT_LEVEL = str(0.1) #Default alpha
TCR_LEVEL = str(0.9) #Default lambda
TCR_LEVEL_LIST = []
X_VAL_LIST = [k for k in range(10)] #Default random seeds
MAX_SAM_SIZE = ARGS[0] if ARGS[0] != "all" else "" 
LEAST_ROB_PERCENTAGE = ARGS[1]


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
          
MAE_GLOBAL = []
RMSE_GLOBAL = []
R2_GLOBAL = []

METHODS_and_ADDONS = [
    ("DENSEWEIGHT", "0.05"),
    ("DENSEWEIGHT", "0.1"),
    ("DENSEWEIGHT", "0.25"),
    ("DENSEWEIGHT", "0.5"),
    ("DENSEWEIGHT", "0.75"),
    ("DENSEWEIGHT", "0.9"),
    ("DENSEWEIGHT", "1.0"),
    ("SMOGN", "5_0.01_0.8"),
    ("AXIL", "X"),
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
D_KEYS = [
("DENSEWEIGHT_0.05"),
("DENSEWEIGHT_0.1"),
("DENSEWEIGHT_0.25"),
("DENSEWEIGHT_0.5"),
("DENSEWEIGHT_0.75"),
("DENSEWEIGHT_0.9"),
("DENSEWEIGHT_1.0")
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


for SET_NAME in SET_NAME_LIST:

    NR_X_VAL = len(X_VAL_LIST)
    print()
    print(SET_NAME)

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
        
    MAE_MEANS = OVERVIEW_MAE.mean(axis=0)
    RMSE_MEANS = OVERVIEW_RMSE.mean(axis=0)
    R2_MEANS = OVERVIEW_R2.mean(axis=0)
 
    
    if HEATMAP:
        if INCLUDE_BEST:
            """############################### METRICS ################################"""
            for (global_list, local_list) in [
            (MAE_GLOBAL, MAE_MEANS), 
            (RMSE_GLOBAL, RMSE_MEANS), 
            (R2_GLOBAL, R2_MEANS)
            ]:


                FINAL_LIST = []    
                FINAL_LIST.append(local_list["DENSEWEIGHT_"+DENSEWEIGHT_LEVEL])

                intermediate_list = []
                for k in D_KEYS:
                    intermediate_list.append(local_list[k])
                intermediate_list = np.array(intermediate_list)

                FINAL_LIST.append(intermediate_list.max())
                FINAL_LIST.append(local_list["SMOGN_5_0.01_0.8"])
                FINAL_LIST.append(local_list["AXIL_X"])
                FINAL_LIST.append(local_list["TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+"_"+TCR_LEVEL])

                intermediate_list = []
                for k in T_KEYS:
                    intermediate_list.append(local_list[k])
                intermediate_list = np.array(intermediate_list)

                FINAL_LIST.append(intermediate_list.max())

                FINAL_LIST_ARGSORT = sorted(range(len(FINAL_LIST)), key=lambda k: FINAL_LIST[k]) #this lists the indices in ascending argument order
                COUNTER = [0]*6
                for i in range(6): #distribute the number of tokens according to placement IF the avg performance is positive; otherwise set number of tokes to 0
                    if FINAL_LIST[FINAL_LIST_ARGSORT[i]] > 0.0:
                        COUNTER[FINAL_LIST_ARGSORT[i]] += i #last place gets 0 tokes, second-to-last place gets 1 token, ..., first place gets 5 tokens
                    else:    
                        COUNTER[FINAL_LIST_ARGSORT[i]] += 0

                global_list.append(COUNTER)
            TCR_LEVEL_LIST.append(T_KEYS[intermediate_list.argmax()])    

        else:
            """############################### METRICS ################################"""
            for (global_list, local_list) in [(MAE_GLOBAL, MAE_MEANS), (RMSE_GLOBAL, RMSE_MEANS), (R2_GLOBAL, R2_MEANS)]:

                FINAL_LIST = []    
                FINAL_LIST.append(local_list["DENSEWEIGHT_"+DENSEWEIGHT_LEVEL])
                FINAL_LIST.append(local_list["SMOGN_5_0.01_0.8"])
                FINAL_LIST.append(local_list["AXIL_X"])
                FINAL_LIST.append(local_list["TCR"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+"_"+TCR_LEVEL])
                
                FINAL_LIST_ARGSORT = sorted(range(len(FINAL_LIST)), key=lambda k: FINAL_LIST[k]) #this lists the indices in ascending argument order
                COUNTER = [0]*4
                for i in range(4): #distribute the number of tokens according to placement IF the avg performance is positive; otherwise set number of tokes to 0
                    if FINAL_LIST[FINAL_LIST_ARGSORT[i]] > 0.0:
                        COUNTER[FINAL_LIST_ARGSORT[i]] += i #last place gets 0 tokes, second-to-last place gets 1 token, ..., first place gets 5 tokens
                    else:    
                        COUNTER[FINAL_LIST_ARGSORT[i]] += 0
                global_list.append(COUNTER)


    if BOXPLOTS:
        matplotlib.rcParams.update({'font.size': 7})    
        fig, ax = plt.subplots(1, 3, figsize=(20, 4))
        fig.suptitle(SET_NAME, size=22).set_y(-0.2)

        WIDTHS = 0.2
        OFFSET = 0.2

        plt.style.use('ggplot')    

        sns.boxplot(data=OVERVIEW_MAE, ax=ax[0]).set(title='MAE ('+SET_NAME+"/"+LEAST_ROB_PERCENTAGE.replace("_", "")+")")
        ax[0].axhline(y=0.0)
        ax[0].axvline(x=-0.5+MAE_MEANS.argmax(), color='r', linestyle='--')
        ax[0].axvline(x=0.5+MAE_MEANS.argmax(), color='r', linestyle='--')
        ax[0].tick_params(axis='x', rotation=80)

        sns.boxplot(data=OVERVIEW_RMSE, ax=ax[1]).set(title='RMSE ('+SET_NAME+"/"+LEAST_ROB_PERCENTAGE.replace("_", "")+")")
        ax[1].axhline(y=0.0)
        ax[1].axvline(x=-0.5+RMSE_MEANS.argmax(), color='r', linestyle='--')
        ax[1].axvline(x=0.5+RMSE_MEANS.argmax(), color='r', linestyle='--')
        ax[1].tick_params(axis='x', rotation=80)

        sns.boxplot(data=OVERVIEW_R2, ax=ax[2]).set(title='R2 ('+SET_NAME+"/"+LEAST_ROB_PERCENTAGE.replace("_", "")+")")
        ax[2].axhline(y=0.0) 
        ax[2].axvline(x=-0.5+R2_MEANS.argmax(), color='r', linestyle='--')
        ax[2].axvline(x=0.5+R2_MEANS.argmax(), color='r', linestyle='--')
        ax[2].tick_params(axis='x', rotation=80)

        plt.subplots_adjust(bottom=0.3)

        plt.savefig(ORIGIN+"/"+SET_NAME+"_"+str(X_VAL_LIST)+"_for_TCR_at"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+".png", dpi=600)
        plt.show()
        

        
if HEATMAP:        
    
    if INCLUDE_BEST:
        print()
        print("TCR_LEVEL_LIST:", TCR_LEVEL_LIST)
        print()
        METHOD_LABELS = ["DW["+DENSEWEIGHT_LEVEL+"]: ", "DW[best]: ", "SMOGN: ", "AXIL: ", "TCR["+TCR_LEVEL+"]: ", "TCR[best]: "]
    else:
        METHOD_LABELS = ["DW["+DENSEWEIGHT_LEVEL+"]: ", "SMOGN: ", "AXIL: ", "TCR["+TCR_LEVEL+"]: "]
    
    MAE_GLOBAL = np.array(MAE_GLOBAL)
    MAE_COUNTS = MAE_GLOBAL.sum(axis=0)
    
    RMSE_GLOBAL = np.array(RMSE_GLOBAL)
    RMSE_COUNTS = RMSE_GLOBAL.sum(axis=0)
    
    R2_GLOBAL = np.array(R2_GLOBAL)
    R2_COUNTS = R2_GLOBAL.sum(axis=0)
    
    
    matplotlib.rcParams.update({'font.size': 7})    
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    plt.style.use('ggplot')  

    if MAX_SAM_SIZE != "":
        mae_title = "MAE ("+MAX_SAM_SIZE.replace("_", "")+"/"+LEAST_ROB_PERCENTAGE.replace("_", "")+")"
        rmse_title = "RMSE ("+MAX_SAM_SIZE.replace("_", "")+"/"+LEAST_ROB_PERCENTAGE.replace("_", "")+")"
        r2_title = "R2 ("+MAX_SAM_SIZE.replace("_", "")+"/"+LEAST_ROB_PERCENTAGE.replace("_", "")+")"
    else:    
        mae_title = "MAE (all/"+LEAST_ROB_PERCENTAGE.replace("_", "")+")"
        rmse_title = "RMSE (all/"+LEAST_ROB_PERCENTAGE.replace("_", "")+")"
        r2_title = "R2 (all/"+LEAST_ROB_PERCENTAGE.replace("_", "")+")"


    sns.heatmap(MAE_GLOBAL, ax=ax[0], annot=True, linewidth=.5, xticklabels=[ml + str(mc) for (ml, mc) in zip(METHOD_LABELS, MAE_COUNTS)], yticklabels=SET_NAME_LIST, cbar=False).set(title=mae_title) 
    sns.heatmap(RMSE_GLOBAL, ax=ax[1], annot=True, linewidth=.5, xticklabels=[ml + str(mc) for (ml, mc) in zip(METHOD_LABELS, RMSE_COUNTS)], yticklabels=[], cbar=False).set(title=rmse_title) 
    sns.heatmap(R2_GLOBAL, ax=ax[2], annot=True, linewidth=.5, xticklabels=[ml + str(mc) for (ml, mc) in zip(METHOD_LABELS, R2_COUNTS)], yticklabels=[], cbar=False).set(title=r2_title) 
    
    ax[0].tick_params(axis='x', rotation=45)
    ax[1].tick_params(axis='x', rotation=45)
    ax[2].tick_params(axis='x', rotation=45)
    
    plt.subplots_adjust(bottom=0.15)

    
    if INCLUDE_BEST:
        plt.savefig(ORIGIN+"/BECNHMARK_OVERVIEW_for_TCR_at"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+".png", dpi=600)
        plt.show()      
    else:
        plt.savefig(ORIGIN+"/BECNHMARK_OVERVIEW_SLIM_for_TCR_at"+MAX_SAM_SIZE+LEAST_ROB_PERCENTAGE+".png", dpi=600)
        plt.show()   
        
        
        
        
    
print("The End")