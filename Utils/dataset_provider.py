import sklearn.datasets
from sklearn import preprocessing
import sklearn.model_selection
import numpy as np
import random
import pandas as pd
import os
from os.path import expanduser
HOME = expanduser("~") #if needed; provides home directory of current user
pd.set_option('display.max_columns', None)


"""
***
DATASETS
***
"""


def dataset_decision(SET_NAME):

    NR_BINS = -1 #safety barrier; does not matter for classification tasks, will be overwritten for regression tasks


    if SET_NAME ==  "Diabetes": 

        SET_HANDLE = "REG"
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        #Dataset-Shape: (442, 10)
        #Labels-Shape: (442,)
        NR_BINS = 10
        
    


    elif SET_NAME == "Treasury":

        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=42367, return_X_y=True)
        #Dataset-Shape: (1049, 15)
        #Labels-Shape: (1049,)
        NR_BINS = 10
        


    
    elif SET_NAME == "Wine_Quality_REG":

        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=287, return_X_y=True)
        #Dataset-Shape: (6497, 11)
        #Labels-Shape: (6497,)
        NR_BINS = 10

        
    
    elif SET_NAME == "Topo21":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=422, return_X_y=True)
        #Dataset-Shape: (8885, 266)
        #Labels-Shape: (8885,)
        NR_BINS = 15


 
    elif SET_NAME == "Bike_Sharing_Demand":

        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=42713, return_X_y=True)
        X_ = X.copy()
        for h in range(len(X_['hour'])):
            x_h = X_['hour'][h]
            if x_h < 6:
                X['hour'][h] = 0
            elif x_h < 12:
                X['hour'][h] = 1
            elif x_h < 15:
                X['hour'][h] = 2
            elif x_h < 18:
                X['hour'][h] = 3
            else:
                X['hour'][h] = 4
        X_ = X.copy()
        X = pd.get_dummies(X_, columns=['season', 'year', 'month', 'hour', 'weekday', 'weather'])
        X = X.drop(['season_winter', 'year_2012', 'month_12', 'hour_4', 'weekday_6', 'weather_rain'], axis = 1)
        #Dataset-Shape: (17379, 39)
        #Labels-Shape: (17379,)
        NR_BINS = 20
        
        

    elif SET_NAME == "Mimic_Los":
        SET_HANDLE = "REG"
        ### MIMIC specific NaN-values ###
        skipper_set = list(set([175,   175,   175,   440,   440,   440,  1925,  1925,  1925,
            3020,  3020,  3020,  4113,  4113,  4113,  5016,  5016,  5016,
            5450,  5450,  5770,  5770,  5770,  6542,  6542,  6542,  7973,
            7973,  7973, 11051, 11051, 11051, 11824, 11824, 11824, 12961,
           12961, 13564, 13564, 13564, 14424, 14424, 14424, 17585, 17585,
           17585]))
        for i in range(len(skipper_set)):
            skipper_set[i] += 1
        dataset_path = HOME + '/Mimic_Data/mimic_set_v2_addon.csv'
        X = pd.read_csv(dataset_path, usecols=range(1, 46), skiprows=skipper_set)
        y = X.pop("los_hours") 
        X.pop("hospital_expire_flag") #this is an addon-label
        X = X.values
        y = y.values
        NR_BINS = 20
        #Dataset-Shape: (19196, 43)
        #Labels-Shape: (19196,)


    
    elif SET_NAME == "California_Housing": 

        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
        #Dataset-Shape: (20640, 8)
        #Labels-Shape: (20640,)
        NR_BINS = 20
        
    



    elif SET_NAME == "Online_News_Popularity":

        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=4545, return_X_y=True)
        X = X.drop(['url', 'timedelta'], axis = 1)
        #Dataset-Shape: (39644, 58)
        #Labels-Shape: (39644,)
        NR_BINS = 20
        

  
        
        
    elif SET_NAME == "Diamonds":

        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=42225, return_X_y=True)

        #print(np.unique(X['cut'])) #['Fair' 'Good' 'Ideal' 'Premium' 'Very Good']
        #print(np.unique(X['color'])) #['D' 'E' 'F' 'G' 'H' 'I' 'J']
        #print(np.unique(X['clarity'])) #['I1' 'IF' 'SI1' 'SI2' 'VS1' 'VS2' 'VVS1' 'VVS2']

        X['cut_c'] = [0.0]*len(X)
        X['color_c'] = [0.0]*len(X)
        X['clarity_c'] = [0.0]*len(X)


        def cut_quality(cq): # no "age"-cluster as this would cause a degenerate rob-distribution
            if cq == 'Fair': 
                return 0.0
            elif cq == 'Good': 
                return 1.0
            elif cq == 'Very Good': 
                return 2.0
            elif cq == 'Premium': 
                return 3.0
            elif cq == 'Ideal': 
                return 4.0
            print("error for", cq)

        def colour_quality(colq): # no "age"-cluster as this would cause a degenerate rob-distribution
            if colq == 'J': 
                return 0.0
            elif colq == 'I': 
                return 1.0
            elif colq == 'H': 
                return 2.0
            elif colq == 'G': 
                return 3.0
            elif colq == 'F': 
                return 4.0
            elif colq == 'E': 
                return 5.0
            elif colq == 'D': 
                return 6.0
            print("error for", colq)

        def clarity_quality(clrq): # no "age"-cluster as this would cause a degenerate rob-distribution    
            if clrq == 'I1': 
                return 0.0
            elif clrq == 'SI2': 
                return 1.0
            elif clrq == 'SI1': 
                return 2.0
            elif clrq == 'VS2': 
                return 3.0
            elif clrq == 'VS1': 
                return 4.0
            elif clrq == 'VVS2': 
                return 5.0
            elif clrq == 'VVS1': 
                return 6.0
            elif clrq == 'IF': 
                return 7.0
            print("error for", clrq)

        X_ = X.copy()
        for i in range(len(X)): #these assignments will trigger a pandas-warning; however, they work exactly as intended
            X['cut_c'][i] = cut_quality(X_['cut'][i])
            X['color_c'][i] = colour_quality(X_['color'][i])   
            X['clarity_c'][i] = clarity_quality(X_['clarity'][i])    
        X = X.drop(['cut', 'color', 'clarity'], axis = 1)
        #Dataset-Shape: (53940, 9)
        #Labels-Shape: (53940,)
        NR_BINS = 20
        


        
    elif SET_NAME == "BNG":

        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=1213, return_X_y=True)
        X_ = X.copy()
        X = pd.get_dummies(X_, columns=['x3', "x7", "x8"])
        X = X.drop(['x3_green', 'x7_yes', 'x8_large'], axis = 1)
        #Dataset-Shape: (78732, 12)
        #Labels-Shape: (78732,)
        NR_BINS = 20
        

   


################################################################ BENCHMARK SETS FROM https://github.com/LeoGrin/tabular-benchmark ################################################################




    #1 
    elif SET_NAME == "Superconduct":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=44148, return_X_y=True)
        NR_BINS = 20


    #2 
    elif SET_NAME == "House_Sales":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=44066, return_X_y=True)
        NR_BINS = 20


    #3 
    elif SET_NAME == "MiamiHousing2016":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=44147, return_X_y=True)
        NR_BINS = 15


    #4 
    elif SET_NAME == "MBGM":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=44061, return_X_y=True)
        X_ = X.copy()
        X = pd.get_dummies(X_, columns=['X3', 'X4', 'X6'])
        X = X.drop(['X3_6', 'X4_3', 'X6_11'], axis = 1)
        NR_BINS = 10


    #5 
    elif SET_NAME == "Yprop41":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=44054, return_X_y=True)
        NR_BINS = 10
              


    #6
    elif SET_NAME == "Elevators":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=44134, return_X_y=True)
        NR_BINS = 15



    #7 
    elif SET_NAME == "Isolet":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=44135, return_X_y=True)
        NR_BINS = 10
      


    #8 
    elif SET_NAME == "CPU_Act":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=44132, return_X_y=True)
        NR_BINS = 10
             


    #9 
    elif SET_NAME == "Pol":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=44133, return_X_y=True)
        NR_BINS = 15



    #10
    elif SET_NAME == "Ailerons":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=44137, return_X_y=True)
        NR_BINS = 15
      
        


################################################################ BIG SETS ################################################################



    #BIG 1
    elif SET_NAME == "Yolanda":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=42705, return_X_y=True)
        NR_BINS = 20



    #BIG 2
    elif SET_NAME == "Year":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=44027, return_X_y=True)
        NR_BINS = 20



    #BIG 3
    elif SET_NAME == "Buzzinsocialmedia_Twitter":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=4549, return_X_y=True)
        NR_BINS = 20



    #BIG 4
    elif SET_NAME == "Rossmann_Store_Sales":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=45647, return_X_y=True)
        NR_BINS = 20
        X_ = X.copy()
        X = pd.get_dummies(X_, columns=['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'IsPromoMonth'])
        X = X.drop(columns=['Store_0', 'DayOfWeek_0', 'Promo_0', 'StateHoliday_0', 'SchoolHoliday_0', 'StoreType_0', 'Assortment_0', 'IsPromoMonth_0'])



    #BIG 5
    elif SET_NAME == "Delays_Zurich_Transport":
        SET_HANDLE = "REG"
        X, y = sklearn.datasets.fetch_openml(data_id=40753, return_X_y=True)
        NR_BINS = 20
        X = X.drop(columns=['line_number', 'time'])
        X_ = X.copy()
        X = pd.get_dummies(X_, columns=['vehicle_type', 'weekday', 'direction', 'stop_id'])
        X = X.drop(columns=['vehicle_type_Bus', 'weekday_1', 'direction_1.0', 'stop_id_1302.0'])

        X, X_test, y, y_test = sklearn.model_selection.train_test_split(X, y, train_size=1_250_000, random_state=1337)

    else:
        print("No Set Defined!")
        print()
        return 
    

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32").reshape(-1, 1)

    print("This dataset has", len(np.unique(X)), "unique elements and", len(np.unique(y)), "unique targets!")
    print()

    return SET_HANDLE, X, y, NR_BINS 






