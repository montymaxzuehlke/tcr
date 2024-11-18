import sys
import os
from os.path import expanduser

HOME = expanduser("~") #if needed; provides home directory of current user
DEFAULT_ORIGIN = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ORIGIN = DEFAULT_ORIGIN + "/../"

for DEFAULT_DIRECTORY in [
DEFAULT_ORIGIN + "automl/", 
DEFAULT_ORIGIN + "Results/", 
DEFAULT_ORIGIN + "Results_REF/", 
DEFAULT_ORIGIN + "Processing/", 
DEFAULT_ORIGIN + "Processing_DA/", 
DEFAULT_ORIGIN + "r_Processing_DA/", 
DEFAULT_ORIGIN + "H2O_Processing/"]:

    for SET_NAME in ["Diabetes", 
                     "Treasury", 
                     "Wine_Quality_REG", 
                     "Topo21",
                     "Bike_Sharing_Demand", 
                     "Mimic_Los",
                     "California_Housing", 
                     "Online_News_Popularity", 
                     "Diamonds",
                     "BNG",
                     
                     "Superconduct",
                     "House_Sales",
                     "MiamiHousing2016",
                     "MBGM",
                     "Yprop41",
                     "Elevators",
                     "Isolet",
                     "CPU_Act",
                     "Pol",
                     "Ailerons",

                     "Yolanda",
                     "Year",
                     "Buzzinsocialmedia_Twitter",
                     "Rossmann_Store_Sales",
                     "Delays_Zurich_Transport"
                    ]:

        if not os.path.exists(DEFAULT_DIRECTORY + SET_NAME):
            os.mkdir(DEFAULT_DIRECTORY + SET_NAME)
            print("Directory " , SET_NAME ,  " Created ")
            print()
        else:
            print("Directory " , SET_NAME ,  " already exists")
            print()

