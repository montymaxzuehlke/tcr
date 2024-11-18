from sklearn.neighbors import KNeighborsRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import random
import pickle
import time
from xgboost import XGBRegressor, XGBClassifier
import xgboost
from sklearn.utils import shuffle, resample



"""
***
NECESSARY FUNCTIONS
***
"""

#A, B are normalised eigenvectors for the eigenvalues A_r, B_r, respectively (the more "misalignged" two vectors are, the larger the arccos(*) of the respective argument)
def cosine_similarity(A, A_r, B, B_r):
    c = 0
    for i in range(min(len(A),len(B))): #sum over all cosine-similarity values weighted by the eigenvalues
        c += np.arccos(np.clip(np.inner(A[i], B[i]), -1.0, 1.0)) * A_r[i] * B_r[i] #A[i] and B[i] are already normalised by construction, A_r[i] and B_r[i] are likewise non-zero by construction; clipping to account for out-of-range values due to imprecise calculations
    return c

#simply union of two lists (using sets); from https://www.geeksforgeeks.org/python-union-two-lists/
def Union(lst_1, lst_2):
    lst1 = lst_1.copy()
    lst2 = lst_2.copy()
    final_list = list(set(lst1) | set(lst2))
    return final_list


 

#basic robustness calculation suited for multiprocessing; works for both training and test data and takes the orders for both the feature space and label space p-metric
def rob_calc(
              ref_data_input,
              ref_labels_input,
              data_input,
              label_input,
              ord_data,
              ord_labels,
              handle,
              bootstrap_iterations,
              bootstraps_length
             ):

    epsilon = 1e-7
    start_time = time.time()

    #only working on copies as we manipulate the data
    ref_data = ref_data_input.copy()
    ref_labels = ref_labels_input.copy()
    data = data_input.copy()

    len_ref_data = len(ref_data)
    len_data = len(data)
    L = [0.0] * len_data


    if label_input is None: # only None for test data; otherwise we will use the true training labels as "proxy labels"; only 1 job for the nn-models as we can not use multiprocessing in child processes here (to be optimised in the future)

        print("Determining Proxy Regression-Labels", "Time Elapsed:", round(time.time()-start_time, 2), "Secs")
        print()
        #print("reflabs before all", ref_labels[:10])
        neigh = KNeighborsRegressor(n_neighbors=1, n_jobs=1) #by definition KNR uses the euclidean metric, equivalent to the Frobenius norm
        neigh.fit(ref_data, ref_labels)
        proxy_labels = neigh.predict(data)
        #print("proxy labs as predictions", proxy_labels[:10])

    else:
        print("Copying Labels")
        print()
        proxy_labels = label_input.copy() #to save computation time


    print("Calculating Sensitivity Lists", "Time Elapsed:", round(time.time()-start_time, 2), "Secs")
    print()


    BOOT = np.array([-1.0]*bootstrap_iterations, dtype='float32')
    

    #main calculation part
    for index in range(len_data):
        if index % 100 == 0:
            print("Percentage:", round(index / len_data,4), "Time Elapsed:", round(time.time()-start_time, 2), "Secs")

        for bs in range(bootstrap_iterations):
            ref_data_boot, ref_labels_boot = shuffle(ref_data, ref_labels)
            ref_data_boot, ref_labels_boot = ref_data_boot[:bootstraps_length], ref_labels_boot[:bootstraps_length]

            MAX = float("-inf")
            for i in range(len(ref_data_boot)):
                norm_labels_difference = LA.norm(ref_labels_boot[i] - proxy_labels[index], ord=ord_labels) #for the CLA case and the p-metric this is either in {0,1} for binary tasks or in {0, "p-th root of 2"} for multilabel classification (either way the label map has sup-norm 1)
                #if i%1000==0:
                    #print(norm_labels_difference)
                if norm_labels_difference > 0:
                    MAX = max(MAX, norm_labels_difference / (LA.norm(ref_data_boot[i]-data[index], ord=ord_data) + epsilon))
            BOOT[bs]=MAX
        L[index] = 1/(np.median(BOOT) + 1) # Robustness value assuming |y|_\infty = 1

        
    return L





def final_heat_map_displayer(val_heat_display, tes_heat_display, indicator, iterator, len_val_tes, len_tra, path, show_graphs, val_subsets, tes_subsets, tra_subsets):


    #Framework copied from matplotlib


    #Validation Heatmaps

    heat_val = np.array(val_heat_display)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(heat_val)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len_val_tes))
    ax.set_yticks(np.arange(len_tra))
    # ... and label them with the respective list entries
    ax.set_xticklabels(val_subsets)
    ax.set_yticklabels(tra_subsets)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len_tra):
        for j in range(len_val_tes):
            text = ax.text(j, i, round(heat_val[i, j], 4),
                           ha="center", va="center", color="w")

    ax.set_title("Validation Performance " + indicator)
    fig.tight_layout()
    plt.savefig(path + '/VAL_' + indicator + '_' + str(iterator) + '.png', dpi=200)
    if show_graphs:
        plt.show()




    #Test Heatmaps

    heat_tes = np.array(tes_heat_display)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(heat_tes)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len_val_tes))
    ax.set_yticks(np.arange(len_tra))
    # ... and label them with the respective list entries
    ax.set_xticklabels(tes_subsets)
    ax.set_yticklabels(tra_subsets)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len_tra):
        for j in range(len_val_tes):
            text = ax.text(j, i, round(heat_tes[i, j], 4),
                           ha="center", va="center", color="w")

    ax.set_title("Test Performance " + indicator)
    fig.tight_layout()
    plt.savefig(path + '/TES_' + indicator + '_' + str(iterator) + '.png', dpi=200)
    if show_graphs:
        plt.show()







def final_decision_heat_map_displayer(decision_heat_display, indicator, iterator, len_val_tes, len_tra, path, show_graphs, tes_subsets, tra_subsets):

    #Train/Test - Selection Heatmaps

    decision_heat = np.array(decision_heat_display)

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(decision_heat)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len_val_tes))
    ax.set_yticks(np.arange(len_tra))
    # ... and label them with the respective list entries
    ax.set_xticklabels(tes_subsets)
    ax.set_yticklabels(tra_subsets)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len_tra):
        for j in range(len_val_tes):
            text = ax.text(j, i, decision_heat[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Train/Test - Selection Percentage " + indicator)
    fig.tight_layout()
    plt.savefig(path + '/' + indicator + '_' + str(iterator) + '.png', dpi=200)
    if show_graphs:
        plt.show()





def reg_label_distribution_displayer(labs, bins_, han, path, itr, legends, show_graphs):

    fig1, ax = plt.subplots(figsize=(20,8))
    plt.hist(labs, bins=bins_, density=True, histtype='step', label=legends)
    plt.xticks(bins_)
    plt.title("Label Distribution " + han)
    plt.legend()
    plt.savefig(path + '/Label_Distribution_' + han + '_' + str(itr) + '.png', dpi=200)
    if show_graphs:
        plt.show()









#the original used for the data analysis
def xgb_runner(A, B):
    #inspired by: https://www.kaggle.com/code/carlmcbrideellis/what-is-adversarial-validation/notebook
    
    X_AB = np.concatenate([A.copy(), B.copy()])
    
    y_A = np.array([0]*len(A))
    y_B = np.array([1]*len(B))
    y_AB = np.concatenate([y_A, y_B])
    
    
    X_AB, y_AB = shuffle(X_AB, y_AB)
    
    
    XGBdata = xgboost.DMatrix(data=X_AB,label=y_AB)
    # our XGBoost parameters
    params = {"objective":"binary:logistic",
              "eval_metric":"logloss",
              'learning_rate': 0.05,
              'max_depth': 10}

    # perform cross validation with XGBoost
    cross_val_results = xgboost.cv(dtrain=XGBdata, 
                               params=params, 
                               nfold=10, 
                               metrics="auc", 
                               num_boost_round=200,
                               early_stopping_rounds=50,
                               as_pandas=False 
                               )
    
    return [cross_val_results["train-auc-mean"][-1], cross_val_results["train-auc-std"][-1], cross_val_results["test-auc-mean"][-1], cross_val_results["test-auc-std"][-1]]
    

    
    
def adversarial_validation(X_low, X_upp, T_low, T_upp, path, i_t_r):
    #inspired by: https://www.kaggle.com/code/carlmcbrideellis/what-is-adversarial-validation/notebook
    
    """#structure ("s":= determined by symmetry, "-":= not interesting)
    -     xgb_01 xgb_02 xgb_03  
    s     -      xgb_12 xgb_13
    s     s      -      xgb_23
    s     s      s      -
    """
    
    """
    legend of the indices:
    "0" := X_low
    "1" := T_low
    "2" := X_upp
    "3" := T_upp
    """
    
    xgb_01 = xgb_runner(X_low, T_low)
    xgb_02 = xgb_runner(X_low, X_upp)  
    xgb_03 = xgb_runner(X_low, T_upp)
    
    xgb_12 = xgb_runner(T_low, X_upp)  
    xgb_13 = xgb_runner(T_low, T_upp)
    
    xgb_23 = xgb_runner(X_upp, T_upp)
    
    for han, xgb in [("xgb_01", xgb_01), 
                     ("xgb_02", xgb_02),
                     ("xgb_03", xgb_03),
                     
                     ("xgb_12", xgb_12),
                     ("xgb_13", xgb_13),
                     
                     ("xgb_23", xgb_23),
        ]:
        
        with open(path + han + '_' + str(i_t_r) + ".pickle", 'wb') as f:
                pickle.dump(xgb, f)

    
  
