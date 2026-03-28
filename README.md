# Manual for Reproducing the Experiments

This is the manual for reproducing the results from:

> Zuehlke, M.-M., & Kudenko, D. (2025). TCR: topologically consistent reweighting for XGBoost in regression tasks. *Machine Learning*, 114(4), 108. Springer.

```bibtex
@article{zuehlke2025tcr,
  title={TCR: topologically consistent reweighting for XGBoost in regression tasks},
  author={Zuehlke, Monty-Maximilian and Kudenko, Daniel},
  journal={Machine Learning},
  volume={114},
  number={4},
  pages={108},
  year={2025},
  publisher={Springer}
}
```

The `tcr` directory contains all necessary files, installation and execution instructions, which are detailed below. The `AXIL` directory contains a file from: https://github.com/pgeertsema/AXIL_paper/tree/main

---

## 1. Installation / Environment Setup

**Step 0.** The entire TCR pipeline produces multiple files along the way that are stored in separate directories. To create the necessary infrastructure, run the `create_infrastructure.sh` shell script from inside the `tcr` directory:

```sh
sh create_infrastructure.sh
```

**Step 1.** Set up the `tcr` conda environment:

```sh
conda env create -f tcr.yml
```

**Step 2.** Activate the environment:

```sh
conda activate tcr
```

---

## 2. Running the Data Pipeline

To calculate the robustness values, run the `data_pipeline.py` script. This script takes two arguments:

- `SET_NAME` (str): The name of the set, loaded from the `dataset_provider.py` script in the Utils directory.
- `ITERATOR` (int): The random seed for the run.

**Example:**

```sh
python3 data_pipeline.py Diabetes 42
```

> **Add-on:** If you want to calculate the robustness values for a subset only and infer the remaining values via a nearest neighbour approach (see Algorithm 2 in the article), set the `MAX_SAM_SIZE` variable in the script header to the desired sample size. The default value is `-1`, which corresponds to using the entire data. Note that this only applies to calculating the robustness values of the training set.

By default, the robustness values of the training set and the test set are determined. Likewise, the input data for both the weighted misalignment heatmaps and the adversarial validation comparison are calculated. However, the calculations can be restricted by setting the corresponding variables (`CALCULATE_ROBUSTNESS_LISTS`, `COMPUTE_COS_SIM_MATRIX`, `ADVERSARIAL_VALIDATION`) in the script header to `False`.

To simulate random splits (instead of splits determined by robustness values), set the `RANDOMISING_LISTS` variable to `True` (default is `False`). This will shuffle the robustness values, leading to random splits.

---

## 3. Visualising the Data Analyses' Results

To visualise the results from the previous step, run the `data_analysis_display.py` script. This script takes one argument:

- `SET_NAME` (str): The name of the set, loaded from the `dataset_provider.py` script in the Utils directory.

**Example:**

```sh
python3 data_analysis_display.py Diabetes
```

Because this script combines results from multiple runs, the script header has a variable (`X_VAL_LIST`) that holds a list of all random seeds for which results are included. Similarly, visualisation methods can be included or excluded by setting the corresponding variables (`ROBUSTNESS_LISTS`, `COSINE_SIMILARITY_HEATMAPS`, `ADVERSARIAL_VALIDATION`) to `True` (default) or `False`.

To split the weighted misalignment heatmaps along the diagonal (so that results based on random splits are included), set the `SPLIT_IN_HALF` variable to `True` (default). To use only the results based on random splits, set the `USE_RANDOM_RESULTS` variable to `True` (default is `False`).

Graphics are saved as `.png` files; statistics from the adversarial validation results are printed to the console (including a LaTeX tabular-friendly format).

---

## 4. Running the H2O Pipeline

To run the hyperparameter search with H2O (that is, train XGBoost regressors with optimised hyperparameters given a time budget of 3 hours), run the `automl_pipeline.py` script. This script takes two arguments:

- `SET_NAME` (str): The name of the set, loaded from the `dataset_provider.py` script in the Utils directory.
- `ITERATOR` (int): The random seed for the run.

**Example:**

```sh
python3 automl_pipeline.py Diabetes 42
```

> **Important:** Before running this script, you need to calculate the robustness values via the `data_pipeline.py` script, because the training and test frames are saved in the order defined by the robustness values. This facilitates the reweighting process in the next script, where we load not only the XGBoost models (more precisely, their hyperparameters) but their training and test data. If you do not want to calculate these lists beforehand, you can create dummy arrays, e.g.:
> ```python
> np.array([0.0]*len_train, dtype="float32")
> np.arange(len_train)
> np.array([0.0]*len_test, dtype="float32")
> np.arange(len_test)
> ```
> Note that the test arrays need to be consistent when using any of the reweighting techniques in `tcr_pipeline.py`.

---

## 5. Running the TCR Pipeline (Incl. Other Reweighting Methods)

To run the TCR pipeline (retraining models with adjusted weights), run the `tcr_pipeline.py` script. This script takes four arguments:

- `SET_NAME` (str): The name of the set, loaded from the `dataset_provider.py` script in the Utils directory.
- `ITERATOR` (int): The random seed for the run.
- `METHOD` (str): The reweighting method. Choose one of: `DENSEWEIGHT`, `SMOGN`, `AXIL`, `TCR`.
- `KEY` (int): The index for the list containing method-specific hyperparameters (alpha for DenseWeight and lambda for TCR). Set this to `0` for `SMOGN` and `AXIL`.

> **Add-on:** This script uses two additional arguments:
>
> - `MAX_SAM_SIZE_KEY` (int): The index of the list value that holds the sample size used in `data_pipeline.py`. The list is found in the script header and can be adjusted. The value `-1` corresponds to the robustness values calculated from the entire data. Using a list-index-argument approach allows streamlining the evaluation across several arguments.
> - `TCR_PERCENTAGE_KEY` (int): The index of the list value that holds the TCR threshold, i.e., the amount of elements that are reweighted (thresholds of 0.05, 0.1, and 0.2 were used in the experiments). As before, a list-index-argument approach is used to allow streamlining. If you want to use any other method, simply provide dummy values (they are disregarded whenever `METHOD != TCR`).

**Example:**

```sh
python3 tcr_pipeline.py Diabetes 42 TCR 8 0 0
```

The list of parameters for DenseWeight is: `[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]`

The list of parameters for TCR is: `[0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]`

These lists can be adjusted to one's liking. For compatibility with job factories (running multiple jobs with arguments provided by a for loop), the `KEY` argument is the index of the list, not the parameter value itself. This also has the advantage of enabling more complex combinations of parameters, as used for `SMOGN`.

---

## 6. Displaying the Comparison Between the Reweighting Methods

To visualise the comparison between the different reweighting methods, run the `stat_comparison_display.py` script. This script takes no command-line arguments, but several variables can be set in the header:

- Set `HEATMAP` and `BOXPLOTS` to `True` to display the respective visualisations.
- When displaying heatmaps, set `INCLUDE_BEST` to `True` (default) to include performances for the optimal parameters.
- You can change the default weighting parameters for DenseWeight and TCR (the conservative values alpha=0.1 and lambda=0.9 were used in the experiments).
- The list `X_VAL_LIST` contains the random seeds for which results are included in the calculations (similar to `data_analysis_display.py`).

To select specific datasets, comment out the remaining ones in the `SET_NAME_LIST` list.

> **Add-on:** This script uses two additional arguments:
>
> - `MAX_SAM_SIZE` (str): A value in `{"all", "_<sample_size>"}`. Use `"all"` to use robustness values calculated from the entire data; otherwise set to the sample size with a leading underscore, e.g., `_5000` for a subset of 5000 elements.
> - `LEAST_ROB_PERCENTAGE` (str): A value in `{"_<tcr_threshold>"}` that determines which portion of the least robust results have been reweighted. For example, if a threshold of `0.1` was used in `tcr_pipeline.py`, set this to `_0.1`.

**Example:**

```sh
python3 stat_comparison_display.py all _0.1
```

---

## 7. Displaying the Scaling and Sensitivity Comparisons

To visualise the scaling and sensitivity comparisons, run the `scaling_comparison_display.py` and `sensitivity_comparison_display.py` scripts. To run these, the TCR results for all weight factors and at least one `LEAST_ROB_PERCENTAGE` (values of 0.05, 0.1, and 0.2 were used) and one `MAX_SAM_SIZE` (the entire set and subsets of 5000, 10000, 20000, and 50000 elements were used) need to exist. The scripts do not require any command-line arguments; all variables are set within the script. The list `X_VAL_LIST` contains the random seeds for which results are included.

**For `scaling_comparison_display.py`:** Select `LEAST_ROB_PERCENTAGE` and the particular dataset in the script header (similar to `stat_comparison_display.py`). Set the list of `MAX_SAM_SIZE` values to consider in line 80. The empty string `""` corresponds to the results based on robustness values calculated from the entire data.

**For `sensitivity_comparison_display.py`:** Select `MAX_SAM_SIZE` (the empty string `""` corresponds to the results based on robustness values calculated from the entire data) and the particular dataset in the script header. Set the list of `LEAST_ROB_PERCENTAGE` values to consider in line 81. If values other than `[0.05, 0.1, 0.2]` are used for `LEAST_ROB_PERCENTAGE`, the `xticklabels` in lines 189, 190, and 191 need to be adjusted.

**Examples:**

```sh
python3 scaling_comparison_display.py
python3 sensitivity_comparison_display.py
```

---

## 8. Extension to Custom Datasets

All of the above can also be run with custom datasets. To do so, add another entry in the `dataset_provider.py` script in the same format. The `NR_BINS` argument defines into how many bins the targets should be distributed to allow stratified training/test splits. Select this value based on the number of samples or unique target values in the dataset.
