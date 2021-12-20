import pandas as pd
import numpy as np

import csv
import json

from utils import *

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)

################
DATA_PATH = 'SET_DATA_PATH_HERE'
OUTPUT_PATH = 'SET_OUTPUT_PATH_HERE'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=False,
                    help='Path to dataset')
parser.add_argument('--output_path',type=str, default=False, 
                    help='Path to write output artifacts')
args = parser.parse_args()
print(args)
#################

data_original = pd.read_parquet(DATA_PATH)
data_original['user_id'] = [x.rsplit('_')[0] for x in data_original.index]
data_original['user_mth'] = [int(x.rsplit('_')[1]) for x in data_original.index]
data_original['ref_label'] = ~data_original.phq9_cat_end.isna()

data_original['user_qtr'] = (data_original.user_mth - 1) // 3
data_original['user_id_qtr'] = data_original.apply(lambda x: str(x.user_id) + '_' + str(x.user_qtr), axis=1)

# only keep rows that can be used for phase 2 (i.e. users that have completed phq9 surveys)
data_original = data_original[data_original.user_id_qtr.isin(data_original[data_original.ref_label].user_id_qtr.unique())]

data_original['decl'] = data_original.phq9_cat_end > data_original.phq9_cat_start

PHASE_1_INPUT_COLS = ['sleep__main_start_hour_adj__score', 'sleep__main_start_hour_adj__intercept', 'sleep__main_start_hour_adj__coeff',
             'sleep_main_start_hour_adj_median', 'sleep_main_start_hour_adj_iqr', 'sleep_main_start_hour_adj_range',
             'sleep__total_asleep_minutes__score', 'sleep__total_asleep_minutes__intercept', 'sleep__total_asleep_minutes__coeff',
             'sleep__awake__sum__score', 'sleep__awake__sum__intercept', 'sleep__awake__sum__coeff','sleep__nap_count__score',
             'sleep__nap_count__intercept', 'sleep__nap_count__coeff', 
             'sleep__total_asleep_minutes__score_', 
             'sleep__total_asleep_minutes__intercept_', 'sleep__total_asleep_minutes__coeff_', 'sleep__main_efficiency__score_', 
             'sleep__main_efficiency__intercept_', 'sleep__main_efficiency__coeff_', 
             'sleep__awake__sum__score_',
             'sleep__awake__sum__intercept_', 'sleep__awake__sum__coeff_', 
             'sleep__total_in_bed_minutes__score_',
             'sleep__total_in_bed_minutes__intercept_', 'sleep__total_in_bed_minutes__coeff_', 
             'steps__awake__sum__score_',
             'steps__awake__sum__intercept_', 'steps__awake__sum__coeff_', 'steps__mvpa__sum__score_',
             'steps__mvpa__sum__intercept_', 'steps__mvpa__sum__coeff_', 'steps__light_activity__sum__score_',
             'steps__light_activity__sum__intercept_', 'steps__light_activity__sum__coeff_',
             'steps_mvpa_sum_recent',
             'sleep_asleep_mean_recent', 'sleep_in_bed_mean_recent', 'sleep_ratio_asleep_in_bed_mean_recent',
             'steps_lpa_sum_recent', 
            ] + [
    x for x in data_original.columns if x.startswith('life') or x.startswith('med') or x.startswith('nonmed')]

PHASE_1_INPUT_COLS = list(set(SCREENER_COLS + PHASE_1_INPUT_COLS))

SELECTED_COLS_1 = {
    2732: ['trauma', 'comorbid_gout', 'nonmed_stop', 'life_meditation','insurance', 'meds_migraine', 'sex', 
           'educ', 'pregnant', 'med_nonmed_dnu', 'comorbid_arthritis', 'race_asian', 'weight', 'money_assistance', 
           'num_migraine_days', 'med_stop', 'household', 'med_dose', 'med_start', 'comorbid_migraines', 'birthyear', 
           'comorbid_neuropathic', 'bmi', 'height', 'race_hispanic', 'race_black', 'money', 
           'sleep_main_start_hour_adj_range'],
    9845: ['trauma', 'comorbid_cancer', 'comorbid_gout', 'nonmed_stop', 'life_meditation', 'insurance',
           'meds_migraine', 'sex', 'educ', 'med_nonmed_dnu', 'comorbid_arthritis', 'weight', 'money_assistance', 
           'num_migraine_days', 'med_stop', 'sleep_ratio_asleep_in_bed_mean_recent', 'med_dose', 'med_start', 
           'comorbid_migraines', 'birthyear', 'comorbid_neuropathic', 'comorbid_diabetes_typ1', 'bmi', 'height', 
           'race_hispanic', 'race_black', 'money', 'sleep_main_start_hour_adj_range'], 
    3264: ['educ', 'weight', 'comorbid_migraines', 'bmi', 'comorbid_neuropathic', 'med_dose', 'birthyear', 
           'race_black', 'trauma', 'comorbid_arthritis', 'nonmed_stop', 'comorbid_cancer', 'race_hispanic', 
           'money_assistance', 'money', 'height', 'race_white', 'med_start', 'meds_migraine', 'insurance', 
           'num_migraine_days', 'sex'], 
    4859: ['trauma', 'comorbid_cancer', 'comorbid_gout', 'nonmed_stop', 'life_meditation', 'insurance', 
           'meds_migraine', 'sex', 'educ', 'pregnant', 'comorbid_arthritis', 'weight', 'money_assistance', 
           'num_migraine_days', 'med_stop', 'household', 'med_dose', 'med_start', 'comorbid_migraines', 
           'birthyear', 'comorbid_neuropathic', 'bmi', 'height', 'race_hispanic', 'race_black', 'money', 
           'sleep__hypersomnia_count_', 'comorbid_osteoporosis'], 
    9225: ['educ', 'weight', 'comorbid_migraines', 'bmi', 'med_nonmed_dnu', 'comorbid_neuropathic', 'med_dose',
           'sleep_main_start_hour_adj_range', 'comorbid_ms', 'birthyear', 'race_black', 'trauma', 'comorbid_arthritis',
           'comorbid_cancer', 'race_hispanic', 'money_assistance', 'money', 'height', 'race_white', 'med_start',
           'meds_migraine', 'insurance', 'num_migraine_days', 'sex', 'comorbid_gout', 'med_stop',]
}

SELECTED_PARAMS_1 = {
    2732: {'n_estimators': 115, 'max_depth': 4, 'drop_rate': 0.2},
    9845: {'n_estimators': 140, 'max_depth': 7, 'drop_rate': 0.05},
    3264: {'n_estimators': 50, 'max_depth': 4, 'drop_rate': 0.0},
    4859: {'n_estimators': 85, 'max_depth': 3, 'drop_rate': 0.0},
    9225: {'n_estimators': 160, 'max_depth': 3, 'drop_rate': 0.15}
}

# selected by getting the most common columns from rfecv on these feature sets separately

SLEEP_CANDIDATE_COLS = ['sleep__awake_regions__countDistinct__coeff', 'sleep_asleep_weekend_mean', 
                        'sleep_ratio_asleep_in_bed_weekday_mean', 'sleep_in_bed_iqr', 'sleep__main_start_hour_adj__intercept', 
                        'sleep__awake__sum__score', 'sleep__total_in_bed_minutes__score', 'sleep__awake_regions__countDistinct__score',
                        'sleep__total_asleep_minutes__score_', 'sleep__main_start_hour_adj__score', 'sleep__main_start_hour_adj__coeff',
                        'sleep__nap_count__intercept', 'sleep__main_efficiency__score', 'sleep__main_efficiency__coeff',
                        'sleep__awake_regions__countDistinct__intercept', 'sleep_ratio_asleep_in_bed__score', 
                        'sleep__total_asleep_minutes__coeff_', 'sleep__awake__sum__score_', 'sleep__awake__sum__coeff_', 
                        'sleep__total_in_bed_minutes__score_', 'sleep_ratio_asleep_in_bed__score_', 'sleep_asleep_mean_recent',]
STEPS_CANDIDATE_COLS = ['steps__mvpa__sum__intercept', 'steps__light_activity__sum__intercept_', 'steps__light_activity__sum__score',
                        'steps_awake_mean', 'steps_lpa_iqr', 'steps_awake_sum_iqr', 'steps__awake__sum__score', 'steps__awake__sum__coeff', 
                        'steps__mvpa__sum__score', 'steps__mvpa__sum__coeff', 'steps__rolling_6_sum__max__score',
                        'steps__rolling_6_sum__max__intercept', 'steps__rolling_6_sum__max__coeff', 'steps__streaks__countDistinct__score',
                        'steps__streaks__countDistinct__intercept', 'steps__streaks__countDistinct__coeff', 'steps__not_moving__sum__score',
                        'steps__not_moving__sum__intercept', 'steps__not_moving__sum__coeff', 'steps_lpa_sum_recent', 
                        'steps_rolling_6_median_recent', 'steps_rolling_6_max_recent',]
SCREENER_CANDIDATE_COLS = ['birthyear', 'educ', 'height', 'weight', 'money', 'money_assistance', 'household', 'comorbid_migraines', 
                           'comorbid_neuropathic', 'comorbid_arthritis', 'sex', 'race_black', 'num_migraine_days', 'insurance', 'race_white',
                           'trauma',]
LMC_CANDIDATE_COLS = ['med_start', 'med_stop', 'med_dose', 'nonmed_start', 'nonmed_stop', 'life_meditation', 'life_stress', 'med_nonmed_dnu',
                      'life_activity_eating', 'life_red_stop_alcoh']

PHASE_2_INPUT_COLS = SLEEP_CANDIDATE_COLS + STEPS_CANDIDATE_COLS + SCREENER_CANDIDATE_COLS + LMC_CANDIDATE_COLS + PROBA_COLS_M1 + PROBA_COLS_M2 + PREV_CAT_COLS

SELECTED_COLS_2 = {
    2732: ['sleep__awake__sum__coeff_', 'sex', 'insurance', 'med_start', 'med_stop', 'med_dose', 'nonmed_stop', 'life_meditation', 'life_stress',
           'med_nonmed_dnu', 'life_red_stop_alcoh', 'cat_m0', 'cat_m2',],
    9845: ['comorbid_arthritis', 'race_black', 'trauma', 'med_start', 'med_stop', 'med_dose', 'nonmed_stop', 'life_meditation', 'life_stress',
           'life_red_stop_alcoh', 'proba_cat_0_m2', 'cat_m0', 'cat_m2',],
    3264: ['comorbid_neuropathic', 'sex', 'race_black', 'insurance', 'med_start', 'med_stop', 'med_dose', 'nonmed_start', 'nonmed_stop',
           'life_meditation', 'life_red_stop_alcoh', 'cat_m0', 'cat_m1',],
    4859: ['sex', 'insurance', 'med_start', 'med_stop', 'med_dose', 'nonmed_stop', 'life_meditation', 'med_nonmed_dnu', 'life_activity_eating',
           'life_red_stop_alcoh', 'proba_cat_1_m2', 'cat_m0', 'cat_m1',],
    9225: ['money_assistance', 'sex', 'insurance', 'med_start', 'med_stop','med_dose', 'nonmed_stop', 'life_meditation', 'life_stress', 
           'life_activity_eating', 'life_red_stop_alcoh', 'cat_m0', 'cat_m1'],
}

RANDOM_SEEDS = [int(x) for x in np.random.randint(0, 10000, 5)]

N_ITER = 50
CV = 5
N_FEATURES_TO_SELECT = 13
IMPORTANCE_TYPE = 'split'

PARAMS_P1 = {
    'selector__n_features_to_select': np.arange(20, 31),
    'clf__n_estimators' : np.arange(40, 170, 10), 
    'clf__max_depth' : np.arange(2, 8),
    'clf__drop_rate': [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
}

feature_importances_1, feature_importances_2 = [], []

results_1 = {'acc': [], 'adjacent_acc': [], 'balanced_acc': [], 'kappa': [], 'f1': []}
results_2 = {'sensitivity': [], 'auroc': [], 'f1': [], 'balanced_acc': [], 'precision': [], 
                     'specificity': [], 'auprc': []}

model_dict = {}

for seed in RANDOM_SEEDS:    
    data = data_original.copy()
    
    print(f"RANDOM SEED {seed}")
    
    print("PHASE 1:")
    
    data_train, data_test = split_participant_data_random(data_original, n_users_test=800, seed=seed)
    
    data_train = data_train.loc[data_train.ref_label]
    data_test = data_test.loc[data_test.ref_label]
    
    print(f"Nb train samples: {len(data_train.index)}, nb test samples: {len(data_test.index)}")
    print(f'Nb generated intermediate labels: {len(data.loc[~data.ref_label].index)}')
    
    X = data[PHASE_1_INPUT_COLS]
    y = data['phq9_cat_end']
    
    X_train, X_test = X.loc[data_train.index], X.loc[data_test.index]
    y_train, y_test = y.loc[data_train.index], y.loc[data_test.index]
    
    X_generate = X.loc[data.loc[~data.ref_label].index]

    # use this to find optimal parameters
    model_1 = phase_1_pipeline(X_train, y_train, 
                               user_id_groups=data.loc[data_train.index, 'user_id'],
                               params=PARAMS_P1, n_iter=N_ITER, cv=CV, seed=seed)
    
    # use this when using selected parameters
#     model_1 = phase_1_model(X_train, y_train, 
#                             n_estimators=SELECTED_PARAMS_1[seed]['n_estimators'], 
#                             max_depth=SELECTED_PARAMS_1[seed]['max_depth'], 
#                             drop_rate=SELECTED_PARAMS_1[seed]['drop_rate'],
#                             seed=seed, importance_type=IMPORTANCE_TYPE)
    
    # PHASE 1: generate labels
    
    y_pp1 = model_1.predict_proba(X_test)
    y_p1 = model_1.predict(X_test)
    
    seed_results = get_model_performance("one", y_test, y_p1, y_pp1)
    
    feature_importances_1.append(model_1.feature_importances_)
    
    print(f"MODEL 1 PERFORMANCE: {seed_results}")
    
    for k, v in seed_results.items():
        results_1[k].append(v)
        
    data.loc[data.loc[~data.ref_label].index, PROBA_COLS] = model_1.predict_proba(X_generate)
    data.loc[data.loc[~data.ref_label].index, 'phq9_cat_end'] = model_1.predict(X_generate)
    
    print(data.phq9_cat_end.value_counts() / len(data))
        
    print("PHASE 2:")
    
    X_2, y_2 = prep_phq_decline_data(data)
    
    X_train_2, X_test_2, y_train_2, y_test_2 = split_train_test_set(
        X_2[PHASE_2_INPUT_COLS],
        y_2, data_train.index, data_test.index)
    
    # to make sure that the same participants are in the train and test sets for phase 1c and phase 2c
#     print(f'Sym diff between indices of training sets for phase 1 and 2 {set(list(X_train_2.index)).symmetric_difference(set(list(X_train.index)))}')
#     print(f'Sym diff between indices of test sets for phase 1 and 2 {set(list(X_test_2.index)).symmetric_difference(set(list(X_test.index)))}')

    # if performing feature selection
    model_2 = phase_2_pipeline(X_train_2, y_train_2, n_features_to_select=N_FEATURES_TO_SELECT,
                           cv=CV, seed=seed)
        
    # if using selected features
#     model_2 = phase_2_model(X_train_2, y_train_2, seed=seed, importance_type=IMPORTANCE_TYPE)      
    
    print(f"Nb train samples: {len(X_train_2)}, nb test samples: {len(X_test_2)}")
    
    y_p2 = model_2.predict(X_test_2)
    y_pp2 = model_2.predict_proba(X_test_2)[:, 1]
    seed_results = get_model_performance("two", y_test_2, y_p2, y_pp2)
    
    feature_importances_2.append(model_2.feature_importances_)
    
    print(f"Actual increase in severity: {y_test_2.sum() / len(y_test_2):.2f}")
    print(f"Predicted increase in severity: {y_p2.sum() / len(y_p2):.2f}")
    
    print(f"MODEL 2 PERFORMANCE: {seed_results}")
    
    for k, v in seed_results.items():
        results_2[k].append(v)

print('SAVING FEATURE IMPORTANCES...')

with open(OUTPUT_PATH + "/importance_" + IMPORTANCE_TYPE + "_1.json", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(feature_importances_1)
    
with open(OUTPUT_PATH + "/importance_" + IMPORTANCE_TYPE + "_2.json", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(feature_importances_2)
    
print('FEATURE IMPORTANCES SAVED!')
        
print('FINAL RESULTS:')

final_performance(results_1)
print()
final_performance(results_2)
    
print('PROCESS COMPLETE')
