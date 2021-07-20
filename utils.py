import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, precision_score,
                             precision_recall_curve, auc)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight


WEAK_LABEL_INPUT_COLS = [
    'sex', 'birthyear', 'educ', 'height', 'weight', 'bmi', 'birth', 'trauma', 'insurance',
    'money', 'money_assistance', 'household', 'comorbid_cancer', 'comorbid_diabetes_typ2',
    'comorbid_gout', 'comorbid_migraines', 'comorbid_osteoporosis', 'comorbid_neuropathic',
    'comorbid_arthritis', 'num_migraine_days', 'meds_migraine', 'med_start', 'med_dose',
    'med_nonmed_dnu', 'steps__active_day_count_', 'sleep__hypersomnia_count_',]

PROBA_COLS = ['proba_cat_0', 'proba_cat_1', 'proba_cat_2', 'proba_cat_3', 'proba_cat_4']
PROBA_COLS_M1 = [x + '_m1' for x in PROBA_COLS]
PROBA_COLS_M2 = [x + '_m2' for x in PROBA_COLS]

PREV_CAT_COLS = ['cat_m0', 'cat_m1', 'cat_m2',]

SCREENER_COLS = [
    'sex', 'race_white', 'race_black', 'race_hispanic', 'race_asian', 'race_other', 'birthyear',
    'educ', 'height', 'weight', 'bmi', 'pregnant', 'birth', 'trauma', 'insurance', 'money',
    'money_assistance', 'household', 'comorbid_cancer', 'comorbid_diabetes_typ1', 'comorbid_diabetes_typ2',
    'comorbid_gout', 'comorbid_migraines', 'comorbid_ms', 'comorbid_osteoporosis', 'comorbid_neuropathic',
    'comorbid_arthritis', 'num_migraine_days', 'meds_migraine',]

PREV_SELECTED_COLS = {
    "0": {"seed": 2732, 
          "selected_columns": [
              "sleep_main_start_hour_adj_range", "birth", "comorbid_diabetes_typ1", "med_dose",
              "life_activity_eating", "proba_cat_1_m1", "proba_cat_2_m1", "proba_cat_3_m1", "proba_cat_4_m1",
              "proba_cat_0_m2", "proba_cat_1_m2", "proba_cat_2_m2", "cat_m0", "cat_m2"]},
    "1": {"seed": 9845,
          "selected_columns": [
              "sleep_main_start_hour_adj_range", "sleep__awake__sum__coeff_", "comorbid_cancer", "med_start",
              "proba_cat_1_m1", "proba_cat_2_m1", "proba_cat_3_m1", "proba_cat_4_m1", "proba_cat_1_m2",
              "proba_cat_2_m2", "proba_cat_3_m2", "proba_cat_4_m2", "cat_m0", "cat_m1"],}, 
    "2": {"seed": 3264, 
          "selected_columns": [
              "comorbid_diabetes_typ1", "comorbid_migraines", "med_dose", "life_meditation", "proba_cat_0_m1",
              "proba_cat_1_m1", "proba_cat_2_m1", "proba_cat_3_m1", "proba_cat_4_m1", "proba_cat_1_m2",
              "proba_cat_2_m2", "proba_cat_3_m2", "cat_m0", "cat_m2"],}, 
    "3": {"seed": 4859,
          "selected_columns": [
              "race_hispanic", "comorbid_gout", "med_start", "med_stop", "life_red_stop_alcoh",
              "proba_cat_1_m1", "proba_cat_2_m1", "proba_cat_3_m1", "proba_cat_4_m1", "proba_cat_0_m2",
              "proba_cat_2_m2", "proba_cat_3_m2", "cat_m0", "cat_m2"],}, 
    "4": {"seed": 9225, 
          "selected_columns": [
              "steps__sedentary_day_count_", "sleep__main_start_hour_adj__intercept_", "race_other", 
              "comorbid_cancer", "proba_cat_1_m1", "proba_cat_2_m1", "proba_cat_3_m1", "proba_cat_4_m1",
              "proba_cat_1_m2", "proba_cat_2_m2", "proba_cat_3_m2", "proba_cat_4_m2", "cat_m0", "cat_m1"],}
}


def sensitivity_score(y_test, y_pred):
    cm = confusion_matrix(y_test,y_pred)
    _, _, fn, tp = cm.ravel()
    return tp/(tp+fn)

def specificity_score(y_test, y_pred):
    cm = confusion_matrix(y_test,y_pred)
    tn, fp, _, _ = cm.ravel()
    return tn/(tn+fp)

def get_model_performance(y_true, y_pred, y_pred_proba):
    perf = {'sensitivity': [], 'auc': [], 'f1': [], 'balanced_acc': [], 
            'precision': [], 'specificity': [], 'auprc': []}
    
    sensitivity = sensitivity_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred, average="weighted")
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    
    prec, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auprc = auc(recall, prec)
        
    perf['sensitivity'] = sensitivity
    perf['auc'] = roc_auc
    perf['f1'] = f1
    perf['balanced_acc'] = balanced_acc
    perf['precision'] = precision
    perf['specificity'] = specificity
    perf['auprc'] = auprc
    
    return perf

def generate_weak_labels(X, y, train_index, test_index):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    model = XGBClassifier(use_label_encoder=False, random_state=10, eval_metric='merror',
                          n_estimators=230, eta=0.11, max_depth=16, subsample=0.8,
                          gamma=0.64, colsample_bytree=0.5)
    model = model.fit(X_train, y_train, sample_weight=classes_weights)

    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    return y_pred_proba, y_pred

def prep_phq_decline_data(input_df):
    label_df = input_df[input_df.ref_label][['user_id', 'user_mth', 'phq9_cat_start', 'phq9_cat_end']].copy()
    label_df.rename(columns={'phq9_cat_start': 'cat_m0', 'phq9_cat_end': 'cat_m3'}, inplace=True)
    label_df[PROBA_COLS_M1 + ['cat_m1'] + PROBA_COLS_M2 + ['cat_m2']] = np.nan
    label_df['decl'] = label_df['cat_m3'] > label_df['cat_m0']

    sub_df = input_df[['user_id', 'user_mth'] + PROBA_COLS + ['ref_label', 'phq9_cat_end']].copy()

    for u_id, m_id in list(zip(label_df.user_id, label_df.user_mth)):
        rows = sub_df[(sub_df.user_id == u_id) & (sub_df.user_mth < m_id) & (sub_df.user_mth > m_id - 3)]
        idx = str(u_id) + '_' + str(m_id)
        if len(rows) > 0:
            if len(rows[rows.user_mth == m_id - 2]) == 1:
                label_df.loc[idx, PROBA_COLS_M1 + ['cat_m1']] = rows[
                    rows.user_mth == m_id - 2][PROBA_COLS + ['phq9_cat_end']].iloc[0].values
            if len(rows[rows.user_mth == m_id - 1]) == 1:
                label_df.loc[idx, PROBA_COLS_M2 + ['cat_m2']] = rows[
                    rows.user_mth == m_id - 1][PROBA_COLS + ['phq9_cat_end']].iloc[0].values
    
    label_df = label_df[label_df.columns[2:]]
    
    decline_df = input_df.merge(label_df, left_index=True, right_index=True)
    
    selected_columns = np.concatenate([
        np.array([x for x in decline_df.columns if x.startswith('sleep_') or x.startswith('steps_') or \
                  (x in SCREENER_COLS) or x.startswith('life_') or x.startswith('med_') or x.startswith('nonmed_')]),
        np.array(PROBA_COLS_M1 + PROBA_COLS_M2 + PREV_CAT_COLS)])
    
    return decline_df[selected_columns], decline_df['decl']

def split_train_test(X, y, seed, test_size=0.2):
    sss = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=seed)
    train_index, test_index = next(sss.split(X, y))

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    return X_train, X_test, y_train, y_test
    

def get_feature_selector(X, y, n_features, seed):
    selector = SequentialFeatureSelector(
        estimator=LGBMClassifier(boosting_type='dart', random_state=seed, metric='auc'),
        n_features_to_select=n_features, scoring='roc_auc', cv=5)
    selector = selector.fit(X, y,)
    
    return selector

def fit_lgbm(X_train, y_train, seed):
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    model = LGBMClassifier(boosting_type='dart', random_state=seed, metric='auc')
    model = model.fit(X_train, y_train, sample_weight=classes_weights)
    
    return model

def predict_lgbm(X_test, model):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    return y_pred_proba, y_pred