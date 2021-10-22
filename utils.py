import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SequentialFeatureSelector, RFE
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             make_scorer, precision_recall_curve, precision_score,
                             roc_auc_score,)

from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight, shuffle

import scipy.stats as stats

from typing import List, Union, Dict, Tuple

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

KAPPA_SCORER = make_scorer(cohen_kappa_score, weights="quadratic")


def kappa_metric(predt, dtrain):
    label = dtrain.get_label()
    y_pred = np.argmax(predt,axis=1)
    return "kappa", cohen_kappa_score(y_pred, label, weights="quadratic")


def split_train_test_set(X, y, train_idx, test_idx):
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    return X_train, X_test, y_train, y_test


def sensitivity_score(y_t: List[int], y_p: List[int]) -> float:
    cm = confusion_matrix(y_t,y_p)
    _, _, fn, tp = cm.ravel()
    return tp/(tp+fn)


def specificity_score(y_t: List[int], y_p: List[int]) -> float:
    cm = confusion_matrix(y_t,y_p)
    tn, fp, _, _ = cm.ravel()
    return tn/(tn+fp)


def get_model_performance(phase: Union["one", "two"],
                          y_true: List[int],
                          y_pred: List[int],
                          y_pred_proba: List[float] = []) -> Dict:
    
    perf = {}
            
    if phase == "one":
        perf['acc'] = accuracy_score(y_true, y_pred)
        perf['adjacent_acc'] = np.sum(np.abs(y_pred - y_true) <= 1) / len(y_true)
        perf['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
        perf['kappa'] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        perf['f1'] = f1_score(y_true, y_pred, average="weighted")

    elif phase == "two":
        perf['sensitivity'] = sensitivity_score(y_true, y_pred)
        perf['auroc'] = roc_auc_score(y_true, y_pred_proba)
        perf['f1'] = f1_score(y_true, y_pred, average="weighted")
        perf['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
        perf['precision'] = precision_score(y_true, y_pred)
        perf['specificity'] = specificity_score(y_true, y_pred)
    
        prec, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        perf['auprc'] = auc(recall, prec)

    return perf


def split_participant_data_random(data: pd.DataFrame,
                                  n_users_test: int = 750,
                                  seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    user_ids_shuffled = shuffle(data.user_id.unique().tolist(), random_state=seed)
    user_ids_train, user_ids_test = user_ids_shuffled[:-n_users_test], user_ids_shuffled[-n_users_test:]
    data_train = data.loc[data.user_id.isin(user_ids_train)]
    data_test = data.loc[data.user_id.isin(user_ids_test)]
    return data_train, data_test


def prep_phq_decline_data(input_df):
    label_df = input_df.loc[input_df.ref_label, ['user_id', 'user_mth', 'phq9_cat_start', 'phq9_cat_end']].copy()
    label_df.rename(columns={'phq9_cat_start': 'cat_m0', 'phq9_cat_end': 'cat_m3'}, inplace=True)
    label_df[PROBA_COLS_M1 + ['cat_m1'] + PROBA_COLS_M2 + ['cat_m2']] = np.nan
    
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


def phase_1_pipeline(X_train: pd.DataFrame,
                     y_train: pd.DataFrame,
                     user_id_groups: pd.DataFrame,
                     params: Dict, 
                     n_iter: int, 
                     cv: int,
                     seed: int) -> RandomizedSearchCV:
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    
    estimator = XGBClassifier(use_label_encoder=False, random_state=seed, eval_metric=KAPPA_SCORER, 
                              disable_default_eval_metric=True,)
    selector = RFE(estimator=estimator, step=2, verbose=2)
    clf = LGBMClassifier(boosting_type='dart', random_state=seed)
    
    pipeline = Pipeline([('selector', selector), ('clf', clf),])
    
    model = RandomizedSearchCV(pipeline, param_distributions=params, n_iter=n_iter, 
                                       scoring=KAPPA_SCORER, cv=GroupKFold(n_splits=cv).split(X_train, y_train, user_id_groups),
                                       verbose=2, random_state=seed,
                                       return_train_score=True,)
    
    model = model.fit(X_train, y_train, clf__sample_weight=classes_weights, clf__eval_metric=kappa_metric)
    
    print(f'Best estimator: {model.best_estimator_}')
    print(f'Best parameters: {model.best_params_}')
    print(f"Selected features: {X_train.columns[model.best_estimator_.named_steps['selector'].get_support()]}")
    
    return model


def phase_2_pipeline(X_train: pd.DataFrame,
                     y_train: pd.DataFrame,
                     n_features_to_select: int,
                     cv: int, 
                     seed: int) -> Pipeline:
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    
    estimator = XGBClassifier(use_label_encoder=False, random_state=seed, eval_metric='auc', disable_default_eval_metric=True,)
    selector = SequentialFeatureSelector(estimator=estimator, n_features_to_select=n_features_to_select, 
                                         cv=cv, scoring='roc_auc')
    clf = LGBMClassifier(boosting_type='dart', metric='auc', random_state=seed, importance_type=importance_type)
    
    pipe = Pipeline([('selector', selector), ('clf', clf),])
        
    pipe.fit(X_train, y_train, clf__sample_weight=classes_weights)
    
    print(f"Selected features: {X_train.columns[pipe.named_steps['selector'].get_support()]}")
    
    return pipe


def phase_1_model(X_train: pd.DataFrame,
                  y_train: pd.DataFrame, 
                  n_estimators: int,
                  max_depth: int,
                  drop_rate: float,
                  seed: int, 
                  importance_type: str ='split') -> LGBMClassifier:
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    clf = LGBMClassifier(boosting_type='dart', n_estimators=n_estimators, max_depth=max_depth, drop_rate=drop_rate, 
                         random_state=seed, importance_type=importance_type)
    clf = clf.fit(X_train, y_train, sample_weight=classes_weights, eval_metric=kappa_metric)
    
    return clf


def phase_2_model(X_train: pd.DataFrame,
                  y_train: pd.DataFrame,
                  seed: int, 
                  importance_type: str = 'split') -> LGBMClassifier:
    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
    clf = LGBMClassifier(boosting_type='dart', metric='auc', random_state=seed, importance_type=importance_type)
    clf = clf.fit(X_train, y_train, sample_weight=classes_weights)
    
    return clf


def final_performance(results_dict: Dict) -> None:
    for k, v in results_dict.items():
        mean = np.mean(v)
        lower, _ = stats.t.interval(
            alpha=0.95, df=len(v)-1, loc=mean, scale=stats.sem(v))
        title = k + ':'
        print(f'  {title:<15}{mean:>6.3f} (± {(mean-lower):.3f})')