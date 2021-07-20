import pandas as pd
import numpy as np
import scipy.stats as stats

import json
import logging
import argparse

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_sel', dest='run_feature_selection', default=False,
                        help='Rerun feature selection')
    parser.add_argument('--n_features', dest='n_features', default=14, type=int,
                        help='If rerunning feature selection, number of features to select')
    parser.add_argument('--source_data', dest='data_path', help='Source data path')
    parser.add_argument('--source_sel_features', dest='sel_features_path',
                        default='results/selected_features.json', 
                        help='Path to pre-selected features')
    parser.add_argument('--dest_sel_features', dest='dest_sel_features_path', default='',
                        help='Path to save selected features, as a JSON file. If undefined, selected features are not saved')
    args = parser.parse_args()
    
    return args

def main(run_feature_selection, n_features, data_path, sel_features_path, dest_sel_features_path):
    logging.basicConfig(level=logging.INFO)
    np.random.seed(0)
    
    logging.info('Loading data...')

    data = pd.read_parquet(data_path)
    data['user_id'] = [x.rsplit('_')[0] for x in data.index]
    data['user_mth'] = [int(x.rsplit('_')[1]) for x in data.index]
    data['ref_label'] = ~data.phq9_cat_end.isna()
    
    logging.info(f'Number of unique ids: {data.user_id.nunique()}')

    logging.info('Generating weak labels...')
    X = data[WEAK_LABEL_INPUT_COLS]
    y = data['phq9_cat_end']

    ref_label_indices = data[data.ref_label].index
    weak_label_indices = data[~data.ref_label].index

    weak_label_pred_proba, weak_label_pred = generate_weak_labels(X, y, ref_label_indices, weak_label_indices)

    data.loc[weak_label_indices, PROBA_COLS] = weak_label_pred_proba
    data.loc[weak_label_indices, 'phq9_cat_end'] = weak_label_pred

    logging.info('Generating phase 2 input data...')

    X_, y_ = prep_phq_decline_data(data)

    metrics_results = {'sensitivity': [], 'auc': [], 'f1': [], 'balanced_acc': [],
                       'precision': [], 'specificity': [], 'auprc': []}

    if run_feature_selection:
        logging.info('Running feature selection...')

        random_seeds = np.random.randint(0, 10000, 5)
        model_params = {}
        i = 0

        for seed in random_seeds:
            X_train, X_test, y_train, y_test = split_train_test(X_, y_, seed)
            selector = get_feature_selector(X_train, y_train, n_features, seed)

            model_params[i] = {'seed': int(seed),
                               'selected_columns': list(X_train.columns[selector.get_support()])}        
            logging.info(model_params[i]['selected_columns'])

            i += 1

            model = fit_lgbm(selector.transform(X_train), y_train, seed)
            y_pred_proba, y_pred = predict_lgbm(selector.transform(X_test), model)
            seed_results = get_model_performance(y_test, y_pred, y_pred_proba)

            for k, v in seed_results.items():
                metrics_results[k].append(v)

        if dest_sel_features_path:
            logging.info('Saving selected features...')
            result = json.dumps(model_params)

            with open(dest_sel_features_path, "w") as f:
                f.write(result)

    else:
        logging.info('Using predefined selected features...')

        with open(sel_features_path) as f:
            model_params = json.load(f)

        for v in model_params.values():
            X_train, X_test, y_train, y_test = split_train_test(X_[v['selected_columns']], y_, v['seed'])

            model = fit_lgbm(X_train, y_train, v['seed'])
            y_pred_proba, y_pred = predict_lgbm(X_test, model)
            seed_results = get_model_performance(y_test, y_pred, y_pred_proba)

            for k, v in seed_results.items():
                metrics_results[k].append(v)

    logging.info('Final performance:')
    for k, v in metrics_results.items():
        mean = np.mean(v)
        lower, _ = stats.t.interval(alpha=0.95, df=len(v)-1, loc=mean, scale=stats.sem(v))
        title = k + ':'
        logging.info(f'  {title:<15}{mean:>6.3f} (± {(mean-lower):.3f})')

    logging.info('Process completed.')


if __name__ == '__main__':
    args = parse_args()
    main(args.run_feature_selection, args.n_features, args.data_path, 
         args.sel_features_path, args.dest_sel_features_path)