import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def read_metadata(json_file):
    meta_df = pd.read_json(json_file)
    return meta_df

def read_submission(csv_file):
    submission_df = pd.read_csv(csv_file)
    return submission_df

def compute_clip_range(truths, preds):
    NUM_POINTS = 201
    best_range = (0.0, 1.0)
    best_loss = log_loss(truths, preds)
    cutpoints = np.linspace(0.0, 1.0, NUM_POINTS)
    for i in range(NUM_POINTS - 1):
        a_min = cutpoints[i]
        for j in range(i+1, NUM_POINTS):
            a_max = cutpoints[j]
            clip_preds = np.clip(preds, a_min, a_max)
            lloss = log_loss(truths, clip_preds)
            if lloss < best_loss:
                best_loss = lloss
                best_range = (a_min, a_max)
    return best_range, best_loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', type=str)
    parser.add_argument('--meta', type=str)
    parser.add_argument('--save', type=str, default=None)
    args = parser.parse_args()

    meta_df = read_metadata(args.meta).transpose()
    submission_df = read_submission(args.submission)
    submission_df.rename(columns={'label':'score'}, inplace=True)
    
    # TODO: check fix for bug in predict_feature_extractor.py which doesn't decode b'str' to str.
    # submission_df['filename'] = submission_df['filename'].apply(lambda x: x[1:])

    merged_df = pd.merge(submission_df, meta_df, left_on='filename', right_index=True, how='inner')
    print(set(submission_df.filename)-set(merged_df.filename))
    #TODO do something with flipped _f samples from tfrecs
    # assert len(merged_df) == len(submission_df)
    merged_df.fillna({'score':0.5}, inplace=True)
    merged_df['binary_label'] = (merged_df['label'] == 'FAKE').astype(int)
    merged_df.to_csv('evaluated_submission.tsv')

    X = merged_df['score'].values.reshape(-1, 1)
    y = merged_df['binary_label'].values

    # print(y, X)
    nll = log_loss(y, X)
    # print(*zip(merged_df['binary_label'], merged_df['score']))
    print('Log loss on submission as is: {}'.format(nll))

    all_half = np.full_like(X, 0.5)
    print('Log loss on all 0.5 submission: {}'.format(log_loss(y, all_half)))

    skf = StratifiedKFold(n_splits=4, random_state=None, shuffle=False)
    lr = LogisticRegression(solver='lbfgs', max_iter=200)
    cv_log_loss_logistic = []
    fold = 1
    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        asis_log_loss = log_loss(y_test, X_test)
        print('Fold {} test log loss as is: {}.'.format(fold, asis_log_loss))

        best_range, best_loss = compute_clip_range(y_train, X_train)
        print('Best fold {} train log loss: {}, with clip range: {}'.format(fold, best_loss, best_range))
        X_test_clipped = np.clip(X_test, best_range[0], best_range[1])
        test_log_loss = log_loss(y_test, X_test_clipped)
        print('Fold {} test clip log loss: {}, with clip range: {}'.format(fold, test_log_loss, best_range))

        # lr.fit(X_train.reshape( -1, 1 ), y_train)
        lr.fit(X_train, y_train)
        # X_test_calibrated = lr.predict_proba(X_test.reshape( -1, 1 ))[:,1]
        X_test_calibrated = lr.predict_proba(X_test)[:,1]
        test_log_loss = log_loss(y_test, X_test_calibrated)
        print('Fold {} test logistic log loss: {}.'.format(fold, test_log_loss))
        cv_log_loss_logistic.append(test_log_loss)
        fold += 1
    
    print('Average logistic calibrated log loss: {}.'.format(np.mean(cv_log_loss_logistic)))

    # Fit on all data and save
    # lr.fit(X.reshape( -1, 1 ), y)
    lr.fit(X, y)
    
    if args.save is not None:
        filename = 'score_calibration.pkl'
        joblib.dump(lr, filename)    
        lr = joblib.load(filename)

    # X_calibrated = lr.predict_proba(X.reshape( -1, 1 ))[:,1]
    X_calibrated = lr.predict_proba(X)[:,1]
    calibrated_log_loss = log_loss(y, X_calibrated)
    print('Platt calibrated log loss: {}.'.format(calibrated_log_loss))

    merged_df['calibrated_score'] = X_calibrated
    
    merged_df[merged_df['label']=='REAL']['score'].hist(bins=100, label='REAL', alpha=0.5)
    merged_df[merged_df['label']=='FAKE']['score'].hist(bins=100, label='FAKE', alpha=0.5)
    plt.legend()
    plt.savefig('score_distribution.png')
    plt.clf()

    merged_df[merged_df['label']=='REAL']['calibrated_score'].hist(bins=100, label='REAL', alpha=0.5)
    merged_df[merged_df['label']=='FAKE']['calibrated_score'].hist(bins=100, label='FAKE', alpha=0.5)
    plt.legend()
    plt.savefig('score_distribution_calibrated.png')
