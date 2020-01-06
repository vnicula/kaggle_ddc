import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

def read_metadata(json_file):
    meta_df = pd.read_json(json_file)
    return meta_df

def read_submission(csv_file):
    submission_df = pd.read_csv(csv_file)
    return submission_df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', type=str)
    parser.add_argument('--meta', type=str)
    args = parser.parse_args()

    meta_df = read_metadata(args.meta).transpose()
    submission_df = read_submission(args.submission)
    submission_df.rename(columns={'label':'score'}, inplace=True)

    merged_df = pd.merge(submission_df, meta_df, left_on='filename', right_index=True, how='inner')
    assert len(merged_df) == len(submission_df)
    merged_df.fillna({'score':0.5}, inplace=True)
    merged_df['binary_label'] = (merged_df['label'] == 'FAKE').astype(int)
    merged_df.to_csv('evaluated_submission.tsv')

    nll = log_loss(merged_df['binary_label'], merged_df['score'])
    # print(*zip(merged_df['binary_label'], merged_df['score']))

    print('Log loss on submission: {}'.format(nll))

    all_half = np.full_like(merged_df['score'], 0.5)
    print('Log loss on 0.5 submission: {}'.format(log_loss(merged_df['binary_label'], all_half)))

    merged_df[merged_df['label']=='REAL']['score'].hist(bins=100, label='REAL', alpha=0.5)
    merged_df[merged_df['label']=='FAKE']['score'].hist(bins=100, label='FAKE', alpha=0.5)
    plt.legend()
    plt.savefig('score_distribution.png')
