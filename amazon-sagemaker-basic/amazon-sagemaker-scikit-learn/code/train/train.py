import argparse
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--metric', type=str, default='minkowski')
    parser.add_argument('--radius', type=float, default=0.4)
    
    # 学習済みモデルは SageMaker の指定のパスに保存する。指定のパスは環境変数から取得する。
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    # 学習データは estimator.fit() 実行時のチャネル名に連動したパスにダウンロードされる。ダウンロードパスは環境変数から取得する。
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    train_csv = os.path.join(args.train, 'wine.csv')

    train_data = pd.read_csv(train_csv, header=None)
    
    n_neighbors = args.n_neighbors
    metric = args.metric
    radius = args.radius

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, radius=radius)
    model_nn = nn.fit(train_data)
    print('Training has finished!')

    # 指定されたパスに学習済みモデルを保存
    # model_dir に保存されたファイル一式は SageMaker によって自動的に S3 にアップロードされる
    joblib.dump(model_nn, os.path.join(args.model_dir, "model.joblib"))