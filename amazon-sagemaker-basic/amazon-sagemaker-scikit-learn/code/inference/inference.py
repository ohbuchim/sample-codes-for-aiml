import argparse
import joblib
import json
import logging
import os
import pandas as pd
import sys
import glob
import shutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--n_neighbors', type=int, default=2, metavar='N',
                        help='N of neighbors')
    args = parser.parse_args()

    # 複数インスタンスを使用した場合に、自分がどのインスタンス（ID）なのかを取得
    # このサンプルでは特に何かに使うわけではなく参考情報
    with open('/opt/ml/config/resourceconfig.json') as f:
        host_settings = json.load(f)
        current_host = host_settings['current_host']
        print('current_host:', current_host)

    # run 実行時に入出力として指定されたパスを取得
    with open('/opt/ml/config/processingjobconfig.json') as f:
        
        # 出力ファイル保存用パスの情報を取得
        processingjobconfig = json.load(f)
        print('processingjobconfig', processingjobconfig)
        output_data_path = ''
        outputs = processingjobconfig['ProcessingOutputConfig']['Outputs']
        for o in outputs:
            if o['OutputName'] == 'result':
                output_data_path = o['S3Output']['LocalPath']

        # 入力ファイル取得用パスの情報を取得
        inputs = processingjobconfig['ProcessingInputs']
        code_path = ''
        input_data_path = ''
        model_path = ''
        for i in inputs:
            if i['InputName'] == 'code':
                code_path = i['S3Input']['LocalPath']
            elif i['InputName'] == 'data':
                input_data_path = i['S3Input']['LocalPath']
            elif i['InputName'] == 'model':
                model_path = i['S3Input']['LocalPath']

    input_files = glob.glob(f"{input_data_path}/*")
    print('input:', str(len(input_files)), input_files[:100])
    print('code:', glob.glob(f"{code_path}/*"))

    # 参考情報としてホストの情報をファイル出力してみる
    log_file = os.path.join(output_data_path, current_host+'.txt')
    with open(log_file, 'w') as f:
        f.write('\n'.join(input_files))

    rawdata_path = os.path.join(input_data_path, 'wine.csv')
    rawdata_df = pd.read_csv(rawdata_path)
    model_download_path = os.path.join(model_path, 'model.tar.gz')
    shutil.unpack_archive(model_download_path, model_path)
    model_file_path = os.path.join(model_path, 'model.joblib')

    knn = joblib.load(model_file_path)
    distances, indices = knn.kneighbors(rawdata_df, args.n_neighbors)

    # 推論結果を CSV にしてから決められたパスに保存
    distances_df = pd.DataFrame(distances)
    distances_df.to_csv(
            os.path.join(output_data_path, 'distances.csv'),
            header=None, index=None)

    indices_df = pd.DataFrame(indices)
    indices_df.to_csv(
            os.path.join(output_data_path, 'indices.csv'),
            header=None, index=None)
