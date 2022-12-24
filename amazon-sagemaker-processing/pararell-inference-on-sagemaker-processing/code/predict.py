
import argparse
import logging
import os
import sys
import glob

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--code-path', type=str,
                        default='/opt/ml/processing/code', metavar='N',
                        help='code path')
    parser.add_argument('--input-data-path', type=str,
                        default='/opt/ml/processing/input/data', metavar='N',
                        help='input data path')
    parser.add_argument('--output-data-path', type=str,
                        default='/opt/ml/processing/output/data', metavar='N',
                        help='output data path')
    parser.add_argument('--model-path', type=str,
                        default='/opt/ml/processing/input/model', metavar='N',
                        help='model path')
    parser.add_argument('--job-id', type=str,
                        default='1', metavar='N',
                        help='job id')
    args = parser.parse_args()

    # [ToDo] 入力ファイルに対応するモデルを使って推論を実行するコードを記述する
    input_files = glob.glob(f"{args.input_data_path}/*")
    print('input files:', input_files)
    print('model files:', glob.glob(f"{args.model_path}/*"))
    print('code:', glob.glob(f"{args.code_path}/*"))
    output_data_path = args.output_data_path

    # 出力ファイルは output_data_path におくこと
    log_file = os.path.join(output_data_path, args.job_id + '.txt')
    with open(log_file, 'w') as f:
        f.write('\n'.join(input_files))
