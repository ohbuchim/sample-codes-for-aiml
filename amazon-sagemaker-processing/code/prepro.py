import argparse
import json
import logging
import os
from PIL import Image
import sys
import glob
import time

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
    parser.add_argument('--scale', type=float, default=1.0, metavar='N',
                        help='scale')
    args = parser.parse_args()

    with open('/opt/ml/config/resourceconfig.json') as f:
        host_settings = json.load(f)
        current_host = host_settings['current_host']
        print('current_host:', current_host)

    with open('/opt/ml/config/processingjobconfig.json') as f:
        processingjobconfig = json.load(f)
        print('processingjobconfig', processingjobconfig)
        output_data_path = ''
        outputs = processingjobconfig['ProcessingOutputConfig']['Outputs']
        for o in outputs:
            if o['OutputName'] == 'result':
                output_data_path = o['S3Output']['LocalPath']

        inputs = processingjobconfig['ProcessingInputs']
        code_path = ''
        input_data_path = ''
        for i in inputs:
            if i['InputName'] == 'code':
                code_path = i['S3Input']['LocalPath']
            elif i['InputName'] == 'data':
                input_data_path = i['S3Input']['LocalPath']

    input_files = glob.glob(f"{args.input_data_path}/*/**")
    print('input:', str(len(input_files)), input_files[:100])
    print('code:', glob.glob(f"{args.code_path}/*"))

    log_file = os.path.join(output_data_path, current_host+'.txt')
    with open(log_file, 'w') as f:
        f.write('\n'.join(input_files))

    scale = float(args.scale)
    for f in input_files:
        img = Image.open(f)
        img_resize = img.resize((int(img.width * scale),
                                 int(img.height * scale)))
        basename = os.path.basename(f)
        img_resize.save(os.path.join(output_data_path, basename))
        time.sleep(1)
