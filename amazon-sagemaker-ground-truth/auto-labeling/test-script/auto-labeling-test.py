import argparse
import boto3
import io
import json
import math
import mxnet as mx
import numpy as np
import os
from PIL import Image
import random
import sagemaker
from sagemaker.amazon.record_pb2 import Record
from sagemaker.inputs import TrainingInput
from sklearn.model_selection import train_test_split
import tempfile
import time

# 事前に入力画像JPGとラベル画像PNGが記載されたmanifestファイルを作成しておき、
# そのファイル名を以下の設定リストのbase_manifest_fileに記載する

base_manifest_file = ''  # 事前に用意した全データが記載されたmanifestファイル
background_class_id = 0
base_label_image_file = ''
num_manual_target = 0  # モデル再学習に必要な追加データセット数
train_ratio = 0.8  # データセットにおける学習データの割合。残りは検証データ
manual_labeled = []
auto_labeled = []
train_manifest_filename = 'train.manifest'
validation_manifest_filename = 'validation.manifest'
batch_manifest_filename = 'batch.manifest'
confidence_thresh = 0
total_confidence_thresh = 0
project_name = ''
bucket_name = ''
prefix = ''
data_output_s3_path = ''
result_output_dir = ''
input_file_dir = ''
gt_job_name = ''  # base_manifest_fileでジョブ名として使用した名前
args = ''
png_prefix = ''
png_viewer_prefix = ''
base_model_path = ''
timestamp = ''
role = ''
html_header = f"""
<!DOCTYPE html>
<html lang=\"ja\">
    <head>
        <meta charset=\"utf-8\">
        <title>Viewer</title>
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <link rel=\"stylesheet\" href=\"style.css\">
        <!-- [if lt IE 9] -->
        <script src=\"http://html5shiv.googlecode.com/svn/trunk/html5.js\"></script>
        <script src=\"http://css3-mediaqueries-js.googlecode.com/svn/trunk/css3-mediaqueries.js\"></script>
        <!-- [endif] -->
        <script src=\"main.js\"></script>
    </head>

    <style type=\"text/css\">
        .vertical{{
            display: flex;
            flex-direction: column;
        }}
        .horizontal{{
            display: flex;
            flex-direction: row;
        }}
    </style>
"""

sess = sagemaker.Session()
s3_client = boto3.client('s3')
s3 = boto3.resource("s3")
bucket = None
training_container_image = sagemaker.image_uris.retrieve(
                            "semantic-segmentation",
                            sess.boto_region_name)

# log_file = open(os.path.join(result_output_dir, 'log.txt'), 'a')
log_filename = ''


def print_log(*message):
    print(message)
    with open(log_filename, 'a') as log_file:
        print(message, file=log_file)


def get_color_palette(filename):
    '''
    ラベル画像 PNG に適用するパレット情報をサンプル画像から取得
    '''
    base_img = Image.open(filename)
    return base_img.getpalette()


def do_manual_labeling(non_labeled_files, labeled_files, num_manual_target):
    '''
    non_labeled_files に記載された未ラベリングのデータから
    num_manual_target で設定された数だけラベル済みデータ labeled_files に移動
    '''
    if len(non_labeled_files) >= num_manual_target:
        sample = random.sample(non_labeled_files, num_manual_target)
    else:
        sample = non_labeled_files.copy()

    new_labeled_files = labeled_files.copy() + sample
    new_non_labeled_files = non_labeled_files.copy()

    for i in sample:
        new_non_labeled_files.remove(i)

    return new_non_labeled_files, new_labeled_files, len(sample)


def create_and_upload_dataset(labeled_files, loop_counter):
    '''
    学習ジョブの入力データとして使用する manifest ファイルを作成する
    学習データ用 manifest ファイルと検証データ用 manifest ファイルを作成して
    Amazon S3 にアップロード
    '''
    target_size = int(len(labeled_files)*train_ratio)
    train, validation = train_test_split(labeled_files,
                                         train_size=target_size)

    with open(train_manifest_filename, mode='w') as f:
        f.writelines(train)

    with open(validation_manifest_filename, mode='w') as f:
        f.writelines(validation)

    manifest_prefix = os.path.join(prefix, str(loop_counter), 'train')

    train_manifest_s3_path = sess.upload_data(
        path=train_manifest_filename, bucket=bucket_name,
        key_prefix=manifest_prefix
    )
    print_log('train_manifest_s3_path:', train_manifest_s3_path)

    validation_manifest_s3_path = sess.upload_data(
        path=validation_manifest_filename,
        bucket=bucket_name, key_prefix=manifest_prefix
    )
    print_log('validation_manifest_s3_path:', validation_manifest_s3_path)

    return train_manifest_s3_path, validation_manifest_s3_path, len(train)


def train_model(
            train_manifest_s3_path,
            validation_manifest_s3_path,
            num_training_samples,
            train_output_s3_path,
            train_job_name):
    '''
    ラベリング済みデータが記載された manifest ファイルを使ってモデルを学習する
    '''
    print_log('Start Model Training...')
    print_log('num_training_samples:', num_training_samples)
    train_data = TrainingInput(
                        train_manifest_s3_path,
                        distribution='FullyReplicated',
                        record_wrapping='RecordIO',
                        content_type='application/x-recordio',
                        s3_data_type='AugmentedManifestFile',
                        attribute_names=['source-ref',
                                         gt_job_name+'-ref'])

    validation_data = TrainingInput(
                        validation_manifest_s3_path,
                        distribution='FullyReplicated',
                        record_wrapping='RecordIO',
                        content_type='application/x-recordio',
                        s3_data_type='AugmentedManifestFile',
                        attribute_names=['source-ref',
                                         gt_job_name+'-ref'])
    # train_output_s3_path = os.path.join(data_output_s3_path, 'train')

    if len(base_model_path) > 0:
        ss_estimator = sagemaker.estimator.Estimator(
            training_container_image,
            role,
            instance_count=1,
            instance_type=args.train_instance_type,
            volume_size=50,
            max_run=360000,
            input_mode='Pipe',
            output_path=train_output_s3_path,
            model_uri=base_model_path,
            sagemaker_session=sess)
    else:
        ss_estimator = sagemaker.estimator.Estimator(
            training_container_image,
            role,
            instance_count=1,
            instance_type=args.train_instance_type,
            volume_size=50,
            max_run=360000,
            input_mode='Pipe',
            output_path=train_output_s3_path,
            sagemaker_session=sess)

    # Setup hyperparameters
    ss_estimator.set_hyperparameters(
        backbone="resnet-50",  # This is the encoder. Other option is resnet-101
        algorithm="fcn",  # This is the decoder. Other options are 'psp' and 'deeplab'
        use_pretrained_model="True",  # Use the pre-trained model.
        crop_size=240,  # Size of image random crop.
        num_classes=int(args.class_num),  # Pascal has 21 classes. This is a mandatory parameter.
        epochs=10,  # Number of epochs to run.
        learning_rate=0.0001,
        optimizer="rmsprop",  # Other options include 'adam', 'rmsprop', 'nag', 'adagrad'.
        lr_scheduler="poly",  # Other options include 'cosine' and 'step'.
        mini_batch_size=16,  # Setup some mini batch size.
        validation_mini_batch_size=16,
        early_stopping=True,  # Turn on early stopping. If OFF, other early stopping parameters are ignored.
        early_stopping_patience=2,  # Tolerate these many epochs if the mIoU doens't increase.
        early_stopping_min_epochs=10,  # No matter what, run these many number of epochs.
        num_training_samples=num_training_samples,  # This is a mandatory parameter, 1464 in this case.
    )

    ss_estimator.fit(
        inputs={'train': train_data, 'validation': validation_data},
        job_name=train_job_name,
        logs=True)

    return ss_estimator


def create_and_upload_manifest_for_batch(non_labeled_files, loop_counter):
    '''
    バッチ推論（自動ラベリング）で使用する manifest ファイルを作成し
    S3 にアップロードする
    manifest ファイルのフォーマットはこちらを参照
    https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_TransformS3DataSource.html#:~:text=A%20manifest%20might%20look%20like%20this%3A%20s3%3A//bucketname/example.manifest
    '''
    manifest_text = []

    data_s3_prefix = json.loads(non_labeled_files[0])['source-ref']
    manifest_text.append({"prefix": os.path.dirname(data_s3_prefix) + '/'})
    for d in non_labeled_files:
        manifest_text.append(os.path.basename(json.loads(d)['source-ref']))

    with open(batch_manifest_filename, mode='w') as f:

        f.write("[")
        for i, d in enumerate(manifest_text):
            if i == 0:
                f.write("%s,\n" %
                        json.dumps(d).replace("'", '"').replace(' ', ''))
            elif i == (len(manifest_text)-1):
                f.write('"%s"\n' % d)
            else:
                f.write('"%s",\n' % d)
        f.write("]")

    manifest_prefix = os.path.join(
                            prefix, str(loop_counter), 'autolabel')

    manifest_s3_path = sess.upload_data(
        path=batch_manifest_filename, bucket=bucket_name,
        key_prefix=manifest_prefix
    )
    print_log('manifest_s3_path:', manifest_s3_path)

    return manifest_s3_path


def do_auto_labeling(ss_estimator, non_labeled_files, loop_counter):
    manifest_s3_path = create_and_upload_manifest_for_batch(non_labeled_files,
                                                            loop_counter)

    output_s3_path = os.path.join(os.path.dirname(manifest_s3_path), 'batch')
    print_log('output_s3_path', output_s3_path)

    ss_transformer = ss_estimator.transformer(
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        output_path=output_s3_path,
    )

    ss_transformer.transform(data=manifest_s3_path,
                             data_type="ManifestFile")

    return output_s3_path


def get_s3_file_list(bucket_name, prefix):
    file_list = []
    next_token = ''
    while True:
        if next_token == '':
            response = s3_client.list_objects_v2(
                                Bucket=bucket_name,
                                Prefix=prefix)
        else:
            response = s3_client.list_objects_v2(
                                Bucket=bucket_name, Prefix=prefix,
                                ContinuationToken=next_token)
        for content in response['Contents']:
            key = content['Key']
            file_list.append(key)
        if 'NextContinuationToken' in response:
            next_token = response['NextContinuationToken']
        else:
            break

    return file_list


def bytes_to_numpy(data):
    # original code is here
    # https://github.com/aws/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/semantic_segmentation_pascalvoc/semantic_segmentation_pascalvoc.ipynb

    rec = Record()
    # mxnet.recordio can only read from files, not in-memory file-like objects,
    # so we buffer the response stream to a file on disk and then read it back:
    with tempfile.NamedTemporaryFile(mode='w+b') as ftemp:
        ftemp.write(data)
        ftemp.seek(0)
        recordio = mx.recordio.MXRecordIO(ftemp.name, 'r')
        rec.ParseFromString(recordio.read())

    values = list(rec.features["target"].float32_tensor.values)
    shape = list(rec.features["shape"].int32_tensor.values)
    # We 'squeeze' away extra dimensions introduced by the fact that
    # the model can operate on batches of images at a time:
    shape = np.squeeze(shape)
    mask = np.reshape(np.array(values), shape)

    return np.squeeze(mask, axis=0)


def concat_images(img1, img2):
    margin = max(3, int(img1.width * 0.01))
    dst = Image.new('P', (img1.width + img2.width +
                          margin, img1.height), (255))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.width + margin, 0))
    return dst


def get_image_from_s3(img_s3_path):
    img_prefix = img_s3_path[6+len(bucket_name):]
    object = bucket.Object(img_prefix)
    body = object.get()["Body"].read()
    return Image.open(io.BytesIO(body))


def upload_png_from_numpy(array, palette, filename, non_labeled_files):
    '''
    numpy データをインデックス付き PNG 画像に変換して S3 に保存する。
    '''
    im = Image.fromarray(array.astype(np.uint8))
    im.putpalette(palette)
    im.save(filename)
    upload_s3_path = 's3://' + os.path.join(bucket_name, png_prefix, filename)
    s3_client.upload_file(
                    Filename=filename,
                    Bucket=bucket_name,
                    Key=os.path.join(png_prefix, filename),
                    ExtraArgs={"ContentType": "image/png"})
    time.sleep(1)

    # 推論結果ファイルと教師画像を 1枚の PNG に連結し、S3 にアップロード
    # 自動ラベリング結果の目視評価に利用
    for d in non_labeled_files:
        if filename in d:
            gt_png_s3_path = json.loads(d)[gt_job_name+'-ref']
            # gt_png_prefix = gt_png_s3_path[6+len(bucket_name):]
            # gt_object = bucket.Object(gt_png_prefix)
            # body = gt_object.get()["Body"].read()
            # gt_image = Image.open(io.BytesIO(body))
            gt_image = get_image_from_s3(gt_png_s3_path)
            input_img_s3_path = json.loads(d)['source-ref']
            in_image = get_image_from_s3(input_img_s3_path)
            in_image_filename = input_img_s3_path.split('/')[-1]
            in_image.save(in_image_filename)

            concat_image = concat_images(im, gt_image)
            concat_image.putpalette(palette)
            concat_filename = 'v_' + filename
            concat_image.save(concat_filename)
            s3_client.upload_file(
                    Filename=concat_filename,
                    Bucket=bucket_name,
                    Key=os.path.join(png_viewer_prefix, concat_filename),
                    ExtraArgs={"ContentType": "image/png"})
            s3_client.upload_file(
                    Filename=in_image_filename,
                    Bucket=bucket_name,
                    Key=os.path.join(png_viewer_prefix, in_image_filename),
                    ExtraArgs={"ContentType": "image/jpeg"})

    time.sleep(1)
    os.remove(filename)
    os.remove(concat_filename)
    os.remove(in_image_filename)

    return upload_s3_path


def update_file_list_by_autolabel(
        non_labeled_files, labeled_files, input_filename, upload_s3_path):
    '''
    採用された自動ラベリングデータを未ラベリングファイルリストから削除し、
    ラベリング済みファイルリストに追加する。
    自動ラベリングデータは次のイテレーションからモデルの学習に使用される。
    '''
    new_labeled_files = labeled_files.copy()
    new_non_labeled_files = non_labeled_files.copy()

    for d in non_labeled_files:
        if input_filename in json.loads(d)['source-ref']:
            new_non_labeled_files.remove(d)
            old_label_image = json.loads(d)[gt_job_name+'-ref']
            d = d.replace(old_label_image, upload_s3_path)
            new_labeled_files.append(d)

    return new_non_labeled_files, new_labeled_files


def is_valid_label(confidence_matrix):
    '''
    confidence_matrix の画素ごとに最大値を算出し、その中で confidence_thresh より
    大きい値の画素数が total_confidence_thresh より多ければそのラベル画像を採用する
    （自信がある画素数が十分にあるという判断）
    '''
    max_confidence = np.max(confidence_matrix, axis=0)
    max_class = np.argmax(confidence_matrix, axis=0)
    num_background = np.count_nonzero(max_class == background_class_id)
    num_non_background = np.count_nonzero(max_class != background_class_id)
    background_ratio = num_background/(num_background+num_non_background)

    # 99% が背景だったら無効扱いにする
    if background_ratio > 0.99:
        return False, 0

    high_confidence_pix_count = np.count_nonzero(
                                    max_confidence > confidence_thresh)
    height = np.shape(confidence_matrix)[1]
    width = np.shape(confidence_matrix)[2]
    num_pixels = width * height
    high_confidence_ratio = high_confidence_pix_count/num_pixels

    if high_confidence_ratio > total_confidence_thresh:
        return True, high_confidence_ratio
    return False, high_confidence_ratio
    # return np.max(confidence_matrix)


def update_file_list(
            auto_labeled_s3_path,
            non_labeled_files,
            labeled_files,
            color_pallete,
            loop_counter):
    '''
    自動ラベリングの結果を評価してファイルリストを更新する。
    '''
    data_uplpoad_prefix = auto_labeled_s3_path[6+len(bucket_name):] + '/'
    file_list = get_s3_file_list(bucket_name, data_uplpoad_prefix)
    well_labeled_counter = 0
    confidence_list = []
    new_non_labeled_files = non_labeled_files.copy()
    new_labeled_files = labeled_files.copy()

    # file_list は自動ラベリング結果の *.jpg.out の一覧
    for file in file_list:

        png_filename = os.path.basename(file)[:-len('.jpg.out')] + '.png'
        response = s3_client.get_object(Bucket=bucket_name, Key=file)
        prob_matrix = bytes_to_numpy(response['Body'].read())

        # 自動ラベリング結果の採用可否を判定
        label_is_valid, high_conf_ratio = is_valid_label(prob_matrix)

        confidence_list.append([file, label_is_valid, high_conf_ratio])

        # 自動ラベリングの結果を accept する場合
        if label_is_valid:
            label_array = np.argmax(prob_matrix, axis=0)  # PNG の素を生成
            well_labeled_counter += 1

            # 推論結果ファイルを PNG に変換し、S3 にアップロードして labeled_files に追加
            # labeled_files に追加したデータは non_labeled_files から削除
            upload_s3_path = upload_png_from_numpy(
                label_array, color_pallete, png_filename, non_labeled_files)

            # 自動ラベリングの結果をファイルリストに反映
            res = update_file_list_by_autolabel(new_non_labeled_files,
                                                new_labeled_files,
                                                os.path.basename(file[:-4]),
                                                upload_s3_path)
            new_non_labeled_files, new_labeled_files = res

    # このイテレーションの結果のファイルリストをファイルとして保存し、
    # S3 にアップロード
    non_labeled_files_filename = 'non_labeled_files.txt'
    labeled_files_filename = 'labeled_files.txt'

    with open(non_labeled_files_filename, mode='w') as f:
        f.writelines(new_non_labeled_files)
    with open(labeled_files_filename, mode='w') as f:
        f.writelines(new_labeled_files)

    file_list_prefix = os.path.join(
        prefix, str(loop_counter), 'updated-list')
    non_labeled_files_s3_path = sess.upload_data(
        path=non_labeled_files_filename, bucket=bucket_name,
        key_prefix=file_list_prefix
    )
    print_log('non_labeled_files_s3_path:', non_labeled_files_s3_path)

    labeled_files_s3_path = sess.upload_data(
        path=labeled_files_filename, bucket=bucket_name,
        key_prefix=file_list_prefix
    )
    print_log('labeled_files_s3_path:', labeled_files_s3_path)

    # 自動ラベリング結果の評価値と評価結果をファイルに保存し、S3 にアップロード
    confidence_filename = 'confidence_list.txt'
    confidence_list.append(['high confidence labels',
                            well_labeled_counter])
    num_low_confidence_labels = len(file_list)-well_labeled_counter
    confidence_list.append(['low confidence labels',
                            num_low_confidence_labels])

    with open(confidence_filename, mode='w') as f:
        for d in confidence_list:
            f.write('%s\n' % str(d))

    prob_list_prefix = os.path.join(prefix, str(loop_counter), 'autolabel')
    upload_s3_path = sess.upload_data(
        path=confidence_filename, bucket=bucket_name,
        key_prefix=prob_list_prefix
    )
    print_log('Confidence file was uploaded to', upload_s3_path)

    return new_non_labeled_files, new_labeled_files, well_labeled_counter


def do_test():

    # あらかじめ用意した入力画像とラベル画像の S3 パスが記載された
    # manifest ファイルを読み込む
    with open(base_manifest_file, "r") as f:
        lines = f.readlines()

    non_labeled_files = lines
    labeled_files = []
    loop_counter = 0
    num_auto_labeled = 0

    # 自動ラベリングの結果をインデックス付き PNG として保存するために
    # 既存の PNG からパレット情報を取得
    color_pallete = get_color_palette(base_label_image_file)

    while True:
        print_log(loop_counter, '-----------------------')
        print_log('non labeled files: ', len(non_labeled_files))

        '''
        前のイテレーションで自動ラベリングしたデータ数が
        num_manual_target より少ない場合、不足分 (new_target) を手動ラベリングで追加する。
        用意してあるラベルデータから new_target 個のデータをピックアップして
        labeled_fileに移動する。
        ピックアップしたデータは non_labeled_file から削除する。
        更新したnon_labeled_files, labeled_filesを返す。
        手動ラベルした想定のファイル数num_manual_labeledを返す。
        最後のイテレーション以外は基本的にnew_targetと同じ値になる。
        '''
        if num_auto_labeled < num_manual_target:
            new_target = num_manual_target - num_auto_labeled

            res = do_manual_labeling(non_labeled_files,
                                     labeled_files, new_target)
            non_labeled_files, labeled_files, num_manual_labeled = res
        else:
            num_manual_labeled = 0

        # マニュアルラベリングしたデータ数を記録
        manual_labeled.append(num_manual_labeled)

        # ラベリングが必要なデータがなくなったら検証終了
        if len(non_labeled_files) == 0:
            print_log('All files are labeled.')
            break

        # ラベリング済みデータを使ってモデルを学習するための manifest ファイルを作成。
        # manifest には入力画像とそれに対応するラベル画像の S3 パスが記載されている。
        res = create_and_upload_dataset(labeled_files, loop_counter)
        train_s3_path, validation_s3_path, num_training_samples = res

        # すべてのラベル付きデータを使ってモデルを学習
        train_output_path = os.path.join(data_output_s3_path,
                                         str(loop_counter), 'train')
        train_job_name = project_name + '-' + timestamp + '-' + str(loop_counter)
        estimator = train_model(
                        train_s3_path,
                        validation_s3_path,
                        num_training_samples,
                        train_output_path,
                        train_job_name)

        # 学習したモデルを使って non_labeled_files のデータに対してバッチ推論
        # バッチ推論＝自動ラベリング
        output_path = do_auto_labeling(
                            estimator,
                            non_labeled_files,
                            loop_counter)

        '''
        バッチ推論の結果を評価し、条件を満たしたデータのみを labeled_files に移動
        labeled_files に移動したデータは non_labeled_files から削除
        最新の non_labeled_files, labeled_files と、
        確信度が高かった自動ラベルファイル数を返す
        '''
        res = update_file_list(
                    output_path,
                    non_labeled_files,
                    labeled_files,
                    color_pallete,
                    loop_counter)
        non_labeled_files, labeled_files, num_auto_labeled = res

        print_log('num_auto_labeled:', num_auto_labeled)

        # 自動ラベリングされたデータ数を記録
        auto_labeled.append(num_auto_labeled)

        loop_counter += 1

    return loop_counter


def add_html_string(body_str, in_name, png_name):
    body_str += f'            {in_name}<br>\n'
    body_str += '            <div class="horizontal">\n'
    body_str += f'                <img src="{in_name}" width="30%" alt=""> \n'
    body_str += f'                <img src="{png_name}" width="60%" alt=""><br>\n'
    body_str += '            </div>\n'
    return body_str


def create_png_view_pages(viewer_images_per_page):
    # 自動ラベリング画像と教師画像を連結した画像のファイルパスを取得
    file_list = get_s3_file_list(bucket_name, png_viewer_prefix)
    png_file_list = [i for i in file_list if '.png' in i]
    jpg_file_list = [i for i in file_list if '.jpg' in i]
    viewer_images_per_page = int(viewer_images_per_page)
    page_num = math.ceil(len(png_file_list)/viewer_images_per_page)
    body_str_head = """
        <body>
            <div class="vertical">
                左：入力画像、中央：自動ラベリング画像、右：教師画像<br>

    """
    body_str = body_str_head
    if page_num <= 1:
        for idx, i in enumerate(png_file_list):
            png_file_name = os.path.basename(i)
            jpg_file_name = os.path.basename(jpg_file_list[idx])
            body_str = add_html_string(body_str, jpg_file_name, png_file_name)

        body_str += """
                </div>
            </body>
        </html>"""
        html_file = 'index.html'
        with open(html_file, 'w') as f:
            f.write(html_header+body_str)
        s3_client.upload_file(
                Filename=html_file,
                Bucket=bucket_name,
                Key=os.path.join(png_viewer_prefix, html_file),
                ExtraArgs={"ContentType": "text/html"})
    else:
        page_list = """
                    <ul>
        """
        for p in range(page_num):
            body_str = body_str_head
            html_file = 'page_' + str(p) + '.html'
            page_list += f"""
                            <li><a href="{html_file}">{html_file}</a>"""
            with open(html_file, 'w') as f:
                start = p*viewer_images_per_page
                end = min(start + viewer_images_per_page, len(png_file_list))
                for i in range(start, end):
                    png_file_name = os.path.basename(png_file_list[i])
                    jpg_file_name = os.path.basename(jpg_file_list[i])
                    body_str = add_html_string(body_str,
                                               jpg_file_name, png_file_name)
                body_str += """
                        </div>
                    </body>
                </html>"""
                f.write(html_header+body_str)
            s3_client.upload_file(
                Filename=html_file,
                Bucket=bucket_name,
                Key=os.path.join(png_viewer_prefix, html_file),
                ExtraArgs={"ContentType": "text/html"})
        html_file = 'index.html'
        page_list += """
                            </ul>
                        </div>
                    </body>
                </html>"""
        with open(html_file, 'w') as f:
            f.write(html_header+page_list)
        s3_client.upload_file(
                Filename=html_file,
                Bucket=bucket_name,
                Key=os.path.join(png_viewer_prefix, html_file),
                ExtraArgs={"ContentType": "text/html"})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-png", type=str, default='')
    parser.add_argument("--manifest-file", type=str, default='')
    parser.add_argument("--role-arn", type=str, default='')
    parser.add_argument("--bucket-name", type=str, default='')
    parser.add_argument("--data-output-path", type=str, default='')
    parser.add_argument("--project-name", type=str, default='')
    parser.add_argument("--gt-job-name", type=str, default='')
    parser.add_argument("--timestamp", type=str, default='')
    parser.add_argument("--viewer-images-per-page", type=str, default='')
    parser.add_argument("--class-num", type=str, default='')
    parser.add_argument("--background-class-id", type=str, default='0')
    parser.add_argument("--confidence-thresh", type=str, default='0.98')
    parser.add_argument("--total-confidence-thresh", type=str, default='0.9')
    parser.add_argument("--num-manual-target", type=str, default='80')
    parser.add_argument("--base-model-path", type=str, default='')
    parser.add_argument("--train_ratio", type=str, default='0.8')
    parser.add_argument("--train-instance-type", type=str,
                        default='ml.p3.2xlarge')
    parser.add_argument("--input-dir", type=str,
                        default='/opt/ml/processing/input')
    parser.add_argument("--output-dir", type=str,
                        default='/opt/ml/processing/output')
    args, _ = parser.parse_known_args()

    input_file_dir = args.input_dir
    base_label_image_file = os.path.join(input_file_dir, args.sample_png)
    result_output_dir = args.output_dir
    base_manifest_file = os.path.join(input_file_dir, args.manifest_file)
    role = args.role_arn
    data_output_s3_path = args.data_output_path
    bucket_name = args.bucket_name
    prefix = data_output_s3_path[6+len(bucket_name):]
    project_name = args.project_name
    confidence_thresh = float(args.confidence_thresh)
    total_confidence_thresh = float(args.total_confidence_thresh)
    gt_job_name = args.gt_job_name
    num_manual_target = int(args.num_manual_target)  # モデル再学習に必要な追加データセット数
    train_ratio = float(args.train_ratio)  # データセットにおける学習データの割合。残りは検証データ
    timestamp = args.timestamp
    png_prefix = os.path.join(prefix, 'png')
    png_viewer_prefix = os.path.join(prefix, 'png_view')
    base_model_path = args.base_model_path
    background_class_id = int(args.background_class_id)
    log_filename = os.path.join(result_output_dir, 'log.txt')
    bucket = s3.Bucket(bucket_name)

    loop_counter = do_test()
    create_png_view_pages(args.viewer_images_per_page)

    total_manual_labeled = sum(manual_labeled)
    total_auto_labeled = sum(auto_labeled)
    total_labels = total_manual_labeled + total_auto_labeled
    auto_labeled_ratio = round(total_auto_labeled*100/total_labels, 2)
    png_s3_path = os.path.join(f's3://{bucket_name}', png_prefix)

    report = []
    report.append(['Number of manual labeled files: ', manual_labeled])
    report.append(['Number of Auto labeled files: ', auto_labeled])
    report.append(['Number of iteration: ', loop_counter])
    report.append(['Auto labeled PNG is here: ', png_s3_path])
    report.append(['Total manual labeled files: ', total_manual_labeled])
    report.append(['Total auto labeled files: ', total_auto_labeled])
    report.append(['Auto labeled ratio: ', auto_labeled_ratio])
    print('==== Report ====')
    print(*report, sep='\n')

    report_filename = 'report.txt'
    with open(os.path.join(result_output_dir, report_filename), mode='w') as f:
        for d in report:
            f.write('%s\n' % str(d))
