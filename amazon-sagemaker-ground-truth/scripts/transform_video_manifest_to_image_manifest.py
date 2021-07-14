#-------------------------------------------------------------------------------------------------------#
# 動画オブジェクト検知ラベリングしたmanifestファイルを
# Rekognition Custom Labels で使える形式の manifest ファイルに変換する。
# このスクリプトが出力した output_video.manifest を S3 にアップロードし、
# Rekognition Custom Labels でデータセットを作成する際に使用する。
# 変換したい output.manifest と SeqLabel.json をこのスクリプトと同じパスにおいて実行する。
# [必須] 引数で画像のサイズをセットする。（$ python modified_json.py -iw 1920 -ih 1080）
# 必要な Ground Truth の出力ファイル格納場所
#    output.manifest 保存場所：ラベリングジョブ出力パス/manifests/output/output.manifest
#    SeqLabel.json 保存場所：ラベリングジョブ出力パス/annotations/consolidated-annotation/output/0/SeqLabel.json
#-------------------------------------------------------------------------------------------------------#
import argparse
import json
import os
import re

def transform_manifest():
    output_file = args.output_file_name
    f_output_file = open(output_file, "w")
    with open(args.output_manifest_name, 'r') as f:
        line = f.read()
        manifest_json = json.loads(line)
        source_ref_dir = os.path.dirname(manifest_json["source-ref"])
        metadata = re.search(r'"([\w-]+-metadata)"', line)
        metadata_str = metadata.group()[1:-1]
        creation_date = manifest_json[metadata_str]['creation-date']
        class_map_json = manifest_json[metadata_str]['class-map']

    with open(args.seq_label_name, 'r') as f:
        line = f.read()
        line_json = json.loads(line)
        for j in line_json["detection-annotations"]:
            annotations_json = j['annotations']
            source_ref = os.path.join(source_ref_dir, j['frame'])
            annotations = []
            class_map = {}
            confidence = []
            for a in annotations_json:
                width = a['width']
                height = a['height']
                top = a['top']
                left = a['left']
                class_id = a['class-id']
                class_map[class_id] = class_map_json[class_id]
                annotations.append({"class_id":int(class_id),"top":top,"left":left,"height":height,"width":width})
                confidence.append({"confidence":0.09})

            info =  {"source-ref":source_ref,"motion-labeling":{"image_size":[{"width":args.image_width,"height":args.image_height,"depth":3}], "annotations":annotations},"motion-labeling-metadata":{"objects":confidence,"class-map":class_map,"type":"groundtruth/object-detection","human-annotated":"yes","creation-date":creation_date,"job-name":"labeling-job/motion-labeling"}}
            info_str = json.dumps(info)
            f_output_file.write(info_str+'\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Transform Ground Truth manifest file')
    parser.add_argument('-om', '--output-manifest-name', type=str, default='output.manifest', help='output.manifest name')
    parser.add_argument('-sl', '--seq-label-name', type=str, default='SeqLabel.json', help='SeqLabel.json name')
    parser.add_argument('-of', '--output-file-name', type=str, default='output_video.manifest', help='output file name')
    parser.add_argument('-iw', '--image-width', type=int, default=0, help='input image width')
    parser.add_argument('-ih', '--image-height', type=int, default=0, help='input image height')
    args = parser.parse_args()
    
    if args.image_width * args.image_height == 0:
        print('[ERROR] --image-width and --image-height are required.')
    else:
        transform_manifest()