# ファイル概要

ML パイプライン作成用の CodeBuild で使用するファイルです。

- buildspec.yml: ML パイプライン用 CodeBuild の設定ファイル
- pipeline.py: ML パイプライン用 CodeBuild が実行する Step Functions ワークフロー作成スクリプト
- requirements.txt: ML パイプライン用 CodeBuild の環境構築用
- setenv.py: パラメタを環境変数に登録するためのスクリプト。setenv.sh から呼ばれる
- setenv.sh: パラメタを環境変数に登録するためのスクリプト
- ml-pipeline: Step Functions ワークフロー作成用のファイル一式
    - ecr-regist-images.sh: 各種コンテナイメージ作成スクリプト
    - data-preparation: データ準備用コンテナイメージ作成ファイル一式
    - train: モデル学習用コンテナイメージ作成ファイル一式
    - model-evaluation: モデル評価用コンテナイメージ作成ファイル一式
    - inference: 推論用コンテナイメージ作成ファイル一式
