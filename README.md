# sample-codes-for-aiml
こちらでは AWS を使った AI/ML のサンプルコードを公開しています。

- amazon-rekognition-image-analysis
  - AWS SDK for Python を使って Amazon Rekognition で画像分析します
- autogluon-tabular-customer-churn
  - [AutoGluon-Tabular](https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html) を使って顧客離反予測します
- lightgbm-byoc
  - Amazon SageMaker でカスタムコンテナで LightGBM を使うサンプルです。学習とリアルタイム推論は [公式のサンプルノートブック](https://github.com/aws-samples/amazon-sagemaker-script-mode/blob/master/lightgbm-byo/lightgbm-byo.ipynb) で方法が紹介されていますので、こちらのサンプルでは Transform API と Amazon SageMaker Processing それぞれを使ってバッチ推論を行う方法をご紹介します
- similar-image-search
  - MXNet の学習済みモデルと Amazon Elasticsearch Service を使って類似画像検索します
- tensorflow2-mnist-byom
  - Amazon SageMaker の機能を使って Tensorflow2 のモデルを学習、デプロイします