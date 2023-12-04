# solafune_classify_genai

- [生成画像とオリジナル画像の分類](https://solafune.com/ja/competitions/05724228-0ac1-4488-a42f-e945f2117632?menu=about&tab=overview)への参加時の解法を掲載。


## セットアップ
### データの準備

- `./tmp/`配下にzipファイルを格納
- `sh unzip_files.sh`

## 環境構築(Dockerを起動)
```shell
./build.sh
./run.sh
```

## 備考
- 実行環境
  - EC2(AMI:Deep Learning Base GPU AMI, instancetype:d4dn.4xlarge, EBS:150GB)
