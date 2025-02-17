# sato-lab

## 環境構築
1. environment.ymlをconda環境で読み込み
'''
conda env create -f environment.yml
'''

2. modules/esm/のファイルをfair-esmのライブラリと置換する
'''
scp -r ./GA/modules/esm usr/anaconda3/envs/GARNA/lib/python3.9/site-packages/
'''

## 実行方法
python GARNA.py