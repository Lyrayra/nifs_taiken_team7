# Forward Model for Spectrum Analysis

このプログラムは、観測された発光スペクトルに対して、理論的な速度分布（ドップラー広がり）と装置関数（Instrument Function）の畳み込みによる順問題モデルを計算・可視化するためのPythonスクリプト群です。

主に、プラズマやガスの分光観測データに対して、バルク速度や熱速度によるドップラー広がりが観測スペクトルにどう影響するかをモデル化し、実測データと比較するために使用します。

## ファイル構成

- `forward_model.py`: メインスクリプトです。理論的なスペクトル $E(\lambda)$ と装置関数 $I(\lambda)$ を畳み込み、モデルスペクトル $D(\lambda) = E(\lambda) * I(\lambda) + bg$ を計算・描画します。
- `sokudobunpu.py`: ガウス型の速度分布関数を計算し、ドップラーシフトを用いて波長分布 $E(\lambda)$ に変換する処理を含みます。また、実測データ（.txtファイル）から平均スペクトルを読み込む機能も備えています。
- `ilambda_builder.py`: 実測データから装置関数 $I(\lambda)$ を抽出・生成するスクリプトです。特定のピークを抽出し、バックグラウンド除去、面積の正規化、および中心化を行います。

## 依存ライブラリ

実行には以下のPythonライブラリが必要です。

- `numpy`
- `matplotlib`
- `scipy`

（インストール例: `pip install numpy matplotlib scipy`）

## 使用方法

### モデルスペクトルの計算と可視化

`forward_model.py` を実行することで、順問題モデルを計算し、実測データとの比較グラフを描画します。

```bash
python forward_model.py
```

実行すると以下の内容を含むプロットが表示されます。
- **Measured $M(\lambda)$**: 実測された平均スペクトル
- **Theory $E(\lambda)$**: 装置広がりを含まない理論的な発光スペクトル（ガウス分布）
- **Forward Model $D(\lambda)$**: 理論スペクトルと装置関数の畳み込み結果
- **Ch 1 - 12**: 各チャンネルの測定データ（スケーリング済み）

### パラメータの調整

`forward_model.py` 内の `plot_forward_model()` 関数にある以下のパラメータを変更することで、モデルを調整できます。

- `v0`: バルク速度 (m/s)
- `dV`: 熱速度の広がり (m/s)
- `A`: 振幅（カウントのスケール）
- `bg`: バックグラウンド値

### 装置関数 $I(\lambda)$ の単独確認

装置関数が正しく抽出できているか確認する場合は、`ilambda_builder.py` を実行します。

```bash
python ilambda_builder.py
```

抽出された装置関数のプロットが表示され、`I_lambda.npz` としてファイルに保存されます。

### 速度分布・波長分布の単独確認

実測スペクトルと理論的な波長分布 $E(\lambda)$ の比較のみを行いたい場合は、`sokudobunpu.py` を実行します。

```bash
python sokudobunpu.py
```

## データの形式について

読み込み対象のデータはテキストファイル（`.txt`）であり、内部に `[data]` セクションまたは `[Data]` セクションを持ち、カンマ区切りでフレーム、X座標、Y座標、カウント値などが記録されている形式（`sokudobunpu.py` および `ilambda_builder.py` の場合）や、各列がチャンネルに対応する形式（`forward_model.py` の `load_dat_spectrum` の場合）を想定しています。
