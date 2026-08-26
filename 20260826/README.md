# NIFS CXS スペクトル解析

LHD荷電交換分光（CXS）の観測データに対して、装置関数 $I(\lambda)$ と理論発光スペクトル $E(\lambda)$ の畳み込みモデルを用いた順問題・逆問題解析を行う Python スクリプト群です。

## 解析フロー

```
a260825_img.txt (Neランプ)
        │
        ├── ilambda_builder.py ──→ 装置関数 I(λ) の抽出（チャンネルごと）
        │                          波長軸のキャリブレーション（サブピクセル精度）
        │
lhdcxs9a_img_sig@... (プラズマ観測)
        │
        ├── forward_model.py ──→ 順問題 D(λ) = E(λ)*I(λ) + bg の計算・可視化
        │
        ├── gyaku.py ──→ 逆問題 curve_fit による v0, dV の抽出（全12ch）
        │                  → dv_r_profile.csv の出力
        │
        └── threedv.py ──→ dV プロファイルの比較プロット (Data a, b, c)
```

## ファイル構成

### コアモジュール

| ファイル | 役割 |
|---|---|
| `ilambda_builder.py` | Ne ランプデータから装置関数 $I(\lambda)$ を抽出。チャンネルごとにサブピクセル精度でピーク位置を検出し、波長キャリブレーションを行う |
| `sokudobunpu.py` | ガウス型速度分布関数の計算、ドップラーシフトによる波長分布 $E(\lambda)$ への変換 |
| `forward_model.py` | 順問題モデル $D(\lambda) = E(\lambda) * I(\lambda) + bg$ の計算・描画、観測データの読み込み |
| `gyaku.py` | 逆問題（`scipy.optimize.curve_fit`）を用いて全12チャンネルのバルク速度 $v_0$ と熱速度幅 $\Delta V$ を抽出 |
| `reff.py` | tsmesh ファイルから実効マイナー半径 $r_{eff}$ を補間計算し、チャンネルと主半径 $R$ の対応を取得 |

### プロット・確認スクリプト

| ファイル | 役割 |
|---|---|
| `threedv.py` | `dv_r_profile.csv`（Data a）、手入力データ（Data b）、`data_n.csv`（Data c）の3種類の $\Delta V$ プロファイルを比較プロット |
| `plot_inst.py` | 装置関数 $I(\lambda)$ の確認用プロット |

### データファイル

| ファイル | 用途 | dV計算に必要 |
|---|---|---|
| `a260825_img.txt` | Ne ランプキャリブレーションデータ（波長較正 + 装置関数抽出） | ✅ |
| `lhdcxs9a_img_sig@189129_t4.44s.txt` | プラズマ観測データ（12ch × 128px） | ✅ |
| `lhdcxs9a_prep@189129.dat` | チャンネル→主半径 $R$ の対応テーブル | ✅ |
| `tsmesh@189129_t4.44s_phi18deg.dat` | $(Z, R)$ 空間での $r_{eff}$ メッシュデータ | ✅ |
| `data_n.csv` | 比較用 $\Delta V$ プロファイル（Data c） | ❌（比較用） |
| `dv_r_profile.csv` | `gyaku.py` の出力（`gyaku.py` 実行で再生成される） | ❌（出力） |
| `b260825_img.txt` | Ne ランプ予備データ | ❌ |
| `aim260825_img.txt` | 画像データ（14MB、未使用） | ❌ |
| `bim260825_img.txt` | 画像データ（14MB、未使用） | ❌ |
| `I_lambda.npz` | `ilambda_builder.py` 単体実行時の保存ファイル | ❌ |
| `opt_disp_result.png` | 分散最適化の出力画像 | ❌ |

## 依存関係

```
gyaku.py  ← dV計算のメインエントリポイント
  ├── ilambda_builder.py  ← 装置関数 I(λ) 抽出 + チャンネルごと波長キャリブレーション
  ├── sokudobunpu.py      ← 理論発光スペクトル E(λ) の生成
  ├── forward_model.py    ← load_dat_spectrum() でプラズマ観測データ読み込み
  │     ├── ilambda_builder.py (波長軸取得)
  │     └── sokudobunpu.py
  └── reff.py             ← チャンネル→主半径 R の変換
        └── tsmesh / prep ファイル
```

## 波長キャリブレーション

Ne ランプデータ（`a260825_img.txt`）の2つの既知輝線を使用：

| 輝線 | 波長 (nm) | 概略ピクセル位置 |
|---|---|---|
| Ne I | 529.81891 | ~34 |
| Ne I | 530.47573 | ~12 |

- **チャンネルごとに独立**してキャリブレーションを実施
- `parabolic_subpixel_peak()` によるサブピクセル精度のピーク検出
- 2点間の線形補間で各ピクセルの波長を決定
- 1 px あたり約 0.0299 nm（波長は降順方向）

## 物理パラメータ

- 光速: $c = 299{,}792{,}458.0$ m/s
- 対象輝線（C VI）: $\lambda_0 = 529.81891$ nm
- 装置関数の抽出幅: $\pm 0.35$ nm

## 依存ライブラリ

```
numpy
matplotlib
scipy
```

## 使用方法

### 1. 逆問題の実行（メイン解析）

```bash
python gyaku.py
```

12チャンネル分のフィッティング結果が表示され、`dv_r_profile.csv` が出力されます。

### 2. dV プロファイルの比較

```bash
python threedv.py
```

Data a（本解析）、Data b（画像書き起こし）、Data c（`data_n.csv`）の3者を1つのグラフに表示します。

### 3. 順問題モデルの確認

```bash
python forward_model.py
```

理論スペクトルと装置関数の畳み込み結果を可視化します。

### 4. 装置関数の確認

```bash
python plot_inst.py       # 装置関数の確認
python ilambda_builder.py # 抽出結果の保存（I_lambda.npz）
```

## データ形式

### 画像データ（`.txt`）

`#[data]` セクション以降に CSV 形式で記録：

```
フレーム番号, X座標(ピクセル), Y座標(チャンネル), カウント値
0,0,0,2102.00
0,0,1,2212.00
...
```

- X: 0–127（波長方向、128ピクセル）
- Y: 0–11（空間方向、12チャンネル）

### 観測データ（`lhdcxs9a_img_sig@...`）

`[Data]` セクション以降にカンマ区切りで 13 列：

```
波長, Ch1, Ch2, ..., Ch12
```
