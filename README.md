# Sleep stage Detection

## コンペ内容

- 主催：Nishika
- ページ：[睡眠段階の判定 〜”睡眠の深さを判別しよう”〜](https://www.nishika.com/competitions/sleep/summary)
- 睡眠ポリグラフ（polysomnography: PSG）から睡眠の深さ（睡眠段階）を予測する
- 終了日：2023/01/20

## 結果

- 暫定：4 位（スコア：0.844833）
- 最終：8 位（スコア：0.849650）

## 解法

- 以下の 2 つのネットワークの出力を結合して予測
  - 波形をスペクトログラム画像に変換し EfficientNet で睡眠段階を予測
  - メタデータを入力し 2 層 NN で睡眠段階を予測
- 詳細は [こちら](https://www.nishika.com/competitions/sleep/topics/445)（この解法内の model2 が本リポジトリです）
