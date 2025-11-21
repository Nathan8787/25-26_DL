<div align="center">
  <span style="color:#111827; font-size:40px; font-weight:700;">
    HW3 · Shakespeare RNN Report
  </span><br>
  <span style="color:#4b5563; font-size:26px; font-weight:600;">
    Nathan (a141251) · 2025-11-25
  </span>
</div>

---

## 0. Overview & Setup
這次的作業是RNN並搭配 Shakespeare corpus (train/valid 依提供檔)的資料。我的框架是：PyTorch + GPU AMP 混合精度，輸出統一在 `plots/`, `outputs/`, `checkpoints/`。亂數種子 113024510，**並且每個 batch 的 hidden state 依作業提示重設為 0。**

重點設定：
- Optimizer: Adam, lr=0.002, clip_grad_norm=5, batch=256
- Epochs: 5（patience=5，保證 5 個 breakpoints）
- Model: RNN / LSTM，hidden size = [64, 128, 256] 與 seq_len = [50, 100, 150]，其中Baseline為 hs=128, seq=100。
- 評估：loss、BPC (=loss/ln2)、acc、err

Observation（整體）：RNN 與 LSTM 都跑 5 epoch；LSTM 在相同設定下收斂更快、指標更好，尤其 hidden size 擴充到 256 時最明顯，以下逐段詳述。

---

## 1. Standard RNN
### 1.1 架構與設定
Embedding → RNN(hidden=128, layers=2, dropout=0.25, batch_first) → Linear → vocab。

### 1.2 學習曲線與指標（RNN vs LSTM 對照）
- ![RNN vs LSTM Loss](plots/comparison_rnn_vs_lstm.png)
- ![RNN vs LSTM Error](plots/comparison_rnn_vs_lstm_error.png)
- ![RNN vs LSTM BPC](plots/comparison_rnn_vs_lstm_bpc.png)

RNN baseline (seq_len=100, hs=128)：
| Epoch | Train Loss | Train BPC | Train Acc | Valid Loss | Valid BPC | Valid Acc |
| --- | --- | --- | --- | --- | --- | --- |
| **best (1)** | 1.6195 | 2.3364 | 0.5137 | **1.6089** | 2.3211 | 0.5210 |
| last (5) | 1.5532 | 2.2408 | 0.5297 | 1.6106 | 2.3236 | 0.5228 |

**Observation（RNN 指標）：最佳在第 1 epoch，之後 valid loss 幾乎不變、acc 只有 0.52 左右，BPC ~2.32，顯示模型快到頂且能力有限。**

### 1.3 5 個訓練截點生成（RNN, 溫度=0.6, seed="The ", 長度=100）
```
[Epoch1] The sense the king with a cleak to a father were when the world.
GLOUCESTER:
My leave of the grow her b

[Epoch2] The great than the set thee hence.
First Corent;
Let the time you to the hands, when have brave me, the

[Epoch3] The enterpain and so mistress and this are and scholer to say the wind of the brother's blessitious, and

[Epoch4] The answer and grace and both of the rest.
DESDEMONA:
She were here that shall stand, that where the lo

[Epoch5] The for the law to the consent, by this sons I am seem that they sent of the death, dead; and thou took 
```
**Observation（RNN 生成）：早期假詞、斷句多；第 3–4 截點開始出現較長片段，但仍有缺字與怪詞；第 5 截點可讀性稍好但雜訊依舊。RNN 對長程依賴掌握有限。**

### 1.4 RNN 超參數比較
Hidden size（seq=100）：
- ![RNN Hidden Size Loss](plots/comparison_hidden_size.png)
- ![RNN Hidden Size Error](plots/comparison_hidden_size_error.png)
- ![RNN Hidden Size BPC](plots/comparison_hidden_size_bpc.png)

最佳驗證 loss：hs64=1.7561(e5), hs128=1.6089(e1), **hs256=1.4975(e3)**。  

**Observation：hs256 提升最明顯；hs64 欠擬合；隨 hidden 變大，loss/err/BPC 三條曲線都下移，但仍不及 LSTM。**

Sequence length（hs=128）：
- ![RNN Seq Len Loss](plots/comparison_seq_len.png)
- ![RNN Seq Len Error](plots/comparison_seq_len_error.png)
- ![RNN Seq Len BPC](plots/comparison_seq_len_bpc.png)

最佳驗證 loss：seq50=1.6221(e5), seq100=1.6089(e1), **seq150=1.5956(e3)**。  

**Observation：seq150 略優於 seq100，seq50 最差；拉長序列幫助有限，仍受 RNN 結構瓶頸。**

---

## 2. LSTM（重做 1–3 並比較）
### 2.1 架構與設定
Embedding → LSTM(hidden=128, layers=2, dropout=0.25) → Linear。

### 2.2 學習曲線與指標
同上圖（RNN vs LSTM），LSTM 曲線顯著低於 RNN。

LSTM baseline (seq_len=100, hs=128)：
| Epoch | Train Loss | Train BPC | Train Acc | Valid Loss | Valid BPC | Valid Acc |
| --- | --- | --- | --- | --- | --- | --- |
| **best (3)** | 1.3468 | 1.9430 | 0.5821 | **1.4423** | 2.0808 | 0.5698 |
| last (5) | 1.3344 | 1.9251 | 0.5851 | 1.4538 | 2.0974 | 0.5692 |

**Observation（LSTM 指標）：驗證 loss 比 RNN 低約 0.17，acc 高約 4–5%，BPC 也顯著下降；在 3–5 epoch 即穩定，收斂較快。**

### 2.3 5 個訓練截點生成（LSTM, 溫度=0.6, seed="The ", 長度=100）
```
[Epoch1] The house of this fair plague, as thou shalt not father.
LUCY:
Let me think it is so spoke more to him.

[Epoch2] The lady so life and love him in her,
So banish'd in my brother with their seas,
Some more from the fres

[Epoch3] The company of his master's reserved,
Appointed the wind, and all the conscience
Is sworn before the Fre

[Epoch4] The counsel will be a patience that is awhile;
So play me and with my heart from her image.
COMINIUS:
C

[Epoch5] The Duke of Salisbury and Montague was,
I will be false to make to the bond.
TOUCHSTONE:
I think there 
```

**Observation（LSTM 生成）：第 1 個截點就能產生可讀台詞；第 3、5 截點句法與語意更自然，角色名常出現，連貫度明顯優於 RNN。**

### 2.4 LSTM 超參數比較
Hidden size：
- ![LSTM Hidden Size Loss](plots/comparison_lstm_hidden_size.png)
- ![LSTM Hidden Size Error](plots/comparison_lstm_hidden_size_error.png)
- ![LSTM Hidden Size BPC](plots/comparison_lstm_hidden_size_bpc.png)

驗證 loss：hs64=1.5718(e5), hs128=1.4423(e3), **hs256=1.3905(e1)**。  

**Observation：hs256 在第 1 個 epoch 就達最佳，容量足且收斂快，因此我會鎖定這組。**

Sequence length：
- ![LSTM Seq Len Loss](plots/comparison_lstm_seq_len.png)
- ![LSTM Seq Len Error](plots/comparison_lstm_seq_len_error.png)
- ![LSTM Seq Len BPC](plots/comparison_lstm_seq_len_bpc.png)

驗證 loss：seq50=1.4948(e5), **seq100=1.4423(e3)**, seq150=1.4443(e5)。  

**Observation：seq100 與 seq150 差異極小，seq50 稍差；考慮效率，我會選擇固定 seq100。**

### 2.5 RNN vs LSTM 總結
- LSTM 數值全面勝出，尤其 hs256：驗證 loss=1.3905、BPC≈2.006、acc≈0.587（5 epoch）。
- Breakpoints 顯示 LSTM 句子流暢、劇本風格明顯；RNN 假詞與斷句較多。
- 收斂速度：LSTM 3–5 epoch 即趨平；RNN 拉長 epoch 也難有突破。

---

## 3. 文字生成 (Prime, 以最佳模型 LSTM_HS_256)
模型：LSTM_HS_256 (seq_len=100)，checkpoint `checkpoints/best_model_LSTM.pth`。  
檔案：`outputs/generation_final.txt`，Prime="JULIET"，溫度=0.6，長度 500（格式化為 12 行）。完整輸出：

```
JULIET:
What, sir?

MARK ANTONY:
A heart, I
 say, you will not a man forth to Cieson
to
 prove my beauty, and his majesty of a worl
d.

TITUS ANDRONICUS:
Come, go with me.

CA
SSIO:
Here, my lord, and the bride-balls an
d moonlight
Of despite of Edward will be de
fend
The tribune to the Fulster's father to
 the
fault of way.

LEONATO:
I cannot be no
t before the former than they know
The stoc
ks of heaven, and think he be made you and

the glory be the end: and the sea was provo
ked before
The danger of a party 
```
**Observation：全篇保持劇本格式，角色名頻繁且對話段落清楚；雖仍有少數斷行與不通順詞，但溫度 0.6 下整體穩定度佳、語氣一致性不錯，較高溫時的雜訊少。**

---

## 4. 結論
- **最佳模型**：LSTM_HS_256（seq_len=100），驗證 loss=1.3905、BPC≈2.006、acc≈0.587（5 epoch）。成本效益最佳。
- **是否要再訓練更久？** 這組在第 1 個 epoch 就接近最優，後續收益有限；其他組合差距大，即使拉長也難追上，我維持現狀。
- **生成溫度**：本次 prime 用 0.6，穩定度高；若要更多樣性可微調，但目前設定已符合需求。

---

_所有圖片在 `plots/`，完整指標與生成輸出在 `outputs/`，checkpoint 於 `checkpoints/`。如需額外細節可直接檢視檔案。_
