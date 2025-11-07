# Feature Map Observations（CIFAR-10 baseline）

> 類別對照：0 airplane、1 automobile、2 bird、3 cat、4 deer、5 dog、6 frog、7 horse、8 ship、9 truck。

我挑選每個類別在 Block 1、Block 2、Block 3 的代表性特徵圖，總結如下：

1. **Class 0 Airplane**  
   - Block 1：大部分通道呈現天空與機身的色彩對比，邊緣沿著機翼亮起。  
   - Block 2：開始只留下機身輪廓；背景雲彩被抑制。  
   - Block 3：僅存機翼與機尾的細長結構，與報告圖 5(b) 相呼應。

2. **Class 1 Automobile**  
   - Block 1：偵測車身紅/藍顏色塊。  
   - Block 2：輪胎的圓形紋理被加強，車窗與車頂分開。  
   - Block 3：只剩車殼外框，背景道路被大幅淡化。

3. **Class 2 Bird**  
   - Block 1：通道亮度集中在翅膀外緣與天空對比。  
   - Block 2：羽毛紋理開始顯現，頭部與尾羽分離。  
   - Block 3：只剩一條弧線勾勒主體，背景樹葉全暗。

4. **Class 3 Cat**  
   - Block 1：偵測耳朵三角形與身體大色塊。  
   - Block 2：毛皮紋理與臉部對稱區域被同時保留。  
   - Block 3：只剩頭部輪廓與眼睛附近亮點，協助區分 cat/dog。

5. **Class 4 Deer**  
   - Block 1：腿部與身體顏色差異清楚。  
   - Block 2：頸部與鹿角的細長結構被強化。  
   - Block 3：多數通道僅留下背脊與鹿角骨架，用來區分 horse。

6. **Class 5 Dog**  
   - Block 1：偵測背部、耳朵與地板。  
   - Block 2：耳朵與口鼻的紋理分離，背景草地逐漸消失。  
   - Block 3：剩下頭部輪廓與口鼻，與 cat 的圓形臉形成對比。

7. **Class 6 Frog**  
   - Block 1：水面/陸地的色塊突顯。  
   - Block 2：腿部與身體紋理被分別點亮。  
   - Block 3：僅剩蟾蜍輪廓，背景水紋完全被抑制。

8. **Class 7 Horse**  
   - Block 1：偵測長條型身軀與四肢。  
   - Block 2：馬頭與鬃毛紋理突出。  
   - Block 3：留下奔跑姿勢的骨架，方便與 deer/ dog 区別。

9. **Class 8 Ship**  
   - Block 1：船體與海面顏色形成明顯對比。  
   - Block 2：船身的水平線與煙囪被不同通道捕捉。  
   - Block 3：僅剩水線與船殼，與 airplane 的水平翼型不同。

10. **Class 9 Truck**  
    - Block 1：車斗的大面積色塊最亮。  
    - Block 2：輪胎與車頭的矩形窗被清楚分開。  
    - Block 3：留下方正的車斗輪廓，讓模型把 truck 與 automobile 區分開。

這些觀察直接支援主報告第 4 節對特徵層級的敘述：愈深的 block 愈傾向保留形狀骨架、移除背景與色彩雜訊。
