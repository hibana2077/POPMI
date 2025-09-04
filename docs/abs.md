# POPMI：**P**lug-and-play **O**rthogonal **P**arameter **M**atching for Model **I**ntegration

— 旋轉等變的 Transformer 融合，用於醫學影像分類

**一句話摘要**：POPMI 提出一個**可插即用（plug-and-play）**的旋轉等變轉接模組，結合**正交參數匹配（orthogonal Procrustes + permutation）**來對齊並融合多個已微調的 Transformer/Conv 模型，在**不增加推論成本**的情況下，提高醫學影像分類的準確度與旋轉穩定性，並給出**理論上界**與**可證明的等變性誤差控制**。我們以 MedMNIST v2 與 ISIC 進行大規模實證與消融。([arXiv][1], [Nature][2])

---

## 研究核心問題

1. 醫學影像（X-ray、皮膚鏡、顯微組織）常出現**任意旋轉**，非等變的模型對旋轉敏感；

2. 實務上常有**多來源/多院別**微調模型，如何**在不疊加推論成本**下融合其知識？現有方法要嘛只做等變（如 G-CNN/等變 ViT），要嘛只做權重融合（如 model soup / re-basin / OT-fusion），**缺乏同時滿足「旋轉等變 + 權重層級融合」的統一框架**。([Proceedings of Machine Learning Research][3], [arXiv][4], [proceedings.neurips.cc][5])

## 研究目標

* 在多資料集、多院別情境下，於**不增加推論時間或參數量**下，提升 AUROC / F1 與**旋轉一致性指標**。
* 提供**可插拔**的等變轉接器，**不限制**骨幹（ViT/Swin/Conv）。
* 以**正交參數匹配** + **權重插值/融合**，在理論上保證：

  1. 等變誤差上界；2) 融合後的損失不劣於已對齊權重的線性路徑上界。

---

## 方法總覽（POPMI）

### (A) Plug-and-play 旋轉等變轉接器（PREA）

* 在**patch embedding**前後與部分注意力頭加入**群等變（E(2)/SE(2)）**轉接器：以**群卷積/等變位置編碼**近似 ViT 的旋轉等變特性；模組可**凍結骨幹**或只微調低秩轉接器。靈感源自 G-CNN 與 E(2)-equivariant ViT 的設計。([arXiv][6], [Proceedings of Machine Learning Research][7])

### (B) Orthogonal Parameter Matching（OPM）

* 對多個已微調的模型，逐層求解**置換 + 正交 Procrustes**對齊：

  $$
  \min_{P\in\Pi,\,R\in O(d)}\ \|W_1 - P\,W_2 R\|_F^2,\quad R^*=\mathrm{UV}^\top\ \text{for SVD}(W_2^\top P^\top W_1=U\Sigma V^\top)
  $$
* 與**Git Re-Basin**（置換對齊）與**OT-fusion / FedMA**（層內對齊）相呼應，但額外引入**正交對齊**以處理注意力/投影層的旋轉自由度，再做權重**插值/平均**得到單一模型，**無額外推論成本**。([arXiv][8], [Personal Robotics Lab][9], [proceedings.neurips.cc][5])

### (C) POPMI-Soup 融合

* 在 OPM 對齊後，做**加權線性插值**（或貪婪選擇）形成單一權重（類 model soup），兼具穩健性與零額外成本。([arXiv][4], [Proceedings of Machine Learning Research][10])

---

## 理論洞見（摘要）

**定義 1（旋轉等變）**：對群 $G\subset SE(2)$，模型 $f$ 等變若 $\forall g\in G,\, f(\rho_X(g)x)=\rho_Y(g)f(x)$。PREA 透過群卷積/等變位置編碼保證對**有限離散旋轉**的等變性。([arXiv][6], [Proceedings of Machine Learning Research][7])

**命題 1（POPMI 等變誤差上界）**：設骨幹非等變，但在前/後級加上 PREA，若群離散化角度集為 $\Theta$，則對任意 $\theta\in\Theta$，

$$
\|f_{\text{POPMI}}(R_\theta x)-R_\theta f_{\text{POPMI}}(x)\|\le \epsilon(\theta)
$$

其中 $\epsilon(\theta)$ 與骨幹破壞等變的模組（如絕對位置編碼）的 Lipschitz 常數與 PREA 校正殘差成正比；對採**相對/群等變位置編碼**之自注意力，可令 $\epsilon$ 收斂為 0。([Proceedings of Machine Learning Research][7])

**命題 2（對齊後插值的損失上界）**：令 $\tilde W_2=P^*W_2R^*$ 為 OPM 對齊解，若損失 $\mathcal L(W)$ 在 $[W_1,\tilde W_2]$ 的線段上 $L$-Lipschitz 並且存在**近似單盆地連通**（linear mode connectivity）性質，則對 $\hat W=\alpha W_1+(1-\alpha)\tilde W_2$ 有

$$
\mathcal L(\hat W)\le \max\{\mathcal L(W_1),\mathcal L(\tilde W_2)\}+ L\,\alpha(1-\alpha)\|W_1-\tilde W_2\|_F,
$$

說明對齊後插值不劣化，呼應 re-basin 與 model soup 觀察。([arXiv][8])

> **證明概念**：命題 1 由群等變層的表示理論與殘差界給出；命題 2 由線段上 Lipschitz 界與 re-basin 單盆地假說（對齊後）推出上界。

---

## 數學推演與證明（精簡）

1. **Procrustes 對齊閉式解**：給定 $A,B\in\mathbb R^{d\times d}$，$\min_{R\in O(d)}\|A-BR\|_F$ 之解為 $R^*=UV^\top$，其中 $U\Sigma V^\top=\mathrm{SVD}(B^\top A)$。我們在置換 $P$ 給定下迭代求解 $R$ 與 $P$（Hungarian/OT 軟匹配）。([proceedings.neurips.cc][5])
2. **群卷積等變性**：若核 $k$ 對群作用閉合，群卷積 $(f*k)(g)=\int f(h)k(h^{-1}g)\,dh$ 對 $G$ 等變；GE-ViT 的群等變位置編碼對 SE(2)/E(2) 亦成立。([arXiv][6], [Proceedings of Machine Learning Research][7])
3. **線性路徑連通性**：re-basin 證實經過**置換對齊**的兩模型存在**零障礙線性連通**，因此插值不會穿越高損失區；我們再加上正交對齊，使注意力/投影層更穩定。([arXiv][8])

---

## 預計使用 Dataset（分類任務）

* **MedMNIST v2**（多模態輕量級 2D/3D，708k 影像）：適合**大規模與多任務**的等變/融合消融。([Nature][2])
* **ISIC 2018/2020**（皮膚鏡分類）：鏡頭取向任意，最能檢驗**旋轉一致性**與泛化。([arXiv][13], [challenge.isic-archive.com][14])

---

## 實驗設計

* **指標**：AUROC、AUPRC、$\Delta_{\text{rot}}$（旋轉一致性：對隨機角度集合 $\Theta$，量測輸出差距）、校準（ECE）。
* **比較**：ViT/Swin 基線、**GE-ViT（E(2)-ViT）**、**Model Soups**、**Git Re-Basin**、**OT-Fusion**、**FedMA**。([Proceedings of Machine Learning Research][7], [arXiv][4], [proceedings.neurips.cc][5])
* **消融**：去掉 PREA / 去掉 OPM / 僅做湯（soup）/ 僅做置換 / 僅做 OT 軟對齊。
* **資料分割**：院別/年份跨域；旋轉受控測試（每張影像隨機旋轉 0–350° 每 10° 評估）。
* **效率**：統計**推論時間/參數量不增加**（相對 ensemble）。

---

## 預期貢獻

1. **首個**將**旋轉等變 plug-and-play 模組**與**權重空間對齊/融合**統一於醫學影像分類；
2. 提出 **OPM**（Permutation + Orthogonal Procrustes）對齊，改善 Transformer 層的融合穩定性；
3. 給出**等變誤差與插值損失的理論上界**；
4. 在 ISIC/MedMNIST v2 實證**零額外推論成本**下優於強基線。([arXiv][4], [Proceedings of Machine Learning Research][7], [Nature][2])

---

## 創新點

* **可插拔**：不改動骨幹架構即可獲得旋轉等變性。
* **雙對齊融合**：結合**置換**（re-basin / FedMA/OT-fusion 思想）與**正交對齊**，專為注意力/投影矩陣設計。([arXiv][8], [proceedings.neurips.cc][5])
* **單模型推論**：像 model soup 一樣**不增加推論成本**，但在對齊後更穩。([arXiv][4])

---

## 與現有研究之區別

* **G-CNN / GE-ViT**：只處理等變，不處理**多模型權重融合**；POPMI 將等變與融合同時做。([arXiv][6], [Proceedings of Machine Learning Research][7])
* **Model Soups**：僅平均權重，未處理**通道/頭的對齊**與等變；POPMI 在**OPM 對齊後**再融合。([arXiv][4])
* **Git Re-Basin / FedMA / OT-Fusion**：能對齊/融合，但**未結合旋轉等變轉接器**；POPMI 在醫影場景加入等變設計與理論界。([arXiv][8], [proceedings.neurips.cc][5])

---

## 實驗成功投稿計畫（TMI）

* **主結果**：三資料集平均 AUROC / F1、$\Delta_{\text{rot}}$ 顯著優於 GE-ViT 與最佳單模型；展示**零推論開銷**。
* **可視化**：旋轉穩定性曲線、權重對齊前後的注意力頭相似度、t-SNE/CKA。
* **臨床價值**：跨院別泛化、標註稀缺時的**小樣本優勢**（因為融合來自多院模型）。
* **開源**：提供 PREA/OPM 參考實作與訓練腳本。

## 失敗投稿計畫（備援）

* 若旋轉等變收益有限：轉為「**低資料/跨域融合**」主題，主打 OPM-soup 在小樣本與跨院泛化的穩健性，改投 **JBHI** / **CMPB** 或 **MICCAI/ISBI** workshop；或縮小到單模態（ISIC）做更深的病理分析。

---

## 可能的局限與風險

* 某些任務**方向資訊本身即為病灶徵象**（如骨科定位），強等變可能傷害表現；可用**軟等變**或只在局部層啟用 PREA。
* 權重對齊對**模型寬度/層設計**敏感；需在設計上限制可比對的層（如 MLP/投影層）並採 OT 軟匹配緩解。([proceedings.neurips.cc][5])

---

### 參考（關鍵負載）

* Plug-and-Play Priors（可插拔思想的經典框架，啟發 PREA 的「解耦」設計）。([docs.lib.purdue.edu][15])
* Group-Equivariant CNN / **E(2)-Equivariant ViT**（旋轉等變的理論與在 Transformer 的落地）。([arXiv][6], [Proceedings of Machine Learning Research][7])
* **Model Soups**（零推論成本的權重平均）。([arXiv][4])
* **Git Re-Basin**（置換對齊與線性連通）。([arXiv][8])
* **OT-Fusion / FedMA**（以 OT/匹配進行層內對齊與融合）。([proceedings.neurips.cc][5], [arXiv][16])
* Datasets：**MedMNIST v2**、**ISIC**。([Nature][2], [arXiv][11], [challenge.isic-archive.com][14])

---

如果你想，我可以把上述材料整理成**論文大綱 + 圖表設計清單（圖 1 方法、圖 2 理論界、圖 3 主結果、圖 4 消融）**與**實驗腳本骨架**（PyTorch + timm + e2cnn/GE-ViT 參考實作），直接可開跑。

[1]: https://arxiv.org/abs/2306.06722? "$E(2)$-Equivariant Vision Transformer"
[2]: https://www.nature.com/articles/s41597-022-01721-8? "MedMNIST v2 - A large-scale lightweight benchmark for 2D ..."
[3]: https://proceedings.mlr.press/v48/cohenc16.html? "Group Equivariant Convolutional Networks"
[4]: https://arxiv.org/abs/2203.05482? "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time"
[5]: https://proceedings.neurips.cc/paper/2020/file/fb2697869f56484404c8ceee2985b01d-Paper.pdf? "Model Fusion via Optimal Transport"
[6]: https://arxiv.org/abs/1602.07576? "Group Equivariant Convolutional Networks"
[7]: https://proceedings.mlr.press/v216/xu23b/xu23b.pdf? "E(2)-Equivariant Vision Transformer"
[8]: https://arxiv.org/abs/2209.04836? "Git Re-Basin: Merging Models modulo Permutation Symmetries"
[9]: https://personalrobotics.cs.washington.edu/publications/ainsworth2023gitrebasin.pdf? "GIT RE-BASIN: MERGING MODELS MODULO PERMU"
[10]: https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf? "averaging weights of multiple fine-tuned models improves ..."
[13]: https://arxiv.org/abs/1902.03368? "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)"
[14]: https://challenge.isic-archive.com/data/? "ISIC Challenge Datasets"
[15]: https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1449&context=ecetr& "Plug-and-Play Priors for Model Based Reconstruction"
[16]: https://arxiv.org/abs/2002.06440? "Federated Learning with Matched Averaging"
