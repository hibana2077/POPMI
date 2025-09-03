# 論文題目（縮寫：POPMI）

**POPMI：Plug-and-Play Orthogonal Projections for Manifold-Invariant Medical Image Classification**
（中文：**以可插即用正交投影實現流形不變性的醫學影像分類**）

---

## 研究核心問題

在醫學影像分類中，同一病灶於不同**幾何變換**（旋轉、反射、尺度、切片角度）下應有相同診斷結論；然而典型做法多仰賴**資料增強**去近似「不變性」，或改造整個網路為**群等變/不變**架構（如 G-CNN）。兩者要嘛缺乏理論保證、要嘛改動成本高。**我們要解決**：如何在**不修改骨幹模型**的前提下，以**可插即用（plug-and-play）**的方式，對**特徵**實作**幾何不變的正交投影**，同時保有可證性與收斂性。參考：PnP/RED/CE 理論基礎、幾何深度學習與不變化理論。([docs.lib.purdue.edu][1], [arXiv][2], [engineering.purdue.edu][3])

## 研究目標

1. 設計一個可插入任意 CNN/ViT 的模組 **POPMI**，在**特徵空間**上實作對給定幾何群 $G$ 的**不變投影** $P_G$，並與現有增強策略互補。
2. 以**共識平衡（Consensus Equilibrium, CE）/PnP**框架包裝 POPMI，使其成為**可收斂**的投影型正則器（prox-like operator）。([arXiv][4], [engineering.purdue.edu][3])
3. 建立**理論界限**：投影後的分類器在群平均分佈下具較佳的一般化風險上界，並連結 PAC-Bayes 結果。([arXiv][5])

## 方法總覽（POPMI 模組）

* **幾何群與雷諾茲算子（Reynolds operator）**：對緊致群 $G$（如平面旋轉反射的有限離散子群），定義

  $$
  P_G f(x)=\int_{g\in G} \rho(g)f(g^{-1}\!\cdot\!x)\,d\mu(g),
  $$

  其中 $\rho(g)$ 為特徵上的群作用，$\mu$ 為 Haar 測度。此算子為**自伴、冪等**，因此是**到 $G$-不變子空間的正交投影**。我們以蒙地卡羅取樣/可學權重近似積分，構成可微分的 layer。([people.kth.se][6], [math.utah.edu][7], [維基百科][8])
* **PnP/CE 封裝**：將 $P_G$ 視為**非擾動、非擴張**的投影/去雜訊算子，放入 CE 或 PnP-ADMM 迭代，與任意骨幹的損失/梯度「達到平衡」；在**有界/非擴張**條件下可證收斂到不動點。([arXiv][2])
* **與增強協同**：保留常見增強（RandAugment、MixUp、CutMix），但 POPMI 在**特徵層**保證不變性，減少靠大量影像級增強「學」到不穩定不變性的風險。([NeurIPS 會議論文集][9], [arXiv][10], [CVF開放存取][11])

## 預期主要貢獻

1. **架構層面的 Plug-and-Play 幾何投影**：首個將**雷諾茲投影**以**PnP/CE**形式導入**醫學影像分類**的通用模組，可插即用於 CNN/ViT。([engineering.purdue.edu][3])
2. **理論層面的風險分析**：在凸損情形下，證明 POPMI 等價於**特徵平均/投影**，相對單純增強可**嚴格收斂**更小一般化誤差上界（接軌 PAC-Bayes）。([arXiv][5])
3. **實證層面的跨模態驗證**：於多來源資料集（X-ray、皮膚鏡、病理、超音波/3D）展現**穩健提升**與**較少增強依賴**。([Nature][12], [arXiv][13], [GitHub][14])

## 創新點

* **不改骨幹、保證不變**：相較等變網路（需改架構）與影像級增強（僅近似不變），**POPMI 以特徵投影直接保證不變性**並具可證收斂。([arXiv][15])
* **理論-工程雙棲**：以**雷諾茲投影＝正交投影**的數學事實連結到**PnP-ADMM/CE 收斂理論**，提供可重現與可證性。([people.kth.se][6], [arXiv][2])
* **與增強/生成式方法互補**：將**擴散式合成增強**作為資料層補充，POPMI 作為特徵層不變約束。([科學直接][16], [arXiv][17])

## 理論洞見（要點）

1. **正交投影性**：對緊致群 $G$ 與 $G$-不變內積，$P_G$ 自伴且 $P_G^2=P_G$，因此為到 $V^G$ 的正交投影；此來自群表示與雷諾茲算子基礎。([people.kth.se][6])
2. **一般化優勢**：對凸損與群作用的資料分佈，**特徵平均/投影**相較僅做資料增強有更緊的 PAC-Bayes 上界，對「學到的不變性在分布外失效」更具魯棒性。([arXiv][5])
3. **收斂性**：若 POPMI layer 以（近似）非擴張映射實作，依 **PnP-ADMM** 與 **CE** 理論，可證到固定點；我們以譜範數正則與動態步長確保條件。([arXiv][2], [engineering.purdue.edu][3])

## 數學推演與關鍵證明綱要

* **命題 1（雷諾茲投影）**：對緊致群 $G$ 有 Haar 測度 $\mu$，定義 $P_G f=\int_G \rho(g)f\,d\mu(g)$。則 $P_G$ 線性、自伴、冪等，且 $\mathrm{Im}(P_G)=V^G$。*證要*：由 $\mu$ 左右不變性得自伴；冪等由 Fubini 與群封閉性；像為不變子空間。([people.kth.se][6])
* **命題 2（風險不增）**：對 $G$-不變目標與凸損 $\ell$，設分類器 $h$ 與 $h\circ P_G$，則在群平均分佈下 $ \mathbb{E}_{(x,y)} \ell(h\circ P_G(x),y) \le \mathbb{E}\ell(\bar h(x),y)$，其中 $\bar h$ 為僅靠資料增強的經驗風險最小解；並給出 PAC-Bayes 界。*證要*：沿用 Lyle et al. 的特徵平均分析，並以 Jensen/凸性與群平均交換性建立上界。([arXiv][18])
* **命題 3（PnP-ADMM/CE 收斂）**：若 POPMI 實作的投影 $\mathcal{P}$ 為非擴張且有固定點集非空，則在**繼續參數**與**界定去雜訊器/投影**條件下，PnP-ADMM/CE 迭代收斂至共識解。*證要*：承襲 Chan 等之有界/非擴張去雜訊器收斂結果並用 CE 等價式。([arXiv][2], [engineering.purdue.edu][3])

## 實作細節

* **群選擇**：2D 醫影常用 $G=C_n$（離散旋轉）× $\{\pm\}$（反射）；病理切片取 $n\in\{8,16\}$，X-ray 取 $n\in\{4,8\}$。
* **近似積分**：隨機抽樣 $k$ 個群元素，做特徵對齊與平均（或加權學習），並以**Gram-Schmidt**在特徵子空間上做**正交化**以提升數值穩定。
* **訓練**：主幹任選（ResNet/ConvNeXt/Swin/ViT-B），POPMI 放在最後 stage 對 1–2 層特徵投影；與 RandAugment/MixUp/CutMix 並用。([NeurIPS 會議論文集][9], [arXiv][10], [CVF開放存取][11])

## 預計使用資料集與評測

* **MedMNIST v2**（2D/3D 多任務輕量基準；做系統性橫評）。([Nature][12], [arXiv][19], [medmnist.com][20])
* **CheXpert**（胸腔 X 光多標籤；評 AUROC、AUPRC）。([arXiv][13], [stanfordmlgroup.github.io][21], [aimi.stanford.edu][22])
* **HAM10000**（皮膚鏡多類；評 Balanced Acc/F1）。([Nature][23], [arXiv][24])
* **PatchCamelyon (PCam)**（病理小塊二分類；速測與消融）。([GitHub][14], [TensorFlow][25])
* **（可選）超音波/3D 任務**作為外部驗證與擴散式增強示範。([科學直接][16])

## 實驗設計（關鍵對照）

1. **骨幹**：ResNet/ViT + 標準增強 vs + POPMI（+ 是否使用生成式擴散增強）。([arXiv][17])
2. **架構對照**：與 **G-CNN/E(2) 等變網路**比較，檢視在**不改骨幹**前提的效益/效率。([arXiv][15])
3. **消融**：群大小 $n$、抽樣數 $k$、正交化/權重學習策略、是否 CE 包裝、是否與 MixUp/CutMix 並用。([arXiv][10], [CVF開放存取][11])
4. **統計檢定**：DeLong test（AUROC）、McNemar/Bootstrap 置信區間。

## 與現有研究之區別

* **非純增強**：文獻證明資料增強可能「學到的」不變性在分佈外失效；POPMI 以**投影**直接強制不變，並在理論上**收斂**與**具上界**。([arXiv][5])
* **非改架構等變**：不同於 G-CNN 必須重寫卷積核與特徵索引，POPMI 僅作為**後段可插 layer**，工程成本低、可移植。([arXiv][15])
* **PnP/CE 移植到分類**：過去 PnP/RED/CE 多用於**重建/去噪**；我們將之**理論化地**引入**分類**作為幾何不變正則。([docs.lib.purdue.edu][1], [PMC][26], [engineering.purdue.edu][3])

## 成功投稿計畫（面向 IEEE TMI）

* **貢獻定位**：提出**通用可插模組 + 可證理論 + 跨模態強實驗**，符合 TMI 對**方法學與臨床影響**並重的取向。
* **實驗包**：

  * **主結果**：CheXpert（多標籤 AUROC）、HAM10000（B-Acc/F1）、PCam（ROC）、MedMNIST v2（多任務平均）。
  * **對照**：Baseline、+RandAugment、+MixUp/CutMix、G-CNN、POPMI（有無 CE）、POPMI+生成式增強。([NeurIPS 會議論文集][9], [arXiv][10], [CVF開放存取][11])
  * **可重現性**：公開程式碼與訓練腳本、固定亂數種子、提供模型權重與推論指令。
* **撰稿結構**：動機→方法（雷諾茲投影＋PnP/CE）→理論→實作→實驗→臨床解讀→限制與未來工作（3D/序列）。
* **風險緩解**：若某單一模態收益有限，強化**跨資料一致性**與**效能-計算效率曲線**，凸顯工程價值。

## 失敗投稿備案（若 TMI 審查未過）

* **技術深化**：聚焦單一模態（如病理或胸腔 X 光）做更深 ablation 與臨床團隊共評。
* **轉投期刊/會議**：

  * **IEEE JBHI**（方法+應用）、**Medical Image Analysis (MedIA)**（若理論與實驗更厚重）、**Computerized Medical Imaging and Graphics**、或 **MICCAI** 術後 workshop／**TMI RAL** 類專欄。
* **補強點**：

  * 擴散式增強的更系統比較與偏差分析。([科學直接][16], [arXiv][17])
  * 與等變網路的**效率-準確**統計顯著性分析。([arXiv][15])

---

### 參考與基礎（精選）

* **Plug-and-Play / RED / CE 基礎與收斂**：Venkatakrishnan et al.（PnP-ADMM）; Chan et al.（固定點收斂條件）; Reehorst & Schniter（RED 與分數匹配、共識視角）; Buzzard et al.（Consensus Equilibrium）。([docs.lib.purdue.edu][1], [arXiv][2], [PMC][26], [engineering.purdue.edu][3])
* **幾何不變/等變**：Cohen & Welling（G-CNN）; Bronstein et al.（Geometric Deep Learning）; 雷諾茲算子/不變投影教材。([arXiv][15], [people.kth.se][6])
* **資料增強（一般與醫影）**：RandAugment、MixUp、CutMix；醫影增強與生成式擴散綜述。([NeurIPS 會議論文集][9], [arXiv][10], [CVF開放存取][11], [科學直接][27], [PMC][28])
* **資料集**：MedMNIST v2、CheXpert、HAM10000、PCam。([Nature][12], [arXiv][19], [stanfordmlgroup.github.io][21], [GitHub][14], [TensorFlow][25])

---

[1]: https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=1449&context=ecetr&utm_source=chatgpt.com "Plug-and-Play Priors for Model Based Reconstruction"
[2]: https://arxiv.org/pdf/1605.01710?utm_source=chatgpt.com "Plug-and-Play ADMM for Image Restoration: Fixed Point ..."
[3]: https://engineering.purdue.edu/~bouman/Plug-and-Play/webdocs/SIIMS01.pdf?utm_source=chatgpt.com "Plug-and-Play Unplugged: Optimization-Free Reconstruction ..."
[4]: https://arxiv.org/abs/1705.08983?utm_source=chatgpt.com "Plug-and-Play Unplugged: Optimization Free Reconstruction using Consensus Equilibrium"
[5]: https://arxiv.org/abs/2005.00178?utm_source=chatgpt.com "On the Benefits of Invariance in Neural Networks"
[6]: https://people.kth.se/~gss/notes/invariant.pdf?utm_source=chatgpt.com "INVARIANT THEORY"
[7]: https://www.math.utah.edu/~bertram/7800/GIT1.pdf?utm_source=chatgpt.com "Course Notes for Math 780-1 (Geometric Invariant Theory)"
[8]: https://en.wikipedia.org/wiki/Reynolds_operator?utm_source=chatgpt.com "Reynolds operator"
[9]: https://proceedings.neurips.cc/paper/2020/file/d85b63ef0ccb114d0a3bb7b7d808028f-Paper.pdf?utm_source=chatgpt.com "RandAugment: Practical Automated Data Augmentation ..."
[10]: https://arxiv.org/pdf/1710.09412?utm_source=chatgpt.com "mixup: BEYOND EMPIRICAL RISK MINIMIZATION"
[11]: https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf?utm_source=chatgpt.com "CutMix: Regularization Strategy to Train Strong Classifiers ..."
[12]: https://www.nature.com/articles/s41597-022-01721-8?utm_source=chatgpt.com "MedMNIST v2 - A large-scale lightweight benchmark for 2D ..."
[13]: https://arxiv.org/abs/1901.07031?utm_source=chatgpt.com "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison"
[14]: https://github.com/basveeling/pcam?utm_source=chatgpt.com "The PatchCamelyon (PCam) deep learning classification ..."
[15]: https://arxiv.org/pdf/1602.07576?utm_source=chatgpt.com "Group Equivariant Convolutional Networks"
[16]: https://www.sciencedirect.com/science/article/abs/pii/S1361841523001068?utm_source=chatgpt.com "Survey paper Diffusion models in medical imaging"
[17]: https://arxiv.org/html/2407.04103v1?utm_source=chatgpt.com "Advances in Diffusion Models for Image Data Augmentation"
[18]: https://arxiv.org/pdf/2005.00178?utm_source=chatgpt.com "On the Benefits of Invariance in Neural Networks"
[19]: https://arxiv.org/abs/2110.14795?utm_source=chatgpt.com "MedMNIST v2 -- A large-scale lightweight benchmark for 2D and 3D biomedical image classification"
[20]: https://medmnist.com/v2?utm_source=chatgpt.com "18x Standardized Datasets for 2D and 3D Biomedical ..."
[21]: https://stanfordmlgroup.github.io/competitions/chexpert/?utm_source=chatgpt.com "CheXpert: A Large Dataset of Chest X-Rays and Competition ..."
[22]: https://aimi.stanford.edu/datasets/chexpert-chest-x-rays?utm_source=chatgpt.com "CheXpert: Chest X-rays"
[23]: https://www.nature.com/articles/sdata2018161?utm_source=chatgpt.com "The HAM10000 dataset, a large collection of multi-source ..."
[24]: https://arxiv.org/abs/1803.10417?utm_source=chatgpt.com "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions"
[25]: https://www.tensorflow.org/datasets/catalog/patch_camelyon?utm_source=chatgpt.com "patch_camelyon - Datasets"
[26]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6801116/?utm_source=chatgpt.com "Regularization by Denoising: Clarifications and New ..."
[27]: https://www.sciencedirect.com/science/article/pii/S277244252400042X?utm_source=chatgpt.com "A systematic review of deep learning data augmentation in ..."
[28]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10027281/?utm_source=chatgpt.com "Medical image data augmentation: techniques ..."
