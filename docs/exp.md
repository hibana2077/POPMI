# POPMI 實驗清單（Markdown）

> 目標：產出 TMI 論文所需的**主結果、消融、魯棒性、效率與統計檢定**。本清單以✅/⬜️追蹤，並給出參數網格與執行指令樣板。

---

## 0. 命名規則與全域設定

* **Run ID**：`<DS>-<Backbone>-<Aug>-POPMI<0/1>-CE<0/1>-it<k>-rho<r>-beta<b>-w<0/1>-k<subset>-seed<s>`
* **種子**：`seed ∈ {0,1,2}`（所有核心實驗皆三次重複，報告平均±95% CI）
* **統計**：分類用 **DeLong**（AUROC）、**McNemar**（ACC/B-ACC），bootstrap 95% CI；多重比較以 Holm–Bonferroni。
* **評測**：

  * CheXpert：AUROC / AUPRC（每病徵+macro）
  * HAM10000：Balanced Accuracy / Macro-F1
  * PCam：AUROC / ACC
  * MedMNIST v2：官方多任務平均（AUC/ACC）
* **腳本參數對應**（對應右側 `POPMI_PyTorch_CE_PnP_minimal_example.py`）：

  * POPMI：`--popmi_beta`, `--popmi_w`（可學權重）, `--popmi_k`（訓練時群子抽樣）
  * CE/PnP：`--ce`, `--ce_iters`, `--ce_rho`
* **樣板指令**（最小可跑 FakeData；改資料集請替換 loader）：

  ```bash
  python POPMI_PyTorch_CE_PnP_minimal_example.py \
    --epochs 10 --batch_size 128 --steps 500 \
    --ce --ce_iters 3 --ce_rho 1.0 \
    --popmi_beta 0.5 --popmi_w --popmi_k 4 \
    --image_size 64 --channels 1 --classes 2 --lr 3e-4
  ```

---

## 1. 核心主結果（每資料集 16 組 × 3 seeds）

> 比較 **Baseline / +RandAugment(RA) / +MixUp / +RA+MixUp** 與 **POPMI on/off、CE on/off** 的組合。

* **骨幹**：`ResNet-18` 與 `ViT-B/16`（或現成 CNN/ViT；最小腳本可先用 TinyBackbone 驗證）
* **POPMI 既定參數**：`beta=0.5`, `w=均勻(0)`, `k=全群`；CE（若開）使用 `iters=3, rho=1.0`
* **總數**：每資料集 4（增強）× 2（POPMI）× 2（CE） = **16 組**

### 1.1 CheXpert（多標籤）

* 指標：AUROC, AUPRC（macro + per-label）
* ✅/⬜️ 清單：

  * ⬜️ A1 Baseline
  * ⬜️ A2 RA
  * ⬜️ A3 MixUp
  * ⬜️ A4 RA+MixUp
  * ⬜️ A1′\~A4′ + **POPMI**
  * ⬜️ A1″\~A4″ + **CE**（無 POPMI，檢查 CE 對自身正則的影響）
  * ⬜️ A1‴\~A4‴ + **POPMI+CE**

### 1.2 HAM10000（多類）

* 指標：Balanced Acc, Macro-F1
* 清單同 1.1（B1\~B16）。

### 1.3 PCam（病理二分類）

* 指標：AUROC, ACC；**小圖**高通量對比。
* 清單同 1.1（C1\~C16）。

### 1.4 MedMNIST v2（多任務彙整）

* 指標：官方 AUC/ACC 平均；跨子集彙整表。
* 清單同 1.1（D1\~D16）。

> **交付物**：每資料集匯總表（±CI）、勝敗圖（POPMI vs Baseline 的提升分佈）、顯著性星號。

---

## 2. 消融研究（以 MedMNIST v2 與 PCam 為主）

> 驗證設計選擇與穩健性，均以 `RA+MixUp` 作為固定增強（除非另註）。

### 2.1 CE 迭代與懲罰參數

* 固定 POPMI on。
* 參數網格：`iters ∈ {1,3,5}`, `rho ∈ {0.5, 1.0, 2.0}` ⇒ **9 組** × 2 seeds。
* ⬜️ E1–E9（MedMNIST） / ⬜️ E1′–E9′（PCam）

### 2.2 殘差係數 β

* `beta ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`（0=純輸入，1=純投影） ⇒ **5 組**。
* ⬜️ F1–F5（MedMNIST） / ⬜️ F1′–F5′（PCam）

### 2.3 群權重（均勻 vs 可學）

* `w ∈ {均勻, 可學}`，CE 固定 `iters=3,rho=1` ⇒ **2 組**。
* ⬜️ G1（均勻） / ⬜️ G2（可學）

### 2.4 訓練期群子抽樣

* `sample_k ∈ {2,4,8(全群)}`，觀察效率/泛化折衷 ⇒ **3 組**。
* ⬜️ H1–H3

### 2.5 放置位置（layer placement）

* 放於：\`

  * 末段(預設)：Global Pooling 前一層
  * 中段：倒數第二個 stage 後
  * 多處：中段+末段（疊加兩層 POPMI）
    \` ⇒ **3 組**。
* ⬜️ I1–I3

### 2.6 Backbone 泛化

* `ResNet-18 / ConvNeXt-T / ViT-B`（或可得骨幹） ⇒ **3 組**。
* ⬜️ J1–J3（固定最佳 POPMI/CE 超參）

### 2.7 直接投影 vs CE 包裝

* 比較 `POPMIProjection` 直通與 `CE_PnP(POPMI)` ⇒ **2 組**。
* ⬜️ K1（直通） / ⬜️ K2（CE 包裝）

---

## 3. 與等變架構的對照（可選 / 若資源允許）

* **模型**：`E(2)-CNN` 或 `G-CNN`（對應 D4）
* **設計**：同 1. 的 A4（RA+MixUp）設定，比較：Baseline vs G-CNN vs **POPMI+CE**
* ⬜️ L1（Baseline） / ⬜️ L2（G-CNN） / ⬜️ L3（POPMI+CE）

---

## 4. 魯棒性與分佈外（OOD）

### 4.1 幾何分佈外測試

* 測試集施加未見角度/組合（例如非 90° 的旋轉、同時旋轉+翻轉）
* 衡量：AUROC/ACC 隨角度曲線；**不變性殘差** `||f(x) - f(g·x)||` 的箱型圖。
* ⬜️ M1（CheXpert 子集） / ⬜️ M2（HAM10000） / ⬜️ M3（PCam）

### 4.2 樣本效率（小樣本）

* 訓練集使用比例 `{5%, 10%, 20%}`；比較 Baseline vs POPMI+CE。
* ⬜️ N1–N3（各資料集）

### 4.3 類別不平衡 / 重加權

* 使用 class-balanced loss / focal loss；觀察 POPMI 的影響。
* ⬜️ O1（CheXpert） / ⬜️ O2（HAM10000）

### 4.4 雜訊與模糊

* 加入高斯雜訊、運動模糊；觀察穩健性。
* ⬜️ P1（PCam） / ⬜️ P2（MedMNIST）

---

## 5. 效率與資源

* **度量**：推論延遲（ms）、FLOPs、參數量、GPU 記憶體峰值（batch=32）
* 比較：Baseline / +POPMI / +POPMI+CE / G-CNN（若執行）
* ⬜️ Q1（PCam） / ⬜️ Q2（CheXpert）

---

## 6. 校準與不確定性

* 指標：**ECE**, **Brier score**，並報告可靠度圖（reliability diagram）
* ⬜️ R1（CheXpert） / ⬜️ R2（HAM10000）

---

## 7. 可解釋性（定性）

* 以 Grad-CAM/Score-CAM 對 `x` 與 `g·x`（幾何變換後）生成可視化，比較熱區是否一致；統計 IoU。
* ⬜️ S1（每資料集抽 100 張）

---

## 8. 論文圖表與表格輸出（最終交付）

* ⬜️ Table 1：四資料集主結果（平均±CI，星號標註顯著）
* ⬜️ Fig 2：POPMI/CE 框架圖（方法示意）
* ⬜️ Fig 3：角度魯棒性曲線（M 系列）
* ⬜️ Fig 4：消融熱力圖（E/F/G/H）
* ⬜️ Fig 5：效率-準確折衷（Q 系列）
* ⬜️ Fig 6：Grad-CAM 對齊示例（S 系列）
* ⬜️ 附錄：完整 seeds 結果與統計檢定明細

---

## 9. 執行備註與指令範例

* **核心主結果（例）**：

  ```bash
  # Baseline+RA+MixUp（無 POPMI/CE）
  python train_med.py --dataset chexpert --backbone resnet18 --ra --mixup --seed 0

  # +POPMI（固定參數）
  python train_med.py --dataset chexpert --backbone resnet18 --ra --mixup \
    --popmi --popmi_beta 0.5 --popmi_w 0 --popmi_k 0 --seed 0

  # +POPMI+CE（iters=3,rho=1）
  python train_med.py --dataset chexpert --backbone resnet18 --ra --mixup \
    --popmi --ce --ce_iters 3 --ce_rho 1.0 --seed 0
  ```
* **消融（例）**：

  ```bash
  # CE 參數網格
  for it in 1 3 5; do for rho in 0.5 1.0 2.0; do \
    python train_med.py --dataset pcam --backbone resnet18 --ra --mixup \
      --popmi --ce --ce_iters $it --ce_rho $rho --seed 0; \
  done; done

  # β 掃描
  for b in 0.0 0.25 0.5 0.75 1.0; do \
    python train_med.py --dataset medmnist --popmi --popmi_beta $b --ce --ce_iters 3 --ce_rho 1.0 --seed 0; \
  done
  ```

---

## 10. 失敗備案（若主結果不顯著）

* ⬜️ 聚焦單一模態（CheXpert 或 PCam）做更細 ablation（E/F/H/I/J/K 全覆蓋）
* ⬜️ 擴充生成式增強（diffusion-based）與偏差分析
* ⬜️ 對照 E(2)-CNN/G-CNN 並做效率-準確統計檢定

> **備註**：若先以右側最小腳本驗證，建議在 `MedMNIST v2 -> PneumoniaMNIST, PathMNIST` 兩子集先跑 `A1~A16 + E/F`，快速形成趨勢後再全量展開。
