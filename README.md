# POPMI
POPMI: Plug-and-Play Orthogonal Projections for Manifold-Invariant Medical Image Classification

核心模組：`POPMIProjection`（群不變雷諾茲投影） + 選用 `CE_PnP`（Consensus Equilibrium 包裝），可直接插入任意 **timm** backbone。

## 安裝依賴

```bash
pip install -r requirements.txt
```

## 快速 FakeData Demo

```bash
python main/popmi.py --arch resnet18 --ce --popmi_beta 0.5 --ce_iters 3 --ce_rho 1.0
```

## MedMNIST 訓練示例

```bash
python scripts/train_med.py \
	--dataset pneumoniamnist --arch resnet18 --epochs 10 --ce \
	--popmi_beta 0.5 --ce_iters 3 --ce_rho 1.0
```

開啟可學群權重與子抽樣：

```bash
python scripts/train_med.py --dataset pneumoniamnist --arch resnet18 \
	--ce --popmi_w --popmi_k 4 --popmi_beta 0.5
```

## 參數對應

| 功能 | 參數 |
|------|------|
| 啟用 CE | `--ce` |
| CE 迭代步 | `--ce_iters` |
| CE rho | `--ce_rho` |
| POPMI 殘差 β | `--popmi_beta` |
| 可學群權重 | `--popmi_w` |
| 群子抽樣 k | `--popmi_k` (0=全群) |

## 架構

```text
popmi/
	datasets/ (MedMNIST, CheXpert/HAM10000 placeholders, PCam placeholder)
	models/   (timm backbone 包裝 + POPMI/CE)
	engine/   (訓練與評估 loop)
	metrics/  (AUROC, Balanced Acc 等)
	statistics/ (預留: DeLong/McNemar/Bootstrap)
scripts/
	train_med.py
main/
	popmi.py (舊 monolith 之兼容 demo)
```

## 實驗矩陣（摘要）

| 類別 | 說明 | 主要腳本 | 備註 |
|------|------|---------|------|
| 主結果 | Baseline / +Aug / +POPMI / +POPMI+CE | `train_med.py` | 依資料集切換 loader |
| CE 消融 | iters × rho | `train_med.py` 迴圈 | 見 docs/exp.md E 區段 |
| β 掃描 | beta ∈ {0,0.25,...,1} | 同上 | F 區段 |
| w 可學 | 均勻 vs 可學 | `--popmi_w` | G |
| 子抽樣 | k ∈ {2,4,8} | `--popmi_k` | H |

完整清單詳見 `docs/exp.md`。

## 待補功能

- CheXpert / PCam 真實 dataset 讀取（目前為 placeholder）。
- 統計檢定：已提供 DeLong / McNemar / Holm-Bonferroni，Bootstrap 待補。
- 效率分析：`scripts/profile_model.py` (FLOPs / latency / 參數 / CUDA peak memory)。

## License

MIT
