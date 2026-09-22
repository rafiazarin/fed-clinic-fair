# FedClinic-Fair

**Transfer-Assisted Personalized Federated Learning for Client-Fair Medical Image Classification**

Federated learning lets several hospitals train one model without sharing patient data. But when each hospital's data looks different, a model can score well *on average* while failing some hospitals badly. This project asks: **can lightweight personalization lift the weakest clients without hurting the average?**

📄 [Read the paper](paper/FedClinic_Fair_Paper.pdf) · 📓 [Notebook](FedClinic_Fair_Final.ipynb)

---

## Key results

We compare 7 federated learning methods on 3 MedMNIST datasets with 5 simulated clients under non-IID (Dirichlet, α = 0.3) data splits. The main metric is **bottom-two Macro-F1**: the average score of the two weakest clients.

| Method | PneumoniaMNIST | BreastMNIST | BloodMNIST |
|---|---|---|---|
| FedAvg (baseline) | 0.4870 | 0.4023 | 0.5034 |
| Head-FT | 0.5550 | **0.4776** | 0.5833 |
| **WarmFedPer** | **0.5874** | 0.4228 | 0.5845 |
| ProxHead | 0.5676 | 0.4228 | **0.5862** |

*Mean across fresh confirmation seeds (10 for PneumoniaMNIST, 5 for the others). Full tables, including mean and worst-client scores for all 7 methods, are in the paper.*

**What we found:**

- **WarmFedPer** (starting personalization from a trained FedAvg model) had the best average rank across datasets. It raised the weakest clients' Macro-F1 by **+0.10** on PneumoniaMNIST and **+0.08** on BloodMNIST over FedAvg. The BreastMNIST difference was within seed noise, so we don't count it.
- **No method won everywhere.** WarmFedPer, Head-FT and ProxHead were within half a rank of each other.
- **Accuracy can hide failures.** On BloodMNIST, Ditto reached 89% accuracy but had the *lowest* Macro-F1 (0.50) of all seven methods, because it ignored minority classes.
- **Raising the floor ≠ narrowing the spread.** ProxHead made clients the most uniform (best-to-worst gap 0.375 → 0.267 on PneumoniaMNIST) without lifting the weakest client highest.

---

## Methods compared

| Method | What it does |
|---|---|
| FedAvg | One global model; all parameters averaged |
| FedProx-style | FedAvg + proximal penalty on all parameters |
| FedPer | Shared adapter, each client keeps its own classifier head |
| Head-FT | Freeze adapter after FedAvg, fine-tune each client's head |
| Ditto-style | Full personalized model per client, pulled toward the global one |
| **WarmFedPer** | Our ablation: FedPer started from the trained FedAvg checkpoint |
| **ProxHead** | Our extension: WarmFedPer + proximal penalty on the head only |

## Setup

- **Backbone:** frozen ImageNet-pretrained ResNet-18 (512-d features, cached once)
- **Trainable part:** small adapter (512 → 128, LayerNorm, ReLU, Dropout) + linear head
- **Training:** 20 global rounds, AdamW, class-weighted cross-entropy; +10 personalization rounds for WarmFedPer and ProxHead
- **Tuning:** hyperparameters chosen on a separate development seed using validation data only
- **Extra checks:** heterogeneity sweep (α = 0.1, 0.3, 1.0), representation sanity check, calibration (ECE), client-level diagnostics

## How to run

The notebook runs on Google Colab (GPU recommended) or locally.

```bash
pip install -r requirements.txt
jupyter notebook FedClinic_Fair_Final.ipynb
```

On Colab it saves outputs to Google Drive; locally it falls back to a local folder. MedMNIST datasets download automatically.

## Limitations

- The 5 clients are **simulated partitions**, not real hospitals.
- Features are computed centrally, so this simulates federated optimization, not a full decentralized system.
- No formal privacy mechanism (secure aggregation, differential privacy) is implemented.
- "Client-fair" here means weaker clients aren't left behind. It is **not** demographic fairness.
- MedMNIST is a research benchmark and not intended for clinical use.

## Team

Course project for **CSE437 Data Science**, BRAC University (Summer 2026).

- Ahetasham Shifat
- Ashique Anjam Rupam
- Rafia Zarin Alisha

**Supervisors:** Md. Sabbir Ahmed, Md. Golam Rabiul Alam
