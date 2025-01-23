# Transfer Survival Forest (TSF)

## 📌 Project Overview

**Transfer Survival Forest (TSF)** is a method designed for **small-sample survival analysis**. It is based on **Random Survival Forest (RSF)** and utilizes **transfer learning** to improve survival prediction accuracy. This project implements TSF training, feature probability calculation, and target forest fine-tuning.

---

## 📂 Code Structure

```bash
TSF/
│── rsf_models/                # Directory to store trained RSF models
│── data/                      # Directory to store datasets (SEER as an example)
│   ├── SEER.csv               # Pretrain data example
│── source_forest.py           # Trains the source forest and saves the model
│── target_forest_finetune.py  # Transfers and fine-tunes the target forest
│── dp_based.py                # DP-based target forest training
│── calculate_dp.py            # Computes feature probabilities
│── model/
│   ├── TransferSurvivalForest.py  # Core TSF implementation
│   ├── TransferTree.py  # Core TSF implementation
│   ├── methods.py                 # Preprocessing and utility functions
│── global_names.py               # Global variables
│── README.md                     # Project documentation
```

---

## 🚀 Installation & Setup

### 1️⃣ Install Dependencies
Run the following command to install necessary Python libraries:
```bash
pip install -r requirements.txt
```

### 2️⃣ Data Preparation
- The **pretrained dataset** and **target dataset** must be aligned (i.e., they must have the same number of features and feature names).
- The repository provides a pretrain data sample (data/SEER.csv)
- Please put your training data in the folder `data`.
---

## 🎯 Usage

### 1️⃣ Train Source Forest
```bash
python train_source_forest.py
```
- This script:
  - Loads the WCH dataset.
  - Trains the **Random Survival Forest (RSF)** (Source Forest).
  - Saves the trained model in `rsf_models/source_forest.pkl`.
  - Computes feature probability and saves it in `dp.csv`.

### 2️⃣ Transfer & Fine-tune Target Forest
```bash
python target_forest_finetune.py
```
- This script:
  - Loads the pretrained `source_forest.pkl`.
  - Fine-tunes the model on the target dataset.
  - Uses **10-fold cross-validation** for evaluation.
  - Outputs the **Concordance Index (CTD)** score.

### 3️⃣ DP-Based Target Forest Training
```bash
python dp_based.py
```
- This script:
  - Computes and applies **Depth Probability (DP) method**.
  - Trains the RSF with transferred structures.
  - Evaluates the model using **10-fold cross-validation** and computes CTD scores.



---

## 📝 Future Improvements
- **Upload public datasets**.
- **Integration of deep learning transfer methods (e.g., DeepSurv, DeepHit)**.

---

## 🔥 Citation

If you use this code in your research, please cite the following paper:
[Zhao, Y., Li, C., Shu, C., Wu, Q., Li, H., Xu, C., Li, T., Wang, Z., Luo, Z., & He, Y. (2025). Tackling small sample survival analysis via transfer learning: A study of colorectal cancer prognosis. arXiv preprint arXiv:2501.12421. https://arxiv.org/abs/2501.12421]

---
### 📩 Contact
For questions or suggestions, please reach out to:
- **Author**: Yonghao Zhao
- **Email**: sc22yz3@leeds.ac.uk

