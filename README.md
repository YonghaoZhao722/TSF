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


## 📊 Code Explanation

### `train_source_forest.py`
- Trains the **source forest (RSF)** and saves `source_forest.pkl`.
- Computes feature usage frequencies and saves them in `dp.csv`.
- Key Functions:
  - `train_source_forest()` - Trains and saves the RSF model.
  - `calculate_and_write_total_feature_probability()` - Computes feature probability.

### `target_forest_finetune.py`
- Loads `source_forest.pkl` and **transfers** forest structures.
- Fine-tunes the model on a **small target dataset**.
- Uses **10-fold cross-validation** to compute **CTD scores**.

### `dp_based.py`
- Uses **Depth Probability (DP) method** to train RSF.
- Performs **10-fold cross-validation** on the WCH dataset.
- Computes **CTD scores**.

### `calculate_dp.py`
- Computes **feature probability distributions** for target forest transfer.
- Key Functions:
  - `save_node_features()` - Extracts feature distributions per tree.
  - `calculate_feature_probability()` - Computes feature probability at different depths.
  - `calculate_and_write_total_feature_probability()` - Saves overall feature probability to CSV.

---

## 📊 Example Results

Example Output:
```
RSF model saved at rsf_models/source_forest.pkl
0.8123
0.8291
0.8045
...
Average CTD: 0.8216
```

---

## 📝 Future Improvements
- **Upload public datasets**.
- **Integration of deep learning transfer methods (e.g., DeepSurv, DeepHit)**.

---

## 🔥 Citation

If you use this code in your research, please cite the following paper:
(waiting for arxiv)

---
### 📩 Contact
For questions or suggestions, please reach out to:
- **Author**: Yonghao Zhao
- **Email**: sc22yz3@leeds.ac.uk

