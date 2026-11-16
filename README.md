# Radiomic Features of Breast MRI as Predictors of NACT Efficacy

Final Year Project, B.E. Biomedical Engineering, MUET Jamshoro (2026)
By Mehak (22BM001) and Ayesha (22BM011). Supervisor: Engr. Rabia Chandio. Co-supervisor: Dr. Amirita Dewani.

Can radiomic features of the baseline breast DCE-MRI predict pathological complete response (pCR) to neoadjuvant chemotherapy in HER2-positive and triple-negative breast cancer?

> Research prototype only. Not a medical device and not for clinical use.

## Data
Public, de-identified data from The Cancer Imaging Archive (TCIA): I-SPY2 and ACRIN-6698 (126 development patients, 12 unseen test patients) and I-SPY1 (19 patients, early check). No patient images or data are stored in this repository.

## What the files do
| Step | Files |
|---|---|
| Data preparation | `ispy2_to_nrrd*.py`, `download_new68_colab.py` |
| Feature extraction (107 PyRadiomics features) | `extract_dev56.py`, `extract_all_70.py`, `merge_features_126.py` |
| Modelling (12 models, nested CV, permutation test, final models) | `train_models.py`, `train_multi_models.py`, `train_advanced_models.py`, `train_nested_tuning.py`, `train_final_models*.py`, `train_subtype_126.py`, `train_robust_126.py` |
| Robustness to mask changes (ICC) | `robustness_extract_126.py`, `robustness_analysis.py` |
| CNN experiment (Google Colab) | `make_patches_126.py`, `cnn_colab_126.py` |
| Prototype app (Streamlit) | `app_mri_126.py`, `autoseg*.py`, `dicom_loader.py`, `config.toml` |
| Early checks on unseen patients | `predict_*.py` |

## Setup
```
pip install -r requirements.txt
streamlit run app_mri_126.py
```
File paths inside the scripts point to the authors' computer and must be edited before use. Random seed is 42.

## Main results (126 development patients)
- Best cross-validated AUC 0.678 (elastic-net logistic regression, permutation p = 0.040)
- Tumor volume alone: AUC 0.691; clinical variables: no signal
- Unseen test (12 patients): accuracy 42%, AUC 0.44
- 66% of features had ICC >= 0.75 across mask perturbations
