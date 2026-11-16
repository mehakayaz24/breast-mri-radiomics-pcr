"""
Advanced machine-learning models on the radiomics features (I-SPY2/ACRIN, 50 patients).

Models: Elastic-Net Logistic Regression, Neural Network (MLP), Extra Trees, Gradient Boosting,
        Histogram Gradient Boosting, and a Stacking ensemble (LR + SVM + RF + XGBoost -> logistic meta-model).

Same fair validation as train_multi_models.py: repeated stratified 5-fold cross-validation at patient level,
every learning step inside the training folds, optional mask-perturbation augmentation of the training folds only
(if the robustness_extract.py output exists).

Put this file in the same folder as train_multi_models.py and train_models.py, then run it.
"""
import os
import train_multi_models as tmm
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# ============================ EDIT THESE ============================
OUT_DIR = r"C:\Users\ntech\Desktop\retrain_126\results_advanced_126"
# ====================================================================

SEED = tmm.SEED
ADVANCED_NAMES = ["ElasticNetLR", "NeuralNetMLP", "ExtraTrees", "GradientBoosting", "HistGradientBoosting", "Stacking"]
_classic = tmm.base_model


def advanced_model(name):
    if name == "ElasticNetLR":
        return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5,
                                  class_weight="balanced", max_iter=5000, random_state=SEED)
    if name == "NeuralNetMLP":
        return MLPClassifier(hidden_layer_sizes=(16,), alpha=0.05, learning_rate_init=0.005, max_iter=1500,
                             random_state=SEED)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=300, max_depth=3, min_samples_leaf=3, class_weight="balanced",
                                    random_state=SEED, n_jobs=1)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8,
                                          random_state=SEED)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=100, l2_regularization=1.0,
                                              min_samples_leaf=5, class_weight="balanced", random_state=SEED)
    if name == "Stacking":
        return StackingClassifier(
            estimators=[("lr", _classic("LogisticRegression")), ("svm", _classic("SVM")),
                        ("rf", _classic("RandomForest")), ("xgb", _classic("XGBoost"))],
            final_estimator=LogisticRegression(C=0.5, max_iter=2000), cv=5, stack_method="predict_proba")
    return _classic(name)


tmm.base_model = advanced_model
tmm.MODEL_NAMES = ADVANCED_NAMES
tmm.OUT_DIR = OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

if __name__ == "__main__":
    tmm.main()
