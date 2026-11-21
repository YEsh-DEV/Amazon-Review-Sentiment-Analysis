"""
Pipeline for E-Commerce Sentiment Analysis: Linear SVM vs. Multinomial Naïve Bayes
Dataset: Amazon Review Polarity (12,000 balanced reviews: 8k Train / 2k Val / 2k Test)
Style: Ponytail Protocol - minimal, idiomatic, necessary code only.
"""

import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
DATA_DIR = "Dataset"
RES_DIR = "Results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Dataset Loading & Partitioning (12,000 Balanced Reviews)
# -----------------------------------------------------------------------------
def prepare_data():
    train_path = os.path.join(DATA_DIR, "train_sample.csv")
    val_path = os.path.join(DATA_DIR, "val_sample.csv")
    test_path = os.path.join(DATA_DIR, "test_sample.csv")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print("Loading cached datasets from CSV...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        return train_df, val_df, test_df

    print("Extracting balanced 12,000 sample from Amazon Polarity...")
    src_parquet = "/tmp/amazon_test.parquet"
    if not os.path.exists(src_parquet):
        raise FileNotFoundError(f"{src_parquet} not found. Please download it first.")

    raw = pd.read_parquet(src_parquet)
    # Combine title and content
    raw["text"] = raw["title"].fillna("") + ". " + raw["content"].fillna("")
    raw["text"] = raw["text"].str.strip()

    # Stratified balance: 6,000 negative (0) and 6,000 positive (1)
    neg = raw[raw["label"] == 0].sample(n=6000, random_state=RANDOM_STATE)
    pos = raw[raw["label"] == 1].sample(n=6000, random_state=RANDOM_STATE)

    # Split: Train (4k each = 8k), Val (1k each = 2k), Test (1k each = 2k)
    train_df = pd.concat([neg.iloc[:4000], pos.iloc[:4000]]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    val_df = pd.concat([neg.iloc[4000:5000], pos.iloc[4000:5000]]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df = pd.concat([neg.iloc[5000:6000], pos.iloc[5000:6000]]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    for df, path in [(train_df, train_path), (val_df, val_path), (test_df, test_path)]:
        df[["text", "label"]].to_csv(path, index=False)

    print(f"Dataset partitions saved to {DATA_DIR}/ (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})")
    return train_df[["text", "label"]], val_df[["text", "label"]], test_df[["text", "label"]]

# -----------------------------------------------------------------------------
# 2. Text Preprocessing (Preserving Negations)
# -----------------------------------------------------------------------------
CONTRACTIONS = {
    r"won\'t": "will not", r"can\'t": "cannot", r"n\'t": " not",
    r"\'re": " are", r"\'s": " is", r"\'d": " would",
    r"\'ll": " will", r"\'t": " not", r"\'ve": " have", r"\'m": " am"
}

def clean_text(text: str) -> str:
    text = str(text).lower()
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"<[^>]+>", " ", text)               # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text) # URLs
    text = re.sub(r"[^a-z\s]", " ", text)              # Keep alphabet only
    return re.sub(r"\s+", " ", text).strip()

# -----------------------------------------------------------------------------
# 3. Exploratory Data Analysis (EDA) Plots
# -----------------------------------------------------------------------------
def plot_eda(train_df):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = train_df["label"].value_counts().rename(index={0: "Negative (0)", 1: "Positive (1)"})
    bars = ax.bar(counts.index, counts.values, color=["#e74c3c", "#2ecc71"], width=0.5, edgecolor="black")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 400, f"{bar.get_height():,}",
                ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    ax.set_title("Training Set Class Distribution (Balanced)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Reviews", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "class_distribution.png"), dpi=300)
    plt.close()

    # Review length distribution
    train_df["word_count"] = train_df["text"].apply(lambda t: len(str(t).split()))
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=train_df, x="word_count", hue="label", palette={0: "#e74c3c", 1: "#2ecc71"},
                 kde=True, bins=50, ax=ax, element="step", common_norm=False)
    ax.set_title("Review Word Count Distribution by Sentiment", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Word Count", fontsize=11)
    ax.set_ylabel("Density / Count", fontsize=11)
    ax.legend(labels=["Positive", "Negative"], title="Sentiment")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "review_length_dist.png"), dpi=300)
    plt.close()
    print("EDA plots saved to Results/")

# -----------------------------------------------------------------------------
# 4. Main Training, Tuning, and Evaluation Pipeline
# -----------------------------------------------------------------------------
def main():
    t_start = time.time()
    train_df, val_df, test_df = prepare_data()
    plot_eda(train_df)

    print("Preprocessing text...")
    X_train_raw = train_df["text"].apply(clean_text).values
    y_train = train_df["label"].values
    X_val_raw = val_df["text"].apply(clean_text).values
    y_val = val_df["label"].values
    X_test_raw = test_df["text"].apply(clean_text).values
    y_test = test_df["label"].values

    print("Fitting TF-IDF Vectorizer on training data...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=25000,
        min_df=2,
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"TF-IDF Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    # Hyperparameter Tuning on Validation Set
    print("\nTuning Linear SVM (C penalty)...")
    best_c, best_svm_f1 = 1.0, 0
    for c in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        m = LinearSVC(C=c, random_state=RANDOM_STATE, dual="auto")
        m.fit(X_train, y_train)
        preds = m.predict(X_val)
        f1 = f1_score(y_val, preds, average="macro")
        if f1 > best_svm_f1:
            best_svm_f1, best_c = f1, c
    print(f"Selected SVM C={best_c} (Val Macro-F1: {best_svm_f1:.4f})")

    print("Tuning Multinomial Naïve Bayes (alpha smoothing)...")
    best_alpha, best_nb_f1 = 1.0, 0
    for a in [0.01, 0.1, 0.5, 1.0, 2.0]:
        m = MultinomialNB(alpha=a)
        m.fit(X_train, y_train)
        preds = m.predict(X_val)
        f1 = f1_score(y_val, preds, average="macro")
        if f1 > best_nb_f1:
            best_nb_f1, best_alpha = f1, a
    print(f"Selected MNB alpha={best_alpha} (Val Macro-F1: {best_nb_f1:.4f})")

    # Train Final Models & Measure Training Time
    print("\nTraining final models on Train set...")
    # Multinomial NB
    t0 = time.perf_counter()
    nb = MultinomialNB(alpha=best_alpha)
    nb.fit(X_train, y_train)
    nb_fit_time = (time.perf_counter() - t0) * 1000

    # Linear SVM: fit base model and calibrated probability estimator
    t0 = time.perf_counter()
    base_svm = LinearSVC(C=best_c, random_state=RANDOM_STATE, dual="auto")
    base_svm.fit(X_train, y_train)
    svm_fit_time = (time.perf_counter() - t0) * 1000

    # Calibrate SVM for true posterior probabilities (using 5-fold CV on Train)
    cal_svm = CalibratedClassifierCV(estimator=LinearSVC(C=best_c, random_state=RANDOM_STATE, dual="auto"), method="sigmoid", cv=5)
    cal_svm.fit(X_train, y_train)

    # Measure Inference Latency (ms per 1,000 samples)
    t0 = time.perf_counter()
    _ = nb.predict_proba(X_test)
    nb_infer_time = ((time.perf_counter() - t0) / len(y_test)) * 1000 * 1000

    t0 = time.perf_counter()
    _ = cal_svm.predict_proba(X_test)
    svm_infer_time = ((time.perf_counter() - t0) / len(y_test)) * 1000 * 1000

    # Validation Threshold Tuning Sweep
    print("Performing threshold optimization sweep on Validation set...")
    val_probs_nb = nb.predict_proba(X_val)[:, 1]
    val_probs_svm = cal_svm.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 81)
    nb_thresh_f1 = [f1_score(y_val, (val_probs_nb >= th).astype(int), average="macro") for th in thresholds]
    svm_thresh_f1 = [f1_score(y_val, (val_probs_svm >= th).astype(int), average="macro") for th in thresholds]

    best_nb_th = thresholds[np.argmax(nb_thresh_f1)]
    best_svm_th = thresholds[np.argmax(svm_thresh_f1)]
    print(f"Optimal Thresholds -> MNB: {best_nb_th:.2f} (F1: {max(nb_thresh_f1):.4f}), SVM: {best_svm_th:.2f} (F1: {max(svm_thresh_f1):.4f})")

    # Plot Threshold Tuning
    plt.figure(figsize=(7, 4.5))
    plt.plot(thresholds, svm_thresh_f1, label=f"Linear SVM (Best th={best_svm_th:.2f})", color="#2980b9", lw=2)
    plt.plot(thresholds, nb_thresh_f1, label=f"Multinomial NB (Best th={best_nb_th:.2f})", color="#e67e22", lw=2, linestyle="--")
    plt.axvline(0.5, color="gray", linestyle=":", label="Default th=0.50")
    plt.title("Validation Macro-F1 across Decision Thresholds", fontsize=12, fontweight="bold")
    plt.xlabel("Positive Class Probability Threshold", fontsize=11)
    plt.ylabel("Validation Macro-F1 Score", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "threshold_tuning.png"), dpi=300)
    plt.close()

    # Final Evaluation on Held-Out Test Set (2,000 samples)
    test_probs_nb = nb.predict_proba(X_test)[:, 1]
    test_probs_svm = cal_svm.predict_proba(X_test)[:, 1]

    # Standard (0.5 threshold) predictions
    nb_preds = (test_probs_nb >= 0.5).astype(int)
    svm_preds = (test_probs_svm >= 0.5).astype(int)

    # Tuned threshold predictions
    nb_preds_tuned = (test_probs_nb >= best_nb_th).astype(int)
    svm_preds_tuned = (test_probs_svm >= best_svm_th).astype(int)

    # Metrics computation
    metrics = {
        "Metric": [
            "Accuracy", "Precision (Positive)", "Recall (Positive)",
            "F1-Score (Positive)", "Macro-F1", "ROC-AUC",
            "Tuned Threshold Macro-F1", "Optimal Threshold",
            "Fit Time (ms)", "Inference Latency (ms/1k reviews)"
        ],
        "Linear SVM": [
            accuracy_score(y_test, svm_preds),
            precision_score(y_test, svm_preds),
            recall_score(y_test, svm_preds),
            f1_score(y_test, svm_preds),
            f1_score(y_test, svm_preds, average="macro"),
            roc_auc_score(y_test, test_probs_svm),
            f1_score(y_test, svm_preds_tuned, average="macro"),
            best_svm_th,
            svm_fit_time,
            svm_infer_time
        ],
        "Multinomial Naïve Bayes": [
            accuracy_score(y_test, nb_preds),
            precision_score(y_test, nb_preds),
            recall_score(y_test, nb_preds),
            f1_score(y_test, nb_preds),
            f1_score(y_test, nb_preds, average="macro"),
            roc_auc_score(y_test, test_probs_nb),
            f1_score(y_test, nb_preds_tuned, average="macro"),
            best_nb_th,
            nb_fit_time,
            nb_infer_time
        ]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(RES_DIR, "metrics_summary.csv"), index=False)
    print("\n--- Held-Out Test Set Results Summary ---")
    print(metrics_df.to_string(index=False))

    # Confusion Matrices Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    cm_svm = confusion_matrix(y_test, svm_preds)
    cm_nb = confusion_matrix(y_test, nb_preds)

    for ax, cm, title, cmap in zip(axes, [cm_svm, cm_nb], ["Linear SVM", "Multinomial Naïve Bayes"], ["Blues", "Oranges"]):
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
                    xticklabels=["Pred Neg", "Pred Pos"], yticklabels=["Actual Neg", "Actual Pos"],
                    annot_kws={"size": 13, "weight": "bold"})
        acc = (cm[0, 0] + cm[1, 1]) / len(y_test)
        ax.set_title(f"{title}\nAccuracy: {acc:.1%}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "confusion_matrices.png"), dpi=300)
    plt.close()

    # ROC Curves Plot
    fpr_svm, tpr_svm, _ = roc_curve(y_test, test_probs_svm)
    fpr_nb, tpr_nb, _ = roc_curve(y_test, test_probs_nb)
    auc_svm = roc_auc_score(y_test, test_probs_svm)
    auc_nb = roc_auc_score(y_test, test_probs_nb)

    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr_svm, tpr_svm, label=f"Linear SVM (AUC = {auc_svm:.4f})", color="#2980b9", lw=2.2)
    plt.plot(fpr_nb, tpr_nb, label=f"Multinomial NB (AUC = {auc_nb:.4f})", color="#e67e22", lw=2.2, linestyle="--")
    plt.plot([0, 1], [0, 1], "k:", label="Random Classifier (AUC = 0.5000)", alpha=0.7)
    plt.title("Held-Out Test Set ROC Curves", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "roc_curves.png"), dpi=300)
    plt.close()

    # Model Performance Comparison Bar Chart
    comp_metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    svm_vals = [accuracy_score(y_test, svm_preds), precision_score(y_test, svm_preds),
                recall_score(y_test, svm_preds), f1_score(y_test, svm_preds), auc_svm]
    nb_vals = [accuracy_score(y_test, nb_preds), precision_score(y_test, nb_preds),
               recall_score(y_test, nb_preds), f1_score(y_test, nb_preds), auc_nb]

    x = np.arange(len(comp_metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rects1 = ax.bar(x - width/2, [v * 100 for v in svm_vals], width, label="Linear SVM", color="#2980b9", edgecolor="black")
    rects2 = ax.bar(x + width/2, [v * 100 for v in nb_vals], width, label="Multinomial NB", color="#e67e22", edgecolor="black")
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Test Performance Comparison: Linear SVM vs. Multinomial NB", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_metrics, fontsize=11)
    ax.set_ylim(60, 100)
    ax.legend(loc="lower right")
    for r in rects1 + rects2:
        h = r.get_height()
        ax.annotate(f"{h:.1f}%", (r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "model_comparison.png"), dpi=300)
    plt.close()

    # Latency Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
    ax1.bar(["Multinomial NB", "Linear SVM"], [nb_fit_time, svm_fit_time], color=["#e67e22", "#2980b9"], edgecolor="black", width=0.5)
    ax1.set_title("Training Fit Time (ms)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Milliseconds (Lower is Faster)")
    for i, v in enumerate([nb_fit_time, svm_fit_time]):
        ax1.text(i, v + 2, f"{v:.1f} ms", ha="center", fontweight="bold")

    ax2.bar(["Multinomial NB", "Linear SVM"], [nb_infer_time, svm_infer_time], color=["#e67e22", "#2980b9"], edgecolor="black", width=0.5)
    ax2.set_title("Inference Latency (ms / 1k reviews)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Milliseconds (Lower is Faster)")
    for i, v in enumerate([nb_infer_time, svm_infer_time]):
        ax2.text(i, v + 0.5, f"{v:.1f} ms", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "latency_comparison.png"), dpi=300)
    plt.close()

    # Top Feature Importance Plot (Interpretability)
    feature_names = np.array(vectorizer.get_feature_names_out())
    svm_coefs = base_svm.coef_.flatten()
    top_pos_idx = np.argsort(svm_coefs)[-12:]
    top_neg_idx = np.argsort(svm_coefs)[:12]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    top_terms = list(feature_names[top_neg_idx]) + list(feature_names[top_pos_idx])
    top_weights = list(svm_coefs[top_neg_idx]) + list(svm_coefs[top_pos_idx])
    bar_colors = ["#e74c3c"] * 12 + ["#2ecc71"] * 12
    ax.barh(top_terms, top_weights, color=bar_colors, edgecolor="black")
    ax.set_title("Top Linear SVM Features (Positive vs. Negative Sentiment)", fontsize=12, fontweight="bold")
    ax.set_xlabel("SVM Model Coefficient (Weight)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, "feature_importance.png"), dpi=300)
    plt.close()

    # Interactive Sentiment Prediction Demonstration
    print("\n--- Live Custom Review Predictions ---")
    sample_tests = [
        "Amazing sound quality and the battery lasts for days! Highly recommended.",
        "Terrible product. Broke after two days of use and customer support was completely useless.",
        "It is not bad for the price, works reasonably well.",
        "Do not waste your money on this garbage."
    ]
    for review in sample_tests:
        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])
        svm_p = cal_svm.predict_proba(vec)[0][1]
        nb_p = nb.predict_proba(vec)[0][1]
        print(f"\nReview: \"{review}\"")
        print(f"  Linear SVM -> Pred: {'Positive' if svm_p >= best_svm_th else 'Negative'} (Prob: {svm_p:.3f})")
        print(f"  Multinomial NB -> Pred: {'Positive' if nb_p >= best_nb_th else 'Negative'} (Prob: {nb_p:.3f})")

    print(f"\nPipeline successfully completed in {time.time() - t_start:.2f} seconds!")
    print(f"All figures and metrics saved to '{RES_DIR}/'")

if __name__ == "__main__":
    main()
