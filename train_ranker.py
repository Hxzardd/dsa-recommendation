import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

BASE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["bkt_mastery_avg","hlr_urgency_avg","p_recall_avg","half_life_avg",
    "days_since_last_review","difficulty_score","acceptance_rate","similarity_score",
    "variety_score","topic_overlap","pool_needs_attention","pool_weak_topic","pool_current_topic"]

def load(path):
    rows = [json.loads(l) for l in open(path)]
    X = pd.DataFrame([[r[k] for k in FEATURES] for r in rows], columns=FEATURES)
    y = np.array([r["relevance"] for r in rows], dtype=float)
    return X, y

def main():
    X, y = load(os.path.join(BASE, "lightgbm_dataset.jsonl"))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMRegressor(objective="regression", n_estimators=400, learning_rate=0.05,
        num_leaves=31, min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1)
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], eval_metric="l2",
              callbacks=[lgb.early_stopping(30, verbose=False)])
    preds = model.predict(Xte)
    print(f"RMSE {mean_squared_error(yte,preds)**0.5:.4f}  R2 {r2_score(yte,preds):.4f}  Spearman {spearmanr(yte,preds)[0]:.4f}")
    for name, gain in sorted(zip(FEATURES, model.booster_.feature_importance(importance_type="gain")), key=lambda x:-x[1]):
        print(f"  {name:26s}{gain:12.1f}")
    model.booster_.save_model(os.path.join(BASE, "ranker_model.txt"))
    print("Model saved to ranker_model.txt")

if __name__ == "__main__":
    main()
    