"""One-shot: build v8_rf_cache.joblib so step4 skips HF reload."""

import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_lib import load_temporal_frames, prepare_xy, train_rf, project_root


def main():
    out_dir = os.path.join(project_root(), "results", "revision_v8")
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "v8_rf_cache.joblib")
    print("Building cache ->", cache)
    df_tr, df_te = load_temporal_frames(te_subsample=None)
    X_tr, y_tr, _ = prepare_xy(df_tr)
    X_te, y_te, lbl_te = prepare_xy(df_te)
    model = train_rf(X_tr, y_tr)
    joblib.dump(
        {"model": model, "X_te": X_te, "X_tr_cols": list(X_tr.columns), "lbl_te": lbl_te},
        cache,
    )
    print("Done. train=", len(X_tr), "test=", len(X_te))


if __name__ == "__main__":
    main()
