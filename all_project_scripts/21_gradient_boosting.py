#!/usr/bin/env python3
"""
Phase 5, Step 4: Gradient Boosting Baselines (XGBoost and LightGBM).

Uses Optuna for hyperparameter optimization.
Strategy: Very small HPO subsample (5K), aggressive early stopping, reduced bins.
Final retraining on full 800K dataset with best hyperparameters.
"""

import sys
import os
import json
import time
import gc
import argparse
import numpy as np
import h5py
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_THREADS = min(os.cpu_count() or 1, 40)


def load_data(h5_path):
    print("Loading data from HDF5...")
    t0 = time.time()
    with h5py.File(h5_path, 'r') as f:
        X_train_kmer = f['train/kmer_features'][:]
        X_train_asm = f['train/assembly_features'][:]
        y_train = f['train/labels'][:]
        X_val_kmer = f['val/kmer_features'][:]
        X_val_asm = f['val/assembly_features'][:]
        y_val = f['val/labels'][:]

    X_train = np.concatenate([X_train_kmer, X_train_asm], axis=1).astype(np.float32)
    X_val = np.concatenate([X_val_kmer, X_val_asm], axis=1).astype(np.float32)
    del X_train_kmer, X_train_asm, X_val_kmer, X_val_asm
    gc.collect()

    rng = np.random.RandomState(42)
    idx_t = rng.choice(len(X_train), 5000, replace=False)
    idx_v = rng.choice(len(X_val), 3000, replace=False)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  HPO subsample: 5,000 train / 3,000 val")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    return (X_train, y_train, X_val, y_val,
            X_train[idx_t].copy(), y_train[idx_t].copy(),
            X_val[idx_v].copy(), y_val[idx_v].copy())


def metrics(y_true, y_pred):
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
    }


def train_lgbm(data, n_trials, output_dir):
    X_train, y_train, X_val, y_val, Xs, ys, Xvs, yvs = data
    print("\n" + "=" * 70)
    print("LightGBM Optuna HPO")
    print("=" * 70)

    def objective(trial):
        p = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 100),
            'max_depth': trial.suggest_int('max_depth', 4, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
            'n_estimators': 200,
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'n_jobs': N_THREADS, 'random_state': 42, 'verbose': -1,
            'subsample_freq': 1, 'max_bin': 63,
        }
        s = 0.0
        for i, w in [(0, 1.0), (1, 1.5)]:
            m = lgb.LGBMRegressor(**p)
            m.fit(Xs, ys[:, i], eval_set=[(Xvs, yvs[:, i])],
                  callbacks=[lgb.log_evaluation(0), lgb.early_stopping(10, verbose=False)])
            s += w * mean_squared_error(yvs[:, i], m.predict(Xvs))
        return s

    t0 = time.time()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    ht = time.time() - t0
    print(f"\nHPO: {ht/60:.1f} min, best={study.best_value:.2f}")
    print(f"Params: {json.dumps(study.best_params, indent=2)}")

    bp = study.best_params.copy()
    bp.update({'n_estimators': 600, 'n_jobs': N_THREADS, 'random_state': 42,
               'verbose': -1, 'subsample_freq': 1, 'max_bin': 255})

    print("\nRetraining on full data...")
    results = {}
    for i, name in [(0, 'completeness'), (1, 'contamination')]:
        t0 = time.time()
        m = lgb.LGBMRegressor(**bp)
        m.fit(X_train, y_train[:, i], eval_set=[(X_val, y_val[:, i])],
              callbacks=[lgb.log_evaluation(50), lgb.early_stopping(30, verbose=True)])
        tt = time.time() - t0
        pred = np.clip(m.predict(X_val), 0, 100)
        met = metrics(y_val[:, i], pred)
        met.update({'train_time_s': tt, 'best_iteration': m.best_iteration_})
        results[name] = met
        print(f"  {name}: MAE={met['mae']:.4f} RMSE={met['rmse']:.4f} R2={met['r2']:.4f} ({tt:.0f}s, iter={m.best_iteration_})")
        mp = os.path.join(output_dir, f'lgbm_{name}.txt')
        m.booster_.save_model(mp)

    with open(os.path.join(output_dir, 'lgbm_optuna_study.json'), 'w') as f:
        json.dump({'best_params': study.best_params, 'best_value': study.best_value,
                    'n_trials': n_trials, 'hpo_time_min': ht/60, 'results': results}, f, indent=2)
    return results


def train_xgb(data, n_trials, output_dir):
    X_train, y_train, X_val, y_val, Xs, ys, Xvs, yvs = data
    print("\n" + "=" * 70)
    print("XGBoost Optuna HPO")
    print("=" * 70)

    def objective(trial):
        p = {
            'max_depth': trial.suggest_int('max_depth', 4, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),
            'n_estimators': 200,
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'tree_method': 'hist', 'device': 'cpu', 'nthread': N_THREADS,
            'random_state': 42, 'early_stopping_rounds': 10, 'max_bin': 64,
        }
        s = 0.0
        for i, w in [(0, 1.0), (1, 1.5)]:
            m = xgb.XGBRegressor(**p)
            m.fit(Xs, ys[:, i], eval_set=[(Xvs, yvs[:, i])], verbose=False)
            s += w * mean_squared_error(yvs[:, i], m.predict(Xvs))
        return s

    t0 = time.time()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    ht = time.time() - t0
    print(f"\nHPO: {ht/60:.1f} min, best={study.best_value:.2f}")
    print(f"Params: {json.dumps(study.best_params, indent=2)}")

    bp = study.best_params.copy()
    bp.update({'n_estimators': 600, 'tree_method': 'hist', 'device': 'cpu',
               'nthread': N_THREADS, 'random_state': 42, 'early_stopping_rounds': 30,
               'max_bin': 256})

    print("\nRetraining on full data...")
    results = {}
    for i, name in [(0, 'completeness'), (1, 'contamination')]:
        t0 = time.time()
        m = xgb.XGBRegressor(**bp)
        m.fit(X_train, y_train[:, i], eval_set=[(X_val, y_val[:, i])], verbose=100)
        tt = time.time() - t0
        pred = np.clip(m.predict(X_val), 0, 100)
        met = metrics(y_val[:, i], pred)
        met.update({'train_time_s': tt, 'best_iteration': int(getattr(m, 'best_iteration', -1))})
        results[name] = met
        print(f"  {name}: MAE={met['mae']:.4f} RMSE={met['rmse']:.4f} R2={met['r2']:.4f} ({tt:.0f}s)")
        m.save_model(os.path.join(output_dir, f'xgboost_{name}.json'))

    with open(os.path.join(output_dir, 'xgboost_optuna_study.json'), 'w') as f:
        json.dump({'best_params': study.best_params, 'best_value': study.best_value,
                    'n_trials': n_trials, 'hpo_time_min': ht/60, 'results': results}, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=50)
    args = parser.parse_args()

    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h5 = os.path.join(proj, 'data', 'features', 'magicc_features.h5')
    out = os.path.join(proj, 'models')
    os.makedirs(out, exist_ok=True)

    data = load_data(h5)
    print(f"\nFeatures: {data[0].shape[1]}, Trials: {args.n_trials}, Threads: {N_THREADS}")

    t0 = time.time()
    lgbm_res = train_lgbm(data, args.n_trials, out)
    lt = time.time() - t0
    print(f"LightGBM total: {lt/60:.1f} min")

    t0 = time.time()
    xgb_res = train_xgb(data, args.n_trials, out)
    xt = time.time() - t0
    print(f"XGBoost total: {xt/60:.1f} min")

    print("\n" + "=" * 70)
    print("GRADIENT BOOSTING RESULTS")
    print("=" * 70)
    for nm, res in [('LightGBM', lgbm_res), ('XGBoost', xgb_res)]:
        for t in ['completeness', 'contamination']:
            m = res[t]
            print(f"  {nm} {t}: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} R2={m['r2']:.4f}")

    with open(os.path.join(out, 'gradient_boosting_results.json'), 'w') as f:
        json.dump({'xgboost': xgb_res, 'lightgbm': lgbm_res,
                    'xgb_time_min': xt/60, 'lgbm_time_min': lt/60}, f, indent=2)


if __name__ == '__main__':
    main()
