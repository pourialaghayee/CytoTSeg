#!/usr/bin/env python3
"""
================================================================================
NESTED CROSS-VALIDATION FOR CLL CLASSIFICATION
================================================================================
Modular Python script converted from Jupyter notebook.
Performs nested CV with hyperparameter tuning and saves all outputs to a 
structured folder.

Author: Converted from notebook
Date: February 2026
================================================================================
"""

import os
import sys
import time
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for all parameters"""

    # Paths
    INPUT_PKL = Path('/mnt/lustre/home/claassen/clala950/DC-TSeg/Data/ICellCNN/Train_tes_split_smaller_model_normal/Test_data_cyto2_finetuned_clean_SINGLE_unet_preds_d1b1_morphology.pkl')
    OUTPUT_ROOT_NAME = 'Classification_results_d1b1'  # <-- change to 'Classification_results_400_mbar_d2b1' for the other run

    # Model selection
    SELECTED_MODEL = 'Unet_preds'  # 'Ground_truth' or 'Unet_preds' or 'Unet_preds_CLL_focus_k1_5' etc.

    # Cross-validation parameters
    RANDOM_STATE = 42
    N_OUTER_FOLDS = 3 #5
    N_INNER_FOLDS = 2 #5
    K_VALUES = [5, 10, 15, 20, 25, 30, 50, 'max']

    # Optimization
    OPTIMIZATION_METRIC = 'auc'
    SECONDARY_METRIC = 'f1'
    AUC_TOLERANCE = 0.02

    # Plotting
    PLOT_CONFIG = {
        'figure_dpi': 300,
        'figure_size_single': (10, 6),
        'figure_size_grid': (16, 12),
        'font_size': 14,
        'title_size': 18,
        'label_size': 14,
        'tick_size': 12,
        'legend_size': 12,
        'line_width': 2,
        'marker_size': 8,
    }

    @classmethod
    def setup_output_directory(cls):
        """Create timestamped output directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = cls.INPUT_PKL.parent / f'{cls.OUTPUT_ROOT_NAME}_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @classmethod
    def setup_plotting(cls):
        """Apply global plotting configuration"""
        plt.rcParams.update({
            'font.size': cls.PLOT_CONFIG['font_size'],
            'axes.titlesize': cls.PLOT_CONFIG['title_size'],
            'axes.labelsize': cls.PLOT_CONFIG['label_size'],
            'xtick.labelsize': cls.PLOT_CONFIG['tick_size'],
            'ytick.labelsize': cls.PLOT_CONFIG['tick_size'],
            'legend.fontsize': cls.PLOT_CONFIG['legend_size'],
            'figure.titlesize': cls.PLOT_CONFIG['title_size'] + 4,
            'lines.linewidth': cls.PLOT_CONFIG['line_width'],
            'lines.markersize': cls.PLOT_CONFIG['marker_size'],
        })

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================

class DataLoader:
    """Handle data loading and feature aggregation"""

    @staticmethod
    def aggregate_morphology_stats(db, model_key, stats=('mean', 'median', 'std', 'cv')):
        """Aggregate morphology statistics across all patients."""
        morph_key = f'{model_key}_morphology'
        patient_ids = [pid for pid in db.keys()
                       if morph_key in db[pid] and isinstance(db[pid][morph_key], pd.DataFrame)]

        if not patient_ids:
            raise ValueError(f"No patients contained a DataFrame under key '{morph_key}'")

        first_df = None
        for pid in patient_ids:
            df = db[pid][morph_key]
            if isinstance(df, pd.DataFrame) and not df.empty:
                first_df = df
                break

        if first_df is None or first_df.empty:
            raise ValueError(f"All '{morph_key}' DataFrames are empty.")

        morph_cols = first_df.columns.tolist()
        if 'Area' not in morph_cols:
            raise ValueError("'Area' column not found in morphology DataFrame.")
        morph_cols = morph_cols[morph_cols.index('Area'):]

        results = {'Patient_ID': []}
        for col in morph_cols:
            for stat in stats:
                results[f'{col}_{stat}'] = []

        for pid in patient_ids:
            results['Patient_ID'].append(pid)
            df = db[pid][morph_key]

            if df is None or df.empty:
                for col in morph_cols:
                    for stat in stats:
                        results[f'{col}_{stat}'].append(np.nan)
                continue

            for col in morph_cols:
                col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if col_data.empty:
                    vals = {s: np.nan for s in stats}
                else:
                    vals = {}
                    for stat in stats:
                        if stat == 'mean':
                            vals[stat] = col_data.mean()
                        elif stat == 'median':
                            vals[stat] = col_data.median()
                        elif stat == 'std':
                            vals[stat] = col_data.std()
                        elif stat == 'cv':
                            mean_val = col_data.mean()
                            std_val = col_data.std()
                            vals[stat] = (std_val / mean_val) if mean_val != 0 else np.nan
                for stat in stats:
                    results[f'{col}_{stat}'].append(vals[stat])

        agg_df = pd.DataFrame(results).set_index('Patient_ID')
        return agg_df

    @staticmethod
    def prepare_data(df):
        """Create X (features) and y (labels) from aggregated morphology."""
        df = df.copy()
        patient_ids = df.index.astype(str).tolist()
        groups = [pid.split('_')[0] for pid in patient_ids]
        y = pd.Series([1 if g.upper().startswith('CLL') else 0 for g in groups],
                      index=df.index, name='Group')
        X = df.copy()
        return X, y

    @classmethod
    def load_and_prepare(cls, pkl_path, model_key):
        """Load pickle and prepare data"""
        print("="*80)
        print("LOADING DATA")
        print("="*80)

        with open(pkl_path, 'rb') as f:
            db = pickle.load(f)
        print(f"✓ Loaded database with {len(db)} patients")

        agg_df = cls.aggregate_morphology_stats(db, model_key, stats=('mean', 'median', 'std', 'cv'))
        print(f"✓ Aggregated shape: {agg_df.shape}")

        X, y = cls.prepare_data(agg_df)

        print(f"\nTotal samples: {len(X)}")
        print(f"Total features: {X.shape[1]}")
        print(f"\nClass distribution:")
        print(f"  Control (0):       {int((y==0).sum())} ({(y==0).mean()*100:.1f}%)")
        print(f"  Disease cases (1): {int((y==1).sum())} ({(y==1).mean()*100:.1f}%)")

        return X, y, db

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class ModelFactory:
    """Factory for creating model pipelines"""

    @staticmethod
    def get_model_pipeline(model_type, random_state=42):
        """Create sklearn pipeline for given model type."""
        if model_type == 'rf':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100, max_depth=4,
                    min_samples_split=3, min_samples_leaf=1,
                    class_weight='balanced',
                    random_state=random_state
                ))
            ])
        elif model_type == 'svm':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    kernel='rbf', probability=True, C=1.0,
                    class_weight='balanced',
                    random_state=random_state
                ))
            ])
        elif model_type == 'lr':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    random_state=random_state,
                    max_iter=1000,
                    class_weight='balanced'
                ))
            ])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# ============================================================================
# FEATURE SELECTION
# ============================================================================

class FeatureSelector:
    """Handle feature selection operations"""

    @staticmethod
    def select_features_for_k(X_train, y_train, X_test, k, feature_names):
        """
        Select top k features using SelectKBest on training data.
        Returns: X_train_reduced, X_test_reduced, selected_feature_names
        """
        k_actual = min(k, X_train.shape[1])
        selector = SelectKBest(f_classif, k=k_actual)
        X_train_reduced = selector.fit_transform(X_train, y_train)
        X_test_reduced = selector.transform(X_test)

        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]

        return X_train_reduced, X_test_reduced, selected_features

# ============================================================================
# NESTED CROSS-VALIDATION
# ============================================================================

class NestedCV:
    """Nested cross-validation implementation"""

    def __init__(self, config):
        self.config = config

    def run_inner_cv_for_k_selection(self, X_train, y_train, k_values, model_type):
        """Run inner CV to select optimal k."""
        feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]

        if not isinstance(X_train, np.ndarray):
            X_train = X_train.values
        if not isinstance(y_train, np.ndarray):
            y_train = y_train.values

        k_scores = {k: [] for k in k_values}
        inner_cv = StratifiedKFold(n_splits=self.config['n_inner_folds'],
                                   shuffle=True,
                                   random_state=self.config['random_state'])

        for k in k_values:
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
                X_inner_train = X_train[inner_train_idx]
                X_inner_val = X_train[inner_val_idx]
                y_inner_train = y_train[inner_train_idx]
                y_inner_val = y_train[inner_val_idx]

                X_inner_train_reduced, X_inner_val_reduced, _ = FeatureSelector.select_features_for_k(
                    X_inner_train, y_inner_train, X_inner_val, k, feature_names
                )

                model = ModelFactory.get_model_pipeline(model_type, self.config['random_state'])
                model.fit(X_inner_train_reduced, y_inner_train)

                y_pred = model.predict(X_inner_val_reduced)
                y_proba = model.predict_proba(X_inner_val_reduced)[:, 1] if hasattr(model, "predict_proba") else y_pred

                metrics = {
                    'accuracy': accuracy_score(y_inner_val, y_pred),
                    'f1': f1_score(y_inner_val, y_pred, zero_division=0),
                    'auc': roc_auc_score(y_inner_val, y_proba) if len(np.unique(y_inner_val)) > 1 else 0.5
                }

                k_scores[k].append(metrics)

        k_avg_scores = {}
        for k in k_values:
            k_avg_scores[k] = {
                'accuracy': np.mean([s['accuracy'] for s in k_scores[k]]),
                'f1': np.mean([s['f1'] for s in k_scores[k]]),
                'auc': np.mean([s['auc'] for s in k_scores[k]]),
            }

        primary_metric = self.config['optimization_metric']
        secondary_metric = self.config['secondary_metric']

        sorted_k = sorted(k_values, key=lambda k: k_avg_scores[k][primary_metric], reverse=True)
        best_k = sorted_k[0]
        best_score = k_avg_scores[best_k][primary_metric]

        candidates = [k for k in sorted_k if abs(k_avg_scores[k][primary_metric] - best_score) <= self.config['auc_tolerance']]

        if len(candidates) > 1:
            best_k = max(candidates, key=lambda k: k_avg_scores[k][secondary_metric])

        return best_k, k_avg_scores

    def run_nested_cv(self, model_type, model_name, X, y):
        """Run nested cross-validation with hyperparameter tuning."""
        print(f"\n{'='*80}")
        print(f"NESTED CV: {model_name}")
        print(f"{'='*80}")

        start_time = time.time()

        feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
        k_values = self.config['k_values_resolved']

        outer_cv = StratifiedKFold(n_splits=self.config['n_outer_folds'],
                                   shuffle=True,
                                   random_state=self.config['random_state'])

        outer_fold_results = []
        all_selected_k = []
        all_selected_features = []
        y_true_all = []
        y_pred_all = []
        y_proba_all = []

        for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            print(f"\n--- Outer Fold {outer_fold_idx}/{self.config['n_outer_folds']} ---")

            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            print(f"  Running inner CV for k selection...")
            optimal_k, k_avg_scores = self.run_inner_cv_for_k_selection(
                X_train, y_train, k_values, model_type
            )
            print(f"  Optimal k selected: {optimal_k}")
            print(f"  Inner CV scores: AUC={k_avg_scores[optimal_k]['auc']:.3f}, F1={k_avg_scores[optimal_k]['f1']:.3f}")

            all_selected_k.append(optimal_k)

            X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
            X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
            y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
            y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test

            X_train_reduced, X_test_reduced, selected_features = FeatureSelector.select_features_for_k(
                X_train_arr, y_train_arr, X_test_arr, optimal_k, feature_names
            )

            all_selected_features.extend(selected_features)

            model = ModelFactory.get_model_pipeline(model_type, self.config['random_state'])
            model.fit(X_train_reduced, y_train_arr)

            y_pred = model.predict(X_test_reduced)
            y_proba = model.predict_proba(X_test_reduced)[:, 1] if hasattr(model, "predict_proba") else y_pred

            y_true_all.extend(y_test_arr)
            y_pred_all.extend(y_pred)
            y_proba_all.extend(y_proba)

            fold_metrics = {
                'accuracy': accuracy_score(y_test_arr, y_pred),
                'precision': precision_score(y_test_arr, y_pred, zero_division=0),
                'recall': recall_score(y_test_arr, y_pred, zero_division=0),
                'f1': f1_score(y_test_arr, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test_arr, y_proba) if len(np.unique(y_test_arr)) > 1 else 0.5
            }

            print(f"  Outer fold performance: Acc={fold_metrics['accuracy']:.3f}, AUC={fold_metrics['auc']:.3f}")

            outer_fold_results.append({
                'fold': outer_fold_idx,
                'optimal_k': optimal_k,
                'selected_features': selected_features,
                'k_scores': k_avg_scores,
                'metrics': fold_metrics,
                'test_indices': test_idx,
            })

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        y_proba_all = np.array(y_proba_all)

        overall_metrics = {
            'accuracy': accuracy_score(y_true_all, y_pred_all),
            'precision': precision_score(y_true_all, y_pred_all, zero_division=0),
            'recall': recall_score(y_true_all, y_pred_all, zero_division=0),
            'f1': f1_score(y_true_all, y_pred_all, zero_division=0),
            'auc': roc_auc_score(y_true_all, y_proba_all) if len(np.unique(y_true_all)) > 1 else 0.5
        }

        cm = confusion_matrix(y_true_all, y_pred_all)
        end_time = time.time()

        k_counter = Counter(all_selected_k)
        consensus_k = k_counter.most_common(1)[0][0]

        feature_counter = Counter(all_selected_features)
        most_common_features = feature_counter.most_common(consensus_k)

        print(f"\n{'='*80}")
        print(f"NESTED CV RESULTS: {model_name}")
        print(f"{'='*80}")
        print(f"Overall Metrics:")
        for metric, value in overall_metrics.items():
            print(f"  {metric.upper():12}: {value:.4f}")
        print(f"\nk Selection Across Folds: {all_selected_k}")
        print(f"Consensus k: {consensus_k} (selected in {k_counter[consensus_k]}/{self.config['n_outer_folds']} folds)")
        print(f"\nExecution time: {end_time - start_time:.2f}s")

        return {
            'model_name': model_name,
            'model_type': model_type,
            'overall_metrics': overall_metrics,
            'confusion_matrix': cm,
            'outer_fold_results': outer_fold_results,
            'all_selected_k': all_selected_k,
            'consensus_k': consensus_k,
            'k_counter': k_counter,
            'feature_counter': feature_counter,
            'most_common_features': most_common_features,
            'y_true': y_true_all,
            'y_pred': y_pred_all,
            'y_proba': y_proba_all,
            'execution_time': end_time - start_time,
            'feature_names': feature_names,
        }

# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Handle all visualization tasks"""

    def __init__(self, output_dir, plot_config):
        self.output_dir = output_dir
        self.plot_config = plot_config

    def plot_k_selection_summary(self, nested_results, config):
        """Plot k selection frequency and elbow curves (no confusion matrix here)."""
        model_name = nested_results['model_name']

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. k Selection Frequency
        k_counter = nested_results['k_counter']
        k_values = sorted(k_counter.keys())
        k_counts = [k_counter[k] for k in k_values]

        axes[0].bar(range(len(k_values)), k_counts, edgecolor='black')
        axes[0].set_xticks(range(len(k_values)))
        axes[0].set_xticklabels(k_values)
        axes[0].set_xlabel('k (Number of Features)')
        axes[0].set_ylabel('Selection Frequency')
        axes[0].set_title(f'{model_name}: k Selection Across Outer Folds')
        axes[0].grid(True, alpha=0.3)

        # 2. Elbow Plot - AUC vs k
        k_values_all = config['k_values_resolved']
        auc_by_k = defaultdict(list)
        f1_by_k = defaultdict(list)

        for fold_result in nested_results['outer_fold_results']:
            k_scores = fold_result['k_scores']
            for k in k_values_all:
                if k in k_scores:
                    auc_by_k[k].append(k_scores[k]['auc'])
                    f1_by_k[k].append(k_scores[k]['f1'])

        k_sorted = sorted(auc_by_k.keys())
        auc_means = [np.mean(auc_by_k[k]) for k in k_sorted]
        auc_stds = [np.std(auc_by_k[k]) for k in k_sorted]

        axes[1].errorbar(k_sorted, auc_means, yerr=auc_stds, marker='o', capsize=5)
        axes[1].set_xlabel('k (Number of Features)')
        axes[1].set_ylabel('AUC (Inner CV)')
        axes[1].set_title(f'{model_name}: Elbow Plot (AUC)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(nested_results['consensus_k'], color='red', linestyle='--',
                        label=f"Consensus k={nested_results['consensus_k']}")
        axes[1].legend()

        # 3. Elbow Plot - F1 vs k
        f1_means = [np.mean(f1_by_k[k]) for k in k_sorted]
        f1_stds = [np.std(f1_by_k[k]) for k in k_sorted]

        axes[2].errorbar(k_sorted, f1_means, yerr=f1_stds, marker='s', capsize=5, color='green')
        axes[2].set_xlabel('k (Number of Features)')
        axes[2].set_ylabel('F1 Score (Inner CV)')
        axes[2].set_title(f'{model_name}: Elbow Plot (F1)')
        axes[2].grid(True, alpha=0.3)
        axes[2].axvline(nested_results['consensus_k'], color='red', linestyle='--',
                        label=f"Consensus k={nested_results['consensus_k']}")
        axes[2].legend()

        plt.tight_layout()
        save_path = self.output_dir / f"nested_cv_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=self.plot_config['figure_dpi'], bbox_inches='tight')
        plt.close()

        return save_path

    def plot_best_model_confusion_matrix(self, all_results):
        """Find the best model by AUC and plot a dedicated confusion matrix for it."""
        best_result = max(all_results, key=lambda r: r['overall_metrics']['f1'])
        model_name = best_result['model_name']
        auc = best_result['overall_metrics']['auc']
        f1 = best_result['overall_metrics']['f1']
        acc = best_result['overall_metrics']['accuracy']
        cm = best_result['confusion_matrix']

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Control', 'Disease cases'],   # ← changed
                    yticklabels=['Control', 'Disease cases'],   # ← changed
                    annot_kws={"size": 16})
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_title(
            f'Confusion Matrix — Best Model: {model_name}\n'
            f'F1={f1:.3f}  |  AUC={auc:.3f}  |  Acc={acc:.3f}',
            fontsize=15, fontweight='bold'
        )

        plt.tight_layout()
        save_path = self.output_dir / "confusion_matrix_best_model.png"
        plt.savefig(save_path, dpi=self.plot_config['figure_dpi'], bbox_inches='tight')
        plt.close()

        print(f"\n✓ Best model: {model_name} (AUC={auc:.3f})")
        print(f"✓ Confusion matrix saved: {save_path.name}")

        return save_path, best_result

    def plot_feature_importance(self, nested_results, config, top_n=30):
        """Plot feature selection frequency across nested CV folds."""
        model_name = nested_results['model_name']
        consensus_k = nested_results['consensus_k']
        feature_counter = nested_results['feature_counter']

        most_common = feature_counter.most_common(top_n)
        if not most_common:
            print(f"No features to plot for {model_name}")
            return None

        features, counts = zip(*most_common)

        fig, ax = plt.subplots(figsize=(10, max(6, 0.4*len(features))))
        y_pos = np.arange(len(features))
        ax.barh(y_pos, counts, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Selection Frequency (Across All Outer Folds)')
        ax.set_title(f'{model_name}: Most Frequently Selected Features\n(Consensus k={consensus_k})')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        save_path = self.output_dir / f"feature_frequency_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=self.plot_config['figure_dpi'], bbox_inches='tight')
        plt.close()

        return save_path

    def plot_model_comparison(self, all_results):
        """Compare all models' nested CV performance."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        model_names = [r['model_name'] for r in all_results]

        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        x = np.arange(len(model_names))
        width = 0.15

        for i, metric in enumerate(metrics):
            values = [r['overall_metrics'][metric] for r in all_results]
            axes[0].bar(x + i*width, values, width, label=metric.upper())

        axes[0].set_xticks(x + width * 2)
        axes[0].set_xticklabels(model_names, rotation=15, ha='right')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim([0, 1.05])
        axes[0].set_title('Nested CV Performance Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        consensus_ks = [r['consensus_k'] for r in all_results]
        axes[1].bar(model_names, consensus_ks, edgecolor='black')
        axes[1].set_ylabel('Consensus k')
        axes[1].set_title('Optimal k Selected by Each Model')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = self.output_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=self.plot_config['figure_dpi'], bbox_inches='tight')
        plt.close()

        return save_path

# ============================================================================
# RESULT EXPORTER
# ============================================================================

class ResultExporter:
    """Export results to CSV and pickle files"""

    def __init__(self, output_dir):
        self.output_dir = output_dir

    def export_performance_summary(self, all_results):
        """Export overall performance comparison"""
        performance_data = []
        for result in all_results:
            row = {'Model': result['model_name'], 'Consensus_k': result['consensus_k']}
            row.update(result['overall_metrics'])
            performance_data.append(row)

        performance_df = pd.DataFrame(performance_data)
        perf_path = self.output_dir / "nested_cv_performance_summary.csv"
        performance_df.to_csv(perf_path, index=False)
        print(f"✓ Performance summary: {perf_path.name}")
        print(performance_df.to_string(index=False))

        return perf_path

    def export_k_selection(self, all_results, n_outer_folds):
        """Export k selection details"""
        k_selection_data = []
        for result in all_results:
            for fold_result in result['outer_fold_results']:
                k_selection_data.append({
                    'Model': result['model_name'],
                    'Outer_Fold': fold_result['fold'],
                    'Selected_k': fold_result['optimal_k'],
                    'Fold_Accuracy': fold_result['metrics']['accuracy'],
                    'Fold_AUC': fold_result['metrics']['auc'],
                    'Fold_F1': fold_result['metrics']['f1'],
                })

        k_selection_df = pd.DataFrame(k_selection_data)
        k_path = self.output_dir / "nested_cv_k_selection.csv"
        k_selection_df.to_csv(k_path, index=False)
        print(f"✓ k selection details: {k_path.name}")

        return k_path

    def export_feature_frequencies(self, all_results, n_outer_folds):
        """Export feature frequency tables"""
        paths = []
        for result in all_results:
            feature_freq_data = []
            for feature, count in result['feature_counter'].most_common(result['consensus_k']):
                feature_freq_data.append({
                    'Feature': feature,
                    'Selection_Count': count,
                    'Selection_Frequency': f"{count}/{n_outer_folds}"
                })

            if feature_freq_data:
                feat_df = pd.DataFrame(feature_freq_data)
                feat_path = self.output_dir / f"feature_frequency_{result['model_name'].lower().replace(' ', '_')}.csv"
                feat_df.to_csv(feat_path, index=False)
                print(f"✓ Feature frequency ({result['model_name']}): {feat_path.name}")
                paths.append(feat_path)

        return paths

    def save_complete_bundle(self, all_results, config, data_info):
        """Save complete results bundle"""
        results_bundle_path = self.output_dir / "nested_cv_complete_results.pkl"

        with open(results_bundle_path, 'wb') as f:
            pickle.dump({
                'all_nested_results': all_results,
                'config': config,
                'data_info': data_info,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, f)

        print(f"✓ Complete results saved: {results_bundle_path.name}")
        return results_bundle_path

# ============================================================================
# MODEL TRAINER
# ============================================================================

class ModelTrainer:
    """Train final production models"""

    def __init__(self, output_dir, random_state):
        self.output_dir = output_dir
        self.random_state = random_state

    def train_final_models(self, all_results, X_full, y):
        """Train final production models on all data"""
        print("\n" + "="*80)
        print("TRAINING FINAL PRODUCTION MODELS")
        print("="*80)

        final_models = {}

        for result in all_results:
            print(f"\n--- {result['model_name']} ---")

            model_type = result['model_type']
            consensus_k = result['consensus_k']

            most_common_features = result['feature_counter'].most_common(consensus_k)
            selected_feature_names = [feat for feat, count in most_common_features]

            print(f"Consensus k: {consensus_k}")
            print(f"Using {len(selected_feature_names)} most frequently selected features")

            X_final = X_full[selected_feature_names].values
            y_final = y.values

            final_model = ModelFactory.get_model_pipeline(model_type, self.random_state)

            start_time = time.time()
            final_model.fit(X_final, y_final)
            train_time = time.time() - start_time

            test_sample = X_final[:1]
            start_time = time.time()
            for _ in range(100):
                _ = final_model.predict(test_sample)
            avg_pred_time = (time.time() - start_time) / 100

            print(f"Training time: {train_time:.3f}s")
            print(f"Average prediction time: {avg_pred_time*1000:.3f}ms")

            model_filename = f"final_model_{result['model_name'].lower().replace(' ', '_')}.pkl"
            model_path = self.output_dir / model_filename

            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': final_model,
                    'selected_features': selected_feature_names,
                    'consensus_k': consensus_k,
                    'nested_cv_metrics': result['overall_metrics'],
                    'training_time': train_time,
                    'prediction_time': avg_pred_time,
                }, f)

            print(f"✓ Model saved: {model_path.name}")

            final_models[result['model_name']] = {
                'model': final_model,
                'path': model_path,
                'features': selected_feature_names,
                'train_time': train_time,
                'pred_time': avg_pred_time,
            }

        return final_models

    def export_speed_comparison(self, final_models):
        """Export prediction speed comparison"""
        speed_data = []
        for name, info in final_models.items():
            speed_data.append({
                'Model': name,
                'Training_Time_s': info['train_time'],
                'Prediction_Time_ms': info['pred_time'] * 1000,
                'Features_Used': len(info['features'])
            })

        speed_df = pd.DataFrame(speed_data)
        print("\n" + "="*80)
        print("PREDICTION SPEED COMPARISON")
        print("="*80)
        print(speed_df.to_string(index=False))

        speed_path = self.output_dir / "prediction_speed_comparison.csv"
        speed_df.to_csv(speed_path, index=False)
        print(f"\n✓ Speed comparison saved: {speed_path.name}")

        return speed_path

# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generate final summary reports"""

    @staticmethod
    def generate_summary_report(all_results, final_models, output_dir, data_info, config, best_model_name):
        """Generate and save comprehensive summary report"""

        report_lines = []
        report_lines.append("="*80)
        report_lines.append("NESTED CROSS-VALIDATION - FINAL SUMMARY REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"\nDataset: {data_info['n_patients']} patients, {data_info['n_features']} features")
        report_lines.append(f"Class distribution: Control={data_info['n_control']}, Disease cases={data_info['n_cll']}")

        report_lines.append(f"\nNested CV Configuration:")
        report_lines.append(f"  Outer folds: {config['n_outer_folds']}")
        report_lines.append(f"  Inner folds: {config['n_inner_folds']}")
        report_lines.append(f"  k values tested: {config['k_values']}")
        report_lines.append(f"  Optimization: {config['optimization_metric'].upper()} (primary), {config['secondary_metric'].upper()} (secondary)")

        report_lines.append(f"\n{'Model':<20} {'Consensus k':<15} {'Accuracy':<12} {'AUC':<12} {'F1':<12}")
        report_lines.append("-" * 80)
        for result in all_results:
            marker = " ← BEST" if result['model_name'] == best_model_name else ""
            report_lines.append(f"{result['model_name']:<20} {result['consensus_k']:<15} "
                  f"{result['overall_metrics']['accuracy']:<12.4f} "
                  f"{result['overall_metrics']['auc']:<12.4f} "
                  f"{result['overall_metrics']['f1']:<12.4f}{marker}")

        report_lines.append(f"\nBest Model (by AUC): {best_model_name}")

        report_lines.append(f"\nFinal Production Models:")
        for name, info in final_models.items():
            report_lines.append(f"  {name}:")
            report_lines.append(f"    Path: {info['path'].name}")
            report_lines.append(f"    Features: {len(info['features'])}")
            report_lines.append(f"    Prediction time: {info['pred_time']*1000:.2f}ms")

        report_lines.append(f"\nAll outputs saved to: {output_dir}")
        report_lines.append(f"\nKey files:")
        report_lines.append(f"  - Performance summary: nested_cv_performance_summary.csv")
        report_lines.append(f"  - k selection details: nested_cv_k_selection.csv")
        report_lines.append(f"  - Feature frequencies: feature_frequency_*.csv")
        report_lines.append(f"  - Final models: final_model_*.pkl")
        report_lines.append(f"  - Complete results: nested_cv_complete_results.pkl")
        report_lines.append(f"  - Best model confusion matrix: confusion_matrix_best_model.png")
        report_lines.append(f"  - This report: summary_report.txt")

        report_lines.append("\n" + "="*80)
        report_lines.append("ANALYSIS COMPLETE!")
        report_lines.append("="*80)

        report_text = "\n".join(report_lines)

        print("\n" + report_text)

        report_path = output_dir / "summary_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(f"\n✓ Summary report saved: {report_path.name}")

        return report_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("="*80)
    print("NESTED CROSS-VALIDATION FOR CLL CLASSIFICATION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Setup
    output_dir = Config.setup_output_directory()
    Config.setup_plotting()

    print(f"\n✓ Output directory: {output_dir}")
    print(f"✓ Input file: {Config.INPUT_PKL}")
    print(f"✓ Selected model: {Config.SELECTED_MODEL}")

    # Load and prepare data
    X_full, y, db = DataLoader.load_and_prepare(Config.INPUT_PKL, Config.SELECTED_MODEL)

    # Resolve k values
    k_values_resolved = []
    for k in Config.K_VALUES:
        if k == 'max':
            k_values_resolved.append(X_full.shape[1])
        else:
            k_values_resolved.append(k)

    # Create config dictionary
    config = {
        'random_state': Config.RANDOM_STATE,
        'n_outer_folds': Config.N_OUTER_FOLDS,
        'n_inner_folds': Config.N_INNER_FOLDS,
        'k_values': Config.K_VALUES,
        'k_values_resolved': k_values_resolved,
        'optimization_metric': Config.OPTIMIZATION_METRIC,
        'secondary_metric': Config.SECONDARY_METRIC,
        'auc_tolerance': Config.AUC_TOLERANCE,
    }

    print(f"\nResolved k values: {k_values_resolved}")

    # Run nested CV for all models
    print("\n" + "="*80)
    print("EXECUTING NESTED CROSS-VALIDATION")
    print("="*80)

    nested_cv = NestedCV(config)
    all_nested_results = []

    rf_results = nested_cv.run_nested_cv('rf', 'RandomForest', X_full, y)
    all_nested_results.append(rf_results)

    svm_results = nested_cv.run_nested_cv('svm', 'SVM', X_full, y)
    all_nested_results.append(svm_results)

    lr_results = nested_cv.run_nested_cv('lr', 'LogisticRegression', X_full, y)
    all_nested_results.append(lr_results)

    print("\n" + "="*80)
    print("ALL NESTED CV COMPLETE")
    print("="*80)

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    visualizer = Visualizer(output_dir, Config.PLOT_CONFIG)

    for result in all_nested_results:
        print(f"\nGenerating plots for {result['model_name']}...")
        visualizer.plot_k_selection_summary(result, config)
        visualizer.plot_feature_importance(result, config, top_n=30)

    visualizer.plot_model_comparison(all_nested_results)

    # ── NEW: single confusion matrix for the best model ──
    _, best_result = visualizer.plot_best_model_confusion_matrix(all_nested_results)
    best_model_name = best_result['model_name']

    print("\n✓ All visualizations generated")

    # Export results
    print("\n" + "="*80)
    print("EXPORTING SUMMARY TABLES")
    print("="*80)

    exporter = ResultExporter(output_dir)
    exporter.export_performance_summary(all_nested_results)
    exporter.export_k_selection(all_nested_results, Config.N_OUTER_FOLDS)
    exporter.export_feature_frequencies(all_nested_results, Config.N_OUTER_FOLDS)

    data_info = {
        'n_patients': len(X_full),
        'n_features': X_full.shape[1],
        'n_control': int((y==0).sum()),
        'n_cll': int((y==1).sum()),
    }

    exporter.save_complete_bundle(all_nested_results, config, data_info)

    # Train final models
    trainer = ModelTrainer(output_dir, Config.RANDOM_STATE)
    final_models = trainer.train_final_models(all_nested_results, X_full, y)
    trainer.export_speed_comparison(final_models)

    # Generate final report (now passes best_model_name)
    ReportGenerator.generate_summary_report(
        all_nested_results, final_models, output_dir, data_info, config, best_model_name
    )

    print("\n" + "="*80)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# """
# ================================================================================
# NESTED CROSS-VALIDATION FOR CLL CLASSIFICATION
# ================================================================================
# Modular Python script converted from Jupyter notebook.
# Performs nested CV with hyperparameter tuning and saves all outputs to a 
# structured folder.

# Author: Converted from notebook
# Date: February 2026
# ================================================================================
# """

# import os
# import sys
# import time
# import warnings
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# from collections import defaultdict, Counter
# from datetime import datetime

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.metrics import (accuracy_score, precision_score, recall_score,
#                              f1_score, roc_auc_score, confusion_matrix)

# warnings.filterwarnings('ignore')

# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# class Config:
#     """Configuration class for all parameters"""

#     # Paths
#     INPUT_PKL = Path('/mnt/lustre/home/claassen/clala950/DC-TSeg/Data/Guck2025/Train_Test_Split/Test_data_cyto2_finetuned_clean_SINGLE_unet_preds_morphology.pkl')
#     OUTPUT_ROOT_NAME = 'Classification_results_student'

#     # Model selection
#     SELECTED_MODEL = 'Unet_preds'  # 'Ground_truth' or 'Unet_preds' or 'Unet_preds_CLL_focus_k1_5' etc.

#     # Cross-validation parameters
#     RANDOM_STATE = 42
#     N_OUTER_FOLDS = 5
#     N_INNER_FOLDS = 5
#     K_VALUES = [5, 10, 15, 20, 25, 30, 50, 'max']

#     # Optimization
#     OPTIMIZATION_METRIC = 'auc'
#     SECONDARY_METRIC = 'f1'
#     AUC_TOLERANCE = 0.02

#     # Plotting
#     PLOT_CONFIG = {
#         'figure_dpi': 300,
#         'figure_size_single': (10, 6),
#         'figure_size_grid': (16, 12),
#         'font_size': 14,
#         'title_size': 18,
#         'label_size': 14,
#         'tick_size': 12,
#         'legend_size': 12,
#         'line_width': 2,
#         'marker_size': 8,
#     }

#     @classmethod
#     def setup_output_directory(cls):
#         """Create timestamped output directory"""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_dir = cls.INPUT_PKL.parent / f'{cls.OUTPUT_ROOT_NAME}_{timestamp}'
#         output_dir.mkdir(parents=True, exist_ok=True)
#         return output_dir

#     @classmethod
#     def setup_plotting(cls):
#         """Apply global plotting configuration"""
#         plt.rcParams.update({
#             'font.size': cls.PLOT_CONFIG['font_size'],
#             'axes.titlesize': cls.PLOT_CONFIG['title_size'],
#             'axes.labelsize': cls.PLOT_CONFIG['label_size'],
#             'xtick.labelsize': cls.PLOT_CONFIG['tick_size'],
#             'ytick.labelsize': cls.PLOT_CONFIG['tick_size'],
#             'legend.fontsize': cls.PLOT_CONFIG['legend_size'],
#             'figure.titlesize': cls.PLOT_CONFIG['title_size'] + 4,
#             'lines.linewidth': cls.PLOT_CONFIG['line_width'],
#             'lines.markersize': cls.PLOT_CONFIG['marker_size'],
#         })

# # ============================================================================
# # DATA LOADING AND PREPARATION
# # ============================================================================

# class DataLoader:
#     """Handle data loading and feature aggregation"""

#     @staticmethod
#     def aggregate_morphology_stats(db, model_key, stats=('mean', 'median', 'std', 'cv')):
#         """Aggregate morphology statistics across all patients."""
#         morph_key = f'{model_key}_morphology'
#         patient_ids = [pid for pid in db.keys()
#                        if morph_key in db[pid] and isinstance(db[pid][morph_key], pd.DataFrame)]

#         if not patient_ids:
#             raise ValueError(f"No patients contained a DataFrame under key '{morph_key}'")

#         first_df = None
#         for pid in patient_ids:
#             df = db[pid][morph_key]
#             if isinstance(df, pd.DataFrame) and not df.empty:
#                 first_df = df
#                 break

#         if first_df is None or first_df.empty:
#             raise ValueError(f"All '{morph_key}' DataFrames are empty.")

#         morph_cols = first_df.columns.tolist()
#         if 'Area' not in morph_cols:
#             raise ValueError("'Area' column not found in morphology DataFrame.")
#         morph_cols = morph_cols[morph_cols.index('Area'):]

#         results = {'Patient_ID': []}
#         for col in morph_cols:
#             for stat in stats:
#                 results[f'{col}_{stat}'] = []

#         for pid in patient_ids:
#             results['Patient_ID'].append(pid)
#             df = db[pid][morph_key]

#             if df is None or df.empty:
#                 for col in morph_cols:
#                     for stat in stats:
#                         results[f'{col}_{stat}'].append(np.nan)
#                 continue

#             for col in morph_cols:
#                 col_data = pd.to_numeric(df[col], errors='coerce').dropna()
#                 if col_data.empty:
#                     vals = {s: np.nan for s in stats}
#                 else:
#                     vals = {}
#                     for stat in stats:
#                         if stat == 'mean':
#                             vals[stat] = col_data.mean()
#                         elif stat == 'median':
#                             vals[stat] = col_data.median()
#                         elif stat == 'std':
#                             vals[stat] = col_data.std()
#                         elif stat == 'cv':
#                             mean_val = col_data.mean()
#                             std_val = col_data.std()
#                             vals[stat] = (std_val / mean_val) if mean_val != 0 else np.nan
#                 for stat in stats:
#                     results[f'{col}_{stat}'].append(vals[stat])

#         agg_df = pd.DataFrame(results).set_index('Patient_ID')
#         return agg_df

#     @staticmethod
#     def prepare_data(df):
#         """Create X (features) and y (labels) from aggregated morphology."""
#         df = df.copy()
#         patient_ids = df.index.astype(str).tolist()
#         groups = [pid.split('_')[0] for pid in patient_ids]
#         y = pd.Series([1 if g.upper().startswith('CLL') else 0 for g in groups],
#                       index=df.index, name='Group')
#         X = df.copy()
#         return X, y

#     @classmethod
#     def load_and_prepare(cls, pkl_path, model_key):
#         """Load pickle and prepare data"""
#         print("="*80)
#         print("LOADING DATA")
#         print("="*80)

#         with open(pkl_path, 'rb') as f:
#             db = pickle.load(f)
#         print(f"✓ Loaded database with {len(db)} patients")

#         agg_df = cls.aggregate_morphology_stats(db, model_key, stats=('mean', 'median', 'std', 'cv'))
#         print(f"✓ Aggregated shape: {agg_df.shape}")

#         X, y = cls.prepare_data(agg_df)

#         print(f"\nTotal samples: {len(X)}")
#         print(f"Total features: {X.shape[1]}")
#         print(f"\nClass distribution:")
#         print(f"  Control (0): {int((y==0).sum())} ({(y==0).mean()*100:.1f}%)")
#         print(f"  CLL (1):     {int((y==1).sum())} ({(y==1).mean()*100:.1f}%)")

#         return X, y, db

# # ============================================================================
# # MODEL DEFINITIONS
# # ============================================================================

# class ModelFactory:
#     """Factory for creating model pipelines"""

#     @staticmethod
#     def get_model_pipeline(model_type, random_state=42):
#         """Create sklearn pipeline for given model type."""
#         if model_type == 'rf':
#             return Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('classifier', RandomForestClassifier(
#                     n_estimators=100, max_depth=4,
#                     min_samples_split=3, min_samples_leaf=1,
#                     class_weight='balanced',
#                     random_state=random_state
#                 ))
#             ])
#         elif model_type == 'svm':
#             return Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('classifier', SVC(
#                     kernel='rbf', probability=True, C=1.0,
#                     class_weight='balanced',
#                     random_state=random_state
#                 ))
#             ])
#         elif model_type == 'lr':
#             return Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('classifier', LogisticRegression(
#                     random_state=random_state,
#                     max_iter=1000,
#                     class_weight='balanced'
#                 ))
#             ])
#         else:
#             raise ValueError(f"Unknown model type: {model_type}")

# # ============================================================================
# # FEATURE SELECTION
# # ============================================================================

# class FeatureSelector:
#     """Handle feature selection operations"""

#     @staticmethod
#     def select_features_for_k(X_train, y_train, X_test, k, feature_names):
#         """
#         Select top k features using SelectKBest on training data.
#         Returns: X_train_reduced, X_test_reduced, selected_feature_names
#         """
#         k_actual = min(k, X_train.shape[1])
#         selector = SelectKBest(f_classif, k=k_actual)

#         X_train_reduced = selector.fit_transform(X_train, y_train)
#         X_test_reduced = selector.transform(X_test)

#         selected_mask = selector.get_support()
#         selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]

#         return X_train_reduced, X_test_reduced, selected_features

# # ============================================================================
# # NESTED CROSS-VALIDATION
# # ============================================================================

# class NestedCV:
#     """Nested cross-validation implementation"""

#     def __init__(self, config):
#         self.config = config

#     def run_inner_cv_for_k_selection(self, X_train, y_train, k_values, model_type):
#         """Run inner CV to select optimal k."""
#         feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]

#         if not isinstance(X_train, np.ndarray):
#             X_train = X_train.values
#         if not isinstance(y_train, np.ndarray):
#             y_train = y_train.values

#         k_scores = {k: [] for k in k_values}
#         inner_cv = StratifiedKFold(n_splits=self.config['n_inner_folds'],
#                                    shuffle=True,
#                                    random_state=self.config['random_state'])

#         for k in k_values:
#             for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
#                 X_inner_train = X_train[inner_train_idx]
#                 X_inner_val = X_train[inner_val_idx]
#                 y_inner_train = y_train[inner_train_idx]
#                 y_inner_val = y_train[inner_val_idx]

#                 X_inner_train_reduced, X_inner_val_reduced, _ = FeatureSelector.select_features_for_k(
#                     X_inner_train, y_inner_train, X_inner_val, k, feature_names
#                 )

#                 model = ModelFactory.get_model_pipeline(model_type, self.config['random_state'])
#                 model.fit(X_inner_train_reduced, y_inner_train)

#                 y_pred = model.predict(X_inner_val_reduced)
#                 y_proba = model.predict_proba(X_inner_val_reduced)[:, 1] if hasattr(model, "predict_proba") else y_pred

#                 metrics = {
#                     'accuracy': accuracy_score(y_inner_val, y_pred),
#                     'f1': f1_score(y_inner_val, y_pred, zero_division=0),
#                     'auc': roc_auc_score(y_inner_val, y_proba) if len(np.unique(y_inner_val)) > 1 else 0.5
#                 }

#                 k_scores[k].append(metrics)

#         k_avg_scores = {}
#         for k in k_values:
#             k_avg_scores[k] = {
#                 'accuracy': np.mean([s['accuracy'] for s in k_scores[k]]),
#                 'f1': np.mean([s['f1'] for s in k_scores[k]]),
#                 'auc': np.mean([s['auc'] for s in k_scores[k]]),
#             }

#         primary_metric = self.config['optimization_metric']
#         secondary_metric = self.config['secondary_metric']

#         sorted_k = sorted(k_values, key=lambda k: k_avg_scores[k][primary_metric], reverse=True)
#         best_k = sorted_k[0]
#         best_score = k_avg_scores[best_k][primary_metric]

#         candidates = [k for k in sorted_k if abs(k_avg_scores[k][primary_metric] - best_score) <= self.config['auc_tolerance']]

#         if len(candidates) > 1:
#             best_k = max(candidates, key=lambda k: k_avg_scores[k][secondary_metric])

#         return best_k, k_avg_scores

#     def run_nested_cv(self, model_type, model_name, X, y):
#         """Run nested cross-validation with hyperparameter tuning."""
#         print(f"\n{'='*80}")
#         print(f"NESTED CV: {model_name}")
#         print(f"{'='*80}")

#         start_time = time.time()

#         feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
#         k_values = self.config['k_values_resolved']

#         outer_cv = StratifiedKFold(n_splits=self.config['n_outer_folds'],
#                                    shuffle=True,
#                                    random_state=self.config['random_state'])

#         outer_fold_results = []
#         all_selected_k = []
#         all_selected_features = []
#         y_true_all = []
#         y_pred_all = []
#         y_proba_all = []

#         for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
#             print(f"\n--- Outer Fold {outer_fold_idx}/{self.config['n_outer_folds']} ---")

#             if isinstance(X, pd.DataFrame):
#                 X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#             else:
#                 X_train, X_test = X[train_idx], X[test_idx]
#             y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#             print(f"  Running inner CV for k selection...")
#             optimal_k, k_avg_scores = self.run_inner_cv_for_k_selection(
#                 X_train, y_train, k_values, model_type
#             )
#             print(f"  Optimal k selected: {optimal_k}")
#             print(f"  Inner CV scores: AUC={k_avg_scores[optimal_k]['auc']:.3f}, F1={k_avg_scores[optimal_k]['f1']:.3f}")

#             all_selected_k.append(optimal_k)

#             X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
#             X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
#             y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
#             y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test

#             X_train_reduced, X_test_reduced, selected_features = FeatureSelector.select_features_for_k(
#                 X_train_arr, y_train_arr, X_test_arr, optimal_k, feature_names
#             )

#             all_selected_features.extend(selected_features)

#             model = ModelFactory.get_model_pipeline(model_type, self.config['random_state'])
#             model.fit(X_train_reduced, y_train_arr)

#             y_pred = model.predict(X_test_reduced)
#             y_proba = model.predict_proba(X_test_reduced)[:, 1] if hasattr(model, "predict_proba") else y_pred

#             y_true_all.extend(y_test_arr)
#             y_pred_all.extend(y_pred)
#             y_proba_all.extend(y_proba)

#             fold_metrics = {
#                 'accuracy': accuracy_score(y_test_arr, y_pred),
#                 'precision': precision_score(y_test_arr, y_pred, zero_division=0),
#                 'recall': recall_score(y_test_arr, y_pred, zero_division=0),
#                 'f1': f1_score(y_test_arr, y_pred, zero_division=0),
#                 'auc': roc_auc_score(y_test_arr, y_proba) if len(np.unique(y_test_arr)) > 1 else 0.5
#             }

#             print(f"  Outer fold performance: Acc={fold_metrics['accuracy']:.3f}, AUC={fold_metrics['auc']:.3f}")

#             outer_fold_results.append({
#                 'fold': outer_fold_idx,
#                 'optimal_k': optimal_k,
#                 'selected_features': selected_features,
#                 'k_scores': k_avg_scores,
#                 'metrics': fold_metrics,
#                 'test_indices': test_idx,
#             })

#         y_true_all = np.array(y_true_all)
#         y_pred_all = np.array(y_pred_all)
#         y_proba_all = np.array(y_proba_all)

#         overall_metrics = {
#             'accuracy': accuracy_score(y_true_all, y_pred_all),
#             'precision': precision_score(y_true_all, y_pred_all, zero_division=0),
#             'recall': recall_score(y_true_all, y_pred_all, zero_division=0),
#             'f1': f1_score(y_true_all, y_pred_all, zero_division=0),
#             'auc': roc_auc_score(y_true_all, y_proba_all) if len(np.unique(y_true_all)) > 1 else 0.5
#         }

#         cm = confusion_matrix(y_true_all, y_pred_all)
#         end_time = time.time()

#         k_counter = Counter(all_selected_k)
#         consensus_k = k_counter.most_common(1)[0][0]

#         feature_counter = Counter(all_selected_features)
#         most_common_features = feature_counter.most_common(consensus_k)

#         print(f"\n{'='*80}")
#         print(f"NESTED CV RESULTS: {model_name}")
#         print(f"{'='*80}")
#         print(f"Overall Metrics:")
#         for metric, value in overall_metrics.items():
#             print(f"  {metric.upper():12}: {value:.4f}")
#         print(f"\nk Selection Across Folds: {all_selected_k}")
#         print(f"Consensus k: {consensus_k} (selected in {k_counter[consensus_k]}/{self.config['n_outer_folds']} folds)")
#         print(f"\nExecution time: {end_time - start_time:.2f}s")

#         return {
#             'model_name': model_name,
#             'model_type': model_type,
#             'overall_metrics': overall_metrics,
#             'confusion_matrix': cm,
#             'outer_fold_results': outer_fold_results,
#             'all_selected_k': all_selected_k,
#             'consensus_k': consensus_k,
#             'k_counter': k_counter,
#             'feature_counter': feature_counter,
#             'most_common_features': most_common_features,
#             'y_true': y_true_all,
#             'y_pred': y_pred_all,
#             'y_proba': y_proba_all,
#             'execution_time': end_time - start_time,
#             'feature_names': feature_names,
#         }

# # ============================================================================
# # VISUALIZATION
# # ============================================================================

# class Visualizer:
#     """Handle all visualization tasks"""

#     def __init__(self, output_dir, plot_config):
#         self.output_dir = output_dir
#         self.plot_config = plot_config

#     def plot_k_selection_summary(self, nested_results, config):
#         """Plot k selection frequency and elbow curves (no confusion matrix here)."""
#         model_name = nested_results['model_name']

#         fig, axes = plt.subplots(1, 3, figsize=(18, 6))

#         # 1. k Selection Frequency
#         k_counter = nested_results['k_counter']
#         k_values = sorted(k_counter.keys())
#         k_counts = [k_counter[k] for k in k_values]

#         axes[0].bar(range(len(k_values)), k_counts, edgecolor='black')
#         axes[0].set_xticks(range(len(k_values)))
#         axes[0].set_xticklabels(k_values)
#         axes[0].set_xlabel('k (Number of Features)')
#         axes[0].set_ylabel('Selection Frequency')
#         axes[0].set_title(f'{model_name}: k Selection Across Outer Folds')
#         axes[0].grid(True, alpha=0.3)

#         # 2. Elbow Plot - AUC vs k
#         k_values_all = config['k_values_resolved']
#         auc_by_k = defaultdict(list)
#         f1_by_k = defaultdict(list)

#         for fold_result in nested_results['outer_fold_results']:
#             k_scores = fold_result['k_scores']
#             for k in k_values_all:
#                 if k in k_scores:
#                     auc_by_k[k].append(k_scores[k]['auc'])
#                     f1_by_k[k].append(k_scores[k]['f1'])

#         k_sorted = sorted(auc_by_k.keys())
#         auc_means = [np.mean(auc_by_k[k]) for k in k_sorted]
#         auc_stds = [np.std(auc_by_k[k]) for k in k_sorted]

#         axes[1].errorbar(k_sorted, auc_means, yerr=auc_stds, marker='o', capsize=5)
#         axes[1].set_xlabel('k (Number of Features)')
#         axes[1].set_ylabel('AUC (Inner CV)')
#         axes[1].set_title(f'{model_name}: Elbow Plot (AUC)')
#         axes[1].grid(True, alpha=0.3)
#         axes[1].axvline(nested_results['consensus_k'], color='red', linestyle='--',
#                         label=f"Consensus k={nested_results['consensus_k']}")
#         axes[1].legend()

#         # 3. Elbow Plot - F1 vs k
#         f1_means = [np.mean(f1_by_k[k]) for k in k_sorted]
#         f1_stds = [np.std(f1_by_k[k]) for k in k_sorted]

#         axes[2].errorbar(k_sorted, f1_means, yerr=f1_stds, marker='s', capsize=5, color='green')
#         axes[2].set_xlabel('k (Number of Features)')
#         axes[2].set_ylabel('F1 Score (Inner CV)')
#         axes[2].set_title(f'{model_name}: Elbow Plot (F1)')
#         axes[2].grid(True, alpha=0.3)
#         axes[2].axvline(nested_results['consensus_k'], color='red', linestyle='--',
#                         label=f"Consensus k={nested_results['consensus_k']}")
#         axes[2].legend()

#         plt.tight_layout()
#         save_path = self.output_dir / f"nested_cv_{model_name.lower().replace(' ', '_')}.png"
#         plt.savefig(save_path, dpi=self.plot_config['figure_dpi'], bbox_inches='tight')
#         plt.close()

#         return save_path

#     def plot_best_model_confusion_matrix(self, all_results):
#         """Find the best model by AUC and plot a dedicated confusion matrix for it."""
#         best_result = max(all_results, key=lambda r: r['overall_metrics']['f1'])
#         model_name = best_result['model_name']
#         auc = best_result['overall_metrics']['auc']
#         f1 = best_result['overall_metrics']['f1']
#         acc = best_result['overall_metrics']['accuracy']
#         cm = best_result['confusion_matrix']

#         fig, ax = plt.subplots(figsize=(7, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
#                     xticklabels=['Control', 'CLL'],
#                     yticklabels=['Control', 'CLL'],
#                     annot_kws={"size": 16})
#         ax.set_xlabel('Predicted Label', fontsize=14)
#         ax.set_ylabel('True Label', fontsize=14)
#         ax.set_title(
#             f'Confusion Matrix — Best Model: {model_name}\n'
#             f'F1={f1:.3f}  |  AUC={auc:.3f}  |  Acc={acc:.3f}',
#             fontsize=15, fontweight='bold'
#         )

#         plt.tight_layout()
#         save_path = self.output_dir / "confusion_matrix_best_model.png"
#         plt.savefig(save_path, dpi=self.plot_config['figure_dpi'], bbox_inches='tight')
#         plt.close()

#         print(f"\n✓ Best model: {model_name} (AUC={auc:.3f})")
#         print(f"✓ Confusion matrix saved: {save_path.name}")

#         return save_path, best_result

#     def plot_feature_importance(self, nested_results, config, top_n=30):
#         """Plot feature selection frequency across nested CV folds."""
#         model_name = nested_results['model_name']
#         consensus_k = nested_results['consensus_k']
#         feature_counter = nested_results['feature_counter']

#         most_common = feature_counter.most_common(top_n)
#         if not most_common:
#             print(f"No features to plot for {model_name}")
#             return None

#         features, counts = zip(*most_common)

#         fig, ax = plt.subplots(figsize=(10, max(6, 0.4*len(features))))
#         y_pos = np.arange(len(features))
#         ax.barh(y_pos, counts, align='center')
#         ax.set_yticks(y_pos)
#         ax.set_yticklabels(features)
#         ax.invert_yaxis()
#         ax.set_xlabel('Selection Frequency (Across All Outer Folds)')
#         ax.set_title(f'{model_name}: Most Frequently Selected Features\n(Consensus k={consensus_k})')
#         ax.grid(True, alpha=0.3, axis='x')

#         plt.tight_layout()
#         save_path = self.output_dir / f"feature_frequency_{model_name.lower().replace(' ', '_')}.png"
#         plt.savefig(save_path, dpi=self.plot_config['figure_dpi'], bbox_inches='tight')
#         plt.close()

#         return save_path

#     def plot_model_comparison(self, all_results):
#         """Compare all models' nested CV performance."""
#         fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#         model_names = [r['model_name'] for r in all_results]

#         metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
#         x = np.arange(len(model_names))
#         width = 0.15

#         for i, metric in enumerate(metrics):
#             values = [r['overall_metrics'][metric] for r in all_results]
#             axes[0].bar(x + i*width, values, width, label=metric.upper())

#         axes[0].set_xticks(x + width * 2)
#         axes[0].set_xticklabels(model_names, rotation=15, ha='right')
#         axes[0].set_ylabel('Score')
#         axes[0].set_ylim([0, 1.05])
#         axes[0].set_title('Nested CV Performance Comparison')
#         axes[0].legend()
#         axes[0].grid(True, alpha=0.3, axis='y')

#         consensus_ks = [r['consensus_k'] for r in all_results]
#         axes[1].bar(model_names, consensus_ks, edgecolor='black')
#         axes[1].set_ylabel('Consensus k')
#         axes[1].set_title('Optimal k Selected by Each Model')
#         axes[1].grid(True, alpha=0.3, axis='y')

#         plt.tight_layout()
#         save_path = self.output_dir / "model_comparison.png"
#         plt.savefig(save_path, dpi=self.plot_config['figure_dpi'], bbox_inches='tight')
#         plt.close()

#         return save_path

# # ============================================================================
# # RESULT EXPORTER
# # ============================================================================

# class ResultExporter:
#     """Export results to CSV and pickle files"""

#     def __init__(self, output_dir):
#         self.output_dir = output_dir

#     def export_performance_summary(self, all_results):
#         """Export overall performance comparison"""
#         performance_data = []
#         for result in all_results:
#             row = {'Model': result['model_name'], 'Consensus_k': result['consensus_k']}
#             row.update(result['overall_metrics'])
#             performance_data.append(row)

#         performance_df = pd.DataFrame(performance_data)
#         perf_path = self.output_dir / "nested_cv_performance_summary.csv"
#         performance_df.to_csv(perf_path, index=False)
#         print(f"✓ Performance summary: {perf_path.name}")
#         print(performance_df.to_string(index=False))

#         return perf_path

#     def export_k_selection(self, all_results, n_outer_folds):
#         """Export k selection details"""
#         k_selection_data = []
#         for result in all_results:
#             for fold_result in result['outer_fold_results']:
#                 k_selection_data.append({
#                     'Model': result['model_name'],
#                     'Outer_Fold': fold_result['fold'],
#                     'Selected_k': fold_result['optimal_k'],
#                     'Fold_Accuracy': fold_result['metrics']['accuracy'],
#                     'Fold_AUC': fold_result['metrics']['auc'],
#                     'Fold_F1': fold_result['metrics']['f1'],
#                 })

#         k_selection_df = pd.DataFrame(k_selection_data)
#         k_path = self.output_dir / "nested_cv_k_selection.csv"
#         k_selection_df.to_csv(k_path, index=False)
#         print(f"✓ k selection details: {k_path.name}")

#         return k_path

#     def export_feature_frequencies(self, all_results, n_outer_folds):
#         """Export feature frequency tables"""
#         paths = []
#         for result in all_results:
#             feature_freq_data = []
#             for feature, count in result['feature_counter'].most_common(result['consensus_k']):
#                 feature_freq_data.append({
#                     'Feature': feature,
#                     'Selection_Count': count,
#                     'Selection_Frequency': f"{count}/{n_outer_folds}"
#                 })

#             if feature_freq_data:
#                 feat_df = pd.DataFrame(feature_freq_data)
#                 feat_path = self.output_dir / f"feature_frequency_{result['model_name'].lower().replace(' ', '_')}.csv"
#                 feat_df.to_csv(feat_path, index=False)
#                 print(f"✓ Feature frequency ({result['model_name']}): {feat_path.name}")
#                 paths.append(feat_path)

#         return paths

#     def save_complete_bundle(self, all_results, config, data_info):
#         """Save complete results bundle"""
#         results_bundle_path = self.output_dir / "nested_cv_complete_results.pkl"

#         with open(results_bundle_path, 'wb') as f:
#             pickle.dump({
#                 'all_nested_results': all_results,
#                 'config': config,
#                 'data_info': data_info,
#                 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             }, f)

#         print(f"✓ Complete results saved: {results_bundle_path.name}")
#         return results_bundle_path

# # ============================================================================
# # MODEL TRAINER
# # ============================================================================

# class ModelTrainer:
#     """Train final production models"""

#     def __init__(self, output_dir, random_state):
#         self.output_dir = output_dir
#         self.random_state = random_state

#     def train_final_models(self, all_results, X_full, y):
#         """Train final production models on all data"""
#         print("\n" + "="*80)
#         print("TRAINING FINAL PRODUCTION MODELS")
#         print("="*80)

#         final_models = {}

#         for result in all_results:
#             print(f"\n--- {result['model_name']} ---")

#             model_type = result['model_type']
#             consensus_k = result['consensus_k']

#             most_common_features = result['feature_counter'].most_common(consensus_k)
#             selected_feature_names = [feat for feat, count in most_common_features]

#             print(f"Consensus k: {consensus_k}")
#             print(f"Using {len(selected_feature_names)} most frequently selected features")

#             X_final = X_full[selected_feature_names].values
#             y_final = y.values

#             final_model = ModelFactory.get_model_pipeline(model_type, self.random_state)

#             start_time = time.time()
#             final_model.fit(X_final, y_final)
#             train_time = time.time() - start_time

#             test_sample = X_final[:1]
#             start_time = time.time()
#             for _ in range(100):
#                 _ = final_model.predict(test_sample)
#             avg_pred_time = (time.time() - start_time) / 100

#             print(f"Training time: {train_time:.3f}s")
#             print(f"Average prediction time: {avg_pred_time*1000:.3f}ms")

#             model_filename = f"final_model_{result['model_name'].lower().replace(' ', '_')}.pkl"
#             model_path = self.output_dir / model_filename

#             with open(model_path, 'wb') as f:
#                 pickle.dump({
#                     'model': final_model,
#                     'selected_features': selected_feature_names,
#                     'consensus_k': consensus_k,
#                     'nested_cv_metrics': result['overall_metrics'],
#                     'training_time': train_time,
#                     'prediction_time': avg_pred_time,
#                 }, f)

#             print(f"✓ Model saved: {model_path.name}")

#             final_models[result['model_name']] = {
#                 'model': final_model,
#                 'path': model_path,
#                 'features': selected_feature_names,
#                 'train_time': train_time,
#                 'pred_time': avg_pred_time,
#             }

#         return final_models

#     def export_speed_comparison(self, final_models):
#         """Export prediction speed comparison"""
#         speed_data = []
#         for name, info in final_models.items():
#             speed_data.append({
#                 'Model': name,
#                 'Training_Time_s': info['train_time'],
#                 'Prediction_Time_ms': info['pred_time'] * 1000,
#                 'Features_Used': len(info['features'])
#             })

#         speed_df = pd.DataFrame(speed_data)
#         print("\n" + "="*80)
#         print("PREDICTION SPEED COMPARISON")
#         print("="*80)
#         print(speed_df.to_string(index=False))

#         speed_path = self.output_dir / "prediction_speed_comparison.csv"
#         speed_df.to_csv(speed_path, index=False)
#         print(f"\n✓ Speed comparison saved: {speed_path.name}")

#         return speed_path

# # ============================================================================
# # REPORT GENERATOR
# # ============================================================================

# class ReportGenerator:
#     """Generate final summary reports"""

#     @staticmethod
#     def generate_summary_report(all_results, final_models, output_dir, data_info, config, best_model_name):
#         """Generate and save comprehensive summary report"""

#         report_lines = []
#         report_lines.append("="*80)
#         report_lines.append("NESTED CROSS-VALIDATION - FINAL SUMMARY REPORT")
#         report_lines.append("="*80)
#         report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         report_lines.append(f"\nDataset: {data_info['n_patients']} patients, {data_info['n_features']} features")
#         report_lines.append(f"Class distribution: Control={data_info['n_control']}, CLL={data_info['n_cll']}")

#         report_lines.append(f"\nNested CV Configuration:")
#         report_lines.append(f"  Outer folds: {config['n_outer_folds']}")
#         report_lines.append(f"  Inner folds: {config['n_inner_folds']}")
#         report_lines.append(f"  k values tested: {config['k_values']}")
#         report_lines.append(f"  Optimization: {config['optimization_metric'].upper()} (primary), {config['secondary_metric'].upper()} (secondary)")

#         report_lines.append(f"\n{'Model':<20} {'Consensus k':<15} {'Accuracy':<12} {'AUC':<12} {'F1':<12}")
#         report_lines.append("-" * 80)
#         for result in all_results:
#             marker = " ← BEST" if result['model_name'] == best_model_name else ""
#             report_lines.append(f"{result['model_name']:<20} {result['consensus_k']:<15} "
#                   f"{result['overall_metrics']['accuracy']:<12.4f} "
#                   f"{result['overall_metrics']['auc']:<12.4f} "
#                   f"{result['overall_metrics']['f1']:<12.4f}{marker}")

#         report_lines.append(f"\nBest Model (by AUC): {best_model_name}")

#         report_lines.append(f"\nFinal Production Models:")
#         for name, info in final_models.items():
#             report_lines.append(f"  {name}:")
#             report_lines.append(f"    Path: {info['path'].name}")
#             report_lines.append(f"    Features: {len(info['features'])}")
#             report_lines.append(f"    Prediction time: {info['pred_time']*1000:.2f}ms")

#         report_lines.append(f"\nAll outputs saved to: {output_dir}")
#         report_lines.append(f"\nKey files:")
#         report_lines.append(f"  - Performance summary: nested_cv_performance_summary.csv")
#         report_lines.append(f"  - k selection details: nested_cv_k_selection.csv")
#         report_lines.append(f"  - Feature frequencies: feature_frequency_*.csv")
#         report_lines.append(f"  - Final models: final_model_*.pkl")
#         report_lines.append(f"  - Complete results: nested_cv_complete_results.pkl")
#         report_lines.append(f"  - Best model confusion matrix: confusion_matrix_best_model.png")
#         report_lines.append(f"  - This report: summary_report.txt")

#         report_lines.append("\n" + "="*80)
#         report_lines.append("ANALYSIS COMPLETE!")
#         report_lines.append("="*80)

#         report_text = "\n".join(report_lines)

#         print("\n" + report_text)

#         report_path = output_dir / "summary_report.txt"
#         with open(report_path, 'w') as f:
#             f.write(report_text)

#         print(f"\n✓ Summary report saved: {report_path.name}")

#         return report_path

# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================

# def main():
#     """Main execution function"""

#     print("="*80)
#     print("NESTED CROSS-VALIDATION FOR CLL CLASSIFICATION")
#     print("="*80)
#     print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("="*80)

#     # Setup
#     output_dir = Config.setup_output_directory()
#     Config.setup_plotting()

#     print(f"\n✓ Output directory: {output_dir}")
#     print(f"✓ Input file: {Config.INPUT_PKL}")
#     print(f"✓ Selected model: {Config.SELECTED_MODEL}")

#     # Load and prepare data
#     X_full, y, db = DataLoader.load_and_prepare(Config.INPUT_PKL, Config.SELECTED_MODEL)

#     # Resolve k values
#     k_values_resolved = []
#     for k in Config.K_VALUES:
#         if k == 'max':
#             k_values_resolved.append(X_full.shape[1])
#         else:
#             k_values_resolved.append(k)

#     # Create config dictionary
#     config = {
#         'random_state': Config.RANDOM_STATE,
#         'n_outer_folds': Config.N_OUTER_FOLDS,
#         'n_inner_folds': Config.N_INNER_FOLDS,
#         'k_values': Config.K_VALUES,
#         'k_values_resolved': k_values_resolved,
#         'optimization_metric': Config.OPTIMIZATION_METRIC,
#         'secondary_metric': Config.SECONDARY_METRIC,
#         'auc_tolerance': Config.AUC_TOLERANCE,
#     }

#     print(f"\nResolved k values: {k_values_resolved}")

#     # Run nested CV for all models
#     print("\n" + "="*80)
#     print("EXECUTING NESTED CROSS-VALIDATION")
#     print("="*80)

#     nested_cv = NestedCV(config)
#     all_nested_results = []

#     rf_results = nested_cv.run_nested_cv('rf', 'RandomForest', X_full, y)
#     all_nested_results.append(rf_results)

#     svm_results = nested_cv.run_nested_cv('svm', 'SVM', X_full, y)
#     all_nested_results.append(svm_results)

#     lr_results = nested_cv.run_nested_cv('lr', 'LogisticRegression', X_full, y)
#     all_nested_results.append(lr_results)

#     print("\n" + "="*80)
#     print("ALL NESTED CV COMPLETE")
#     print("="*80)

#     # Generate visualizations
#     print("\n" + "="*80)
#     print("GENERATING VISUALIZATIONS")
#     print("="*80)

#     visualizer = Visualizer(output_dir, Config.PLOT_CONFIG)

#     for result in all_nested_results:
#         print(f"\nGenerating plots for {result['model_name']}...")
#         visualizer.plot_k_selection_summary(result, config)
#         visualizer.plot_feature_importance(result, config, top_n=30)

#     visualizer.plot_model_comparison(all_nested_results)

#     # ── NEW: single confusion matrix for the best model ──
#     _, best_result = visualizer.plot_best_model_confusion_matrix(all_nested_results)
#     best_model_name = best_result['model_name']

#     print("\n✓ All visualizations generated")

#     # Export results
#     print("\n" + "="*80)
#     print("EXPORTING SUMMARY TABLES")
#     print("="*80)

#     exporter = ResultExporter(output_dir)
#     exporter.export_performance_summary(all_nested_results)
#     exporter.export_k_selection(all_nested_results, Config.N_OUTER_FOLDS)
#     exporter.export_feature_frequencies(all_nested_results, Config.N_OUTER_FOLDS)

#     data_info = {
#         'n_patients': len(X_full),
#         'n_features': X_full.shape[1],
#         'n_control': int((y==0).sum()),
#         'n_cll': int((y==1).sum()),
#     }

#     exporter.save_complete_bundle(all_nested_results, config, data_info)

#     # Train final models
#     trainer = ModelTrainer(output_dir, Config.RANDOM_STATE)
#     final_models = trainer.train_final_models(all_nested_results, X_full, y)
#     trainer.export_speed_comparison(final_models)

#     # Generate final report (now passes best_model_name)
#     ReportGenerator.generate_summary_report(
#         all_nested_results, final_models, output_dir, data_info, config, best_model_name
#     )

#     print("\n" + "="*80)
#     print("ALL TASKS COMPLETED SUCCESSFULLY!")
#     print("="*80)
#     print(f"\nAll results saved to: {output_dir}")
#     print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# """
# ================================================================================
# NESTED CROSS-VALIDATION FOR CLL CLASSIFICATION
# ================================================================================
# Modular Python script converted from Jupyter notebook.
# Performs nested CV with hyperparameter tuning and saves all outputs to a 
# structured folder.

# Author: Converted from notebook
# Date: February 2026
# ================================================================================
# """

# import os
# import sys
# import time
# import warnings
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# from collections import defaultdict, Counter
# from datetime import datetime

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
#                              f1_score, roc_auc_score, confusion_matrix)

# warnings.filterwarnings('ignore')

# # ============================================================================
# # CONFIGURATION
# # ============================================================================

# class Config:
#     """Configuration class for all parameters"""

#     # Paths
#     INPUT_PKL = Path('/mnt/lustre/home/claassen/clala950/DC-TSeg/Data/ETH/march2026/morphology_1500_mbar_results.pkl')
#     OUTPUT_ROOT_NAME = 'Classification_results_1500mbar'  # Will be created with timestamp

#     # Model selection
#     SELECTED_MODEL = 'unet'  # 'Ground_truth' or 'Unet_preds' or 'Unet_preds_CLL_focus_k1_5' or 'Unet_preds_CLL_focus_k2_0' or 'Unet_preds_CLL_focus_k2_5' or 'Unet_preds_CLL_focus_k3_0'
#     # Cross-validation parameters
#     RANDOM_STATE = 42
#     N_OUTER_FOLDS = 5
#     N_INNER_FOLDS = 5
#     K_VALUES = [5, 10, 15, 20, 25, 30, 50, 'max']

#     # Optimization
#     OPTIMIZATION_METRIC = 'auc'
#     SECONDARY_METRIC = 'f1'
#     AUC_TOLERANCE = 0.02

#     # Plotting
#     PLOT_CONFIG = {
#         'figure_dpi': 300,
#         'figure_size_single': (10, 6),
#         'figure_size_grid': (16, 12),
#         'font_size': 14,
#         'title_size': 18,
#         'label_size': 14,
#         'tick_size': 12,
#         'legend_size': 12,
#         'line_width': 2,
#         'marker_size': 8,
#     }

#     @classmethod
#     def setup_output_directory(cls):
#         """Create timestamped output directory"""
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_dir = cls.INPUT_PKL.parent / f'{cls.OUTPUT_ROOT_NAME}_{timestamp}'
#         output_dir.mkdir(parents=True, exist_ok=True)
#         return output_dir

#     @classmethod
#     def setup_plotting(cls):
#         """Apply global plotting configuration"""
#         plt.rcParams.update({
#             'font.size': cls.PLOT_CONFIG['font_size'],
#             'axes.titlesize': cls.PLOT_CONFIG['title_size'],
#             'axes.labelsize': cls.PLOT_CONFIG['label_size'],
#             'xtick.labelsize': cls.PLOT_CONFIG['tick_size'],
#             'ytick.labelsize': cls.PLOT_CONFIG['tick_size'],
#             'legend.fontsize': cls.PLOT_CONFIG['legend_size'],
#             'figure.titlesize': cls.PLOT_CONFIG['title_size'] + 4,
#             'lines.linewidth': cls.PLOT_CONFIG['line_width'],
#             'lines.markersize': cls.PLOT_CONFIG['marker_size'],
#         })

# # ============================================================================
# # DATA LOADING AND PREPARATION
# # ============================================================================

# class DataLoader:
#     """Handle data loading and feature aggregation"""

#     @staticmethod
#     def aggregate_morphology_stats(db, model_key, stats=('mean', 'median', 'std', 'cv')):
#         """Aggregate morphology statistics across all patients."""
#         morph_key = f'{model_key}_morphology'
#         patient_ids = [pid for pid in db.keys() 
#                        if morph_key in db[pid] and isinstance(db[pid][morph_key], pd.DataFrame)]

#         if not patient_ids:
#             raise ValueError(f"No patients contained a DataFrame under key '{morph_key}'")

#         # Find morphology columns
#         first_df = None
#         for pid in patient_ids:
#             df = db[pid][morph_key]
#             if isinstance(df, pd.DataFrame) and not df.empty:
#                 first_df = df
#                 break

#         if first_df is None or first_df.empty:
#             raise ValueError(f"All '{morph_key}' DataFrames are empty.")

#         morph_cols = first_df.columns.tolist()
#         if 'Area' not in morph_cols:
#             raise ValueError("'Area' column not found in morphology DataFrame.")
#         morph_cols = morph_cols[morph_cols.index('Area'):]

#         results = {'Patient_ID': []}
#         for col in morph_cols:
#             for stat in stats:
#                 results[f'{col}_{stat}'] = []

#         for pid in patient_ids:
#             results['Patient_ID'].append(pid)
#             df = db[pid][morph_key]

#             if df is None or df.empty:
#                 for col in morph_cols:
#                     for stat in stats:
#                         results[f'{col}_{stat}'].append(np.nan)
#                 continue

#             for col in morph_cols:
#                 col_data = pd.to_numeric(df[col], errors='coerce').dropna()
#                 if col_data.empty:
#                     vals = {s: np.nan for s in stats}
#                 else:
#                     vals = {}
#                     for stat in stats:
#                         if stat == 'mean':
#                             vals[stat] = col_data.mean()
#                         elif stat == 'median':
#                             vals[stat] = col_data.median()
#                         elif stat == 'std':
#                             vals[stat] = col_data.std()
#                         elif stat == 'cv':
#                             mean_val = col_data.mean()
#                             std_val = col_data.std()
#                             vals[stat] = (std_val / mean_val) if mean_val != 0 else np.nan
#                 for stat in stats:
#                     results[f'{col}_{stat}'].append(vals[stat])

#         agg_df = pd.DataFrame(results).set_index('Patient_ID')
#         return agg_df

#     @staticmethod
#     def prepare_data(df):
#         """Create X (features) and y (labels) from aggregated morphology."""
#         df = df.copy()
#         patient_ids = df.index.astype(str).tolist()
#         groups = [pid.split('_')[0] for pid in patient_ids]
#         y = pd.Series([1 if g.upper().startswith('CLL') else 0 for g in groups],
#                       index=df.index, name='Group')

#         X = df.copy()

#         return X, y

#     @classmethod
#     def load_and_prepare(cls, pkl_path, model_key):
#         """Load pickle and prepare data"""
#         print("="*80)
#         print("LOADING DATA")
#         print("="*80)

#         with open(pkl_path, 'rb') as f:
#             db = pickle.load(f)
#         print(f"✓ Loaded database with {len(db)} patients")

#         agg_df = cls.aggregate_morphology_stats(db, model_key, stats=('mean', 'median', 'std', 'cv'))
#         print(f"✓ Aggregated shape: {agg_df.shape}")

#         X, y = cls.prepare_data(agg_df)

#         print(f"\nTotal samples: {len(X)}")
#         print(f"Total features: {X.shape[1]}")
#         print(f"\nClass distribution:")
#         print(f"  Control (0): {int((y==0).sum())} ({(y==0).mean()*100:.1f}%)")
#         print(f"  CLL (1):     {int((y==1).sum())} ({(y==1).mean()*100:.1f}%)")

#         return X, y, db

# # ============================================================================
# # MODEL DEFINITIONS
# # ============================================================================

# class ModelFactory:
#     """Factory for creating model pipelines"""

#     @staticmethod
#     def get_model_pipeline(model_type, random_state=42):
#         """Create sklearn pipeline for given model type."""
#         if model_type == 'rf':
#             return Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('classifier', RandomForestClassifier(
#                     n_estimators=100, max_depth=4,
#                     min_samples_split=3, min_samples_leaf=1,
#                     class_weight='balanced', 
#                     random_state=random_state
#                 ))
#             ])
#         elif model_type == 'svm':
#             return Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('classifier', SVC(
#                     kernel='rbf', probability=True, C=1.0,
#                     class_weight='balanced',
#                     random_state=random_state
#                 ))
#             ])
#         elif model_type == 'lr':
#             return Pipeline([
#                 ('scaler', StandardScaler()),
#                 ('classifier', LogisticRegression(
#                     random_state=random_state, 
#                     max_iter=1000, 
#                     class_weight='balanced'
#                 ))
#             ])
#         else:
#             raise ValueError(f"Unknown model type: {model_type}")

# # ============================================================================
# # FEATURE SELECTION
# # ============================================================================

# class FeatureSelector:
#     """Handle feature selection operations"""

#     @staticmethod
#     def select_features_for_k(X_train, y_train, X_test, k, feature_names):
#         """
#         Select top k features using SelectKBest on training data.
#         Returns: X_train_reduced, X_test_reduced, selected_feature_names
#         """
#         k_actual = min(k, X_train.shape[1])
#         selector = SelectKBest(f_classif, k=k_actual)

#         X_train_reduced = selector.fit_transform(X_train, y_train)
#         X_test_reduced = selector.transform(X_test)

#         selected_mask = selector.get_support()
#         selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]

#         return X_train_reduced, X_test_reduced, selected_features

# # ============================================================================
# # NESTED CROSS-VALIDATION
# # ============================================================================

# class NestedCV:
#     """Nested cross-validation implementation"""

#     def __init__(self, config):
#         self.config = config

#     def run_inner_cv_for_k_selection(self, X_train, y_train, k_values, model_type):
#         """Run inner CV to select optimal k."""
#         feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'f{i}' for i in range(X_train.shape[1])]

#         if not isinstance(X_train, np.ndarray):
#             X_train = X_train.values
#         if not isinstance(y_train, np.ndarray):
#             y_train = y_train.values

#         k_scores = {k: [] for k in k_values}
#         inner_cv = StratifiedKFold(n_splits=self.config['n_inner_folds'], 
#                                    shuffle=True, 
#                                    random_state=self.config['random_state'])

#         for k in k_values:
#             for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
#                 X_inner_train = X_train[inner_train_idx]
#                 X_inner_val = X_train[inner_val_idx]
#                 y_inner_train = y_train[inner_train_idx]
#                 y_inner_val = y_train[inner_val_idx]

#                 # Select features
#                 X_inner_train_reduced, X_inner_val_reduced, _ = FeatureSelector.select_features_for_k(
#                     X_inner_train, y_inner_train, X_inner_val, k, feature_names
#                 )

#                 # Train model
#                 model = ModelFactory.get_model_pipeline(model_type, self.config['random_state'])
#                 model.fit(X_inner_train_reduced, y_inner_train)

#                 # Evaluate
#                 y_pred = model.predict(X_inner_val_reduced)
#                 y_proba = model.predict_proba(X_inner_val_reduced)[:, 1] if hasattr(model, "predict_proba") else y_pred

#                 metrics = {
#                     'accuracy': accuracy_score(y_inner_val, y_pred),
#                     'f1': f1_score(y_inner_val, y_pred, zero_division=0),
#                     'auc': roc_auc_score(y_inner_val, y_proba) if len(np.unique(y_inner_val)) > 1 else 0.5
#                 }

#                 k_scores[k].append(metrics)

#         # Average scores across inner folds
#         k_avg_scores = {}
#         for k in k_values:
#             k_avg_scores[k] = {
#                 'accuracy': np.mean([s['accuracy'] for s in k_scores[k]]),
#                 'f1': np.mean([s['f1'] for s in k_scores[k]]),
#                 'auc': np.mean([s['auc'] for s in k_scores[k]]),
#             }

#         # Select optimal k
#         primary_metric = self.config['optimization_metric']
#         secondary_metric = self.config['secondary_metric']

#         sorted_k = sorted(k_values, key=lambda k: k_avg_scores[k][primary_metric], reverse=True)
#         best_k = sorted_k[0]
#         best_score = k_avg_scores[best_k][primary_metric]

#         candidates = [k for k in sorted_k if abs(k_avg_scores[k][primary_metric] - best_score) <= self.config['auc_tolerance']]

#         if len(candidates) > 1:
#             best_k = max(candidates, key=lambda k: k_avg_scores[k][secondary_metric])

#         return best_k, k_avg_scores

#     def run_nested_cv(self, model_type, model_name, X, y):
#         """Run nested cross-validation with hyperparameter tuning."""
#         print(f"\n{'='*80}")
#         print(f"NESTED CV: {model_name}")
#         print(f"{'='*80}")

#         start_time = time.time()

#         feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])]
#         k_values = self.config['k_values_resolved']

#         outer_cv = StratifiedKFold(n_splits=self.config['n_outer_folds'], 
#                                    shuffle=True, 
#                                    random_state=self.config['random_state'])

#         # Storage for results
#         outer_fold_results = []
#         all_selected_k = []
#         all_selected_features = []
#         y_true_all = []
#         y_pred_all = []
#         y_proba_all = []

#         for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
#             print(f"\n--- Outer Fold {outer_fold_idx}/{self.config['n_outer_folds']} ---")

#             # Split data
#             if isinstance(X, pd.DataFrame):
#                 X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#             else:
#                 X_train, X_test = X[train_idx], X[test_idx]
#             y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#             # Inner CV: Select optimal k
#             print(f"  Running inner CV for k selection...")
#             optimal_k, k_avg_scores = self.run_inner_cv_for_k_selection(
#                 X_train, y_train, k_values, model_type
#             )
#             print(f"  Optimal k selected: {optimal_k}")
#             print(f"  Inner CV scores: AUC={k_avg_scores[optimal_k]['auc']:.3f}, F1={k_avg_scores[optimal_k]['f1']:.3f}")

#             all_selected_k.append(optimal_k)

#             # Train model with optimal k on outer training data
#             X_train_arr = X_train.values if hasattr(X_train, 'values') else X_train
#             X_test_arr = X_test.values if hasattr(X_test, 'values') else X_test
#             y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
#             y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test

#             X_train_reduced, X_test_reduced, selected_features = FeatureSelector.select_features_for_k(
#                 X_train_arr, y_train_arr, X_test_arr, optimal_k, feature_names
#             )

#             all_selected_features.extend(selected_features)

#             model = ModelFactory.get_model_pipeline(model_type, self.config['random_state'])
#             model.fit(X_train_reduced, y_train_arr)

#             # Predict on outer test fold
#             y_pred = model.predict(X_test_reduced)
#             y_proba = model.predict_proba(X_test_reduced)[:, 1] if hasattr(model, "predict_proba") else y_pred

#             # Store predictions
#             y_true_all.extend(y_test_arr)
#             y_pred_all.extend(y_pred)
#             y_proba_all.extend(y_proba)

#             # Calculate fold metrics
#             fold_metrics = {
#                 'accuracy': accuracy_score(y_test_arr, y_pred),
#                 'precision': precision_score(y_test_arr, y_pred, zero_division=0),
#                 'recall': recall_score(y_test_arr, y_pred, zero_division=0),
#                 'f1': f1_score(y_test_arr, y_pred, zero_division=0),
#                 'auc': roc_auc_score(y_test_arr, y_proba) if len(np.unique(y_test_arr)) > 1 else 0.5
#             }

#             print(f"  Outer fold performance: Acc={fold_metrics['accuracy']:.3f}, AUC={fold_metrics['auc']:.3f}")

#             outer_fold_results.append({
#                 'fold': outer_fold_idx,
#                 'optimal_k': optimal_k,
#                 'selected_features': selected_features,
#                 'k_scores': k_avg_scores,
#                 'metrics': fold_metrics,
#                 'test_indices': test_idx,
#             })

#         # Calculate overall nested CV metrics
#         y_true_all = np.array(y_true_all)
#         y_pred_all = np.array(y_pred_all)
#         y_proba_all = np.array(y_proba_all)

#         overall_metrics = {
#             'accuracy': accuracy_score(y_true_all, y_pred_all),
#             'precision': precision_score(y_true_all, y_pred_all, zero_division=0),
#             'recall': recall_score(y_true_all, y_pred_all, zero_division=0),
#             'f1': f1_score(y_true_all, y_pred_all, zero_division=0),
#             'auc': roc_auc_score(y_true_all, y_proba_all) if len(np.unique(y_true_all)) > 1 else 0.5
#         }

#         cm = confusion_matrix(y_true_all, y_pred_all)
#         end_time = time.time()

#         # Determine consensus k
#         k_counter = Counter(all_selected_k)
#         consensus_k = k_counter.most_common(1)[0][0]

#         # Count feature selection frequency
#         feature_counter = Counter(all_selected_features)
#         most_common_features = feature_counter.most_common(consensus_k)

#         print(f"\n{'='*80}")
#         print(f"NESTED CV RESULTS: {model_name}")
#         print(f"{'='*80}")
#         print(f"Overall Metrics:")
#         for metric, value in overall_metrics.items():
#             print(f"  {metric.upper():12}: {value:.4f}")
#         print(f"\nk Selection Across Folds: {all_selected_k}")
#         print(f"Consensus k: {consensus_k} (selected in {k_counter[consensus_k]}/{self.config['n_outer_folds']} folds)")
#         print(f"\nExecution time: {end_time - start_time:.2f}s")

#         return {
#             'model_name': model_name,
#             'model_type': model_type,
#             'overall_metrics': overall_metrics,
#             'confusion_matrix': cm,
#             'outer_fold_results': outer_fold_results,
#             'all_selected_k': all_selected_k,
#             'consensus_k': consensus_k,
#             'k_counter': k_counter,
#             'feature_counter': feature_counter,
#             'most_common_features': most_common_features,
#             'y_true': y_true_all,
#             'y_pred': y_pred_all,
#             'y_proba': y_proba_all,
#             'execution_time': end_time - start_time,
#             'feature_names': feature_names,
#         }

# # ============================================================================
# # VISUALIZATION
# # ============================================================================

# class Visualizer:
#     """Handle all visualization tasks"""

#     def __init__(self, output_dir, plot_config):
#         self.output_dir = output_dir
#         self.plot_config = plot_config

#     def plot_k_selection_summary(self, nested_results, config):
#         """Plot k selection frequency and elbow curves."""
#         model_name = nested_results['model_name']

#         fig, axes = plt.subplots(2, 2, figsize=self.plot_config['figure_size_grid'])

#         # 1. k Selection Frequency
#         k_counter = nested_results['k_counter']
#         k_values = sorted(k_counter.keys())
#         k_counts = [k_counter[k] for k in k_values]

#         axes[0, 0].bar(range(len(k_values)), k_counts, edgecolor='black')
#         axes[0, 0].set_xticks(range(len(k_values)))
#         axes[0, 0].set_xticklabels(k_values)
#         axes[0, 0].set_xlabel('k (Number of Features)')
#         axes[0, 0].set_ylabel('Selection Frequency')
#         axes[0, 0].set_title(f'{model_name}: k Selection Across Outer Folds')
#         axes[0, 0].grid(True, alpha=0.3)

#         # 2. Elbow Plot - AUC vs k
#         k_values_all = config['k_values_resolved']
#         auc_by_k = defaultdict(list)
#         f1_by_k = defaultdict(list)

#         for fold_result in nested_results['outer_fold_results']:
#             k_scores = fold_result['k_scores']
#             for k in k_values_all:
#                 if k in k_scores:
#                     auc_by_k[k].append(k_scores[k]['auc'])
#                     f1_by_k[k].append(k_scores[k]['f1'])

#         k_sorted = sorted(auc_by_k.keys())
#         auc_means = [np.mean(auc_by_k[k]) for k in k_sorted]
#         auc_stds = [np.std(auc_by_k[k]) for k in k_sorted]

#         axes[0, 1].errorbar(k_sorted, auc_means, yerr=auc_stds, marker='o', capsize=5)
#         axes[0, 1].set_xlabel('k (Number of Features)')
#         axes[0, 1].set_ylabel('AUC (Inner CV)')
#         axes[0, 1].set_title(f'{model_name}: Elbow Plot (AUC)')
#         axes[0, 1].grid(True, alpha=0.3)
#         axes[0, 1].axvline(nested_results['consensus_k'], color='red', linestyle='--', 
#                           label=f"Consensus k={nested_results['consensus_k']}")
#         axes[0, 1].legend()

#         # 3. Elbow Plot - F1 vs k
#         f1_means = [np.mean(f1_by_k[k]) for k in k_sorted]
#         f1_stds = [np.std(f1_by_k[k]) for k in k_sorted]

#         axes[1, 0].errorbar(k_sorted, f1_means, yerr=f1_stds, marker='s', capsize=5, color='green')
#         axes[1, 0].set_xlabel('k (Number of Features)')
#         axes[1, 0].set_ylabel('F1 Score (Inner CV)')
#         axes[1, 0].set_title(f'{model_name}: Elbow Plot (F1)')
#         axes[1, 0].grid(True, alpha=0.3)
#         axes[1, 0].axvline(nested_results['consensus_k'], color='red', linestyle='--', 
#                           label=f"Consensus k={nested_results['consensus_k']}")
#         axes[1, 0].legend()

#         # 4. Confusion Matrix
#         cm = nested_results['confusion_matrix']
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
#                     xticklabels=['Control', 'CLL'], yticklabels=['Control', 'CLL'])
#         axes[1, 1].set_xlabel('Predicted')
#         axes[1, 1].set_ylabel('True')
#         axes[1, 1].set_title(f'{model_name}: Confusion Matrix (Nested CV)')

#         plt.tight_layout()
#         save_path = self.output_dir / f"nested_cv_{model_name.lower().replace(' ', '_')}.png"
#         plt.savefig(save_path, dpi=self.plot_config['figure_dpi'], bbox_inches='tight')
#         plt.close()

#         return save_path

#     def plot_feature_importance(self, nested_results, config, top_n=30):
#         """Plot feature selection frequency across nested CV folds."""
#         model_name = nested_results['model_name']
#         consensus_k = nested_results['consensus_k']
#         feature_counter = nested_results['feature_counter']

#         most_common = feature_counter.most_common(top_n)
#         if not most_common:
#             print(f"No features to plot for {model_name}")
#             return None

#         features, counts = zip(*most_common)

#         fig, ax = plt.subplots(figsize=(10, max(6, 0.4*len(features))))
#         y_pos = np.arange(len(features))
#         ax.barh(y_pos, counts, align='center')
#         ax.set_yticks(y_pos)
#         ax.set_yticklabels(features)
#         ax.invert_yaxis()
#         ax.set_xlabel('Selection Frequency (Across All Outer Folds)')
#         ax.set_title(f'{model_name}: Most Frequently Selected Features\n(Consensus k={consensus_k})')
#         ax.grid(True, alpha=0.3, axis='x')

#         plt.tight_layout()
#         save_path = self.output_dir / f"feature_frequency_{model_name.lower().replace(' ', '_')}.png"
#         plt.savefig(save_path, dpi=self.plot_config['figure_dpi'], bbox_inches='tight')
#         plt.close()

#         return save_path

#     def plot_model_comparison(self, all_results):
#         """Compare all models' nested CV performance."""
#         fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#         model_names = [r['model_name'] for r in all_results]

#         # Metrics comparison
#         metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
#         x = np.arange(len(model_names))
#         width = 0.15

#         for i, metric in enumerate(metrics):
#             values = [r['overall_metrics'][metric] for r in all_results]
#             axes[0].bar(x + i*width, values, width, label=metric.upper())

#         axes[0].set_xticks(x + width * 2)
#         axes[0].set_xticklabels(model_names, rotation=15, ha='right')
#         axes[0].set_ylabel('Score')
#         axes[0].set_ylim([0, 1.05])
#         axes[0].set_title('Nested CV Performance Comparison')
#         axes[0].legend()
#         axes[0].grid(True, alpha=0.3, axis='y')

#         # Consensus k comparison
#         consensus_ks = [r['consensus_k'] for r in all_results]
#         axes[1].bar(model_names, consensus_ks, edgecolor='black')
#         axes[1].set_ylabel('Consensus k')
#         axes[1].set_title('Optimal k Selected by Each Model')
#         axes[1].grid(True, alpha=0.3, axis='y')

#         plt.tight_layout()
#         save_path = self.output_dir / "model_comparison.png"
#         plt.savefig(save_path, dpi=self.plot_config['figure_dpi'], bbox_inches='tight')
#         plt.close()

#         return save_path

# # ============================================================================
# # RESULT EXPORTER
# # ============================================================================

# class ResultExporter:
#     """Export results to CSV and pickle files"""

#     def __init__(self, output_dir):
#         self.output_dir = output_dir

#     def export_performance_summary(self, all_results):
#         """Export overall performance comparison"""
#         performance_data = []
#         for result in all_results:
#             row = {'Model': result['model_name'], 'Consensus_k': result['consensus_k']}
#             row.update(result['overall_metrics'])
#             performance_data.append(row)

#         performance_df = pd.DataFrame(performance_data)
#         perf_path = self.output_dir / "nested_cv_performance_summary.csv"
#         performance_df.to_csv(perf_path, index=False)
#         print(f"✓ Performance summary: {perf_path.name}")
#         print(performance_df.to_string(index=False))

#         return perf_path

#     def export_k_selection(self, all_results, n_outer_folds):
#         """Export k selection details"""
#         k_selection_data = []
#         for result in all_results:
#             for fold_result in result['outer_fold_results']:
#                 k_selection_data.append({
#                     'Model': result['model_name'],
#                     'Outer_Fold': fold_result['fold'],
#                     'Selected_k': fold_result['optimal_k'],
#                     'Fold_Accuracy': fold_result['metrics']['accuracy'],
#                     'Fold_AUC': fold_result['metrics']['auc'],
#                     'Fold_F1': fold_result['metrics']['f1'],
#                 })

#         k_selection_df = pd.DataFrame(k_selection_data)
#         k_path = self.output_dir / "nested_cv_k_selection.csv"
#         k_selection_df.to_csv(k_path, index=False)
#         print(f"✓ k selection details: {k_path.name}")

#         return k_path

#     def export_feature_frequencies(self, all_results, n_outer_folds):
#         """Export feature frequency tables"""
#         paths = []
#         for result in all_results:
#             feature_freq_data = []
#             for feature, count in result['feature_counter'].most_common(result['consensus_k']):
#                 feature_freq_data.append({
#                     'Feature': feature,
#                     'Selection_Count': count,
#                     'Selection_Frequency': f"{count}/{n_outer_folds}"
#                 })

#             if feature_freq_data:
#                 feat_df = pd.DataFrame(feature_freq_data)
#                 feat_path = self.output_dir / f"feature_frequency_{result['model_name'].lower().replace(' ', '_')}.csv"
#                 feat_df.to_csv(feat_path, index=False)
#                 print(f"✓ Feature frequency ({result['model_name']}): {feat_path.name}")
#                 paths.append(feat_path)

#         return paths

#     def save_complete_bundle(self, all_results, config, data_info):
#         """Save complete results bundle"""
#         results_bundle_path = self.output_dir / "nested_cv_complete_results.pkl"

#         with open(results_bundle_path, 'wb') as f:
#             pickle.dump({
#                 'all_nested_results': all_results,
#                 'config': config,
#                 'data_info': data_info,
#                 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             }, f)

#         print(f"✓ Complete results saved: {results_bundle_path.name}")
#         return results_bundle_path

# # ============================================================================
# # MODEL TRAINER
# # ============================================================================

# class ModelTrainer:
#     """Train final production models"""

#     def __init__(self, output_dir, random_state):
#         self.output_dir = output_dir
#         self.random_state = random_state

#     def train_final_models(self, all_results, X_full, y):
#         """Train final production models on all data"""
#         print("\n" + "="*80)
#         print("TRAINING FINAL PRODUCTION MODELS")
#         print("="*80)

#         final_models = {}

#         for result in all_results:
#             print(f"\n--- {result['model_name']} ---")

#             model_type = result['model_type']
#             consensus_k = result['consensus_k']

#             # Get most frequent features
#             most_common_features = result['feature_counter'].most_common(consensus_k)
#             selected_feature_names = [feat for feat, count in most_common_features]

#             print(f"Consensus k: {consensus_k}")
#             print(f"Using {len(selected_feature_names)} most frequently selected features")

#             # Prepare data with selected features
#             X_final = X_full[selected_feature_names].values
#             y_final = y.values

#             # Train final model on ALL data
#             final_model = ModelFactory.get_model_pipeline(model_type, self.random_state)

#             start_time = time.time()
#             final_model.fit(X_final, y_final)
#             train_time = time.time() - start_time

#             # Test prediction speed
#             test_sample = X_final[:1]
#             start_time = time.time()
#             for _ in range(100):
#                 _ = final_model.predict(test_sample)
#             avg_pred_time = (time.time() - start_time) / 100

#             print(f"Training time: {train_time:.3f}s")
#             print(f"Average prediction time: {avg_pred_time*1000:.3f}ms")

#             # Save model
#             model_filename = f"final_model_{result['model_name'].lower().replace(' ', '_')}.pkl"
#             model_path = self.output_dir / model_filename

#             with open(model_path, 'wb') as f:
#                 pickle.dump({
#                     'model': final_model,
#                     'selected_features': selected_feature_names,
#                     'consensus_k': consensus_k,
#                     'nested_cv_metrics': result['overall_metrics'],
#                     'training_time': train_time,
#                     'prediction_time': avg_pred_time,
#                 }, f)

#             print(f"✓ Model saved: {model_path.name}")

#             final_models[result['model_name']] = {
#                 'model': final_model,
#                 'path': model_path,
#                 'features': selected_feature_names,
#                 'train_time': train_time,
#                 'pred_time': avg_pred_time,
#             }

#         return final_models

#     def export_speed_comparison(self, final_models):
#         """Export prediction speed comparison"""
#         speed_data = []
#         for name, info in final_models.items():
#             speed_data.append({
#                 'Model': name,
#                 'Training_Time_s': info['train_time'],
#                 'Prediction_Time_ms': info['pred_time'] * 1000,
#                 'Features_Used': len(info['features'])
#             })

#         speed_df = pd.DataFrame(speed_data)
#         print("\n" + "="*80)
#         print("PREDICTION SPEED COMPARISON")
#         print("="*80)
#         print(speed_df.to_string(index=False))

#         speed_path = self.output_dir / "prediction_speed_comparison.csv"
#         speed_df.to_csv(speed_path, index=False)
#         print(f"\n✓ Speed comparison saved: {speed_path.name}")

#         return speed_path

# # ============================================================================
# # REPORT GENERATOR
# # ============================================================================

# class ReportGenerator:
#     """Generate final summary reports"""

#     @staticmethod
#     def generate_summary_report(all_results, final_models, output_dir, data_info, config):
#         """Generate and save comprehensive summary report"""

#         report_lines = []
#         report_lines.append("="*80)
#         report_lines.append("NESTED CROSS-VALIDATION - FINAL SUMMARY REPORT")
#         report_lines.append("="*80)
#         report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         report_lines.append(f"\nDataset: {data_info['n_patients']} patients, {data_info['n_features']} features")
#         report_lines.append(f"Class distribution: Control={data_info['n_control']}, CLL={data_info['n_cll']}")

#         report_lines.append(f"\nNested CV Configuration:")
#         report_lines.append(f"  Outer folds: {config['n_outer_folds']}")
#         report_lines.append(f"  Inner folds: {config['n_inner_folds']}")
#         report_lines.append(f"  k values tested: {config['k_values']}")
#         report_lines.append(f"  Optimization: {config['optimization_metric'].upper()} (primary), {config['secondary_metric'].upper()} (secondary)")

#         report_lines.append(f"\n{'Model':<20} {'Consensus k':<15} {'Accuracy':<12} {'AUC':<12} {'F1':<12}")
#         report_lines.append("-" * 80)
#         for result in all_results:
#             report_lines.append(f"{result['model_name']:<20} {result['consensus_k']:<15} "
#                   f"{result['overall_metrics']['accuracy']:<12.4f} "
#                   f"{result['overall_metrics']['auc']:<12.4f} "
#                   f"{result['overall_metrics']['f1']:<12.4f}")

#         report_lines.append(f"\nFinal Production Models:")
#         for name, info in final_models.items():
#             report_lines.append(f"  {name}:")
#             report_lines.append(f"    Path: {info['path'].name}")
#             report_lines.append(f"    Features: {len(info['features'])}")
#             report_lines.append(f"    Prediction time: {info['pred_time']*1000:.2f}ms")

#         report_lines.append(f"\nAll outputs saved to: {output_dir}")
#         report_lines.append(f"\nKey files:")
#         report_lines.append(f"  - Performance summary: nested_cv_performance_summary.csv")
#         report_lines.append(f"  - k selection details: nested_cv_k_selection.csv")
#         report_lines.append(f"  - Feature frequencies: feature_frequency_*.csv")
#         report_lines.append(f"  - Final models: final_model_*.pkl")
#         report_lines.append(f"  - Complete results: nested_cv_complete_results.pkl")
#         report_lines.append(f"  - This report: summary_report.txt")

#         report_lines.append("\n" + "="*80)
#         report_lines.append("ANALYSIS COMPLETE!")
#         report_lines.append("="*80)

#         report_text = "\n".join(report_lines)

#         # Print to console
#         print("\n" + report_text)

#         # Save to file
#         report_path = output_dir / "summary_report.txt"
#         with open(report_path, 'w') as f:
#             f.write(report_text)

#         print(f"\n✓ Summary report saved: {report_path.name}")

#         return report_path

# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================

# def main():
#     """Main execution function"""

#     print("="*80)
#     print("NESTED CROSS-VALIDATION FOR CLL CLASSIFICATION")
#     print("="*80)
#     print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print("="*80)

#     # Setup
#     output_dir = Config.setup_output_directory()
#     Config.setup_plotting()

#     print(f"\n✓ Output directory: {output_dir}")
#     print(f"✓ Input file: {Config.INPUT_PKL}")
#     print(f"✓ Selected model: {Config.SELECTED_MODEL}")

#     # Load and prepare data
#     X_full, y, db = DataLoader.load_and_prepare(Config.INPUT_PKL, Config.SELECTED_MODEL)

#     # Resolve k values
#     k_values_resolved = []
#     for k in Config.K_VALUES:
#         if k == 'max':
#             k_values_resolved.append(X_full.shape[1])
#         else:
#             k_values_resolved.append(k)

#     # Create config dictionary
#     config = {
#         'random_state': Config.RANDOM_STATE,
#         'n_outer_folds': Config.N_OUTER_FOLDS,
#         'n_inner_folds': Config.N_INNER_FOLDS,
#         'k_values': Config.K_VALUES,
#         'k_values_resolved': k_values_resolved,
#         'optimization_metric': Config.OPTIMIZATION_METRIC,
#         'secondary_metric': Config.SECONDARY_METRIC,
#         'auc_tolerance': Config.AUC_TOLERANCE,
#     }

#     print(f"\nResolved k values: {k_values_resolved}")

#     # Run nested CV for all models
#     print("\n" + "="*80)
#     print("EXECUTING NESTED CROSS-VALIDATION")
#     print("="*80)

#     nested_cv = NestedCV(config)
#     all_nested_results = []

#     # RandomForest
#     rf_results = nested_cv.run_nested_cv('rf', 'RandomForest', X_full, y)
#     all_nested_results.append(rf_results)

#     # SVM
#     svm_results = nested_cv.run_nested_cv('svm', 'SVM', X_full, y)
#     all_nested_results.append(svm_results)

#     # Logistic Regression
#     lr_results = nested_cv.run_nested_cv('lr', 'LogisticRegression', X_full, y)
#     all_nested_results.append(lr_results)

#     print("\n" + "="*80)
#     print("ALL NESTED CV COMPLETE")
#     print("="*80)

#     # Generate visualizations
#     print("\n" + "="*80)
#     print("GENERATING VISUALIZATIONS")
#     print("="*80)

#     visualizer = Visualizer(output_dir, Config.PLOT_CONFIG)

#     for result in all_nested_results:
#         print(f"\nGenerating plots for {result['model_name']}...")
#         visualizer.plot_k_selection_summary(result, config)
#         visualizer.plot_feature_importance(result, config, top_n=30)

#     visualizer.plot_model_comparison(all_nested_results)
#     print("\n✓ All visualizations generated")

#     # Export results
#     print("\n" + "="*80)
#     print("EXPORTING SUMMARY TABLES")
#     print("="*80)

#     exporter = ResultExporter(output_dir)
#     exporter.export_performance_summary(all_nested_results)
#     exporter.export_k_selection(all_nested_results, Config.N_OUTER_FOLDS)
#     exporter.export_feature_frequencies(all_nested_results, Config.N_OUTER_FOLDS)

#     data_info = {
#         'n_patients': len(X_full),
#         'n_features': X_full.shape[1],
#         'n_control': int((y==0).sum()),
#         'n_cll': int((y==1).sum()),
#     }

#     exporter.save_complete_bundle(all_nested_results, config, data_info)

#     # Train final models
#     trainer = ModelTrainer(output_dir, Config.RANDOM_STATE)
#     final_models = trainer.train_final_models(all_nested_results, X_full, y)
#     trainer.export_speed_comparison(final_models)

#     # Generate final report
#     ReportGenerator.generate_summary_report(
#         all_nested_results, final_models, output_dir, data_info, config
#     )

#     print("\n" + "="*80)
#     print("ALL TASKS COMPLETED SUCCESSFULLY!")
#     print("="*80)
#     print(f"\nAll results saved to: {output_dir}")
#     print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# if __name__ == "__main__":
#     main()
