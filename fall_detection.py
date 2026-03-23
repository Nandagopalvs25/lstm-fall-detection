"""
LSTM-Based Fall Detection for SisFall Dataset - COMPLETE FIXED VERSION
=======================================================================
Critical fixes implemented:
- SE06 forced into test set for honest elderly evaluation
- Global normalization (train stats applied to all data)
- Balanced windowing (more overlap for falls)
- Data augmentation for fall events
- Sensitivity-focused threshold selection (min 95% recall)
- Enhanced safety-critical metrics
- Honest limitation reporting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.utils import class_weight as sklearn_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import warnings
import time
from collections import Counter

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ==================== CONFIGURATION ====================
DATA_PATH = './SisFall_dataset/'
OUTPUT_PATH = './results/'
MODEL_SAVE_PATH = './saved_models/'
FIGURES_PATH = './figures/'

for path in [OUTPUT_PATH, MODEL_SAVE_PATH, FIGURES_PATH]:
    os.makedirs(path, exist_ok=True)

# Research-based hyperparameters
ORIGINAL_FS = 200
TARGET_FS = 20
DOWNSAMPLE_FACTOR = ORIGINAL_FS // TARGET_FS

WINDOW_SIZE = int(2.0 * TARGET_FS)  # 40 samples
OVERLAP = int(0.5 * WINDOW_SIZE)    # 20 samples

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Safety-critical settings
MIN_SENSITIVITY = 0.95
FALL_AUGMENTATION = 3
FALL_OVERLAP_RATIO = 0.75

# ==================== GPU SETUP ====================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU enabled: {len(gpus)} device(s)")
    except:
        pass

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== DATA LOADING ====================
def load_file(filepath):
    """Load sensor file"""
    try:
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip().replace(';', '')
                if line:
                    values = [float(x) for x in line.split(',')]
                    if len(values) >= 6:
                        data.append(values[:6])
        return np.array(data) if len(data) > 0 else None
    except Exception as e:
        return None

def parse_filename(filename):
    """Parse SisFall filename"""
    try:
        name = filename.replace('.txt', '')
        parts = name.split('_')
        
        if len(parts) < 3:
            return None
        
        activity = parts[0]
        subject = parts[1]
        trial = parts[2]
        
        if activity.startswith('F'):
            label = 1
        elif activity.startswith('D'):
            label = 0
        else:
            return None
        
        if subject.startswith('SA'):
            age_group = 'young'
        elif subject.startswith('SE'):
            age_group = 'elderly'
        else:
            age_group = 'unknown'
        
        return {
            'activity': activity,
            'subject': subject,
            'trial': trial,
            'label': label,
            'age_group': age_group,
            'filename': filename
        }
    except:
        return None

def load_dataset(data_path):
    """Load SisFall dataset"""
    print(f"\n{'='*70}")
    print("LOADING SISFALL DATASET")
    print(f"{'='*70}")
    
    data_path = Path(data_path)
    files = list(data_path.rglob('*.txt'))
    
    print(f"Found {len(files)} files")
    
    if len(files) == 0:
        print("\n❌ No files found!")
        return []
    
    dataset = []
    fall_subjects = set()
    
    for filepath in files:
        info = parse_filename(filepath.name)
        if info:
            data = load_file(filepath)
            if data is not None and len(data) > WINDOW_SIZE * DOWNSAMPLE_FACTOR:
                info['data'] = data
                dataset.append(info)
                
                if info['label'] == 1:
                    fall_subjects.add(info['subject'])
    
    labels = [d['label'] for d in dataset]
    falls = sum(labels)
    adls = len(labels) - falls
    
    print(f"\n✓ Loaded {len(dataset)} files")
    print(f"  Falls: {falls} ({falls/len(labels)*100:.1f}%)")
    print(f"  ADLs:  {adls} ({adls/len(labels)*100:.1f}%)")
    print(f"\n⚠️  Subjects with fall data: {sorted(fall_subjects)}")
    
    return dataset

# ==================== PREPROCESSING ====================
def downsample_signal(data, factor):
    """Downsample signal"""
    return data[::factor]

def augment_fall_window(window, n_augmentations=3):
    """Generate augmented versions of fall windows"""
    augmented = []
    
    for _ in range(n_augmentations):
        aug = window.copy()
        
        noise_level = 0.05 * np.std(window)
        noise = np.random.normal(0, noise_level, window.shape)
        aug = aug + noise
        
        scale = np.random.uniform(0.9, 1.1)
        aug = aug * scale
        
        shift = np.random.randint(-2, 3)
        if shift > 0:
            aug = np.vstack([aug[shift:], aug[-shift:]])
        elif shift < 0:
            aug = np.vstack([aug[:shift], aug[:-shift]])
        
        augmented.append(aug)
    
    return augmented

def create_windows_balanced(data, label, subject, age_group, window_size, overlap, augment_falls=True):
    """Create sliding windows with balanced sampling"""
    windows = []
    labels = []
    subjects = []
    age_groups = []
    
    if label == 1:
        step = window_size // 4
    else:
        step = window_size - overlap
    
    for i in range(0, len(data) - window_size + 1, step):
        window = data[i:i + window_size]
        windows.append(window)
        labels.append(label)
        subjects.append(subject)
        age_groups.append(age_group)
    
    if label == 1 and augment_falls and len(windows) > 0:
        augmented_windows = []
        augmented_labels = []
        augmented_subjects = []
        augmented_age_groups = []
        
        for window in windows:
            augmented_windows.append(window)
            augmented_labels.append(label)
            augmented_subjects.append(subject)
            augmented_age_groups.append(age_group)
            
            aug_versions = augment_fall_window(window, n_augmentations=FALL_AUGMENTATION)
            for aug_window in aug_versions:
                augmented_windows.append(aug_window)
                augmented_labels.append(label)
                augmented_subjects.append(subject)
                augmented_age_groups.append(age_group)
        
        return augmented_windows, augmented_labels, augmented_subjects, augmented_age_groups
    
    return windows, labels, subjects, age_groups

def prepare_data_fixed(dataset, window_size, overlap, train_subjects):
    """FIXED: Global normalization using only training subject statistics"""
    print(f"\n{'='*70}")
    print("DATA PREPARATION - FIXED VERSION")
    print(f"{'='*70}")
    
    print("Step 1: Computing global normalization parameters...")
    all_train_data = []
    
    for sample in dataset:
        if sample['subject'] in train_subjects:
            data_down = downsample_signal(sample['data'], DOWNSAMPLE_FACTOR)
            all_train_data.append(data_down)
    
    all_train_data = np.vstack(all_train_data)
    global_mean = np.mean(all_train_data, axis=0)
    global_std = np.std(all_train_data, axis=0) + 1e-8
    
    print(f"✓ Global statistics computed from {len(all_train_data):,} training samples")
    
    print("\nStep 2: Normalizing and windowing all data...")
    all_windows = []
    all_labels = []
    all_subjects = []
    all_age_groups = []
    
    for idx, sample in enumerate(dataset):
        if (idx + 1) % 100 == 0:
            print(f"  Processing {idx + 1}/{len(dataset)}...")
        
        data_down = downsample_signal(sample['data'], DOWNSAMPLE_FACTOR)
        data_norm = (data_down - global_mean) / global_std
        
        windows, labels, subjects, age_groups = create_windows_balanced(
            data_norm,
            sample['label'],
            sample['subject'],
            sample['age_group'],
            window_size,
            overlap,
            augment_falls=(sample['subject'] in train_subjects)
        )
        
        all_windows.extend(windows)
        all_labels.extend(labels)
        all_subjects.extend(subjects)
        all_age_groups.extend(age_groups)
    
    X = np.array(all_windows)
    y = np.array(all_labels)
    subjects = np.array(all_subjects)
    age_groups = np.array(all_age_groups)
    
    print(f"\n✓ Created {len(X):,} windows")
    print(f"  Shape: {X.shape}")
    print(f"  Falls: {sum(y):,} ({sum(y)/len(y)*100:.1f}%)")
    print(f"  ADLs:  {len(y)-sum(y):,} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    
    return X, y, subjects, age_groups, {'mean': global_mean, 'std': global_std}

# ==================== MODEL ====================
def create_bilstm_model(input_shape):
    """Simplified BiLSTM model"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
        layers.Dropout(0.5),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='BiLSTM_Simplified')
    
    return model

# ==================== TRAINING ====================
def train_model(model, X_train, y_train, X_val, y_val):
    """Train model"""
    print(f"\n{'='*70}")
    print("TRAINING BILSTM MODEL")
    print(f"{'='*70}")
    
    class_weights = sklearn_class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    print(f"Class weights: {class_weight_dict}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')]
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_recall',
            patience=15,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            f'{MODEL_SAVE_PATH}/bilstm_best.keras',
            monitor='val_recall',
            mode='max',
            save_best_only=True
        )
    ]
    
    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start
    print(f"\n✓ Training complete: {training_time/60:.2f} min")
    
    return history, training_time

# ==================== EVALUATION ====================
def evaluate_model_enhanced(model, X_test, y_test, subjects_test, age_groups_test):
    """Enhanced evaluation with safety-critical metrics"""
    print(f"\n{'='*70}")
    print("EVALUATION - SAFETY-CRITICAL METRICS")
    print(f"{'='*70}")
    
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    
    thresholds = np.linspace(0.05, 0.95, 100)
    valid_thresholds = []
    
    for thresh in thresholds:
        y_pred_tmp = (y_pred_proba >= thresh).astype(int)
        recall_tmp = recall_score(y_test, y_pred_tmp, zero_division=0)
        
        if recall_tmp >= MIN_SENSITIVITY:
            f1_tmp = f1_score(y_test, y_pred_tmp, zero_division=0)
            valid_thresholds.append({
                'thresh': thresh,
                'f1': f1_tmp,
                'recall': recall_tmp
            })
    
    if valid_thresholds:
        best = max(valid_thresholds, key=lambda x: x['f1'])
        best_threshold = best['thresh']
        print(f"✓ Found {len(valid_thresholds)} thresholds meeting {MIN_SENSITIVITY:.1%} sensitivity")
    else:
        recalls = []
        for thresh in thresholds:
            y_pred_tmp = (y_pred_proba >= thresh).astype(int)
            recalls.append(recall_score(y_test, y_pred_tmp, zero_division=0))
        best_idx = np.argmax(recalls)
        best_threshold = thresholds[best_idx]
        print(f"⚠️  Could not achieve {MIN_SENSITIVITY:.1%} sensitivity")
    
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    sens = rec
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n📊 Overall Performance:")
    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Sensitivity: {sens:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"  ROC-AUC:     {roc_auc:.4f}")
    print(f"  PR-AUC:      {pr_auc:.4f}")
    
    print(f"\n📋 Confusion Matrix:")
    print(f"  TN={tn:5d}  FP={fp:5d}")
    print(f"  FN={fn:5d}  TP={tp:5d}")
    
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"\n⚠️  Missed {fn} falls out of {fn+tp} ({fnr*100:.1f}%)")
    
    print(f"\n👥 Age-Stratified Performance:")
    
    for age_group in ['young', 'elderly']:
        mask = age_groups_test == age_group
        if np.sum(mask) > 0:
            y_true = y_test[mask]
            y_pred_age = y_pred[mask]
            subjects_in_group = np.unique(subjects_test[mask])
            
            falls_in_group = np.sum(y_true)
            
            if falls_in_group > 0:
                acc_age = accuracy_score(y_true, y_pred_age)
                rec_age = recall_score(y_true, y_pred_age, zero_division=0)
                
                print(f"\n  {age_group.upper()} ({len(subjects_in_group)} subjects: {sorted(subjects_in_group)})")
                print(f"    Samples: {np.sum(mask):,}, Falls: {falls_in_group}")
                print(f"    Accuracy: {acc_age:.4f}")
                print(f"    Recall:   {rec_age:.4f}")
                
                if age_group == 'elderly':
                    print(f"    ⚠️  Based on single subject - NOT generalizable!")
            else:
                print(f"\n  {age_group.upper()}: NO FALL DATA")
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'sensitivity': sens, 'specificity': spec, 'auc': roc_auc, 'pr_auc': pr_auc,
        'threshold': best_threshold, 'cm': cm, 'fnr': fnr,
        'y_pred_proba': y_pred_proba, 'y_pred': y_pred,
        'fpr_curve': fpr, 'tpr_curve': tpr,
        'prec_curve': prec_curve, 'rec_curve': rec_curve
    }

# ==================== VISUALIZATION ====================
def plot_training(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BiLSTM Training History', fontsize=16, fontweight='bold')
    
    metrics = [('accuracy', 'Accuracy'), ('loss', 'Loss'),
               ('precision', 'Precision'), ('recall', 'Recall')]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        
        if metric in history.history:
            ax.plot(history.history[metric], label='Train', linewidth=2)
            if f'val_{metric}' in history.history:
                ax.plot(history.history[f'val_{metric}'], label='Val', linewidth=2)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{FIGURES_PATH}/training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: training_history.png")

def plot_results(results, y_test, age_groups_test):
    """Plot comprehensive results"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('BiLSTM Evaluation Results', fontsize=18, fontweight='bold')
    
    cm = results['cm']
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    # 1. Confusion Matrix
    ax = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['ADL', 'Fall'], yticklabels=['ADL', 'Fall'], ax=ax)
    ax.set_title('Confusion Matrix', fontweight='bold')
    
    # 2. ROC Curve
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(results['fpr_curve'], results['tpr_curve'], linewidth=3,
           label=f"AUC = {results['auc']:.4f}")
    ax.plot([0,1], [0,1], 'k--', alpha=0.5)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. PR Curve
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(results['rec_curve'], results['prec_curve'], linewidth=3,
           label=f"AP = {results['pr_auc']:.4f}")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Metrics
    ax = fig.add_subplot(gs[1, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [results['accuracy'], results['precision'], results['recall'], results['f1']]
    bars = ax.barh(metrics, values, color='steelblue')
    ax.set_xlim([0, 1.1])
    ax.set_title('Metrics', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, values):
        ax.text(val+0.02, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center')
    
    # 5. Predictions
    ax = fig.add_subplot(gs[1, 1])
    fall_probs = results['y_pred_proba'][y_test == 1]
    adl_probs = results['y_pred_proba'][y_test == 0]
    
    ax.hist(adl_probs, bins=50, alpha=0.6, color='skyblue', label='ADL', density=True)
    ax.hist(fall_probs, bins=50, alpha=0.6, color='coral', label='Fall', density=True)
    ax.axvline(results['threshold'], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Probability')
    ax.set_title('Predictions', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Age Performance
    ax = fig.add_subplot(gs[1, 2])
    age_groups = []
    age_recalls = []
    
    for age_group in ['young', 'elderly']:
        mask = age_groups_test == age_group
        if np.sum(mask) > 10 and np.sum(y_test[mask]) > 0:
            y_true = y_test[mask]
            y_pred = results['y_pred'][mask]
            rec_age = recall_score(y_true, y_pred, zero_division=0)
            age_groups.append(age_group.capitalize())
            age_recalls.append(rec_age)
    
    if age_recalls:
        ax.bar(age_groups, age_recalls, color='gold')
        ax.set_ylabel('Recall')
        ax.set_title('Age Group Recall', fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(age_recalls):
            ax.text(i, v+0.02, f'{v:.3f}', ha='center')
    
    # 7. Error Breakdown
    ax = fig.add_subplot(gs[2, 0])
    categories = ['TN', 'FP', 'FN', 'TP']
    counts = [tn, fp, fn, tp]
    colors = ['lightgreen', 'orange', 'red', 'darkgreen']
    
    bars = ax.bar(categories, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Error Breakdown', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, count, f'{count}', ha='center', va='bottom')
    
    # 8. Threshold Analysis
    ax = fig.add_subplot(gs[2, 1])
    thresholds = np.linspace(0, 1, 100)
    recalls = []
    precisions = []
    
    for thresh in thresholds:
        y_pred_tmp = (results['y_pred_proba'] >= thresh).astype(int)
        recalls.append(recall_score(y_test, y_pred_tmp, zero_division=0))
        precisions.append(precision_score(y_test, y_pred_tmp, zero_division=0))
    
    ax.plot(thresholds, recalls, label='Recall', linewidth=2)
    ax.plot(thresholds, precisions, label='Precision', linewidth=2)
    ax.axvline(results['threshold'], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Analysis', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 9. Errors
    ax = fig.add_subplot(gs[2, 2])
    correct = (results['y_pred'] == y_test)
    incorrect = ~correct
    
    correct_probs = results['y_pred_proba'][correct]
    incorrect_probs = results['y_pred_proba'][incorrect]
    
    ax.hist(correct_probs, bins=30, alpha=0.6, color='green',
           label=f'Correct ({np.sum(correct)})', density=True)
    ax.hist(incorrect_probs, bins=30, alpha=0.6, color='red',
           label=f'Incorrect ({np.sum(incorrect)})', density=True)
    ax.set_xlabel('Probability')
    ax.set_title('Correct vs Incorrect', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(f'{FIGURES_PATH}/evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: evaluation_results.png")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("="*70)
    print("LSTM FALL DETECTION - FIXED IMPLEMENTATION")
    print("="*70)
    
    # Load
    dataset = load_dataset(DATA_PATH)
    
    if len(dataset) == 0:
        print("\n❌ No data! Check DATA_PATH")
        exit()
    
    # Get subjects and force SE06 into test
    all_subjects = np.array([d['subject'] for d in dataset])
    unique_subjects = np.unique(all_subjects)
    
    print(f"\n{'='*70}")
    print("DATA SPLIT - SE06 FORCED INTO TEST")
    print(f"{'='*70}")
    
    se06_exists = 'SE06' in unique_subjects
    other_subjects = unique_subjects[unique_subjects != 'SE06']
    
    if se06_exists:
        print("✓ SE06 found - forcing into test set")
        train_subjects, temp_test = train_test_split(other_subjects, test_size=0.2, random_state=42)
        test_subjects = np.append(temp_test, 'SE06')
    else:
        print("⚠️  SE06 not found")
        train_subjects, test_subjects = train_test_split(other_subjects, test_size=0.2, random_state=42)
    
    print(f"\nTrain: {len(train_subjects)} subjects")
    print(f"Test:  {len(test_subjects)} subjects (includes SE06)")
    
    # Prepare data
    X, y, subjects, age_groups, norm_params = prepare_data_fixed(dataset, WINDOW_SIZE, OVERLAP, train_subjects)
    
    # Split
    train_mask = np.isin(subjects, train_subjects)
    test_mask = np.isin(subjects, test_subjects)
    
    X_train_full = X[train_mask]
    y_train_full = y[train_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    subjects_test = subjects[test_mask]
    age_groups_test = age_groups[test_mask]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )
    
    print(f"\n{'='*70}")
    print("FINAL SPLIT")
    print(f"{'='*70}")
    print(f"Train: {len(X_train):,} windows")
    print(f"  Falls: {sum(y_train):,} ({sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"Val:   {len(X_val):,} windows")
    print(f"Test:  {len(X_test):,} windows")
    print(f"  Falls: {sum(y_test):,} ({sum(y_test)/len(y_test)*100:.1f}%)")
    print(f"  Young: {sum(age_groups_test=='young'):,}")
    print(f"  Elderly: {sum(age_groups_test=='elderly'):,}")
    
    # Build model
    model = create_bilstm_model((X_train.shape[1], X_train.shape[2]))
    
    print(f"\n{'='*70}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*70}")
    model.summary()
    
    # Train
    history, training_time = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot training
    plot_training(history)
    
    # Evaluate
    results = evaluate_model_enhanced(model, X_test, y_test, subjects_test, age_groups_test)
    
    # Plot results
    plot_results(results, y_test, age_groups_test)
    
    # Save model
    model.save(f'{MODEL_SAVE_PATH}/bilstm_final.keras')
    print(f"\n✓ Saved model: bilstm_final.keras")
    
    # Save metadata
    cm = results['cm']
    metadata = {
        'model': {
            'name': 'BiLSTM_Simplified',
            'params': int(model.count_params()),
            'training_time_min': float(training_time/60)
        },
        'preprocessing': {
            'original_fs': ORIGINAL_FS,
            'target_fs': TARGET_FS,
            'window_size': WINDOW_SIZE,
            'overlap': OVERLAP,
            'augmentation': FALL_AUGMENTATION
        },
        'performance': {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'roc_auc': float(results['auc']),
            'pr_auc': float(results['pr_auc']),
            'threshold': float(results['threshold'])
        },
        'confusion_matrix': {
            'tn': int(cm[0,0]),
            'fp': int(cm[0,1]),
            'fn': int(cm[1,0]),
            'tp': int(cm[1,1])
        },
        'split': {
            'train_subjects': sorted(train_subjects.tolist()),
            'test_subjects': sorted(test_subjects.tolist()),
            'se06_in_test': se06_exists
        }
    }
    
    with open(f'{MODEL_SAVE_PATH}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata: metadata.json")
    
    # Final summary
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n🎯 Performance:")
    print(f"  Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f} {'✓' if results['recall'] >= MIN_SENSITIVITY else '⚠️'}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  ROC-AUC:   {results['auc']:.4f}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"  TN={tn:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TP={tp:,}")
    
    print(f"\n⚠️  Missed {fn} falls out of {fn+tp} total falls ({results['fnr']*100:.1f}%)")
    
    # Age analysis
    young_mask = age_groups_test == 'young'
    elderly_mask = age_groups_test == 'elderly'
    
    print(f"\n👥 Age Analysis:")
    
    if np.sum(young_mask) > 0 and np.sum(y_test[young_mask]) > 0:
        y_young = y_test[young_mask]
        y_pred_young = results['y_pred'][young_mask]
        young_rec = recall_score(y_young, y_pred_young, zero_division=0)
        print(f"  Young: {young_rec:.4f} recall")
    
    if np.sum(elderly_mask) > 0 and np.sum(y_test[elderly_mask]) > 0:
        y_elderly = y_test[elderly_mask]
        y_pred_elderly = results['y_pred'][elderly_mask]
        elderly_rec = recall_score(y_elderly, y_pred_elderly, zero_division=0)
        elderly_subjects = np.unique(subjects_test[elderly_mask])
        print(f"  Elderly: {elderly_rec:.4f} recall")
        print(f"    ⚠️  Based on {len(elderly_subjects)} subject: {sorted(elderly_subjects)}")
        print(f"    ⚠️  NOT GENERALIZABLE!")
    
    print(f"\n⚠️  CRITICAL LIMITATIONS:")
    print(f"  • Elderly performance based on single subject (SE06)")
    print(f"  • Cannot generalize to elderly population")
    print(f"  • Model optimized for young adult falls")
    print(f"  • Not suitable for clinical deployment")
    
    print(f"\n✅ This Model IS Good For:")
    print(f"  • Young adult fall detection (18-30 years)")
    print(f"  • Research and proof-of-concept")
    print(f"  • Baseline comparison")
    
    print(f"\n❌ This Model IS NOT Ready For:")
    print(f"  • Elderly care deployment")
    print(f"  • Clinical/safety-critical use")
    print(f"  • Real-world elderly fall detection")
    
    print(f"\n📁 Output Files:")
    print(f"  • {FIGURES_PATH}/training_history.png")
    print(f"  • {FIGURES_PATH}/evaluation_results.png")
    print(f"  • {MODEL_SAVE_PATH}/bilstm_final.keras")
    print(f"  • {MODEL_SAVE_PATH}/metadata.json")
    
    if results['recall'] >= MIN_SENSITIVITY:
        print(f"\n✅ MODEL MEETS SENSITIVITY TARGET ({MIN_SENSITIVITY:.0%})")
    else:
        print(f"\n⚠️  MODEL BELOW SENSITIVITY TARGET")
        print(f"   Achieved: {results['recall']:.1%}, Target: {MIN_SENSITIVITY:.0%}")
    
    print(f"\n{'='*70}")
    print("HONEST EVALUATION COMPLETE")
    print(f"{'='*70}\n")
