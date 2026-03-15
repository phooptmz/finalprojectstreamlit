import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import time
import os
import json

# ✅ ใช้ Optuna สำหรับ Hyperparameter Tuning
try:
    import optuna
    from optuna.integration import sklearn as optuna_sklearn
    USE_OPTUNA = True
except ImportError:
    USE_OPTUNA = False
    print("⚠️  Optuna not found. Using GridSearchCV/RandomizedSearchCV instead.")

warnings.filterwarnings("ignore")

# ==================== 0) CONFIG ====================
CSV_PATH = r"C:\Users\pho\Documents\newtrainmodel\combined_data\ac_training_comfort_balanced.csv"
FEATURES = ['OCI', 'delta_OCI', 'temp_indoor', 'humi_indoor']
TARGET = 'ac_mode_class'
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5  # จำนวน Fold ใน Cross-Validation

# ✅ เลือกโมเดลที่ต้องการเทรน
MODELS_TO_TRAIN = {
    "RandomForest": True,
    "XGBoost": True,         
    "LightGBM": True,        
    "CatBoost": True,        
    "LogisticRegression": False
}

# ==================== 1) Load & Prepare Data ====================
print("📊 กำลังโหลดข้อมูล...")
start_time = time.time()

try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print(f"❌ Error: File not found at {CSV_PATH}")
    print("   โปรดรัน merge_data.py ก่อนเพื่อสร้างไฟล์รวม")
    exit()

# ตรวจสอบและจัดการ Missing Values
missing = df.isnull().sum()
if missing.any():
    print(f"⚠️  พบค่าว่าง:\n{missing[missing > 0]}")
    df = df.dropna(subset=FEATURES + [TARGET])

# ตรวจสอบ Class Balance
class_dist = df[TARGET].value_counts().sort_index()
print("\n📈 Class Distribution:")
for cls, count in class_dist.items():
    pct = (count / len(df)) * 100
    print(f"   Class {cls}: {count} ({pct:.2f}%)")

# เตือนถ้า Class ไม่สมดุลมาก
if class_dist.min() < (len(df) * 0.1):
    print("\n⚠️  คำเตือน: Class ไม่สมดุล! จะใช้ class_weight='balanced' อัตโนมัติ")
    USE_CLASS_WEIGHT = True
else:
    USE_CLASS_WEIGHT = False

# แยก Features และ Target
X = df[FEATURES].values
y = df[TARGET].astype(int).values

# แบ่ง Train/Test ด้วย Stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y
)

print(f"\n✂️  แบ่งข้อมูล: Train={len(X_train)}, Test={len(X_test)}")
print(f"⏱️  เวลาโหลดข้อมูล: {time.time() - start_time:.2f} วินาที")

# ==================== 2) Define Models & Hyperparameters ====================
models_config = {}

# --- Random Forest ---
if MODELS_TO_TRAIN.get("RandomForest", True):
    models_config["RandomForest"] = {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                random_state=RANDOM_STATE, 
                n_jobs=-1,
                class_weight='balanced' if USE_CLASS_WEIGHT else None
            ))
        ]),
        "params": {
            "rf__n_estimators": [100, 200, 300, 400, 500],
            "rf__max_depth": [10, 15, 20, 25, None],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf": [1, 2, 4],
            "rf__bootstrap": [True, False]
        }
    }

# --- XGBoost ---
if MODELS_TO_TRAIN.get("XGBoost", False):
    try:
        from xgboost import XGBClassifier
        models_config["XGBoost"] = {
            "model": Pipeline([
                ('scaler', StandardScaler()),
                ('xgb', XGBClassifier(
                    objective='multi:softmax', 
                    num_class=3, 
                    eval_metric='mlogloss', 
                    n_jobs=-1, 
                    random_state=RANDOM_STATE,
                ))
            ]),
            "params": {
                "xgb__n_estimators": [100, 200, 300, 400],
                "xgb__max_depth": [3, 5, 7, 9],
                "xgb__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
                "xgb__colsample_bytree": [0.7, 0.8, 0.9, 1.0]
            }
        }
    except ImportError:
        print("⚠️  XGBoost not installed. Skipping...")

# --- LightGBM ---
if MODELS_TO_TRAIN.get("LightGBM", False):
    try:
        from lightgbm import LGBMClassifier
        models_config["LightGBM"] = {
            "model": Pipeline([
                ('scaler', StandardScaler()),
                ('lgb', LGBMClassifier(
                    objective='multiclass',
                    num_class=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight='balanced' if USE_CLASS_WEIGHT else None,
                    verbose=-1 
                ))
            ]),
            "params": {
                "lgb__n_estimators": [100, 200, 300, 400],
                "lgb__max_depth": [-1, 5, 10, 15],
                "lgb__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "lgb__num_leaves": [31, 50, 100],
                "lgb__subsample": [0.7, 0.8, 0.9, 1.0]
            }
        }
    except ImportError:
        print("⚠️  LightGBM not installed. Skipping...")

# --- CatBoost ---
if MODELS_TO_TRAIN.get("CatBoost", False):
    try:
        from catboost import CatBoostClassifier
        models_config["CatBoost"] = {
            "model": Pipeline([
                ('scaler', StandardScaler()),
                ('cat', CatBoostClassifier(
                    loss_function='MultiClass',
                    random_state=RANDOM_STATE,
                    thread_count=-1,
                    verbose=False, 
                    auto_class_weights='Balanced' if USE_CLASS_WEIGHT else 'None'
                ))
            ]),
            "params": {
                "cat__iterations": [100, 200, 300, 400],
                "cat__depth": [4, 6, 8, 10],
                "cat__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "cat__l2_leaf_reg": [1, 3, 5, 7, 9]
            }
        }
    except ImportError:
        print("⚠️  CatBoost not installed. Skipping...")

# ==================== 3) Training Function ====================
def train_and_evaluate(model_name, model, params):
    print(f"🌲 กำลังเทรน: {model_name} (กรุณารอสักครู่...)")
    start_train = time.time()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    if USE_OPTUNA and len(params) > 2:
        optuna_params = {}
        for key, values in params.items():
            optuna_params[key] = optuna.distributions.CategoricalDistribution(values)
            
        search = optuna_sklearn.OptunaSearchCV(
            estimator=model, 
            param_distributions=optuna_params, 
            n_trials=30,  
            cv=cv, 
            n_jobs=-1,
            random_state=RANDOM_STATE, 
            scoring='f1_weighted',
            verbose=0
        )
    else:
        search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=params, 
            n_iter=30, 
            cv=cv, 
            n_jobs=-1, 
            random_state=RANDOM_STATE, 
            scoring='f1_weighted', 
            verbose=0
        )
    
    # เทรน
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    train_time = time.time() - start_train
    
    # ประเมินผล
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    cm = confusion_matrix(y_test, y_pred)
    
    # ดึงค่า Feature Importance
    feature_importance_list = []
    model_step = best_model.steps[-1][1]
    if hasattr(model_step, 'feature_importances_'):
        fi = pd.DataFrame({
            'Feature': FEATURES,
            'Importance': model_step.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        max_val = fi['Importance'].max()
        for _, row in fi.iterrows():
            # ปรับให้กราฟแท่งมีความยาวสูงสุดที่ 40 ตัวอักษรเสมอ ไม่ทะลุจอ
            bar_len = int((row['Importance'] / max_val) * 40) if max_val > 0 else 0
            bar = "█" * bar_len
            feature_importance_list.append(f"   {row['Feature']:15} {row['Importance']:8.4f} {bar}")
    
    # บันทึกโมเดล
    safe_name = model_name.replace(" ", "_")
    filename = f"{safe_name}_model.joblib"
    dump(best_model, filename)
    
    metadata = {
        "model_name": model_name, "best_params": search.best_params_,
        "metrics": {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1_score": float(f1), "cv_mean": float(cv_mean), "cv_std": float(cv_std)},
        "training_time": float(train_time), "features": FEATURES, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(f"{safe_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # ส่งคืนค่ากลับไปเก็บไว้รอปริ้นท์ตอนจบ
    return {
        "name": model_name, "model": best_model, "train_time": train_time,
        "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
        "cv_mean": cv_mean, "cv_std": cv_std, "best_params": search.best_params_,
        "cm": cm, "feature_importances": feature_importance_list, "filename": filename
    }

# ==================== 4) Run Training ====================
print("\n🚀 Starting Training & Evaluation...")
print("="*60)
results = []
for name, config in models_config.items():
    result = train_and_evaluate(name, config["model"], config["params"])
    results.append(result)
    print(f"   ✅ เทรน {name} เสร็จสิ้น!")

# ==================== 5) Print All Results Together ====================
print("\n" + "="*60)
print("📊 สรุปผลการประเมินทุกโมเดล (ALL MODELS SUMMARY)")
print("="*60)

for res in results:
    print(f"\n🟢 {res['name']} Results:")
    print(f"   ⏱️  Training Time: {res['train_time']:.2f} วินาที")
    print(f"   📊 Accuracy:  {res['accuracy']:.4f} ({res['accuracy']*100:.2f}%)")
    print(f"   🎯 Precision: {res['precision']:.4f} ({res['precision']*100:.2f}%)")
    print(f"   📥 Recall:    {res['recall']:.4f} ({res['recall']*100:.2f}%)")
    print(f"   🏆 F1-Score:  {res['f1_score']:.4f} ({res['f1_score']*100:.2f}%)")
    print(f"   🔄 CV Score:  {res['cv_mean']:.4f} (+/- {res['cv_std']*2:.4f})")
    
    print(f"\n   🔧 Best Parameters:")
    for param, value in res['best_params'].items():
        print(f"      - {param}: {value}")
    
    print(f"\n   📋 Confusion Matrix:\n{res['cm']}")
    
    if res['feature_importances']:
        print(f"\n   🌟 Feature Importance:")
        for line in res['feature_importances']:
            print(line)
    
    print("-" * 50)

# ==================== 6) Select Best Model ====================
print("\n" + "="*60)
print("🏆 FINAL BEST MODEL")
print("="*60)

best_result = max(results, key=lambda x: x['f1_score'])

print(f"\n🥇 WINNER: {best_result['name']}")
print(f"   - F1-Score: {best_result['f1_score']:.4f}")
print(f"   - Accuracy: {best_result['accuracy']:.4f}")
print(f"   - CV Score: {best_result['cv_mean']:.4f}")

# เซฟไฟล์โมเดลที่ดีที่สุดแยกไว้
safe_best_name = best_result['name'].replace(" ", "_")
final_model_path = f"{safe_best_name}_best_model.joblib"
dump(best_result['model'], final_model_path)
print(f"\n💾 Saved BEST model to: {final_model_path}")

# ==================== 7) Visualization ====================
plt.figure(figsize=(10, 5))
x = np.arange(len(results))
width = 0.35

plt.bar(x - width/2, [r['accuracy'] for r in results], width, label='Accuracy', color='skyblue')
plt.bar(x + width/2, [r['f1_score'] for r in results], width, label='F1-Score', color='coral')

plt.xticks(x, [r['name'] for r in results])
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)

y_pred_best = best_result['model'].predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Eco', 'Normal', 'High'], 
            yticklabels=['Eco', 'Normal', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_result["name"]}')
plt.tight_layout()
plt.savefig('confusion_matrix_best.png', dpi=150)

print("\n✅ TRAINING COMPLETE! เซฟรูปภาพกราฟเปรียบเทียบและ Confusion Matrix เรียบร้อยแล้วครับ")