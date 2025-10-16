import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Load data
data = pd.read_csv(r'C:\Users\galan\Music\projek_new\gabung.csv', sep=';')

# Pisahkan fitur (X) dan label (y)
X = data[['LBP_0', 'LBP_1', 'LBP_2', 'LBP_3', 'LBP_4', 'LBP_5', 'LBP_6', 'LBP_7', 'LBP_8', 'LBP_9',
          'fch_feature_0', 'fch_feature_1', 'fch_feature_2', 'fch_feature_3', 'fch_feature_4',
          'fch_feature_5', 'fch_feature_6', 'fch_feature_7', 'fch_feature_8', 'fch_feature_9']]
y = data['label']

# Encode label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Parameter grid diperluas untuk semua jenis kernel
param_grid = [
    # Kernel RBF
    {
        'kernel': ['rbf'],
        'C': [1, 10, 100],
        'gamma': ['scale', 'auto']
    },
    # Kernel Polynomial
    {
        'kernel': ['poly'],
        'C': [1, 10],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'],
        'coef0': [0, 0.1, 0.5, 1]
    },
    # Kernel Sigmoid
    {
        'kernel': ['sigmoid'],
        'C': [1, 10],
        'gamma': ['scale', 'auto'],
        'coef0': [0, 0.1, 0.5, 1]
    },
    # Kernel Linear
    {
        'kernel': ['linear'],
        'C': [1, 10, 100]
    }
]


# Rasio data latih
ratios = [0.60, 0.70, 0.80]

# Inisialisasi variabel terbaik
best_f1_score = -1
best_model = None
best_ratio = None
best_kernel = None

# Simpan metrik per kelas
metrics_data = {
    'ratio': [],
    'class': [],
    'precision': [],
    'recall': [],
    'f1_score': []
}

for ratio in ratios:
    print(f"\nEvaluasi dengan rasio train-test: {ratio}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)

    # Grid Search CV
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)
    current_f1_score = f1_score(y_test, y_pred, average='macro')

    # Simpan model dan label encoder
    model_filename = f'svm_model_ratio_{ratio:.2f}.pkl'
    joblib.dump(best_svm, model_filename)
    label_filename = f'svm_label_encoder_ratio_{ratio:.2f}.pkl'
    joblib.dump(label_encoder, label_filename)

    print(f"Model disimpan sebagai: {model_filename}")
    print("Best Parameters:", grid_search.best_params_)
    print("F1 Score:", current_f1_score)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Train Ratio {ratio})')
    plt.tight_layout()
    plt.show()

    # Simpan metrik
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    for cls in label_encoder.classes_:
        metrics_data['ratio'].append(ratio)
        metrics_data['class'].append(cls)
        metrics_data['precision'].append(report[cls]['precision'])
        metrics_data['recall'].append(report[cls]['recall'])
        metrics_data['f1_score'].append(report[cls]['f1-score'])

    # Cek model terbaik
    if current_f1_score > best_f1_score:
        best_f1_score = current_f1_score
        best_model = best_svm
        best_ratio = ratio
        best_kernel = grid_search.best_params_['kernel']

# Simpan model terbaik
if best_model is not None:
    best_model_filename = f'svm_best_model_ratio_{best_ratio:.2f}_{best_kernel}_best.pkl'
    joblib.dump(best_model, best_model_filename)
    print(f"\nModel terbaik disimpan sebagai: {best_model_filename}")
    print(f"Rasio terbaik: {best_ratio:.2f}, Kernel: {best_kernel}, F1 Score terbaik: {best_f1_score:.4f}")

# ========================
# VISUALISASI HASIL
# ========================
metrics_df = pd.DataFrame(metrics_data)

# Precision
plt.figure(figsize=(14, 8))
sns.lineplot(data=metrics_df, x='ratio', y='precision', hue='class', marker='o', linewidth=2.5)
plt.title('Precision per Kelas (SVM)', fontsize=14)
plt.xlabel('Rasio Training Data')
plt.ylabel('Precision')
plt.xticks(ratios)
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Recall
plt.figure(figsize=(14, 8))
sns.lineplot(data=metrics_df, x='ratio', y='recall', hue='class', marker='o', linewidth=2.5)
plt.title('Recall per Kelas (SVM)', fontsize=14)
plt.xlabel('Rasio Training Data')
plt.ylabel('Recall')
plt.xticks(ratios)
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# F1-Score
plt.figure(figsize=(14, 8))
sns.lineplot(data=metrics_df, x='ratio', y='f1_score', hue='class', marker='o', linewidth=2.5)
plt.title('F1-Score per Kelas (SVM)', fontsize=14)
plt.xlabel('Rasio Training Data')
plt.ylabel('F1-Score')
plt.xticks(ratios)
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# F1-Score Macro Average
f1_macro_scores = metrics_df.groupby('ratio')['f1_score'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=f1_macro_scores, x='ratio', y='f1_score', marker='o', color='red', linewidth=3)
plt.title('F1-Score Macro Average (SVM)')
plt.xlabel('Rasio Training Data')
plt.ylabel('F1-Score Macro Average')
plt.xticks(ratios)
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()
