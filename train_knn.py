import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Load data
data = pd.read_csv(r'C:\Users\galan\Music\projek_new\gabung.csv', sep=';')

# Pisahkan fitur dan label
X = data[['LBP_0', 'LBP_1', 'LBP_2', 'LBP_3', 'LBP_4', 'LBP_5', 'LBP_6', 'LBP_7', 'LBP_8', 'LBP_9', 
          'fch_feature_0', 'fch_feature_1', 'fch_feature_2', 'fch_feature_3', 'fch_feature_4', 
          'fch_feature_5', 'fch_feature_6', 'fch_feature_7', 'fch_feature_8', 'fch_feature_9']]
y = data['label']

# Encode label
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Parameter Grid untuk KNN
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
    'p': [1, 2],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

ratios = [0.60, 0.70, 0.80]

best_f1_score = -1
best_model = None
best_ratio = None

metrics_data = {
    'ratio': [],
    'class': [],
    'precision': [],
    'recall': [],
    'f1_score': []
}

for ratio in ratios:
    print(f"\nEvaluasi dengan rasio train-test: {ratio}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=ratio, random_state=42)
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)
    
    current_f1_score = f1_score(y_test, y_pred, average='macro')
    
    model_filename = f'knn_model_ratio_{ratio:.2f}.pkl'
    joblib.dump(best_knn, model_filename)
    print(f"Model disimpan sebagai: {model_filename}")
    
    label_encoder_filename = f'knn_label_encoder_ratio_{ratio:.2f}.pkl'
    joblib.dump(label_encoder, label_encoder_filename)
    
    print("Best Parameters:", grid_search.best_params_)
    print("F1 Score:", current_f1_score)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Train Ratio {ratio})')
    plt.show()
    
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    for cls in label_encoder.classes_:
        metrics_data['ratio'].append(ratio)
        metrics_data['class'].append(cls)
        metrics_data['precision'].append(report[cls]['precision'])
        metrics_data['recall'].append(report[cls]['recall'])
        metrics_data['f1_score'].append(report[cls]['f1-score'])
    
    if current_f1_score > best_f1_score:
        best_f1_score = current_f1_score
        best_model = best_knn
        best_ratio = ratio

# Simpan model terbaik
if best_model is not None:
    best_model_filename = f'knn_best_model_ratio_{best_ratio:.2f}_best.pkl'
    joblib.dump(best_model, best_model_filename)
    print(f"\nModel terbaik disimpan sebagai: {best_model_filename}")
    print(f"Rasio train-test terbaik: {best_ratio:.2f}")
    print(f"F1 Score terbaik: {best_f1_score:.4f}")

# Visualisasi metrik
metrics_df = pd.DataFrame(metrics_data)

# Plot Precision
plt.figure(figsize=(14, 8))
sns.lineplot(data=metrics_df, x='ratio', y='precision', hue='class', marker='o', linewidth=2.5)
plt.title('Precision per Kelas (KNN)', fontsize=14)
plt.xlabel('Rasio Training Data', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.xticks(ratios)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot Recall
plt.figure(figsize=(14, 8))
sns.lineplot(data=metrics_df, x='ratio', y='recall', hue='class', marker='o', linewidth=2.5)
plt.title('Recall per Kelas (KNN)', fontsize=14)
plt.xlabel('Rasio Training Data', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.xticks(ratios)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot F1-Score
plt.figure(figsize=(14, 8))
sns.lineplot(data=metrics_df, x='ratio', y='f1_score', hue='class', marker='o', linewidth=2.5)
plt.title('F1-Score per Kelas (KNN)', fontsize=14)
plt.xlabel('Rasio Training Data', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.xticks(ratios)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Macro Average F1
f1_macro_scores = metrics_df.groupby('ratio')['f1_score'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(data=f1_macro_scores, x='ratio', y='f1_score', marker='o', color='red', linewidth=3)
plt.title('F1-Score Macro Average (KNN)', fontsize=14)
plt.xlabel('Rasio Training Data', fontsize=12)
plt.ylabel('F1-Score Macro Average', fontsize=12)
plt.xticks(ratios)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
