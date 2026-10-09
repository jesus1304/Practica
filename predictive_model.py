import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
df = pd.read_csv('predictive_challenge.csv')
df['drop_future_visibility'] = (df['future_visibility_w2'] < df['future_visibility_w1']).astype(int)

X = df[['age_group', 'role', 'work_mode', 'learning_appetite_w1', 'learning_capacity_w1',
        'development_access_w1', 'professional_agency_w1', 'knowledge_exchange_w1', 'future_visibility_w1']]
y = df['drop_future_visibility']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

cat_cols = ['age_group', 'role', 'work_mode']
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 4, 6],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [2, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_proba = best_model.predict_proba(X_test)[:, 1]

y_proba_train = best_model.predict_proba(X_train)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_train, y_proba_train)

f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]

y_pred_optimal = (y_proba >= best_threshold).astype(int)

print(f"=== MEJORES HIPERPARÁMETROS ===")
print(grid_search.best_params_)
print(f"\nUmbral de decisión óptimo: {best_threshold:.3f}\n")

print("=== RANDOM FOREST MEJORADO (CON UMBRAL AJUSTADO) ===")
print(classification_report(y_test, y_pred_optimal))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

ohe_names = best_model.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_cols)
features = num_cols + list(ohe_names)

df_importance = pd.DataFrame({
    'Feature': features,
    'Importance': best_model.named_steps['model'].feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== IMPORTANCIA DE CARACTERÍSTICAS ===")
print(df_importance)