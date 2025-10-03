# ML Dashboard Project
# Author: Antile Kaba
# Date: 2025-10-02

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# 1. Load Sample Data
# -------------------------
np.random.seed(42)
data = pd.DataFrame({
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100),
    'Feature3': np.random.randn(100),
    'Target': np.random.choice([0,1], size=100)
})

# -------------------------
# 2. Data Preprocessing
# -------------------------
X = data[['Feature1','Feature2','Feature3']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 3. Model Training
# -------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# 4. Evaluation
# -------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# -------------------------
# 5. Visualization
# -------------------------
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

plt.figure(figsize=(8,5))
sns.barplot(x=X.columns, y=model.feature_importances_)
plt.title('Feature Importances')
plt.savefig('feature_importances.png')
plt.close()

print("\nML Dashboard Project Completed!")
