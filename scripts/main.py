# ML Dashboard Project
# Author: Antile Kaba
# Date: 2025-10-02

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------
# 1. Load Sample Data
# -------------------------
data = pd.DataFrame({
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Feature3': np.random.randint(0, 10, 100),
    'Target': np.random.choice([0,1], 100)
})

# -------------------------
# 2. Split Data
# -------------------------
X = data[['Feature1','Feature2','Feature3']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 3. Train Model
# -------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------------
# 4. Evaluate Model
# -------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print("Confusion Matrix:")
print(cm)

# -------------------------
# 5. Visualization
# -------------------------
sns.set_style('whitegrid')
plt.figure(figsize=(6,4))
sns.barplot(x=['Class 0','Class 1'], y=cm.sum(axis=1))
plt.title('Predicted vs Actual Counts')
plt.savefig('predicted_vs_actual.png')
plt.close()

print("ML Dashboard Project Completed!")
