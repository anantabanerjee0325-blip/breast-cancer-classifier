from sklearn.datasets import load_breast_cancer
import pandas as pd
data = load_breast_cancer()
df=pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target 

print(df.shape)
print(df.head())
print(data.target_names)
print(df.groupby('diagnosis')['mean radius'].mean())

import matplotlib.pyplot as plt
import seaborn as sns
sns.boxplot(x='diagnosis', y='mean radius', data=df)
plt.xlabel('Diagnosis (0 = malignant, 1 = benign)')
plt.ylabel('Mean Radius')
plt.title('Tumor Radius by Diagnosis')
plt.savefig('radius_boxplot.png')

feature_names = data.feature_names
differences = {}

for feature in feature_names:
    group_means = df.groupby('diagnosis')[feature].mean()
    difference = group_means[0] - group_means[1]
    differences[feature] = difference

sorted_differences = sorted(differences.items(), key=lambda x: abs(x[1]), reverse=True)

for feature, diff in sorted_differences[:10]:
    print(feature, diff)

plt.figure(figsize=(12,10))
correlation_matrix = df.drop('diagnosis', axis=1) .corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Breast Cancer Dataset')
plt.colorbar(label='Diagnosis (0 = malignant, 1 = benign)')
plt.savefig('pca_plot.png')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(X_train.shape)
print(X_test.shape)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=data.target_names))

coefficients = pd.DataFrame({
    'feature': feature_names,
    'coefficient': model.coef_[0]
})

coefficients['abs_coefficient'] = coefficients['coefficient'].abs()
coefficients = coefficients.sort_values('abs_coefficient', ascending=False)

print(coefficients.head(10))

plt.figure(figsize=(10, 6))
top_10 = coefficients.head(10)
colors = ['red' if c < 0 else 'blue' for c in top_10['coefficient']]
plt.barh(top_10['feature'], top_10['coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.title('Top 10 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png')