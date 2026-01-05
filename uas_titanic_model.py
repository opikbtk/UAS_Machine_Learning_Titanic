import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. Load Dataset
# ==========================================
print("Loading Titanic Dataset...")
# Menggunakan dataset titanic builtin dari seaborn
df = sns.load_dataset('titanic')
print(f"Dataset shape awal: {df.shape}")
print(df.head())

# Simpan ke CSV agar kita punya file fisiknya (untuk dikumpulkan)
df.to_csv('titanic.csv', index=False)
print("Dataset berhasil disimpan sebagai 'titanic.csv'")

# ==========================================
# 2. Exploratory Data Analysis (EDA) Singkat
# ==========================================
print("\n--- Info Dataset ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Distribusi Target (Survived) ---")
print(df['survived'].value_counts(normalize=True))

# ==========================================
# 3. Preprocessing
# ==========================================
print("\nDoing Preprocessing...")

# Drop kolom yang duplikat/tidak perlu/terlalu banyak missing value untuk model sederhana
# deck: banyak null
# embark_town: sama dengan embarked
# alive: sama dengan target (leaking)
# who, adult_male: redundant dengan sex/age
cols_to_drop = ['deck', 'embark_town', 'alive', 'who', 'adult_male', 'class'] 
df_clean = df.drop(columns=cols_to_drop)

# Handling Missing Values
# Age: isi dengan median
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
# Embarked: isi dengan modus (most frequent)
df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

# Encoding Categorical Variables
# Menggunakan Label Encoding untuk Sex dan Embarked
le = LabelEncoder()
df_clean['sex'] = le.fit_transform(df_clean['sex']) # male:1, female:0 (usually)
df_clean['embarked'] = le.fit_transform(df_clean['embarked'])

# Cek data siap pakai
print("\nData siap training (5 baris pertama):")
print(df_clean.head())

# ==========================================
# 4. Train/Test Split
# ==========================================
X = df_clean.drop('survived', axis=1)
y = df_clean['survived']

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simpan Data Training dan Testing ke CSV (Bukti pemisahan data)
print("\nSaving Train & Test sets to CSV...")
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('titanic_train.csv', index=False)
test_data.to_csv('titanic_test.csv', index=False)
print("Berhasil disimpan: 'titanic_train.csv' dan 'titanic_test.csv'")

# ==========================================
# 5. Build Model (Decision Tree)
# ==========================================
# Menggunakan max_depth=3 agar pohon mudah dibaca dan tidak overfitting
dt_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# ==========================================
# 6. Evaluation
# ==========================================
y_pred = dt_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 7. Visualization
# ==========================================
print("\n--- Generating Tree Visualization ---")
plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=X.columns, class_names=['Not Survived', 'Survived'], filled=True, rounded=True)
plt.title("Decision Tree Visualization (Titanic)")
plt.savefig('titanic_tree_structure.png')
print("Tree visualization saved as 'titanic_tree_structure.png'")

# Export text representation
tree_rules = export_text(dt_model, feature_names=list(X.columns))
print("\nRules of the tree:")
print(tree_rules)
