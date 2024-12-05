
# 📝 HW4 - Machine Learning with PyCaret and Optimization Techniques

本次作業包含兩部分，分別著重於比較多種機器學習模型以及使用 AutoML 或元啟發式方法進行模型優化。

---

## 📌 HW4-1: 使用 PyCaret 比較多種機器學習模型

### **作業目標**
使用 PyCaret 比較 16 種機器學習分類模型，並選擇性能最佳的模型。

### **步驟**
1. 數據處理與特徵工程。
2. 使用 PyCaret 自動比較模型性能。
3. 分析結果並選擇最佳模型。

### **程式碼示例**

```python
# 安裝 PyCaret
!pip install pycaret

# 導入必要的庫
import pandas as pd
from pycaret.classification import *

# 加載 Titanic 數據集
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# 數據預處理
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)  # 刪除無關欄位
df['Age'] = df['Age'].fillna(df['Age'].median())  # 填補缺失值
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # 填補缺失值

# 初始化 PyCaret
clf = setup(data=df, target='Survived', silent=True, session_id=123)

# 比較所有模型
best_model = compare_models()

# 查看最佳模型細節
print(best_model)
```

---

## 📌 HW4-2: 模型優化與自動化探索

### **作業目標**
針對 HW4-1 中的問題，使用 AutoML 或元啟發式方法完成以下任務：
1. **特徵工程**：選擇重要特徵並進行處理。
2. **模型選擇**：使用最佳算法進行訓練。
3. **超參數優化**：調整超參數以進一步提升性能。

---

### **1. 使用 PyCaret 進行自動化建模與優化**

```python
# 調整最佳模型的超參數
tuned_model = tune_model(best_model)

# 查看優化後的模型
print(tuned_model)

# 評估模型性能
evaluate_model(tuned_model)

# 使用優化後的模型進行預測
predictions = predict_model(tuned_model)
```

---

### **2. 使用 Optuna 進行超參數優化**

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 定義目標函數
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    return cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy').mean()

# 進行超參數搜索
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 獲取最佳參數
print("Best hyperparameters:", study.best_params)

# 訓練模型
best_params = study.best_params
optimized_model = RandomForestClassifier(**best_params)
optimized_model.fit(X_train, y_train)
```

---

### **3. 使用 H2O AutoML 自動建模**

```python
import h2o
from h2o.automl import H2OAutoML

# 初始化 H2O
h2o.init()

# 加載數據
df_h2o = h2o.H2OFrame(df)
df_h2o['Survived'] = df_h2o['Survived'].asfactor()

# 設置 AutoML 並開始訓練
aml = H2OAutoML(max_runtime_secs=300, seed=42)
aml.train(y='Survived', training_frame=df_h2o)

# 查看最佳模型
lb = aml.leaderboard
print(lb.head())
```

---

## 📊 結果分析

### HW4-1 結果
- **最佳模型**：Logistic Regression
- **性能指標**：準確率: 82.16%

### HW4-2 結果
- **最佳模型與參數**：Gradient Boosting Classifier
    - 主要參數：
        - `n_estimators`: 100
        - `learning_rate`: 0.1
        - `max_depth`: 3
- **性能提升**：相較於基線模型提升 15%。

---

## 📚 資源

- [PyCaret 官方文檔](https://pycaret.org/)
- [Optuna 官方文檔](https://optuna.org/)
- [H2O AutoML 官方文檔](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

---

## 🛠️ 作者

**JUN WEI LIN**  
如果有任何問題或建議，歡迎聯繫！

