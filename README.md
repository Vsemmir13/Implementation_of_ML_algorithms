# Implementation_of_ML_algorithms

  Один из самых эффективных способов разобраться как работает тот или иной алгоритм - реализовать его самому.

  Задача: реализовать основные алгоритмы классического машинного обучения на Python, используя только Pandas и NumPy.

  Обозреваемые алгоритмы:
  1. Линейный модели:
  - Линейная регрессия (Linear Regression)
  - Логистическая регрессия (Logistic Regression) 
  - Метод опорных векторов (Support Vector Machine, SVM)
    
  2. Метрические алгоритмы:
  - Метод k-ближайших соседей (k-nearest neighbors algorithm, k-NN)
    
  3. Деревья решений (Decision Trees)
  
  4. Ансамбли:
  - Бэггинг (Bagging)
  - Случайный лес (Random Forest)
  - Бустинг над деревьями решений (Boosting)
    
  5. Кластеризация:
  - Метод k-средних (K-means)
  - Агломеративная кластеризация (Agglomerative Clustering)
  - DBSCAN
    
  6. Понижение размерности:
  - Метод главных компонент (Principal Component Analysis, PCA)
    
Помимо этого рассмотрены такие функции как:
- Функции потерь для классификации и регрессии
- Метрики качества для классификации и регрессии
- Различные регуляризации
- Стохастический градиентный спуск

Датасеты:
- Для регрессии
```python   
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]
```
- Для классификации - Banknote Authentication
  
https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip

```python 
df = pd.read_csv('banknote+authentication.zip', header=None)
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
X, y = df.iloc[:,:4], df['target']
```
- Для регрессии - Diabetes
  
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes
```python 
from sklearn.datasets import load_diabetes
data = load_diabetes(as_frame=True)
X, y = data['data'], data['target']
```
