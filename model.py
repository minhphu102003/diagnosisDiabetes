### Load libraries

import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics # Import scikit-learn metrics module for accuracy calculation

from utils import config

### Loading data

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
# load dataset
pima = pd.read_csv(config.pathData, header=None, names=col_names)


### Feature selection

feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
pima[feature_cols] = pima[feature_cols].apply(pd.to_numeric, errors='coerce')
pima['label'] = pd.to_numeric(pima['label'], errors='coerce')

# Loại bỏ các hàng chứa giá trị NaN
pima = pima.dropna()

X = pima[feature_cols]  # Features
y = pima['label']  # Target variable

### Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# Dùng random_state để mỗi lần chạy kết quả đều cho ra giá trị đồng nhất

### Building decision tree

clf = DecisionTreeClassifier(criterion="entropy", max_depth=config.maxDepth)

clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

### Visualize decision tree

from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf,out_file=dot_data,filled=True, rounded=True, special_characters=True, feature_names=feature_cols, class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())

