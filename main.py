from sklearn import model_selection 
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier 
import pandas as pd 
  
url = "wine_data.xlsx"
dataframe = pd.read_excel(url) 
arr = dataframe.values 
X = arr[:, 1:14] 
Y = arr[:, 0] 
  
seed = 8
kfold = model_selection.KFold(n_splits = 3, 
                       random_state = seed) 
  
 
base_cls = DecisionTreeClassifier() 
  
 
num_trees = 500
  

model = BaggingClassifier(base_estimator = base_cls, 
                          n_estimators = num_trees, 
                          random_state = seed) 
  
results = model_selection.cross_val_score(model, X, Y, cv = kfold) 
print("accuracy :") 
print(results.mean())
