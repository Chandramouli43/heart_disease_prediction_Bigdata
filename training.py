from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
l1 = ['SEQN', 'Gender', 'Age', 'Annual-Family-Income', 'Ratio-Family-Income-Poverty', 'X60-sec-pulse', 'Systolic', 'Diastolic', 'Weight', 'Height', 'Body-Mass-Index', 'White-Blood-Cells', 'Lymphocyte', 'Monocyte', 'Eosinophils', 'Basophils', 'Red-Blood-Cells', 'Hemoglobin', 'Mean-Cell-Vol', 'Mean-Cell-Hgb-Conc.', 'Mean-cell-Hemoglobin', 'Platelet-count', 'Mean-Platelet-Vol', 'Segmented-Neutrophils', 'Hematocrit', 'Red-Cell-Distribution-Width', 'Albumin', 'ALP', 'AST', 'ALT', 'Cholesterol', 'Creatinine', 'Glucose', 'GGT', 'Iron', 'LDH', 'Phosphorus', 'Bilirubin', 'Protein', 'Uric.Acid', 'Triglycerides', 'Total-Cholesterol', 'HDL', 'Glycohemoglobin', 'Vigorous-work', 'Moderate-work', 'Health-Insurance', 'Diabetes', 'Blood-Rel-Diabetes', 'Blood-Rel-Stroke']

tr=pd.read_csv("./CardiacPrediction.csv")
X= tr[l1]
y = tr[["CoronaryHeartDisease"]]
X, X_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

from sklearn import tree
clf3 = tree.DecisionTreeClassifier()
clf3 = clf3.fit(X,y)
from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier()
clf4 = clf4.fit(X,np.ravel(y))
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb=gnb.fit(X,np.ravel(y))
joblib.dump(clf3,"dtc.pkl")
joblib.dump(clf4,"rfc.pkl")
joblib.dump(gnb,"gnb.pkl")

