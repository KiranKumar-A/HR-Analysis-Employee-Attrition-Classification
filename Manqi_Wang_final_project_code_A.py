# Import neccessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
%matplotlib inline
from sklearn.decomposition import RandomizedPCA
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from time import time
from pandas import concat

# Import CSV file and drop columns "EmployeeCount", "EmployeeNumber","Over18","StandardHours"
df = pd.read_csv('IBM_HR_Employee_Attrition.csv')
df = df.drop(df.columns[[8, 9, 21, 26]], axis=1)
# Convert "Attrition" 'Yes' into 1 and 'No' into 0
df["Attrition"] = df["Attrition"].astype('category').cat.codes
# create dummy variables for categorical features
df = pd.get_dummies(df)

# Create X and y
target_name = 'Attrition'
X = df.drop('Attrition', axis=1)
y = df[target_name]

# Split the dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y.ravel(), test_size=0.2, random_state=0,stratify=y)
X.head()

# First create a copy of your dataframe
X_train = X_train.copy()
X_test = X_test.copy()
# Don't include the categorical columns in the transformation
col_names = ['Age', 'DailyRate', 'HourlyRate', 'MonthlyIncome','MonthlyRate', 'PercentSalaryHike']
X_train_scaled = X_train[col_names]
X_test_scaled = X_test[col_names]
scaler = StandardScaler()
scaler.fit(X_train_scaled)
X_train_scaled = scaler.transform(X_train_scaled)
X_test_scaled = scaler.transform(X_test_scaled)
# Now, don't create a new dataframe but assign the result to those scaled columns
X_train[col_names] = X_train_scaled
X_test[col_names] = X_test_scaled
#print(X_train)
#print(X_test)

# Create base rate model
t0 = time()
def base_rate_model(X) :
    y = np.zeros(X.shape[0])
    return y

print ("---Base Model---")
base_roc_auc = roc_auc_score(y_test, base_rate_model(X_test))
print ("Base Rate roc_auc = %2.2f" % base_roc_auc)
print(classification_report(y_test, base_rate_model(X_test)))
print("done in %0.3fs" % (time() - t0))

# Logistic Regression Model
print ("---Logistic Model---")
t0 = time()
tuned_parameters = [{ 
                    'C' :[0.01, 0.1, 1, 10, 100],
                    'penalty' :['l1', 'l2']}
                ]
logreg = GridSearchCV(LogisticRegression(class_weight = "balanced"), tuned_parameters, 
                       scoring='roc_auc')
logreg.fit(X_train, y_train)
print("Best parameters set found on development set:")
print()
print(logreg.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in logreg.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
            % (mean_score, scores.std() * 2, params))
print()
print("Detailed classification report:")
y_true, y_pred_logreg = y_test, logreg.predict(X_test)
logreg_roc_auc = roc_auc_score(y_true, y_pred_logreg)
print ("Random Forest roc_auc = %2.2f" % logreg_roc_auc)
print(classification_report(y_true, y_pred_logreg))
print()
print("done in %0.3fs" % (time() - t0))

#logreg = LogisticRegression(penalty='l1', C=1, class_weight = "balanced")
#logreg.fit(X_train, y_train)

# Random Forests
print ("---Random Forest Model---")
t0 = time()
tuned_parameters = [{ 
                    'max_depth' :[6,8,12,16,20,24,28,32],
                    'n_estimators': [50,100,200,400,800]},
                ]

rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, 
                       scoring='roc_auc')
rf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(rf.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in rf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
            % (mean_score, scores.std() * 2, params))
print()

print("Detailed classification report:")
y_true, y_pred_rf = y_test, rf.predict(X_test)
rf_roc_auc = roc_auc_score(y_true, y_pred_rf)
print ("Random Forest roc_auc = %2.2f" % rf_roc_auc)
print(classification_report(y_true, y_pred_rf))
print()
print("done in %0.3fs" % (time() - t0))

# Adaboost
print ("---Adaboost Model---")
t0 = time()
tuned_parameters = [{ 
                    'learning_rate' :[0.6, 0.8, 1, 1.2],
                    'n_estimators': [50,100,200,400,800]},
                ]


ada = GridSearchCV(AdaBoostClassifier(), tuned_parameters, 
                       scoring='roc_auc')
ada.fit(X_train, y_train)


print("Best parameters set found on development set:")
print()
print(ada.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in ada.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
            % (mean_score, scores.std() * 2, params))
print()

print("Detailed classification report:")
y_true, y_pred_ada = y_test, ada.predict(X_test)
ada_roc_auc = roc_auc_score(y_true, y_pred_ada)
print ("Adaboost roc_auc = %2.2f" % ada_roc_auc)
print(classification_report(y_true, y_pred_ada))
print()
print("done in %0.3fs" % (time() - t0))

# Create ROC Graph
#logreg_fpr, logreg_tpr, logreg_thresholds = roc_curve(y_test, y_pred_logreg)
#rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, y_pred_rf)
#ada_fpr, ada_tpr, ada_thresholds = roc_curve(y_test, y_pred_ada)

logreg_fpr, logreg_tpr, logreg_thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
ada_fpr, ada_tpr, ada_thresholds = roc_curve(y_test, ada.predict_proba(X_test)[:,1])

plt.figure()
# Plot Logistic Regression ROC
plt.plot(logreg_fpr, logreg_tpr, label='Logistic Regression (area = %0.2f)' % logreg_roc_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)

# Plot AdaBoost ROC
plt.plot(ada_fpr, ada_tpr, label='AdaBoost (area = %0.2f)' % ada_roc_auc)

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()
