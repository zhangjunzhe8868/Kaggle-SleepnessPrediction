from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import precision_recall_curve,auc
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


train,test = train_test_split(df_final,test_size =0.3 ,random_state = 111)

train_x=train.drop(['drowsy'],axis=1)
test_x=test.drop(['drowsy'],axis=1)
train_y=train['drowsy']
test_y=test['drowsy']

logit  = LogisticRegression(C=1.0, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
logit.fit(train_x,train_y)
predictions   = logit.predict(test_x)
probabilities = logit.predict_proba(test_x)

RFC=RandomForestClassifier(n_estimators = 500)
RFC.fit(train_x,train_y)
predictions_rf   = RFC.predict(test_x)
probabilities_rf = RFC.predict_proba(test_x)