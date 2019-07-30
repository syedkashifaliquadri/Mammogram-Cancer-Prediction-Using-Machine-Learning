from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler


print("STARTED")
# data = pd.read_csv('datat.csv')
df = pd.read_csv('datat.csv', names = ["BI_RADS", "Age", "Shape", "Margin", "Density", "Severity"], na_values=["?"])
df['BI_RADS'].fillna(4.277134, inplace=True) #NOT REALLY NECESSARY SINCE BI-RADS ISN'T USED!
df['Age'].fillna(55.416490, inplace=True)
df['Shape'].fillna(2.721505, inplace=True)
df['Margin'].fillna(2.792035, inplace=True)
df['Density'].fillna(2.908571, inplace=True)

X = df.drop(['Severity', 'BI_RADS'], axis=1)
y = df['Severity']
features = ['Age', 'Shape', 'Margin', 'Density']
X_array = X.as_matrix() # creating an array
y_array = y.as_matrix() # creating an array
scaler = StandardScaler().fit(X)

scaled_data = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(scaled_data,y, test_size=0.25, random_state=42)

y = df['Severity'] #this is what we're trying to predict!
X = df[features]
print("===============================================")
dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(X_train, y_train)

dtc_pred = dtc.predict(X_test)
print(accuracy_score( y_test, dtc_pred))
print(dtc.predict([[46,1,1,1]]))
print("===============================================")
print("random start ")

forest_reg = RandomForestClassifier(random_state=42)
forest_reg.fit(X_train, y_train)
forest_reg_pred = forest_reg.predict(X_test)
print(accuracy_score( y_test, forest_reg_pred))
print("random end")

print("===============================================")
print("svm start ")

svm_linear = SVC( kernel = 'linear')
svm_linear.fit(X_train, y_train)
svm_linear_pred = svm_linear.predict(X_test)
print(accuracy_score(y_test, svm_linear_pred))
print("svm end")
print("===============================================")


print("KNN start ")
neigh = KNeighborsClassifier(n_neighbors=21)
neigh.fit(X_train, y_train)
neigh_pred = neigh.predict(X_test)
print(accuracy_score( y_test, neigh_pred))
print("KNN end")
print("===============================================")


print("Naive Base start ")
scaler = MinMaxScaler()
X_minmax = scaler.fit_transform(X)

mnb = MultinomialNB()
X_train_MinMax, X_test_MinMax, y_train, y_test = train_test_split(X_minmax,y, test_size=0.25, random_state=42)
mnb.fit(X_train_MinMax, y_train)
mnb_pred = mnb.predict(X_test_MinMax)
print(accuracy_score( y_test, mnb_pred))
print("Naive Base end")
print("===============================================")


# X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.3)
#

# #Using Random Forest Classifier
# clf_rf = RandomForestClassifier(random_state=43)
# clf_rf = clf_rf.fit(X_train,y_train)
#
# ac = accuracy_score(y_test,clf_rf.predict(X_test))
# print('Accuracy is : ',ac)
# print(clf_rf.predict([[21.57,546.1,0.06373,20.98,0.02045,0.01795,0.006399,621.2,0.114,0.1667,0.1212,0.05614,0.2637]]))

# X = data.drop('diagnosis', axis=1)
# y = data['diagnosis']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# clf_rf = RandomForestClassifier(random_state=43)
# clf_rf = clf_rf.fit(X_train, y_train)

print("Trained")
# X1 = X.drop(['radius_mean','perimeter_mean','radius_se','perimeter_se','radius_worst','perimeter_worst'],axis=1)
# X1.drop(['fractal_dimension_mean','smoothness_se','texture_se','symmetry_se','fractal_dimension_se'],axis=1,inplace=True)
#
# X1.drop(['fractal_dimension_worst'],axis=1,inplace=True)
# X1.drop(['concave points_mean','texture_worst','concavity_mean'],axis=1,inplace=True)
#
# X1.drop(['smoothness_mean','symmetry_mean','id'],axis=1,inplace=True)


# X1 = (X1 - X1.mean())/(X1.std())
# data_final = X1.copy()
# data_final['diagnosis'] = y
# data_final
# data_final = pd.melt(data_final,id_vars='diagnosis',
#                                 var_name='features',
#                                 value_name='value')


# X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.3)


app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':
        print("YES")
        a_1=request.form.get('ex1')
        a_2=request.form.get('ex2')
        a_3=request.form.get('ex3')
        a_4=request.form.get('ex4')
        a_5=request.form.get('ex5')
        a_6=request.form.get('ex6')
        a_7=request.form.get('ex7')
        a_8=request.form.get('ex8')
        a_9=request.form.get('ex9')
        ac = clf_rf.predict([[a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9]])
        print(ac)
        if ac == [1]:
            # print('Random Classifier Accuracy is : ', ac)
            print('Malignant Tumor')
            return render_template('index.html', ac='Malignant Tumor')
        else:
            print('Benign Tumor')
            return render_template('index.html', ac='Benign Tumor')

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


app.run(debug=True)

