import pandas as pd
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df
pd.options.display.max_columns = 500
df
# Binary: Attrition, Gender, Over18, OverTime
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x== 'Male' else 0)
df['Over18'] = df['Over18'].apply(lambda x: 1 if 'Y' else 0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x== 'Yes' else 0 )
df
df = df.join(pd.get_dummies(df['BusinessTravel'])).drop('BusinessTravel', axis=1)
df = df.join(pd.get_dummies(df['Department'],prefix='Department')).drop('Department', axis=1)
df = df.join(pd.get_dummies(df['EducationField'],prefix='Education')).drop('EducationField', axis=1)
df = df.join(pd.get_dummies(df['JobRole'],prefix='Role')).drop('JobRole', axis=1)
df = df.join(pd.get_dummies(df['MaritalStatus'],prefix='Status')).drop('MaritalStatus', axis=1)
df
df = df.map(lambda x: 1 if x is True else 0 if x is False else x)

df
df = df.drop('EmployeeNumber' , axis = 1)
df
import matplotlib.pyplot as plt

df.hist(figsize=(20 ,15))
plt.tight_layout()
plt.show()
df = df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1)
df
### Model Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = df.drop('Attrition', axis=1), df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_jobs=-1)

model.fit(X_train, y_train)
model.score(X_test, y_test)
sorted_importances = dict(sorted(zip(model.feature_names_in_, model.feature_importances_), key=lambda x: x[1], reverse=True))
import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))
plt.bar(sorted_importances.keys(), sorted_importances.values())
plt.xticks(rotation=45, ha='right')

plt.show()
