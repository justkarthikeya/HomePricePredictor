import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("/Users/karthikeyakammili/Downloads/Data Sets/Real Estate Project/"
                 "bengaluru_house_prices_cleaned_2.csv")

# print(df.isnull().sum())

df = df[~df['del_by_bath'].str.contains('del')]
# print(df.head().to_string())
df = df[~df['del_by_sqft'].str.contains('del')]
# print(df.head().to_string())

df['locationid'] = df['location'].factorize()[0]
# print(df.head().to_string())

X = df[['locationid', 'bhk', 'bath', 'tot_sqft']]
y = df['house_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

print(f'Accuracy of RFC Model: {clf.score(X_test, y_test)}')
print(f'Predicted price by RFC Model: {clf.predict([[0, 4, 4, 2100]])}')
