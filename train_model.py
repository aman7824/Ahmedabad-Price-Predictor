import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

df = pd.read_csv("ahmedabad_house.csv")
df = df.drop(columns=['Unnamed: 0','Title','description'], errors='ignore')
df = df.dropna()

encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBRegressor(n_estimators=200, learning_rate=0.05)
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl","wb"))
pickle.dump(encoders, open("encoders.pkl","wb"))
pickle.dump(X.columns.tolist(), open("columns.pkl","wb"))

print("Model trained with XGBoost")
