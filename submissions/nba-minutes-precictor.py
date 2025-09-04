import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("hf://datasets/Hatman/NBA-Player-Career-Stats/nba_career_stats.csv")

target = "MIN"
features = df.select_dtypes(include=[np.number]).drop(columns=[target])

X = features
y = df[target]

X = X.fillna(0)
y = y.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

def evaluate(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"  MSE: {mse:.2f}")
    print(f"  R^2: {r2:.3f}\n")

evaluate(y_test, y_pred_rf, "Random Forest")
evaluate(y_test, y_pred_lr, "Linear Regression")

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 important features:\n", importances.head(10))

def predict_minutes(player_name, model="rf"):
    row = df[df["FULL_NAME"] == player_name]
    if row.empty:
        return f"Player '{player_name}' not found in dataset."
    
    X_row = row.select_dtypes(include=[np.number]).drop(columns=[target]).fillna(0)
    
    if model == "rf":
        pred = rf.predict(X_row)[0]
    else:
        pred = lr.predict(X_row)[0]
    
    actual = row[target].values[0]
    
    print(f"Predicted Minutes: {pred:.0f}")
    print(f"Actual Minutes: {actual}")
    return ""

print(predict_minutes("Cody Zeller", model="rf"))
print(predict_minutes("Cody Zeller", model="lr"))