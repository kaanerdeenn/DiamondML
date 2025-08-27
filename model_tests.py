import pickle
import pandas as pd

with open("30-diamond_model_complete.pkl","rb") as f:
    saved_data=pickle.load(f)



model=saved_data["model"]

X_test_scaled=pd.read_csv("30-testdatascaled.csv")
print(model.predict(X_test_scaled))



