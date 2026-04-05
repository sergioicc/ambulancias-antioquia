import joblib

joblib.dump(km, "modelo_kmeans.pkl")
joblib.dump(scaler, "scaler_kmeans.pkl")