import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from openpyxl import Workbook
from joblib import dump, load
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor


def entrenamiento():
    data = pd.read_excel("/home/arcgis/Escritorio/images_python/metricas_nuevas_2.xlsx")
    
    #data.dropna(inplace=True)
    X = data[["largo", "ancho", "area"]]
    y = data["espiguillas"]
    # Eliminar filas con NaN en y
    
    # Dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    #model = RandomForestRegressor(n_estimators=1100, random_state=42)
    #model = KNeighborsRegressor(n_neighbors=33)
    #model = LinearRegression()
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    #model = AdaBoostRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
   
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
     # Métricas del modelo
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Coefficient of Determination R2:", r2_score(y_test, y_pred))
    # Guardar el modelo entrenado 
    joblib.dump(model, 'model_NUEVO_SVR.joblib') 

#para iniciar la funcion 
entrenamiento()