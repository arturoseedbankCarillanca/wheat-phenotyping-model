import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from openpyxl import Workbook
from joblib import dump, load

def entrenamiento():
    data = pd.read_excel("/home/arcgis/Escritorio/images_python/pruebas_im/data.xlsx")

    #data.dropna(inplace=True)
    X = data[["alto_cm", "ancho_cm", "area"]]
    y = data["espiguillas"]

    # Dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Iniciamos y entrenamos el modelo prueba 1
    #model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Iniciamos y entrenamos el modelo
    model = KNeighborsRegressor(n_neighbors=15)
   
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
     # Métricas del modelo
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Coefficient of Determination R2:", r2_score(y_test, y_pred))
    # Guardar el modelo entrenado 
    joblib.dump(model, 'model.joblib')


def prediccion(area_cm, w_cm, h_cm):
 
    model = joblib.load('model.joblib')
    # Predicting the Test set results
    #h_cm=7.44
    #w_cm=1.69
    #area_cm=12.57
    # creamos un dataframe con los datos de entrada
    data_pred = pd.DataFrame([[h_cm, w_cm, area_cm]], columns=["alto_cm", "ancho_cm", "area"])
    # Predice la cantidad de espiguillas
    prediccion_espi= model.predict(data_pred)
    
    return prediccion_espi[0]

