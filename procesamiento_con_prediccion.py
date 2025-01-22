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
import cv2 as cv
import numpy as np
import pandas as pd
lista_n_ima=list(range(56,86)) + list(range(101,146)) + list(range(151,187))+ list(range(188,229))+ list(range(231,240))+ list(range(241,250))+ list(range(251,426))+ list(range(431,511))+ list(range(512,555))+ list(range(556,560))+ list(range(561,605))+ list(range(606,773))+ list(range(774,785))+ list(range(786,841))+ list(range(847,863))+ list(range(864,891))+ list(range(896,911))+ list(range(912,950))+ list(range(951,983))+ list(range(984,991))+ list(range(1001,1096))+ list(range(1106,1111))+ list(range(1116,1121))+ list(range(1126,1131))+ list(range(1136,1141))\
+ list(range(1151,1213))+ list(range(1214,1224))+ list(range(1225,1227))+ list(range(1228,1278))+ list(range(1279,1313))+ list(range(1331,1341))+ list(range(1471,1530))+ list(range(1531,1543))+ list(range(1544,1546))+ list(range(1751,1791))+ list(range(1846,1866))+ list(range(1868,1869))+ list(range(1871,1875))+ list(range(1880,1889))+ list(range(1890,1961))

lista_negra=[116,117,118,119,120,169,172,202,203,205,270,341,342,343,346,348,350,358,359,360,362,364,367,368,388,389,390,391,392,396,397,399,400,401,402,406,414,418,429,452,459,460,464,513,613,674,698,890,956,961,962,963,964,965,977,978,979,982,1005,1008,1010,1022,1023,1027,1028,1032,1033,1036,1037,1038,1039,1040,1041,1042,1048,1060,1073,1074,1075,1093,1116,1117,1119,1136,1163,1164,1172,1173,1178,1179,1180,1198,1208,1215,1221,1225,1238,1242,1263,1268,1269,1271,1272,1273,1274,1275,1282,1284,1294,1304,1340,1534,1778,1311,1312]
lista_n_ima_nueva = [item for item in lista_n_ima if item not in lista_negra]
def prediccion_con_img_prediccion():
    base_datos=pd.DataFrame()
    for numero in lista_n_ima_nueva:
        data_=pd.DataFrame()
        imagen1 = cv.imread(f'/home/arcgis/Escritorio/images_python/pruebas_im/Espigas_arturo/JPG_trim_high_resolution/{numero}.JPG') #Esto cambiar a imagenes de prueba que no sean las que usamos en el modelo y que se tengan datos

        # Ajusta las dimensiones de la imagen 
        scale_percent = 50
        width = int(imagen1.shape[1] * scale_percent / 100)
        height = int(imagen1.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # Redimensiona la imagen9
        resized = cv.resize(imagen1, dim, interpolation = cv.INTER_AREA)

        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        _, threshold = cv.threshold(gray, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # Crea un kernel circular para las operaciones morfológica
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))

        # Utiliza operaciones morfológicas para mejorar la imagen
        opening = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel, iterations=5)
        dilated = cv.dilate(opening, kernel, iterations=4)
        eroded = cv.erode(dilated, kernel, iterations=3)

        # Invierte los colores
        mask = cv.bitwise_not(eroded)

        # Encuentra los contornos
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        pixel_scale = 0.01351749  # Escala de píxel (cm/píxel), reemplazar con el valor real

        area_suma=[]
        largo_suma=[]
        ancho_mayor=[]
        for contour in contours:
            area_pixel = cv.contourArea(contour)
             
            if 1000000 > area_pixel > 500:
                # Encuentra el rectángulo de área mínima
                rect = cv.minAreaRect(contour)
                # Obtiene el ancho y el alto del rectángulo de área mínima
                _, (w, h), _ = rect
                
                #x, y, w, h = cv.boundingRect(contour)
                
                # Cálculo de ancho y largo en cm con pixel_scale
                w_cm = pixel_scale * min(w, h)
                h_cm = pixel_scale * max(w, h)          
                area_pixel = cv.contourArea(contour)
                area_cm = pixel_scale * pixel_scale * area_pixel # Área en cm^2
                area_suma.append(area_cm)
                largo_suma.append(h_cm)
                ancho_mayor.append(w_cm)

        area_total=sum(area_suma)
        largo_total=sum(largo_suma)
        ancho_mayor_=max(ancho_mayor)
        
        
        #prediccion

        model = joblib.load('model_NUEVO_SVR.joblib')
        # Predicting the Test set results
        #h_cm=7.44
        #w_cm=1.69
        #area_cm=12.57
        # creamos un dataframe con los datos de entrada
        data_pred = pd.DataFrame([[h_cm, w_cm, area_cm]], columns=["largo", "ancho", "area"])
        # Predice la cantidad de espiguillas
        prediccion_espi= model.predict(data_pred)
    

        print(f"Area del contorno en cm^2: {area_total}")
        print(f"Dimensiones de la caja delimitadora en cm: {largo_total} x {ancho_mayor_}")
        print(f"Cantidad de espigullas: {prediccion_espi}")

        data_['Numero']=[numero]
        data_['area']=[area_total]
        data_['ancho']=[ancho_mayor_]
        data_['largo']=[largo_total]
        data_['espiguillas']=[largo_total]
        if base_datos.empty:
            base_datos=data_
        else:
            base_datos=pd.concat([base_datos,data_],axis=0)
    base_datos.to_excel('metricas_predicción.xlsx')

#para iniciar función
prediccion_con_img_prediccion()