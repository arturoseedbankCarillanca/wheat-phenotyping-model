import cv2 as cv
import numpy as np
from modelo import prediccion, entrenamiento
import pandas as pd


def prediccion_con_img():
    base_datos=pd.DataFrame()
    for numero in range(78,86):
        numero=594
        data_=pd.DataFrame()
        imagen1 = cv.imread(f'/home/arcgis/Escritorio/images_python/pruebas_im/Espigas_arturo/JPG_trim_high_resolution/{numero}.JPG')

        # Ajusta las dimensiones de la imagen 
        scale_percent = 50
        width = int(imagen1.shape[1] * scale_percent / 100)
        height = int(imagen1.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # Redimensiona la imagen
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

        for contour in contours:
            area_pixel = cv.contourArea(contour)
            if 1000000 > area_pixel > 500:
                cv.drawContours(resized, contour, -1, (0, 255, 0), 3)  # Dibuja el contorno
                x, y, w, h = cv.boundingRect(contour)
                
                w_cm = pixel_scale * w
                h_cm = pixel_scale * h
                area_cm = pixel_scale * pixel_scale * w*h  # Área en cm^2
                print(f"Area del contorno en cm^2: {area_cm}")
                print(f"Dimensiones de la caja delimitadora en cm: {w_cm} x {h_cm}")

        prediccion_espi=prediccion(area_cm, w_cm, h_cm)
        data_['Numero']=[numero]
        data_['area']=[area_cm]
        data_['ancho']=[w_cm]
        data_['largo']=[h_cm]
        if base_datos.empty:
            base_datos=data_
        else:
            base_datos=pd.concat(base_datos,data_,axis=0)


        print(f'la cantidad de espiguillas son: {prediccion_espi}')
        cv.imshow('Contours', resized)
        cv.waitKey(0)
        cv.destroyAllWindows()

#entrenamiento()
prediccion_con_img()
