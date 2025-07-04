===============================================================================
🌤️ PREDICTOR DE CLIMA A PRIORI - INSTRUCCIONES DE USO
===============================================================================

Proyecto: BIY7121_EV3_Clustering_RiquelmeFrancisco_MarceloLeandro_HernándezVíctor
Fecha: 2024
Modelo: Algoritmo A PRIORI para combinaciones climáticas

===============================================================================
📁 ESTRUCTURA DEL PROYECTO
===============================================================================

├── app.py                     # API Flask con el modelo A PRIORI
├── modelo_entrenado.pkl       # Modelo serializado (generado automáticamente)
├── entrenamiento_modelo.ipynb # Notebook para entrenar el modelo
├── index.html                 # Interfaz web que consume la API
├── ejemplo.csv                # Archivo CSV de prueba
└── README.txt                 # Este archivo de instrucciones

===============================================================================
🔧 REQUISITOS PREVIOS
===============================================================================

1. Python 3.8 o superior
2. Pip (gestor de paquetes de Python)
3. Navegador web moderno (Chrome, Firefox, Safari, Edge)

===============================================================================
📦 INSTALACIÓN DE DEPENDENCIAS
===============================================================================

1. Abrir terminal/consola de comandos
2. Navegar al directorio del proyecto
3. Ejecutar el siguiente comando:

   pip install flask flask-cors pandas numpy scikit-learn mlxtend joblib

   O alternativamente, si tienes un archivo requirements.txt:
   pip install -r requirements.txt

===============================================================================
🚀 PASOS PARA EJECUTAR EL SISTEMA
===============================================================================

PASO 1: ENTRENAR EL MODELO
--------------------------
1. Abrir el archivo 'entrenamiento_modelo.ipynb' en Jupyter Notebook
2. Ejecutar todas las celdas del notebook
3. Esto generará el archivo 'modelo_entrenado.pkl' automáticamente

PASO 2: INICIAR LA API
----------------------
1. Abrir terminal en el directorio del proyecto
2. Ejecutar el comando:
   python app.py
   
3. Deberías ver un mensaje similar a:
   "🚀 Iniciando API en puerto 5000"
   "📊 Modelo disponible: ✅"
   
4. La API estará disponible en: http://localhost:5000

PASO 3: VERIFICAR LA API
------------------------
1. Abrir navegador web
2. Ir a: http://localhost:5000
3. Deberías ver la página de información de la API
4. Para verificar el estado: http://localhost:5000/health

PASO 4: USAR LA INTERFAZ WEB
----------------------------
1. Abrir el archivo 'index.html' en tu navegador web
2. Verificar que la URL de la API esté configurada como: http://localhost:5000
3. Cargar el archivo 'ejemplo.csv' usando el botón "Seleccionar archivo CSV"
4. Hacer clic en "Procesar Datos"
5. Hacer clic en "Generar Predicciones"
6. Ver los resultados en gráficos y tablas

===============================================================================
📊 FORMATO DE DATOS CSV
===============================================================================

El archivo CSV debe contener las siguientes columnas:
- MaxTemp: Temperatura máxima del día (°C)
- Humidity3pm: Humedad a las 3 PM (%)
- WindGustSpeed: Velocidad máxima del viento (km/h)
- Sunshine: Horas de sol (horas)

Ejemplo de formato:
MaxTemp,Humidity3pm,WindGustSpeed,Sunshine
22.9,22.0,44.0,7.04
25.1,25.0,44.0,8.7
...

===============================================================================
🔧 CONFIGURACIÓN AVANZADA
===============================================================================

CAMBIAR PUERTO DE LA API:
- Modificar la línea final de app.py
- Cambiar: app.run(host='0.0.0.0', port=5000)
- Por: app.run(host='0.0.0.0', port=PUERTO_DESEADO)

HABILITAR MODO DEBUG:
- Ejecutar: python app.py
- O establecer variable de entorno: export DEBUG=true

CONFIGURAR PARA PRODUCCIÓN:
- Usar un servidor WSGI como Gunicorn
- Comando: gunicorn -w 4 -b 0.0.0.0:5000 app:app

===============================================================================
🧪 PRUEBAS Y VALIDACIÓN
===============================================================================

PROBAR LA API DIRECTAMENTE:
1. Usar curl o Postman
2. Endpoint: POST http://localhost:5000/predict
3. Body JSON:
   {
     "MaxTemp": 25.5,
     "Humidity3pm": 60.0,
     "WindGustSpeed": 35.0,
     "Sunshine": 8.5
   }

PROBAR CON LOTE DE DATOS:
1. Endpoint: POST http://localhost:5000/predict_batch
2. Body JSON:
   {
     "data": [
       {"MaxTemp": 25.5, "Humidity3pm": 60.0, "WindGustSpeed": 35.0, "Sunshine": 8.5},
       {"MaxTemp": 30.0, "Humidity3pm": 40.0, "WindGustSpeed": 50.0, "Sunshine": 12.0}
     ]
   }

===============================================================================
📈 INTERPRETACIÓN DE RESULTADOS
===============================================================================

MÉTRICAS PRINCIPALES:
- Soporte: Frecuencia de aparición de una combinación
- Confianza: Probabilidad de que ocurra la consecuencia dado el antecedente
- Lift: Cuántas veces es más probable que ocurra la combinación vs. al azar

VALORES TÍPICOS:
- Confianza > 0.5 (50%): Regla confiable
- Lift > 1.0: Correlación positiva
- Lift = 1.0: Independencia estadística
- Lift < 1.0: Correlación negativa

===============================================================================
🐛 SOLUCIÓN DE PROBLEMAS
===============================================================================

ERROR: "Modelo no disponible"
- Verificar que existe el archivo 'modelo_entrenado.pkl'
- Ejecutar el notebook de entrenamiento completo

ERROR: "Port already in use"
- Cambiar el puerto en app.py
- O cerrar la aplicación que esté usando el puerto 5000

ERROR: "CORS policy"
- Verificar que flask-cors esté instalado
- La API ya incluye configuración CORS

ERROR: "File not found"
- Verificar que todos los archivos estén en el mismo directorio
- Revisar permisos de archivos

===============================================================================
💡 CONSEJOS Y MEJORES PRÁCTICAS
===============================================================================

1. DATOS DE ENTRADA:
   - Usar valores realistas para las variables climáticas
   - Verificar que no haya valores nulos o extremos
   - El modelo funciona mejor con datos dentro de rangos normales

2. INTERPRETACIÓN:
   - Enfocarse en reglas con alta confianza (>0.7)
   - Considerar el lift para evaluar la fuerza de la asociación
   - Analizar las combinaciones frecuentes para identificar patrones

3. RENDIMIENTO:
   - Para datasets grandes, procesar en lotes pequeños
   - Limitar las predicciones a ~100 registros por solicitud
   - Usar el endpoint /predict para casos individuales

===============================================================================
📞 SOPORTE Y CONTACTO
===============================================================================

En caso de problemas técnicos:
1. Revisar los logs de la API en la consola
2. Verificar que todas las dependencias estén instaladas
3. Consultar la documentación de la API en http://localhost:5000

Desarrollado por:
- Francisco Riquelme
- Leandro Marcelo  
- Víctor Hernández

Curso: BIY7121 - Evaluación N°3
Fecha: 2024

===============================================================================
🎯 OBJETIVOS DE APRENDIZAJE CUMPLIDOS
===============================================================================

✅ Puesta en producción de modelo de Machine Learning
✅ Creación de API REST con Flask
✅ Consumo de API desde interfaz web
✅ Visualización dinámica de resultados
✅ Manejo de datos CSV en tiempo real
✅ Implementación de modelo A PRIORI
✅ Análisis de reglas de asociación
✅ Interpretación de métricas de minería de datos

===============================================================================