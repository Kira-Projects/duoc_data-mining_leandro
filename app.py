#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Flask para modelo A PRIORI - Clima Australia
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# ============================================================================
# CONFIGURACIÓN Y CARGA DEL MODELO
# ============================================================================

class ModeloApriori:
    def __init__(self, modelo_path='./modelo_entrenado.pkl'):
        """Inicializa el modelo A PRIORI"""
        self.modelo = None
        self.cargar_modelo(modelo_path)
    
    def cargar_modelo(self, modelo_path):
        """Carga el modelo serializado"""
        try:
            with open(modelo_path, 'rb') as f:
                self.modelo = pickle.load(f)
            logger.info("✅ Modelo cargado exitosamente")
        except FileNotFoundError:
            logger.error(f"❌ Archivo {modelo_path} no encontrado")
            raise
        except AttributeError as e:
            # Si hay un error de atributo, probablemente es por incompatibilidad de clase
            logger.error(f"❌ Error de compatibilidad al cargar modelo: {str(e)}")
            logger.info("🔄 Intentando cargar con método alternativo...")
            self._cargar_modelo_alternativo(modelo_path)
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {str(e)}")
            raise
    
    def _cargar_modelo_alternativo(self, modelo_path):
        """Método alternativo para cargar el modelo con compatibilidad"""
        try:
            # Crear una clase temporal compatible
            class ModeloAPRIORI:
                pass
            
            # Agregar la clase al módulo actual
            import sys
            sys.modules[__name__].ModeloAPRIORI = ModeloAPRIORI
            
            # Cargar el modelo
            with open(modelo_path, 'rb') as f:
                modelo_obj = pickle.load(f)
            
            # Si el modelo es una instancia de ModeloAPRIORI, convertir a dict
            if isinstance(modelo_obj, ModeloAPRIORI):
                # Mapear los atributos del modelo a los nombres esperados por el código
                self.modelo = {
                    'variables': modelo_obj.variables_clima,
                    'discretization_maps': modelo_obj.rangos,
                    'frequent_itemsets': modelo_obj.frequent_itemsets,
                    'rules': modelo_obj.rules
                }
                logger.info("✅ Modelo convertido a dict desde ModeloAPRIORI con mapeo de atributos")
            else:
                self.modelo = modelo_obj
                logger.info("✅ Modelo cargado exitosamente con compatibilidad (no fue necesario convertir)")
            
        except Exception as e:
            logger.error(f"❌ Error en carga alternativa: {str(e)}")
            raise
    
    def discretizar_valor(self, variable, valor):
        """Discretiza un valor numérico basado en los rangos del modelo"""
        if variable not in self.modelo['discretization_maps']:
            return None
        
        rangos = self.modelo['discretization_maps'][variable]
        
        # Determinar la categoría basada en los rangos (tuples)
        for categoria, rango in rangos.items():
            # Los rangos son tuples (min, max)
            min_val, max_val = rango
            if min_val <= valor <= max_val:
                # Generar el formato correcto: Variable_Cat_Categoria
                return f"{variable}_Cat_{categoria}"
        
        # Si no está en ningún rango, usar el más cercano
        mejor_categoria = None
        menor_distancia = float('inf')
        
        for categoria, rango in rangos.items():
            min_val, max_val = rango
            distancia = min(abs(valor - min_val), abs(valor - max_val))
            if distancia < menor_distancia:
                menor_distancia = distancia
                mejor_categoria = categoria
        
        if mejor_categoria:
            return f"{variable}_Cat_{mejor_categoria}"
        
        return None
    
    def predecir_asociaciones(self, datos_entrada):
        """Predice asociaciones basadas en los datos de entrada"""
        try:
            # Discretizar datos de entrada
            datos_discretos = []
            for variable, valor in datos_entrada.items():
                if variable in self.modelo['variables']:
                    categoria = self.discretizar_valor(variable, valor)
                    if categoria:
                        datos_discretos.append(categoria)
            
            # Buscar reglas aplicables
            reglas_aplicables = []
            
            for _, regla in self.modelo['rules'].iterrows():
                antecedentes = set(regla['antecedents'])
                
                # Verificar si los antecedentes están en los datos de entrada
                if antecedentes.issubset(set(datos_discretos)):
                    consecuentes = list(regla['consequents'])
                    reglas_aplicables.append({
                        'antecedents': list(antecedentes),
                        'consequents': consecuentes,
                        'confidence': float(regla['confidence']),
                        'lift': float(regla['lift']),
                        'support': float(regla['support'])
                    })
            
            # Ordenar por confianza
            reglas_aplicables.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Buscar combinaciones frecuentes que incluyan los datos de entrada
            combinaciones_frecuentes = []
            for _, item in self.modelo['frequent_itemsets'].iterrows():
                itemset = set(item['itemsets'])
                entrada_set = set(datos_discretos)
                
                # Si hay intersección entre el itemset y los datos de entrada
                if entrada_set.intersection(itemset):
                    combinaciones_frecuentes.append({
                        'itemset': list(itemset),
                        'support': float(item['support']),
                        'match_level': len(entrada_set.intersection(itemset)) / len(entrada_set)
                    })
            
            # Ordenar por nivel de coincidencia y soporte
            combinaciones_frecuentes.sort(key=lambda x: (x['match_level'], x['support']), reverse=True)
            
            return {
                'datos_entrada_discretos': datos_discretos,
                'reglas_aplicables': reglas_aplicables[:10],  # Top 10
                'combinaciones_frecuentes': combinaciones_frecuentes[:10],  # Top 10
                'total_reglas': len(reglas_aplicables),
                'total_combinaciones': len(combinaciones_frecuentes)
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            raise

# Inicializar modelo global
try:
    print("inicializando")
    modelo_global = ModeloApriori()
except Exception as e:
    logger.error(f"Error inicializando modelo: {str(e)}")
    modelo_global = None

# ============================================================================
# RUTAS DE LA API
# ============================================================================

@app.route('/')
def home():
    """Página de inicio con información de la API"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Modelo A PRIORI - Clima Australia</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #27ae60; font-weight: bold; }
            .url { color: #2980b9; font-family: monospace; }
            pre { background: #2c3e50; color: white; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .status { text-align: center; padding: 20px; }
            .status.ok { color: #27ae60; }
            .status.error { color: #e74c3c; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌤️ API Modelo A PRIORI - Clima Australia</h1>
            
            <div class="status {{ 'ok' if modelo_ok else 'error' }}">
                {{ '✅ Modelo cargado correctamente' if modelo_ok else '❌ Error: Modelo no disponible' }}
            </div>
            
            <h2>📚 Endpoints Disponibles</h2>
            
            <div class="endpoint">
                <div><span class="method">GET</span> <span class="url">/</span></div>
                <p>Página de inicio con información de la API</p>
            </div>
            
            <div class="endpoint">
                <div><span class="method">GET</span> <span class="url">/health</span></div>
                <p>Verificar el estado de la API y el modelo</p>
            </div>
            
            <div class="endpoint">
                <div><span class="method">POST</span> <span class="url">/predict</span></div>
                <p>Predecir asociaciones climáticas basadas en datos de entrada</p>
                <h4>Formato de entrada:</h4>
                <pre>{
    "MaxTemp": 25.5,
    "Humidity3pm": 60.0,
    "WindGustSpeed": 35.0,
    "Sunshine": 8.5
}</pre>
            </div>
            
            <div class="endpoint">
                <div><span class="method">POST</span> <span class="url">/predict_batch</span></div>
                <p>Predicción en lote para múltiples registros</p>
                <h4>Formato de entrada:</h4>
                <pre>{
    "data": [
        {"MaxTemp": 25.5, "Humidity3pm": 60.0, "WindGustSpeed": 35.0, "Sunshine": 8.5},
        {"MaxTemp": 30.0, "Humidity3pm": 40.0, "WindGustSpeed": 50.0, "Sunshine": 12.0}
    ]
}</pre>
            </div>
            
            <h2>🔧 Información del Modelo</h2>
            <div class="endpoint">
                <p><strong>Tipo:</strong> Algoritmo A PRIORI</p>
                <p><strong>Variables:</strong> MaxTemp, Humidity3pm, WindGustSpeed, Sunshine</p>
                <p><strong>Categorías:</strong> Baja, Media, Alta</p>
                <p><strong>Métricas:</strong> Soporte, Confianza, Lift</p>
            </div>
            
            <h2>📊 Ejemplo de Uso</h2>
            <pre>curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"MaxTemp": 25.5, "Humidity3pm": 60.0, "WindGustSpeed": 35.0, "Sunshine": 8.5}'</pre>
            
            <div style="text-align: center; margin-top: 30px; color: #7f8c8d;">
                <p>🎓 BIY7121_EV3_Clustering_MarceloLeandro</p>
                <p>{{ timestamp }}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html_content, 
                                modelo_ok=modelo_global is not None,
                                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@app.route('/health', methods=['GET'])
def health_check():
    """Verificar el estado de la API"""
    try:
        status = {
            'status': 'healthy' if modelo_global is not None else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'modelo_cargado': modelo_global is not None,
            'version': '1.0.0'
        }
        
        if modelo_global is not None:
            status['modelo_info'] = {
                'variables': modelo_global.modelo['variables'],
                'total_reglas': len(modelo_global.modelo['rules']),
                'total_itemsets': len(modelo_global.modelo['frequent_itemsets'])
            }
        
        return jsonify(status), 200 if modelo_global is not None else 503
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint principal para predicciones"""
    try:
        if modelo_global is None:
            return jsonify({
                'error': 'Modelo no disponible',
                'message': 'El modelo no se ha cargado correctamente'
            }), 503
        
        # Validar datos de entrada
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Datos inválidos',
                'message': 'Se requieren datos JSON en el body'
            }), 400
        
        # Validar variables requeridas
        required_vars = ['MaxTemp', 'Humidity3pm', 'WindGustSpeed', 'Sunshine']
        missing_vars = [var for var in required_vars if var not in data]
        
        if missing_vars:
            return jsonify({
                'error': 'Variables faltantes',
                'message': f'Se requieren las variables: {missing_vars}'
            }), 400
        
        # Validar tipos de datos
        for var in required_vars:
            try:
                data[var] = float(data[var])
            except (ValueError, TypeError):
                return jsonify({
                    'error': 'Tipo de dato inválido',
                    'message': f'La variable {var} debe ser numérica'
                }), 400
        
        # Realizar predicción
        resultado = modelo_global.predecir_asociaciones(data)
        
        # Construir respuesta
        respuesta = {
            'timestamp': datetime.now().isoformat(),
            'datos_entrada': data,
            'prediccion': resultado,
            'interpretacion': _generar_interpretacion(resultado)
        }
        
        return jsonify(respuesta), 200
        
    except Exception as e:
        logger.error(f"Error en /predict: {str(e)}")
        return jsonify({
            'error': 'Error interno',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Endpoint para predicciones en lote"""
    try:
        if modelo_global is None:
            return jsonify({
                'error': 'Modelo no disponible',
                'message': 'El modelo no se ha cargado correctamente'
            }), 503
        
        # Validar datos de entrada
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({
                'error': 'Datos inválidos',
                'message': 'Se requiere un array "data" con los registros'
            }), 400
        
        batch_data = data['data']
        if not isinstance(batch_data, list):
            return jsonify({
                'error': 'Formato inválido',
                'message': 'El campo "data" debe ser un array'
            }), 400
        
        # Procesar cada registro
        resultados = []
        errores = []
        
        required_vars = ['MaxTemp', 'Humidity3pm', 'WindGustSpeed', 'Sunshine']
        
        for idx, registro in enumerate(batch_data):
            try:
                # Validar variables requeridas
                missing_vars = [var for var in required_vars if var not in registro]
                if missing_vars:
                    errores.append({
                        'index': idx,
                        'error': f'Variables faltantes: {missing_vars}'
                    })
                    continue
                
                # Validar tipos de datos
                registro_limpio = {}
                for var in required_vars:
                    try:
                        registro_limpio[var] = float(registro[var])
                    except (ValueError, TypeError):
                        errores.append({
                            'index': idx,
                            'error': f'Variable {var} debe ser numérica'
                        })
                        continue
                
                # Realizar predicción
                resultado = modelo_global.predecir_asociaciones(registro_limpio)
                
                resultados.append({
                    'index': idx,
                    'datos_entrada': registro_limpio,
                    'prediccion': resultado
                })
                
            except Exception as e:
                errores.append({
                    'index': idx,
                    'error': str(e)
                })
        
        # Construir respuesta
        respuesta = {
            'timestamp': datetime.now().isoformat(),
            'total_procesados': len(batch_data),
            'exitosos': len(resultados),
            'errores': len(errores),
            'resultados': resultados,
            'errores_detalle': errores
        }
        
        return jsonify(respuesta), 200
        
    except Exception as e:
        logger.error(f"Error en /predict_batch: {str(e)}")
        return jsonify({
            'error': 'Error interno',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def _generar_interpretacion(resultado):
    """Genera una interpretación legible de los resultados"""
    interpretacion = {
        'resumen': '',
        'reglas_principales': [],
        'recomendaciones': []
    }
    
    # Resumen
    total_reglas = resultado['total_reglas']
    total_combinaciones = resultado['total_combinaciones']
    
    interpretacion['resumen'] = f"Se encontraron {total_reglas} reglas aplicables y {total_combinaciones} combinaciones frecuentes relacionadas con las condiciones ingresadas."
    
    # Reglas principales
    if resultado['reglas_aplicables']:
        for regla in resultado['reglas_aplicables'][:3]:  # Top 3
            interpretacion['reglas_principales'].append({
                'descripcion': f"Si {', '.join(regla['antecedents'])} entonces es probable {', '.join(regla['consequents'])}",
                'confianza': f"{regla['confidence']:.1%}",
                'fuerza': f"{regla['lift']:.2f}x más probable que al azar"
            })
    
    # Recomendaciones
    if resultado['combinaciones_frecuentes']:
        mejor_combinacion = resultado['combinaciones_frecuentes'][0]
        interpretacion['recomendaciones'].append(
            f"La combinación más frecuente relacionada incluye: {', '.join(mejor_combinacion['itemset'])}"
        )
    
    return interpretacion

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Iniciando API en puerto {port}")
    logger.info(f"📊 Modelo disponible: {'✅' if modelo_global else '❌'}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)