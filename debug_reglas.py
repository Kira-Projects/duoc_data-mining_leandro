#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para debuggear las reglas y itemsets del modelo
"""

import pickle
import sys
import pandas as pd

def debug_reglas():
    """Debuggear las reglas y itemsets"""
    try:
        print("🔍 Debuggeando reglas y itemsets...")
        
        # Crear clase compatible
        class ModeloAPRIORI:
            pass
        
        # Agregar al módulo actual
        sys.modules[__name__].ModeloAPRIORI = ModeloAPRIORI
        
        # Cargar el modelo
        with open('./modelo_entrenado.pkl', 'rb') as f:
            modelo_obj = pickle.load(f)
        
        # Mapear como lo hace la aplicación
        modelo = {
            'variables': modelo_obj.variables_clima,
            'discretization_maps': modelo_obj.rangos,
            'frequent_itemsets': modelo_obj.frequent_itemsets,
            'rules': modelo_obj.rules
        }
        
        print("=== INFORMACIÓN GENERAL ===")
        print(f"Total de reglas: {len(modelo['rules'])}")
        print(f"Total de itemsets frecuentes: {len(modelo['frequent_itemsets'])}")
        
        print("\n=== MUESTRA DE REGLAS ===")
        if len(modelo['rules']) > 0:
            print("Primeras 5 reglas:")
            for i in range(min(5, len(modelo['rules']))):
                regla = modelo['rules'].iloc[i]
                print(f"Regla {i+1}:")
                print(f"  Antecedentes: {regla['antecedents']} (tipo: {type(regla['antecedents'])})")
                print(f"  Consecuentes: {regla['consequents']} (tipo: {type(regla['consequents'])})")
                print(f"  Confianza: {regla['confidence']:.3f}")
                print(f"  Soporte: {regla['support']:.3f}")
                print()
        
        print("\n=== MUESTRA DE ITEMSETS FRECUENTES ===")
        if len(modelo['frequent_itemsets']) > 0:
            print("Primeros 5 itemsets:")
            for i in range(min(5, len(modelo['frequent_itemsets']))):
                itemset = modelo['frequent_itemsets'].iloc[i]
                print(f"Itemset {i+1}:")
                print(f"  Items: {itemset['itemsets']} (tipo: {type(itemset['itemsets'])})")
                print(f"  Soporte: {itemset['support']:.3f}")
                print()
        
        print("\n=== PRUEBA CON DATOS DE ENTRADA ===")
        # Probar con los datos que no funcionaron
        datos_prueba = {
            "MaxTemp": 22.9,
            "Humidity3pm": 22.0,
            "WindGustSpeed": 44.0,
            "Sunshine": 7.04
        }
        
        # Discretizar manualmente
        datos_discretos = []
        for variable, valor in datos_prueba.items():
            if variable in modelo['variables']:
                rangos = modelo['discretization_maps'][variable]
                for categoria, rango in rangos.items():
                    min_val, max_val = rango
                    if min_val <= valor <= max_val:
                        datos_discretos.append(f"{variable}_Cat_{categoria}")
                        break
        
        print(f"Datos discretizados: {datos_discretos}")
        
        # Buscar reglas que coincidan
        reglas_coincidentes = []
        for _, regla in modelo['rules'].iterrows():
            antecedentes = set(regla['antecedents'])
            if antecedentes.issubset(set(datos_discretos)):
                reglas_coincidentes.append(regla)
        
        print(f"Reglas coincidentes encontradas: {len(reglas_coincidentes)}")
        
        # Buscar itemsets que coincidan
        itemsets_coincidentes = []
        for _, item in modelo['frequent_itemsets'].iterrows():
            itemset = set(item['itemsets'])
            entrada_set = set(datos_discretos)
            if entrada_set.intersection(itemset):
                itemsets_coincidentes.append(item)
        
        print(f"Itemsets coincidentes encontrados: {len(itemsets_coincidentes)}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_reglas() 