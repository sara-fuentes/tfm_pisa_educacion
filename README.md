# Proyecto PISA con MLflow

Este proyecto utiliza MLflow para el seguimiento y gestión de experimentos de machine learning en el análisis de datos PISA.

## Requisitos Previos

1. Python 3.8 o superior
2. pip (gestor de paquetes de Python)

## Instalación

1. Clonar el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd tfm_pisa_educacion
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Configuración de MLflow

1. Iniciar el servidor MLflow:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

2. Acceder a la interfaz web de MLflow:
- Abrir un navegador y visitar: http://localhost:5000

## Estructura del Proyecto

- `mlflow_example.py`: Script de ejemplo que muestra cómo usar MLflow
- `mlruns/`: Directorio donde MLflow almacena los experimentos
- `data/`: Directorio con los datos del proyecto
- `*.ipynb`: Notebooks de Jupyter con análisis y modelos

## Uso de MLflow

### 1. Seguimiento de Experimentos

Para registrar un experimento:

```python
import mlflow

# Configurar el experimento
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Nombre_Experimento")

# Iniciar un run
with mlflow.start_run():
    # Registrar parámetros
    mlflow.log_params({"param1": value1, "param2": value2})
    
    # Entrenar modelo y obtener métricas
    # ...
    
    # Registrar métricas
    mlflow.log_metrics({"metric1": value1, "metric2": value2})
    
    # Guardar modelo
    mlflow.sklearn.log_model(model, "model")
```

### 2. Visualización de Resultados

1. Acceder a la interfaz web de MLflow (http://localhost:5000)
2. Seleccionar el experimento deseado
3. Explorar las diferentes ejecuciones y sus métricas
4. Comparar diferentes modelos y sus parámetros

### 3. Gestión de Modelos

- Los modelos se guardan automáticamente en el directorio `mlruns/`
- Puedes cargar modelos guardados usando:
```python
loaded_model = mlflow.sklearn.load_model("runs:/<run_id>/model")
```

## Mejores Prácticas

1. Siempre registrar:
   - Parámetros del modelo
   - Métricas de evaluación
   - Artefactos importantes (gráficos, datos de validación)
   - El modelo entrenado

2. Usar nombres descriptivos para los experimentos y runs

3. Documentar los cambios importantes en los parámetros

4. Mantener un registro de las mejores ejecuciones

## Recursos Adicionales

- [Documentación oficial de MLflow](https://mlflow.org/docs/latest/index.html)
- [Tutorial de MLflow](https://mlflow.org/docs/latest/tutorials-and-examples/index.html) 