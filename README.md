He dividido la solución principalmente en tres componentes:
•	pipeline.py: contiene la lógica principal de procesamiento y análisis.
•	main.py: expone los resultados mediante una API REST utilizando FastAPI.
•	predictive_model.py: contiene el modelo predictivo basado en Random Forest.
Instalaciones:
pip install pandas numpy
pip install fastapi uvicorn
Requisitos
Las principales dependencias son:
•	pandas
•	numpy
•	fastapi
•	uvicorn
•	scikit-learn
Estructura del proyecto
La estructura que utilizo es:
project/
│
├── pipeline.py
├── main.py
├── predictive_model.py
├── responses.csv
├── item_config.csv
├── predictive_challenge.csv
Ejecución del pipeline
Para ejecutar el procesamiento completo utilizo: python pipeline.py
Mi pipeline realiza las siguientes etapas:
1.	Cargo los CSV.
2.	Elimino respuestas duplicadas.
3.	Valido las respuestas.
4.	Aplico reverse scoring cuando corresponde.
5.	Transformo las respuestas de escala 1–5 a una escala 0–100.
6.	Calculo las Signals utilizando los pesos definidos en item_config.csv.
7.	Calculo los Moments.
8.	Genero el análisis de las diferentes cohortes.
Ejecución de la API
Para levantar y ejecutar la API en un entorno local de desarrollo se utiliza el servidor ASGI Uvicorn mediante el comando uvicorn api:app --reload. Una vez iniciado el servicio, la aplicación queda disponible en http://127.0.0.1:8000, ofreciendo además una interfaz de documentación interactiva en http://127.0.0.1:8000/docs.
La solución expone cuatro endpoints principales diseñados para consultar la información procesada por el pipeline. El endpoint http://127.0.0.1:8000/signals retorna el desglose individual de todas las señales calculadas para cada participante, mientras que http://127.0.0.1:8000/moments entrega los resultados correspondientes a los momentos de carrera y vida. Por su parte, http://127.0.0.1:8000/cohorts consolida el análisis global de inteligencia de cohortes sobre el total de señales. Finalmente, la ruta http://127.0.0.1:8000/cohorts/role/manager permite filtrar los datos de forma granular seleccionando una dimensión concreta junto con su valor específico.
Ejecución del modelo predictivo
Para ejecutar el modelo predictivo utilizo:python predictive_model.py
Como modelo principal he utilizado RandomForestClassifier 
Para evaluar el modelo utilizo:
•	Classification report: precision, recall, F1-score y accuracy.
•	ROC-AUC: mide la capacidad del modelo para distinguir entre las dos clases. Se obtiene un valor de 0.762.
•	Importancia de las variables: permite conocer qué variables tienen mayor peso en las predicciones.
Riesgo de overfitting
Existe riesgo de overfitting, especialmente porque el dataset no es muy grande. Para reducirlo utilizo una separación 80/20 entre entrenamiento y prueba, validación cruzada de 5 folds y limito la complejidad de los árboles mediante max_depth, min_samples_split y min_samples_leaf.
Limitaciones del dataset
El dataset tiene un número reducido de observaciones, por lo que las métricas pueden variar dependiendo de la división de los datos. Además, existe cierto desequilibrio entre las clases, por lo que no es suficiente analizar únicamente la accuracy.
Por último, el modelo identifica relaciones predictivas, no causales. Es decir, que una variable tenga mucha importancia no significa que sea la causa directa del cambio en la visibilidad futura.
Tests
Compruebo:
•	Que las respuestas fuera del rango válido se traten correctamente.
•	Que se eliminen duplicados.
•	Que la transformación 1–5 a 0–100 sea correcta.
•	Que el reverse scoring funcione correctamente.
•	Que se respeten los pesos de las Signals.
•	Que las fórmulas de Moments sean correctas.
•	Que las cohortes con menos de 10 respuestas no sean reportables.
•	Que las cohortes con al menos 10 respuestas sí sean reportables.
