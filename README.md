# Sistema RAG aplicado a documentacion publica fiscal

Repositorio tecnico del Trabajo Fin de Grado. El codigo se organiza en dos versiones para que pueda verse la evolucion del sistema desde el prototipo inicial hasta el RAG final.

## Estructura del repositorio

- `v1/`: prototipo base. Incluye la primera ingesta sobre el informe anual de la AEAT, recuperacion inicial, interfaz de pruebas, artefactos intermedios y visualizacion de embeddings.
- `v2/`: sistema final. Incorpora las mejoras de ingesta, chunking padre-hijo, embeddings actualizados, recuperacion hibrida fiscal, soporte para normativa BOE, soporte para ficheros XLSX presupuestarios, respuestas controladas y tests ampliados.

## Evolucion tecnica

La carpeta `v1` conserva el punto de partida usado para evaluar las limitaciones iniciales: corpus reducido, pipeline mas simple y recuperacion menos especializada.

La carpeta `v2` contiene el resultado final del desarrollo:

- ingesta de PDFs y XLSX;
- persistencia local en JSONL;
- recuperacion densa, BM25 y fusion hibrida fiscal;
- tratamiento especifico de consultas juridicas;
- extraccion determinista de celdas en tablas presupuestarias;
- interfaz Streamlit;
- pruebas unitarias de ingesta, recuperacion y generacion.

## Ejecutar el sistema final

Desde la raiz del repositorio:

```powershell
cd .\v2
streamlit run .\interfaz\app.py
```

Para ejecutar los tests del sistema final:

```powershell
cd .\v2
python -m unittest discover -s tests
```

## Credenciales

No se versiona ninguna clave privada. Para ejecutar llamadas reales al servidor Llamus, define `LLAMUS_API_KEY` o crea un fichero local `.llamus_api_key` a partir de `v2/.llamus_api_key.example`.
