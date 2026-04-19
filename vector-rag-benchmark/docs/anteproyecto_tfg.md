# Anteproyecto de TFG

## Título provisional

**Benchmark reproducible de estrategias de recuperación documental para sistemas RAG en entornos de recursos moderados**

## Resumen

El presente Trabajo Fin de Grado propone estudiar el papel de las técnicas de recuperación documental en sistemas basados en embeddings y arquitecturas Retrieval-Augmented Generation (RAG), con un enfoque específicamente orientado a la evaluación experimental y reproducible. El objetivo principal es comparar distintas estrategias de recuperación sobre un corpus de documentación técnica, evaluando su rendimiento en términos de precisión, latencia, coste computacional y viabilidad práctica en un entorno de recursos moderados.

El trabajo tendrá una orientación aplicada y experimental. Se plantea construir un prototipo funcional capaz de indexar documentación técnica, recuperar fragmentos relevantes ante una consulta del usuario y generar respuestas apoyadas en dichas evidencias. No obstante, la contribución principal no será el chatbot en sí, sino el diseño de un benchmark comparativo que permita estudiar con criterios cuantitativos tres enfoques de recuperación: búsqueda léxico-estadística, búsqueda vectorial y búsqueda híbrida. Además, se analizarán motores ligeros orientados a hardware no especializado, como FAISS y HNSWlib, dejando abierta la posibilidad de incorporar alguna alternativa adicional si el desarrollo y los recursos disponibles lo permiten.

## Justificación del tema

La aparición de modelos de lenguaje de gran tamaño y el uso extensivo de embeddings han convertido la recuperación vectorial en un componente clave de numerosos sistemas actuales de IA. Sin embargo, en muchos casos se da por hecho que la recuperación vectorial es siempre superior a otras técnicas, cuando en realidad su comportamiento depende del dominio, del tipo de consulta, del modelo de embedding utilizado y del motor de indexación elegido.

Este TFG pretende abordar esa cuestión desde una perspectiva práctica y académica. Por una parte, permite estudiar conceptos relevantes de recuperación de información, estructuras de indexación y búsqueda aproximada de vecinos más cercanos. Por otra, ofrece un caso de uso actual y claramente justificable: la consulta asistida sobre documentación técnica. La diferenciación del trabajo no residirá en desarrollar un asistente conversacional genérico, sino en construir un marco de evaluación reproducible que permita responder con datos a una pregunta concreta: cuándo compensa emplear recuperación vectorial, cuándo resulta preferible una estrategia híbrida y qué coste tiene cada decisión en un entorno ligero.

## Problema de investigación

La pregunta principal que guiará el trabajo será la siguiente:

**¿Qué estrategia de recuperación ofrece el mejor equilibrio entre precisión, latencia, coste y utilidad final en un sistema RAG sobre documentación técnica ejecutable en un entorno de recursos moderados?**

A partir de esta pregunta general se derivan las siguientes subcuestiones:

1. ¿En qué tipos de consulta mejora la búsqueda vectorial frente a una búsqueda léxico-estadística tradicional?
2. ¿Cómo cambia el rendimiento al emplear recuperación híbrida en lugar de recuperación puramente vectorial o puramente léxica?
3. ¿Qué diferencias prácticas presentan distintos motores ligeros de indexación vectorial en un entorno de recursos moderados?
4. ¿Hasta qué punto la calidad de la recuperación condiciona la calidad final de las respuestas generadas en un sistema RAG?

## Objetivo general

Analizar y comparar distintas estrategias de recuperación de información sobre documentación técnica, implementando un benchmark reproducible y un prototipo funcional que permitan medir su comportamiento y valorar su aplicabilidad real en sistemas RAG ejecutables sin infraestructura pesada.

## Objetivos específicos

1. Estudiar los fundamentos teóricos de los embeddings, la recuperación vectorial y los algoritmos de búsqueda aproximada de vecinos más cercanos.
2. Seleccionar un corpus de documentación técnica adecuado para una evaluación reproducible.
3. Construir un pipeline de ingesta, fragmentación e indexación del corpus.
4. Implementar una línea base de búsqueda léxico-estadística.
5. Implementar al menos dos alternativas de recuperación vectorial ligera.
6. Implementar una estrategia de recuperación híbrida que combine evidencia léxica y vectorial.
7. Definir un protocolo de evaluación reproducible, con consultas, criterios de relevancia y condiciones de ejecución claramente documentadas.
8. Desarrollar un prototipo web sencillo que permita consultar el sistema y visualizar resultados y fuentes.
9. Evaluar el sistema mediante métricas de recuperación, tiempos de respuesta y coste computacional en hardware modesto.
10. Analizar las ventajas, limitaciones y escenarios de uso de cada enfoque.

## Alcance del trabajo

El alcance previsto del TFG es moderado y realista para un desarrollo individual. El trabajo se centrará en un corpus de documentación técnica bien estructurada y de tamaño contenido, suficiente para realizar comparaciones representativas sin requerir infraestructura de alto rendimiento. La evaluación se planteará a escala académica y de prototipo, no a escala industrial.

No forma parte del alcance inicial entrenar modelos propios ni ejecutar grandes modelos de lenguaje en local. Para reducir la dependencia del hardware, se contempla un enfoque mixto: procesamiento local para la mayor parte del flujo y uso puntual de servicios en la nube o API para tareas de embeddings o generación de respuestas, si fuera necesario.

Tampoco forma parte del objetivo principal desarrollar un asistente conversacional complejo desde el punto de vista de producto. El prototipo se utilizará como demostrador del análisis experimental, de modo que el peso académico del trabajo recaerá sobre la comparativa, la metodología de evaluación y la interpretación de resultados.

## Metodología

La metodología del trabajo se organizará en cuatro bloques:

### 1. Revisión teórica

Se realizará una revisión del estado del arte sobre:

- embeddings y representación semántica de texto;
- bases de datos vectoriales y estructuras de indexación;
- algoritmos ANN, prestando especial atención a hiperparámetros clave que equilibran latencia y precisión, por ejemplo `M` y `efSearch` en HNSW, o `nlist` y `nprobe` en FAISS;
- sistemas RAG y su dependencia de la etapa de recuperación;
- frameworks de evaluación automatizada basados en LLMs-as-a-judge, por ejemplo RAGAS, para la medición de métricas como *Faithfulness* y *Context Precision*.

### 2. Diseño e implementación del prototipo

Se construirá un prototipo compuesto por:

- un backend en Python para la ingesta, fragmentación, indexación y evaluación;
- varios módulos de recuperación para comparar enfoques léxicos, vectoriales e híbridos;
- una interfaz web sencilla en JavaScript para realizar consultas y visualizar resultados y respuestas con citas.

### 3. Evaluación experimental

La comparativa se apoyará en una batería de consultas definidas sobre el corpus, junto con criterios de relevancia previamente establecidos y un entorno de ejecución documentado para facilitar la repetibilidad. Se utilizarán, al menos, las siguientes métricas:

- Recall@k;
- MRR o nDCG;
- tiempo de indexación;
- tiempo medio de consulta;
- uso aproximado de memoria y coste computacional asociado a cada estrategia;
- observaciones cualitativas sobre la calidad de las respuestas generadas.

### 4. Análisis y discusión

Finalmente, se discutirán los resultados obtenidos, identificando en qué casos cada estrategia es más adecuada y qué compromisos existen entre precisión, coste computacional y complejidad de implementación.

## Tecnologías previstas

- **Python** para el backend y la evaluación experimental.
- **JavaScript** para la interfaz web del prototipo.
- **Embeddings y LLM** mediante API, si resultan necesarios para la parte RAG.
- Motores vectoriales ligeros como **FAISS** y **HNSWlib**, junto con una línea base de recuperación tradicional y un enfoque híbrido.

## Viabilidad

El trabajo se considera viable dentro de un plazo aproximado de tres a cuatro meses, siempre que se mantenga un alcance contenido y una selección razonable de motores y corpus. El proyecto está planteado deliberadamente para un entorno de hardware modesto, evitando dependencias de GPU local y apoyándose en servicios externos solo cuando aporten valor claro.

Como medida de control del alcance, la versión base del trabajo se apoyará en:

- una línea base léxica;
- dos motores o estrategias vectoriales ligeras;
- una estrategia híbrida;
- un prototipo web funcional, pero contenido.

Si el desarrollo avanza favorablemente, podrán ampliarse algunos apartados, por ejemplo comparando más motores, evaluando distintos embeddings o refinando el sistema RAG. Si surgieran limitaciones de tiempo o recursos, se priorizará la parte comparativa y experimental frente a funcionalidades accesorias de la interfaz. Esta decisión metodológica refuerza además la diferenciación del TFG, al situar el valor principal en el benchmark y no en la complejidad del demostrador.

## Plan de trabajo preliminar

1. Delimitación final del tema, del corpus y de la pregunta de investigación.
2. Revisión bibliográfica inicial y definición del marco teórico.
3. Construcción del pipeline de ingesta y fragmentación.
4. Implementación de la línea base léxica.
5. Implementación de la recuperación vectorial y de la recuperación híbrida.
6. Desarrollo del prototipo web de consulta.
7. Diseño y ejecución de la evaluación experimental.
8. Análisis de resultados.
9. Redacción de la memoria y preparación de la defensa.

## Resultados esperados

Se espera obtener:

- una comparativa clara entre estrategias de recuperación;
- un protocolo de evaluación reutilizable para comparar enfoques de recuperación en futuros trabajos;
- un prototipo funcional que sirva como demostrador del trabajo, pero subordinado al análisis experimental;
- un análisis razonado sobre la utilidad real de las bases vectoriales en un caso concreto de recuperación documental;
- conclusiones sobre qué soluciones ofrecen mejor equilibrio entre calidad y coste en un entorno de recursos moderados.

## Valor y aplicabilidad en el ámbito empresarial

Más allá de su interés académico, este proyecto tiene una aplicación directa en contextos empresariales donde existe una gran cantidad de documentación interna o técnica que debe consultarse con rapidez y fiabilidad. Muchas organizaciones están incorporando asistentes basados en IA, pero con frecuencia lo hacen sin haber evaluado previamente qué estrategia de recuperación documental resulta más adecuada para su caso concreto. En este sentido, el valor principal del proyecto no sería únicamente el prototipo resultante, sino el marco comparativo que permite tomar decisiones técnicas con criterio antes de invertir en una solución de producción.

El trabajo puede resultar útil, por ejemplo, en empresas que gestionan manuales técnicos, documentación de producto, bases de conocimiento internas, procedimientos operativos, catálogos normativos o repositorios de soporte. En estos escenarios, una comparación entre recuperación léxica, vectorial e híbrida permite identificar qué enfoque ofrece mejor equilibrio entre precisión, latencia, complejidad y coste computacional. Esto es especialmente relevante para organizaciones pequeñas o medianas, o para departamentos concretos dentro de una empresa, que no disponen de infraestructura pesada ni de presupuesto elevado para desplegar soluciones avanzadas de IA.

Además, el proyecto puede servir como base para futuros desarrollos en áreas como buscadores inteligentes de documentación, asistentes internos de soporte técnico, sistemas de ayuda al empleado, herramientas de consulta normativa o asistentes de atención al cliente basados en fuentes verificables. Desde una perspectiva empresarial, el TFG aporta valor porque reduce incertidumbre técnica, favorece decisiones de implantación más informadas y ofrece una metodología reutilizable para evaluar tecnologías de recuperación documental antes de integrarlas en productos o procesos reales.

## Bibliografía inicial orientativa

[1] P. Lewis *et al.*, "Retrieval-augmented generation for knowledge-intensive NLP tasks," en *Advances in Neural Information Processing Systems*, vol. 33, pp. 9459-9474, 2020.  
[2] J. Johnson, M. Douze y H. Jégou, "Billion-scale similarity search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535-547, 2019.  
[3] Y. A. Malkov y D. A. Yashunin, "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 42, no. 4, pp. 824-836, 2020.  
[4] S. Robertson y H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333-389, 2009.  
[5] S. Es, J. James, L. Espinosa-Anke y S. Schockaert, "RAGAS: Automated Evaluation of Retrieval Augmented Generation," *arXiv preprint arXiv:2309.15217*, 2023.

## Observaciones finales

La propuesta busca un equilibrio entre actualidad tecnológica, valor académico y viabilidad práctica. El interés principal del TFG no será solamente construir una aplicación, sino analizar con criterio qué ventajas y limitaciones presentan distintas estrategias de recuperación cuando se aplican a un caso realista, medible y ejecutable en condiciones de hardware accesibles. Esa orientación comparativa, reproducible y centrada en entornos ligeros constituye la principal seña de identidad del trabajo.
