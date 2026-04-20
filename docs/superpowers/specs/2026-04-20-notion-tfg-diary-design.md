# Diseno del Diario TFG en Notion

## Objetivo

Crear un espacio en Notion que funcione como diario academico-operativo del TFG y deje evidencia clara de la progresion diaria del trabajo. El sistema debe servir a tres usos simultaneos:

1. seguimiento cotidiano del trabajo tecnico;
2. supervision del profesor con permiso de comentario;
3. reutilizacion posterior del contenido en la memoria.

## Alcance

La solucion cubre en su fase inicial:

- una pagina contenedora principal del TFG en Notion;
- una base de datos `Diario TFG` con una entrada por sesion;
- vistas orientadas a seguimiento, cronologia, memoria y bloqueos;
- una forma consistente de actualizar el diario con ayuda del asistente;
- preparacion del espacio para compartirlo con el profesor.

La solucion deja prevista para una fase posterior:

- una base de datos `Fuentes TFG` con una entrada por documento o recurso consultado cuando el corpus y las fuentes reales esten disponibles.

La solucion no cubre:

- automatizacion completa de comparticion por email si Notion no expone esa operacion via MCP;
- generacion automatica de entradas sin validacion minima del usuario;
- un sistema complejo de tareas, hitos o Kanban independiente del diario.

## Principios de diseno

- Una sesion de trabajo equivale a una entrada del diario.
- Cada entrada debe explicar que se hizo, por que, con que evidencias y cual es el siguiente paso.
- Las fuentes consultadas deben quedar trazables y enlazadas con las sesiones donde se usaron.
- La estructura debe ser ligera de mantener para que no compita con el propio TFG.
- El profesor debe poder seguir el progreso sin riesgo de editar contenido.

## Arquitectura de informacion

### 1. Pagina contenedora

Se creara una pagina principal llamada `TFG` o equivalente, que actuara como punto de entrada para el profesor y para el propio alumno. Esta pagina incluira:

- una descripcion breve del proyecto;
- una seccion de estado actual;
- una vista incrustada de la base `Diario TFG`;
- una vista incrustada de la base `Fuentes TFG`;
- espacio opcional para enlaces futuros a memoria, datasets, reuniones o entregables.

### 2. Base de datos `Diario TFG`

Habra una fila por sesion real de trabajo.

Propiedades previstas:

- `Nombre`: titulo de la sesion.
- `Fecha`: fecha de trabajo.
- `Estado`: `Planificado`, `En curso`, `Hecho`, `Bloqueado`.
- `Bloque del TFG`: `Planteamiento`, `Corpus y fuentes`, `Baseline RAG`, `Retrieval avanzado`, `Evaluacion`, `Servidor/Ollama`, `Memoria`, `Reunion/tutoria`, `Administrativo`.
- `Objetivo de la sesion`: texto corto.
- `Trabajo realizado`: texto largo.
- `Problemas encontrados`: texto largo.
- `Decisiones tomadas`: texto largo.
- `Siguiente paso`: texto largo.
- `Aporte a memoria`: texto largo.
- `Fuentes relacionadas`: relacion con `Fuentes TFG`.

Contenido recomendado dentro de cada pagina de sesion:

- objetivo de la sesion;
- trabajo realizado;
- fuentes consultadas;
- problemas encontrados;
- decisiones tomadas;
- resultado de la sesion;
- siguiente paso;
- notas para la memoria.

### 3. Base de datos `Fuentes TFG` (fase posterior)

Habra una fila por documento o recurso consultado.

Propiedades previstas:

- `Nombre`: titulo de la fuente.
- `Tipo de fuente`: `PDF`, `paper`, `web oficial`, `benchmark`, `documentacion`, `dataset`.
- `Origen`: organismo, autor o entidad.
- `URL`: enlace principal.
- `Bloque del TFG`: categoria principal de uso.
- `Descripcion breve`: resumen de utilidad.
- `Usada en memoria`: checkbox.
- `Sesiones relacionadas`: relacion con `Diario TFG`.

Esta base se activara cuando existan fuentes reales suficientemente estables. Permitira responder despues a preguntas como que documentos sustentan una decision, que fuentes se usaron para un bloque concreto o que referencias acabaron siendo utiles en la memoria.

## Vistas

### Vistas para `Diario TFG`

- `Hoy y recientes`: orden descendente por fecha.
- `Por bloque`: agrupada por `Bloque del TFG`.
- `Cronologia`: orden ascendente por fecha.
- `Para la memoria`: filtrada para entradas con contenido util en `Aporte a memoria`.
- `Bloqueos y problemas`: filtrada por `Estado = Bloqueado` o sesiones con problemas relevantes.

### Vistas para `Fuentes TFG` (fase posterior)

- `Todas las fuentes`: vista general.
- `PDFs`: filtrada por `Tipo de fuente = PDF`.
- `Oficiales`: filtrada por organismos o webs institucionales.
- `Usadas en memoria`: filtrada por `Usada en memoria = true`.

## Flujo operativo

### Flujo diario

1. El alumno trabaja en una sesion del TFG.
2. El asistente redacta o actualiza la entrada correspondiente en `Diario TFG`.
3. Mientras no exista la base de fuentes, las referencias relevantes se anotan de forma ligera dentro de la propia sesion.
4. Cuando la base `Fuentes TFG` exista, se migraran o enlazaran esas referencias de forma estructurada.
5. La entrada queda lista para revision del profesor.

### Flujo de tutorias

1. El profesor entra en la pagina contenedora del TFG.
2. Revisa las vistas de progreso y los comentarios sobre bloqueos o decisiones.
3. Puede dejar comentarios sin modificar directamente la estructura ni el contenido.

## Comparticion con el profesor

Objetivo de permisos:

- profesor con acceso por invitacion por email;
- permiso de `comentario`;
- sin permiso de edicion.

Limitacion tecnica:

- el MCP de Notion disponible en esta sesion permite buscar, crear y actualizar contenido, pero no expone una operacion directa de compartir por email o gestionar permisos de workspace.

Decision:

- la estructura del espacio se creara desde el asistente;
- la invitacion al profesor se hara manualmente desde la interfaz de Notion una vez creada la pagina contenedora o la base compartida.

## Criterio editorial del asistente

El asistente actualizara las entradas:

- en primera persona;
- con tono tecnico-academico;
- breve pero util para memoria y supervision;
- sin convertir cada entrada en un texto largo o artificial.

El objetivo no es fingir una autoria opaca, sino ayudarte a mantener un registro coherente y de calidad que tu puedas revisar y usar.

## Riesgos y mitigaciones

### Riesgo 1: exceso de mantenimiento

Si la estructura exige demasiados campos, el diario dejara de actualizarse.

Mitigacion:

- mantener solo dos bases de datos;
- limitar estados y categorias;
- reutilizar plantillas de entrada.

### Riesgo 2: fuentes duplicadas

El mismo PDF o paper podria aparecer varias veces con titulos ligeramente distintos.

Mitigacion:

- comprobar antes si la fuente ya existe;
- actualizar el registro existente y relacionarlo con nuevas sesiones.

### Riesgo 3: el profesor no ve contexto

Si se comparte solo una tabla, la experiencia sera pobre.

Mitigacion:

- compartir una pagina contenedora con introduccion y vistas embebidas.

### Riesgo 4: dependencia de capacidades MCP

La comparticion por email puede no ser automatizable desde el conector.

Mitigacion:

- dejar el espacio listo y hacer el paso final de invitacion manualmente en Notion.

## Plan de implementacion posterior

Una vez aprobado este diseno:

1. crear pagina contenedora del TFG en Notion;
2. crear base `Diario TFG`;
3. ajustar propiedades y vistas;
4. dejar una primera entrada de ejemplo;
5. indicar el paso manual para invitar al profesor por email con permiso de comentario.

Fase posterior, cuando existan fuentes reales:

1. crear base `Fuentes TFG`;
2. anadir relaciones entre `Diario TFG` y `Fuentes TFG`;
3. migrar o normalizar referencias ya registradas dentro de las sesiones.

## Criterio de exito

La solucion se considerara correcta si:

- existe una pagina principal clara para el TFG;
- el diario permite registrar sesiones reales con valor para la memoria;
- las fuentes PDF y demas referencias tienen un camino claro de incorporacion posterior sin bloquear el arranque del diario;
- el profesor puede revisar la evolucion del trabajo mediante comentarios;
- el mantenimiento diario resulta suficientemente ligero como para sostenerse durante el TFG.
