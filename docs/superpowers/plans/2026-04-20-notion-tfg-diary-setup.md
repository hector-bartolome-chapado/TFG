# Notion TFG Diary Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Montar en Notion la estructura inicial del diario del TFG con una página contenedora, una base de datos `Diario TFG`, vistas útiles y una primera entrada de ejemplo.

**Architecture:** La implementación creará primero una página contenedora del TFG, después una base de datos hija con el esquema mínimo acordado, y por último ajustará vistas y contenido inicial. La compartición con el profesor queda como paso manual final desde la UI de Notion porque el MCP no expone permisos por email.

**Tech Stack:** Notion MCP (`notion_create_pages`, `notion_create_database`, `notion_fetch`, `notion_create_view`, `notion_update_view`, `notion_update_page`)

---

### Task 1: Crear la página contenedora del TFG

**Files:**
- Create: `Notion page: TFG`
- Verify: `Notion fetch output for parent page`

- [ ] **Step 1: Crear una página padre del TFG**

Crear una página titulada `TFG` con una introducción breve, objetivo del espacio y nota sobre seguimiento por parte del profesor.

- [ ] **Step 2: Verificar que la página existe y recuperar su `page_id`**

Usar `notion_fetch` sobre la nueva página para confirmar el contenido y reutilizar su `page_id` como padre del diario.

### Task 2: Crear la base de datos `Diario TFG`

**Files:**
- Create: `Notion database: Diario TFG`
- Verify: `Notion fetch output for database/data source`

- [ ] **Step 1: Crear la base de datos bajo la página `TFG`**

Crear una base de datos `Diario TFG` con este esquema:

```sql
CREATE TABLE (
  "Nombre" TITLE,
  "Fecha" DATE,
  "Estado" STATUS,
  "Bloque del TFG" SELECT(
    'Planteamiento':gray,
    'Corpus y fuentes':blue,
    'Baseline RAG':green,
    'Retrieval avanzado':purple,
    'Evaluacion':orange,
    'Servidor/Ollama':red,
    'Memoria':yellow,
    'Reunion/tutoria':pink,
    'Administrativo':brown
  ),
  "Objetivo de la sesion" RICH_TEXT,
  "Trabajo realizado" RICH_TEXT,
  "Problemas encontrados" RICH_TEXT,
  "Decisiones tomadas" RICH_TEXT,
  "Siguiente paso" RICH_TEXT,
  "Aporte a memoria" RICH_TEXT
)
```

- [ ] **Step 2: Verificar esquema y recuperar `data_source_id`**

Usar `notion_fetch` sobre la base creada y confirmar que el `data_source_id` y las propiedades coinciden con el diseño aprobado.

### Task 3: Crear vistas útiles del diario

**Files:**
- Modify: `Notion database views for Diario TFG`

- [ ] **Step 1: Crear la vista `Hoy y recientes`**

Tipo `table`, mostrando columnas principales y ordenando por `Fecha` descendente.

- [ ] **Step 2: Crear la vista `Por bloque`**

Tipo `board` o `table` agrupada por `Bloque del TFG`, según permita mejor lectura académica.

- [ ] **Step 3: Crear la vista `Cronologia`**

Tipo `table` ordenada por `Fecha` ascendente.

- [ ] **Step 4: Crear la vista `Bloqueos y problemas`**

Tipo `table` filtrada por `Estado = Bloqueado`.

### Task 4: Añadir una primera entrada de ejemplo

**Files:**
- Create: `Notion database row: primera entrada del Diario TFG`

- [ ] **Step 1: Crear una entrada inicial**

Crear una sesión ejemplo con fecha de hoy, bloque `Planteamiento` o `Servidor/Ollama`, y contenido breve que refleje el arranque del diario.

- [ ] **Step 2: Comprobar renderizado y legibilidad**

Validar que las propiedades y el contenido dentro de la página de la entrada tienen buen formato para uso diario y para memoria.

### Task 5: Dejar lista la operativa final

**Files:**
- Modify: `Notion page: TFG`

- [ ] **Step 1: Añadir una nota operativa sobre uso**

Incluir en la página `TFG` una nota corta explicando que el diario lo iré actualizando contigo sesión a sesión.

- [ ] **Step 2: Añadir instrucción manual para compartir con el profesor**

Dejar una línea visible recordando que debes invitar al profesor por email con permiso de comentario desde la interfaz de Notion.
