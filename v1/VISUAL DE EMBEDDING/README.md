# Visual de Embedding

La visualización de embeddings del proyecto se hace ahora con **TensorBoard Embedding Projector**.

## Flujo

1. Generas embeddings con el extractor simple.
2. Ejecutas el exportador de esta carpeta.
3. Abres TensorBoard apuntando al `logdir` generado.
4. En TensorBoard, entras en la pestaña **Projector**.

## Exportar

Desde la raíz de `TFG`:

```powershell
python ".\\VISUAL DE EMBEDDING\\export_to_projector.py" --embeddings ".\\RESULTADOS EMBEDDING\\embeddings\\AEAT_informe_anual_2024.jsonl"
```

Esto crea artefactos en:

```text
RESULTADOS EMBEDDING/projector/AEAT_informe_anual_2024/
```

## Abrir TensorBoard

```powershell
tensorboard --logdir ".\\RESULTADOS EMBEDDING\\projector"
```

Después abre en el navegador:

```text
http://localhost:6006/
```

Y entra en:

```text
Projector
```

## Qué se genera

- `metadata.tsv`
- `projector_config.pbtxt`
- `embedding.ckpt-*`

## Idea clave

El exportador no vuelve a calcular embeddings. Solo transforma tu `.jsonl` al formato que TensorBoard Projector sabe leer.
