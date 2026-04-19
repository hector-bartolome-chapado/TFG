const fs = require("fs");
const path = require("path");
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  HeadingLevel,
  Packer,
  PageBreak,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} = require("docx");

const projectRoot = path.resolve(__dirname, "..");
const markdownPath = path.join(projectRoot, "docs", "anteproyecto_tfg.md");
const outputPath = path.join(projectRoot, "docs", "Anteproyecto_TFG_Vector_RAG.docx");

const raw = fs.readFileSync(markdownPath, "utf8");

const accent = "1F4E79";
const accentSoft = "EAF2F8";
const muted = "5B6770";
const borderGrey = "D9E2EC";
const dark = "22303C";

function extractSection(title) {
  const escaped = title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const nextHeading = String.raw`\n##\s+`;
  const regex = new RegExp(`## ${escaped}\\n\\n([\\s\\S]*?)(?=${nextHeading}|$)`);
  const match = raw.match(regex);
  return match ? match[1].trim() : "";
}

function extractTitle() {
  const match = raw.match(/## Título provisional\s+\n+\*\*(.+?)\*\*/s);
  return match ? match[1].trim() : "Anteproyecto de TFG";
}

function cleanInline(text) {
  return text.replace(/\*\*(.+?)\*\*/g, "$1").trim();
}

function createSectionChildren(markdownBlock) {
  const children = [];
  const lines = markdownBlock.split(/\r?\n/);
  let paragraphBuffer = [];
  let orderedCount = 0;

  const flushParagraph = () => {
    if (!paragraphBuffer.length) {
      return;
    }
    const text = cleanInline(paragraphBuffer.join(" ").replace(/\s+/g, " "));
    if (text) {
      children.push(
        new Paragraph({
          style: "BodyText",
          children: [new TextRun({ text, color: dark })],
        })
      );
    }
    paragraphBuffer = [];
  };

  for (const line of lines) {
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      orderedCount = 0;
      continue;
    }

    if (/^###\s+/.test(trimmed)) {
      flushParagraph();
      children.push(
        new Paragraph({
          heading: HeadingLevel.HEADING_2,
          children: [new TextRun(cleanInline(trimmed.replace(/^###\s+/, "")))],
        })
      );
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      flushParagraph();
      orderedCount += 1;
      children.push(
        new Paragraph({
          style: "BodyList",
          children: [new TextRun({ text: `${orderedCount}. ${cleanInline(trimmed.replace(/^\d+\.\s+/, ""))}`, color: dark })],
        })
      );
      continue;
    }

    if (/^- /.test(trimmed)) {
      flushParagraph();
      children.push(
        new Paragraph({
          style: "BodyList",
          children: [new TextRun({ text: `- ${cleanInline(trimmed.replace(/^- /, ""))}`, color: dark })],
        })
      );
      continue;
    }

    paragraphBuffer.push(trimmed);
  }

  flushParagraph();
  return children;
}

function fullWidthCell(text, options = {}) {
  return new TableCell({
    width: { size: 9360, type: WidthType.DXA },
    margins: { top: 140, bottom: 140, left: 180, right: 180 },
    borders: options.borders || {
      top: { style: BorderStyle.SINGLE, color: borderGrey, size: 1 },
      bottom: { style: BorderStyle.SINGLE, color: borderGrey, size: 1 },
      left: { style: BorderStyle.SINGLE, color: borderGrey, size: 1 },
      right: { style: BorderStyle.SINGLE, color: borderGrey, size: 1 },
    },
    shading: options.shading,
    children: [
      new Paragraph({
        style: options.style || "BodyText",
        alignment: options.alignment || AlignmentType.LEFT,
        children: [new TextRun({ text, bold: !!options.bold, color: options.color || dark })],
      }),
    ],
  });
}

function keyValueRow(label, value) {
  return new TableRow({
    children: [
      new TableCell({
        width: { size: 2500, type: WidthType.DXA },
        margins: { top: 100, bottom: 100, left: 140, right: 140 },
        shading: { fill: accentSoft, type: ShadingType.CLEAR },
        borders: {
          top: { style: BorderStyle.SINGLE, color: borderGrey, size: 1 },
          bottom: { style: BorderStyle.SINGLE, color: borderGrey, size: 1 },
          left: { style: BorderStyle.SINGLE, color: borderGrey, size: 1 },
          right: { style: BorderStyle.SINGLE, color: borderGrey, size: 1 },
        },
        children: [
          new Paragraph({
            style: "TableLabel",
            children: [new TextRun({ text: label, bold: true, color: accent })],
          }),
        ],
      }),
      new TableCell({
        width: { size: 6860, type: WidthType.DXA },
        margins: { top: 100, bottom: 100, left: 160, right: 160 },
        borders: {
          top: { style: BorderStyle.SINGLE, color: borderGrey, size: 1 },
          bottom: { style: BorderStyle.SINGLE, color: borderGrey, size: 1 },
          left: { style: BorderStyle.SINGLE, color: borderGrey, size: 1 },
          right: { style: BorderStyle.SINGLE, color: borderGrey, size: 1 },
        },
        children: [
          new Paragraph({
            style: "BodyText",
            children: [new TextRun({ text: value, color: dark })],
          }),
        ],
      }),
    ],
  });
}

const title = extractTitle();
const resumen = extractSection("Resumen");

const bodySections = [
  "Justificación del tema",
  "Problema de investigación",
  "Objetivo general",
  "Objetivos específicos",
  "Alcance del trabajo",
  "Metodología",
  "Tecnologías previstas",
  "Viabilidad",
  "Plan de trabajo preliminar",
  "Resultados esperados",
  "Valor y aplicabilidad en el ámbito empresarial",
  "Bibliografía inicial orientativa",
  "Observaciones finales",
].flatMap((sectionTitle) => {
  const block = extractSection(sectionTitle);
  return [
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [new TextRun(sectionTitle)],
    }),
    ...createSectionChildren(block),
  ];
});

const doc = new Document({
  creator: "OpenAI Codex",
  title,
  description: "Anteproyecto de TFG sobre bases de datos vectoriales y sistemas RAG",
  styles: {
    default: {
      document: {
        run: { font: "Arial", size: 22, color: dark },
        paragraph: { spacing: { line: 276 } },
      },
    },
    paragraphStyles: [
      {
        id: "TitleStyle",
        name: "TitleStyle",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: "Arial", size: 34, bold: true, color: accent },
        paragraph: { spacing: { before: 120, after: 180 }, alignment: AlignmentType.CENTER },
      },
      {
        id: "MetaStyle",
        name: "MetaStyle",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: "Arial", size: 18, color: muted, italics: true },
        paragraph: { spacing: { after: 100 }, alignment: AlignmentType.CENTER },
      },
      {
        id: "BodyText",
        name: "BodyText",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: "Arial", size: 22, color: dark },
        paragraph: { spacing: { after: 160 }, line: 320 },
      },
      {
        id: "BodyList",
        name: "BodyList",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: "Arial", size: 22, color: dark },
        paragraph: { spacing: { after: 80 }, line: 300, indent: { left: 360 } },
      },
      {
        id: "TableLabel",
        name: "TableLabel",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: "Arial", size: 20, bold: true, color: accent },
        paragraph: { spacing: { after: 20 } },
      },
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: "Arial", size: 28, bold: true, color: accent },
        paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 0 },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { font: "Arial", size: 23, bold: true, color: "355C7D" },
        paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 1 },
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: "Anteproyecto TFG · propuesta preliminar", color: muted, size: 18 }),
              ],
            }),
          ],
        }),
      },
      children: [
        new Paragraph({
          style: "MetaStyle",
          spacing: { before: 1200, after: 80 },
          children: [new TextRun({ text: "ANTEPROYECTO DE TFG", bold: true, color: accent })],
        }),
        new Paragraph({
          style: "TitleStyle",
          children: [new TextRun(title)],
        }),
        new Paragraph({
          style: "MetaStyle",
          spacing: { before: 120, after: 280 },
          children: [new TextRun("Recuperación de información · Bases vectoriales · Sistemas RAG")],
        }),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [9360],
          rows: [
            new TableRow({
              children: [
                fullWidthCell("Propuesta academica preliminar para validacion con profesorado", {
                  alignment: AlignmentType.CENTER,
                  bold: true,
                  color: "FFFFFF",
                  shading: { fill: accent, type: ShadingType.CLEAR },
                  borders: {
                    top: { style: BorderStyle.NONE, size: 0, color: accent },
                    bottom: { style: BorderStyle.NONE, size: 0, color: accent },
                    left: { style: BorderStyle.NONE, size: 0, color: accent },
                    right: { style: BorderStyle.NONE, size: 0, color: accent },
                  },
                }),
              ],
            }),
          ],
        }),
        new Paragraph({ style: "BodyText", alignment: AlignmentType.CENTER, children: [new TextRun("")] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [2500, 6860],
          rows: [
            keyValueRow("Enfoque", "Análisis comparativo con prototipo funcional y evaluación experimental."),
            keyValueRow("Dominio", "Documentación técnica estructurada para recuperación y consulta asistida."),
            keyValueRow("Tecnologías", "Python, JavaScript, embeddings por API y motores vectoriales ligeros."),
            keyValueRow("Viabilidad", "Pensado para hardware modesto con apoyo puntual de servicios en la nube."),
          ],
        }),
        new Paragraph({ children: [new PageBreak()] }),
        new Paragraph({
          heading: HeadingLevel.HEADING_1,
          children: [new TextRun("Resumen ejecutivo")],
        }),
        new Table({
          width: { size: 9360, type: WidthType.DXA },
          columnWidths: [9360],
          rows: [
            new TableRow({
              children: [
                fullWidthCell(cleanInline(resumen), {
                  shading: { fill: accentSoft, type: ShadingType.CLEAR },
                }),
              ],
            }),
          ],
        }),
        ...bodySections,
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outputPath, buffer);
  console.log(`DOCX created at ${outputPath}`);
});
