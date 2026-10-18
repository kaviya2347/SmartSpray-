// Builds docs/SmartSpray_Project_Documentation.docx from the Markdown chapters in
// docs/project_doc/*.md (a small Markdown subset: headings, paragraphs, lists,
// tables, code blocks, block quotes, images, **bold**, *italic*, `code`).
//
//   node docs/project_doc/build_docx.js            (run from the repository root)
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, AlignmentType, ImageRun, PageBreak, TableOfContents, Footer, Header,
  PageNumber, LevelFormat, TabStopType, VerticalAlign, Bookmark, InternalHyperlink,
  PositionalTab, PositionalTabAlignment, PositionalTabRelativeTo, PositionalTabLeader, Tab, LeaderType,
} = require('docx');

const ROOT = process.cwd();
const SRC = path.join(ROOT, 'docs', 'project_doc');
const OUT = process.argv[2] || path.join(ROOT, 'docs', 'SmartSpray_Project_Documentation.docx');
const PAGES = process.argv[3] && fs.existsSync(process.argv[3]) ? JSON.parse(fs.readFileSync(process.argv[3], 'utf8')) : {};
const tocEntries = [];
const FONT = 'Calibri', MONO = 'Consolas';
const GREEN = '1B5E20', INK = '1F2A24', MUTED = '5F6B66';
const CONTENT_W = 9026;           // A4 width 11906 - 2 x 1440 margins (DXA)
const MAX_IMG_PX = 600;           // ~6.25 in

// ── image size from file header ───────────────────────────────────────────
function imageSize(buf, file) {
  if (buf[0] === 0x89 && buf[1] === 0x50) return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20), type: 'png' };
  if (buf[0] === 0xff && buf[1] === 0xd8) {
    let i = 2;
    while (i < buf.length) {
      if (buf[i] !== 0xff) { i++; continue; }
      const m = buf[i + 1];
      const len = buf.readUInt16BE(i + 2);
      if ([0xc0, 0xc1, 0xc2].includes(m)) return { h: buf.readUInt16BE(i + 5), w: buf.readUInt16BE(i + 7), type: 'jpg' };
      i += 2 + len;
    }
  }
  throw new Error(`unknown image format: ${file}`);
}

// ── inline formatting ────────────────────────────────────────────────────
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*?\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith('`')) out.push(new TextRun({ text: t.slice(1, -1), font: MONO, size: (base.size || 21) - 2, color: '2E4A3A', ...base, bold: base.bold }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

// ── blocks ───────────────────────────────────────────────────────────────
const border = { style: BorderStyle.SINGLE, size: 4, color: 'C9D3CE' };
const cellBorders = { top: border, bottom: border, left: border, right: border };

function table(rows) {
  const cells = rows.map((r) => r.replace(/^\|/, '').replace(/\|\s*$/, '').split('|').map((c) => c.trim()));
  const header = cells[0];
  const body = cells.slice(2);
  const n = header.length;
  const strip = (s) => s.replace(/\*\*|`|\*/g, '');
  const lens = [], mins = [];
  for (let j = 0; j < n; j++) {
    const col = [header[j], ...body.map((r) => r[j] || '')];
    const all = col.map((s) => strip(s).length);
    const avg = all.reduce((a, b) => a + b, 0) / all.length;
    lens.push(Math.max(6, Math.min(70, 0.6 * Math.max(...all) + 0.4 * avg)));
    // the longest unbreakable token (code paths, numbers) sets a minimum width
    const tok = Math.max(...col.map((s) => Math.max(0, ...strip(s).split(/[\s]+/).map((w) => w.length))));
    const htok = Math.max(...strip(header[j]).split(/\s+/).map((w) => w.length));
    mins.push(Math.min(3200, Math.max(200 + tok * 95, 260 + htok * 105)));
  }
  const total = lens.reduce((a, b) => a + b, 0);
  let base = mins.map((m) => Math.max(700, m));
  const baseSum = base.reduce((a, b) => a + b, 0);
  let widths;
  if (baseSum >= CONTENT_W) {
    widths = base.map((b) => Math.floor(b * CONTENT_W / baseSum));
  } else {
    // give every column its minimum, then share the rest by content length
    const spare = CONTENT_W - baseSum;
    const want = lens.map((l, j) => Math.max(0, CONTENT_W * l / total - base[j]));
    const wantSum = want.reduce((a, b) => a + b, 0) || 1;
    widths = base.map((b, j) => Math.floor(b + spare * want[j] / wantSum));
  }
  widths[widths.length - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const mk = (row, isHead) => new TableRow({
    tableHeader: isHead,
    cantSplit: true,
    children: Array.from({ length: n }, (_, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA },
      borders: cellBorders,
      verticalAlign: VerticalAlign.CENTER,
      shading: isHead ? { type: ShadingType.CLEAR, fill: 'E3F0E5', color: 'auto' } : undefined,
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      children: [new Paragraph({ spacing: { before: 0, after: 0, line: 252 },
        children: runs(row[j] || '', { size: 18, bold: isHead || undefined, color: INK }) })],
    })),
  });
  return [new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [mk(header, true), ...body.map((r) => mk(r, false))] }),
  new Paragraph({ spacing: { before: 0, after: 120 }, children: [] })];
}

function codeBlock(lines) {
  const shade = { type: ShadingType.CLEAR, fill: 'F2F5F3', color: 'auto' };
  const bd = { style: BorderStyle.SINGLE, size: 12, color: '7EB38A', space: 6 };
  return lines.map((l, i) => new Paragraph({
    shading: shade,
    border: { left: bd },
    indent: { left: 120, right: 120 },
    spacing: { before: i === 0 ? 80 : 0, after: i === lines.length - 1 ? 160 : 0, line: 240 },
    keepLines: true,
    children: [new TextRun({ text: l.length ? l.replace(/\t/g, '    ') : ' ', font: MONO, size: 15, color: '233129' })],
  }));
}

let figNo = 0;
function image(caption, file) {
  const p = path.join(ROOT, file);
  if (!fs.existsSync(p)) { console.warn('missing image', file); return []; }
  const buf = fs.readFileSync(p);
  const s = imageSize(buf, file);
  let w = Math.min(MAX_IMG_PX, s.w), h = Math.round(s.h * w / s.w);
  const MAX_H = 360;                                  // keep tall figures from filling a page
  if (h > MAX_H) { w = Math.round(w * MAX_H / h); h = MAX_H; }
  figNo++;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ data: buf, type: s.type, transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: path.basename(file) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: 200 },
      children: [new TextRun({ text: `Figure ${figNo}. `, bold: true, size: 18, color: MUTED }),
        ...runs(caption, { size: 18, italics: true, color: MUTED })] }),
  ];
}

function quote(text) {
  return [new Paragraph({
    shading: { type: ShadingType.CLEAR, fill: 'EEF6F0', color: 'auto' },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: '2E7D32', space: 8 } },
    indent: { left: 160, right: 160 },
    spacing: { before: 120, after: 180, line: 276 },
    children: runs(text, { size: 20, color: '1F3B28' }),
  })];
}

// ── markdown → paragraphs ─────────────────────────────────────────────────
let chapter = 0, section = 0, sub = 0, listInstance = 0, bm = 0;
function headingPara(level, label, pageBreak) {
  const id = `h${++bm}`;
  if (level <= 2) tocEntries.push({ level, label, id });
  const hl = [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3][level - 1];
  return new Paragraph({ heading: hl, pageBreakBefore: pageBreak || undefined,
    children: [new Bookmark({ id, children: [new TextRun({ text: label })] })] });
}
function heading(level, text) {
  text = text.replace(/\*\*|`/g, '');
  const isAppendix = /^Appendix/.test(text);
  if (level === 1) {
    section = 0; sub = 0;
    let label = text;
    if (!isAppendix) { chapter++; label = `${chapter}  ${text}`; }
    return [headingPara(1, label, true)];
  }
  const inApp = /^Appendix/.test(currentH1);
  if (level === 2) {
    section++; sub = 0;
    return [headingPara(2, inApp ? text : `${chapter}.${section}  ${text}`)];
  }
  sub++;
  return [headingPara(3, inApp ? text : `${chapter}.${section}.${sub}  ${text}`)];
}

function tocParagraphs() {
  return tocEntries.map((e) => new Paragraph({
    spacing: { before: e.level === 1 ? 110 : 0, after: e.level === 1 ? 30 : 10, line: 250 },
    indent: { left: e.level === 1 ? 0 : 420 },
    tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W, leader: LeaderType.DOT }],
    children: [new InternalHyperlink({ anchor: e.id, children: [
      new TextRun({ text: e.label, bold: e.level === 1, size: e.level === 1 ? 20 : 18, color: e.level === 1 ? GREEN : INK }),
      new TextRun({ children: [new Tab(), String(PAGES[e.label] ?? '00')], size: e.level === 1 ? 20 : 18,
        bold: e.level === 1, color: INK }),
    ] })],
  }));
}

let currentH1 = '';
function convert(md) {
  const out = [];
  const lines = md.split('\n');
  let i = 0, para = [];
  const flush = () => {
    if (!para.length) return;
    out.push(new Paragraph({ spacing: { before: 0, after: 140, line: 288 }, children: runs(para.join(' '), { size: 21, color: INK }) }));
    para = [];
  };
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      flush();
      const buf = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      out.push(...codeBlock(buf));
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      flush();
      if (h[1].length === 1) currentH1 = h[2];
      out.push(...heading(Math.min(3, h[1].length), h[2].trim()));
      i++; continue;
    }
    if (/^\|/.test(line)) {
      flush();
      const rows = [];
      while (i < lines.length && /^\|/.test(lines[i])) rows.push(lines[i++]);
      out.push(...table(rows));
      continue;
    }
    const img = line.match(/^!\[(.*)\]\((.*)\)\s*$/);
    if (img) { flush(); out.push(...image(img[1], img[2])); i++; continue; }
    if (/^>\s?/.test(line)) {
      flush();
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ''));
      out.push(...quote(buf.join(' ')));
      continue;
    }
    const li = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
    if (li) {
      flush();
      const numbered = /\d+\./.test(li[2]);
      listInstance++;
      const inst = listInstance;
      while (i < lines.length) {
        const m = lines[i].match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
        if (!m) {
          // continuation line of the previous item
          if (lines[i].trim() && /^\s{2,}\S/.test(lines[i]) && out.length) {
            const last = out[out.length - 1];
            last.addChildElement(new TextRun({ text: ' ' }));
            runs(lines[i].trim(), { size: 21, color: INK }).forEach((r) => last.addChildElement(r));
            i++; continue;
          }
          break;
        }
        const level = Math.min(1, Math.floor(m[1].length / 2));
        const isNum = /\d+\./.test(m[2]);
        out.push(new Paragraph({
          numbering: isNum ? { reference: 'numbers', level, instance: inst } : { reference: 'bullets', level },
          spacing: { before: 0, after: 70, line: 276 },
          children: runs(m[3], { size: 21, color: INK }),
        }));
        i++;
      }
      out.push(new Paragraph({ spacing: { before: 0, after: 60 }, children: [] }));
      continue;
    }
    if (!line.trim()) { flush(); i++; continue; }
    para.push(line.trim());
    i++;
  }
  flush();
  return out;
}

// ── title page ───────────────────────────────────────────────────────────
function titlePage() {
  const c = (text, opts = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 0, after: opts.after ?? 120 },
    children: [new TextRun({ text, font: FONT, ...opts })] });
  return [
    new Paragraph({ spacing: { before: 1500, after: 0 }, children: [] }),
    c('SMARTSPRAY', { size: 30, bold: true, color: '2E7D32', characterSpacing: 60, after: 240 }),
    c('Vision-Guided Autonomous UAV for', { size: 44, bold: true, color: INK, after: 40 }),
    c('Plant-Level Disease Detection and', { size: 44, bold: true, color: INK, after: 40 }),
    c('Variable-Rate Spraying', { size: 44, bold: true, color: INK, after: 360 }),
    new Paragraph({ alignment: AlignmentType.CENTER, border: { top: { style: BorderStyle.SINGLE, size: 8, color: '7EB38A', space: 12 } },
      spacing: { before: 0, after: 200 }, children: [] }),
    c('Complete technical documentation', { size: 28, color: '2E4A3A', after: 60 }),
    c('architecture · algorithms · code · results · operating guide · review preparation', { size: 22, color: MUTED, after: 1000 }),
    c('Team 134 · B.Tech Data Science', { size: 24, bold: true, color: INK, after: 80 }),
    c('Vellore Institute of Technology, Chennai', { size: 24, color: INK, after: 300 }),
    c('Riddhima Singh  ·  Rounak Handa  ·  Sharan K', { size: 24, color: INK, after: 120 }),
    c('Guide: Rajesh M', { size: 24, color: INK, after: 900 }),
    c('Version 1.0 · October 2026 · implementation status ≈ 75 %', { size: 20, color: MUTED, after: 60 }),
    c('Repository: SmartSpray-main · all figures and numbers are reproducible from the files cited', { size: 18, color: MUTED, after: 0 }),
  ];
}

// ── assemble ─────────────────────────────────────────────────────────────
const files = fs.readdirSync(SRC).filter((f) => /^\d+_.*\.md$/.test(f)).sort();
const body = [];
for (const f of files) body.push(...convert(fs.readFileSync(path.join(SRC, f), 'utf8')));

const doc = new Document({
  creator: 'Team 134, VIT Chennai',
  title: 'SmartSpray — Vision-Guided Autonomous UAV for Plant-Level Disease Detection and Variable-Rate Spraying',
  description: 'Complete technical documentation of the SmartSpray project',
  styles: {
    default: { document: { run: { font: FONT, size: 21, color: INK } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 34, bold: true, color: GREEN, font: FONT },
        paragraph: { spacing: { before: 120, after: 240 }, outlineLevel: 0,
          border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: '7EB38A', space: 6 } } } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 27, bold: true, color: '2E5E36', font: FONT },
        paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 23, bold: true, color: '3B5245', font: FONT },
        paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: {
    config: [
      { reference: 'bullets', levels: [
        { level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } },
        { level: 1, format: LevelFormat.BULLET, text: '◦', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 940, hanging: 260 } } } },
      ] },
      { reference: 'numbers', levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 320 } } } },
        { level: 1, format: LevelFormat.LOWER_LETTER, text: '%2)', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 940, hanging: 320 } } } },
      ] },
    ],
  },
  sections: [
    { properties: { titlePage: true,
        page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1300, left: 1440, right: 1440 } } },
      headers: {
        first: new Header({ children: [new Paragraph({ children: [] })] }),
        default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: 'SmartSpray · Vision-Guided Autonomous UAV — technical documentation', size: 16, color: '8A9590' })] })] }) },
      footers: {
        first: new Footer({ children: [new Paragraph({ children: [] })] }),
        default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: ['Page ', PageNumber.CURRENT, ' of ', PageNumber.TOTAL_PAGES], size: 16, color: '8A9590' })] })] }) },
      children: [
        ...titlePage(),
        new Paragraph({ pageBreakBefore: true, children: [new TextRun({ text: 'Contents', bold: true, size: 34, color: GREEN })], spacing: { after: 160 } }),
        ...tocParagraphs(),
        ...body,
      ] },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT} (${(buf.length / 1024 / 1024).toFixed(2)} MB, ${figNo} figures, ${chapter} chapters)`);
});
