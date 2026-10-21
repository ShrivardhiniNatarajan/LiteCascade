// Builds paper/LiteCascade_IEEE.docx (IEEE conference layout, US Letter, two columns).
// Every result number is read from figures/numbers.json (written by scripts/make_paper_figures.py).
//   cd paper && node build_paper.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, SectionType, TabStopType, ShadingType, VerticalAlign, LineRuleType,
} = require("docx");

const N = JSON.parse(fs.readFileSync(path.join(__dirname, "figures", "numbers.json"), "utf8"));
const FONT = "Times New Roman";
const COL = 5040; // column width in DXA (3.5 in)

// ------------------------------------------------------------------ number helpers
const f3 = (v) => v.toFixed(3);
const f4 = (v) => v.toFixed(4);
const pc = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const k = (v) => (v >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : `${(v / 1e3).toFixed(1)}K`);
const int = (v) => Math.round(v).toLocaleString("en-US");
const card = (id) => N.cards.find((c) => c.id === id);
const R = N.stages_R, T = N.stages_T;
const RF = R[R.length - 1], TF = T[T.length - 1];
const gate = RF.gate;
const abl = Object.fromEntries(N.ablations.map((a) => [a.name.split(" (")[0], a]));
const D = N.data.windows;

// ------------------------------------------------------------------ text helpers
// Inline markup: *italic*, **bold**, _{sub}, ^{sup}
// IEEE numbering: references are renumbered in order of first citation in the body.
const SRC = fs.readFileSync(__filename, "utf8");
const BODY_SRC = SRC.slice(SRC.lastIndexOf("// I. Introduction"), SRC.lastIndexOf("// References"));
const CITE_ORDER = [];
for (const m of BODY_SRC.matchAll(/(?<=[\s,(])\[(\d+)\]/g)) if (!CITE_ORDER.includes(m[1])) CITE_ORDER.push(m[1]);
const cite = (n) => String(CITE_ORDER.indexOf(String(n)) + 1);
function runs(text, base = {}) {
  text = text.replace(/\[(\d+)\]/g, (_, n) => `[${CITE_ORDER.includes(n) ? cite(n) : n}]`);
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(2, -1), superScript: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const para = (text, opts = {}) =>
  new Paragraph({
    children: runs(text, { size: opts.size ?? 20 }),
    alignment: opts.align ?? AlignmentType.JUSTIFIED,
    indent: opts.noIndent ? undefined : { firstLine: 202 },
    spacing: { after: opts.after ?? 0, before: opts.before ?? 0, line: 228, lineRule: LineRuleType.AUTO },
    keepNext: opts.keepNext,
  });
let secNo = 0, subNo = 0;
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
const section = (title) => {
  subNo = 0;
  return new Paragraph({
    children: [new TextRun({ text: `${ROMAN[secNo++]}. ${title}`, font: FONT, size: 20, smallCaps: true })],
    alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
  });
};
const sub = (title) =>
  new Paragraph({
    children: [new TextRun({ text: `${String.fromCharCode(65 + subNo++)}. ${title}`, font: FONT, size: 20, italics: true })],
    spacing: { before: 120, after: 60 }, keepNext: true,
  });
let eqNo = 0;
const eq = (text) =>
  new Paragraph({
    children: [new TextRun({ text: "\t", font: FONT }), ...runs(text, { size: 20 }), new TextRun({ text: `\t(${++eqNo})`, font: FONT, size: 20 })],
    tabStops: [{ type: TabStopType.CENTER, position: COL / 2 }, { type: TabStopType.RIGHT, position: COL }],
    spacing: { before: 60, after: 60 },
  });
const eqLine = (text) =>
  new Paragraph({
    children: [new TextRun({ text: "\t", font: FONT }), ...runs(text, { size: 20 })],
    tabStops: [{ type: TabStopType.CENTER, position: COL / 2 }],
    spacing: { before: 60 },
    keepNext: true,
  });
const bullet = (text, n) =>
  new Paragraph({
    children: runs(`${n ? `${n})` : "•"} ${text}`, { size: 20 }),
    alignment: AlignmentType.JUSTIFIED, indent: { left: 202, hanging: 0 }, spacing: { after: 30 },
  });

// ------------------------------------------------------------------ tables
let tabNo = 0;
const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(caption, header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, isHead, last, highlight) =>
    new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      verticalAlign: VerticalAlign.CENTER,
      margins: { top: 20, bottom: 20, left: 50, right: 50 },
      shading: highlight ? { type: ShadingType.CLEAR, fill: "E8EEF7", color: "auto" } : undefined,
      borders: { top: isHead ? thin : none, bottom: isHead || last ? thin : none, left: none, right: none },
      children: [new Paragraph({
        alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: runs(String(text), { size: 14, bold: isHead }),
      })],
    });
  const t = new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true, false, false)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false, ri === rows.length - 1, opts.highlight?.(ri))) })),
    ],
  });
  tabNo++;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
      children: [new TextRun({ text: `TABLE ${ROMAN[tabNo - 1]}`, font: FONT, size: 16 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true,
      children: [new TextRun({ text: caption, font: FONT, size: 16, smallCaps: true })] }),
    t,
    ...(opts.note ? [new Paragraph({ children: runs(opts.note, { size: 14 }), spacing: { before: 40, after: 120 } })] : [new Paragraph({ spacing: { after: 100 }, children: [] })]),
  ];
}

// ------------------------------------------------------------------ figures
let figNo = 0;
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(file, caption) {
  const p = path.join(__dirname, "figures", file);
  const { w, h } = pngSize(p);
  const width = 320; // px at 96 dpi = 3.33 in
  figNo++;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(p), transformation: { width, height: Math.round((width * h) / w) },
        altText: { title: `Fig. ${figNo}`, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 40, after: 140 },
      children: [new TextRun({ text: `Fig. ${figNo}. `, font: FONT, size: 16 }), ...runs(caption, { size: 16 })] }),
  ];
}

// ------------------------------------------------------------------ title block
const authors = [
  ["Anish P", "Reg. No. 24BCE2394"],
  ["Shrivardhini N", "Reg. No. 24BCE2515"],
  ["Vividha Senthil Mohan", "Reg. No. 24BCB0159"],
  ["Nidhi Satish", "Reg. No. 24BAI0267"],
];
const authorTable = new Table({
  width: { size: 10440, type: WidthType.DXA },
  columnWidths: [2610, 2610, 2610, 2610],
  rows: [new TableRow({
    children: authors.map(([name, reg]) => new TableCell({
      width: { size: 2610, type: WidthType.DXA },
      borders: { top: none, bottom: none, left: none, right: none },
      children: [name, reg, "[Department], [Institution]", "[City, Country]", "[email]"].map((l, i) =>
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: l, font: FONT, size: i === 0 ? 22 : 18, italics: i === 2 })] })),
    })),
  })],
});

const lcR = card("fixture_R_int8"), lcT = card("fixture_T_int8"), teacher = card("fixture_teacher");
const zooDL = N.cards.filter((c) => c.kind === "baseline");
const best = [...N.cards].sort((a, b) => b.metrics.f1_macro - a.metrics.f1_macro)[0];

const title = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: "LiteCascade: A Risk-Controlled Early-Exit Cascade of Distilled Lightweight Deep Networks for On-Device IoT Intrusion and Malware Detection", font: FONT, size: 44 })] }),
  authorTable,
  new Paragraph({ spacing: { after: 200 }, children: [] }),
];

// ------------------------------------------------------------------ body
const body = [];
const add = (...x) => body.push(...x.flat());

add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 }, children: [
  new TextRun({ text: "Abstract", font: FONT, size: 18, bold: true, italics: true }),
  new TextRun({ text: "—", font: FONT, size: 18, bold: true }),
  ...runs(
    `Deep sequence models detect IoT botnet traffic well but are expensive to run on every window of traffic, most of which is benign. ` +
    `We present LiteCascade, a two-exit deep network in which a PCA-initialised projection and a depthwise-separable stem feed a tiny ` +
    `Sentinel classifier and a distilled temporal Analyst. A temperature-calibrated gate releases a window at the Sentinel only when a ` +
    `one-sided Clopper–Pearson bound guarantees that the extra misses caused by exiting early stay below a user-set budget ε with ` +
    `confidence 1−α. The network is trained jointly with teacher-to-Analyst and Analyst-to-Sentinel distillation, then ` +
    `structurally pruned, quantized to INT8 with quantization-aware training, and re-calibrated. We release an open pipeline with ` +
    `leakage-safe device-level splits, loaders for N-BaIoT, CICIoT2023 and IoT-23, eleven deep-learning comparison models, and an ` +
    `interactive dashboard that runs all models live. On a deterministic synthetic benchmark in the N-BaIoT format (two unseen test ` +
    `devices, ${int(D.test.windows)} windows), the ${int(lcR.metrics.params)}-parameter INT8 LiteCascade-R reaches macro-F1 ` +
    `${f3(lcR.metrics.f1_macro)} versus ${f3(teacher.metrics.f1_macro)} for a ${int(teacher.metrics.params)}-parameter CNN-BiLSTM teacher, ` +
    `occupies ${lcR.metrics.size_kb.toFixed(1)} KB, lets ${pc(lcR.metrics.rho)} of windows skip the Analyst with a measured ` +
    `exit-induced miss rate of ${pc(lcR.metrics.exit_induced_miss_rate, 2)} (ε = 0.5%), and is ${lcR.metrics.measured_speedup.toFixed(2)}× ` +
    `faster than always running both exits. Several small baselines are equally accurate on this synthetic data; evaluation on the real ` +
    `datasets and on device hardware remains future work.`,
    { size: 18, bold: true }),
] }));
add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 }, children: [
  new TextRun({ text: "Index Terms", font: FONT, size: 18, bold: true, italics: true }),
  new TextRun({ text: "—IoT security, intrusion detection, deep learning, early-exit networks, knowledge distillation, conformal risk control, quantization, TinyML.", font: FONT, size: 18, bold: true }),
] }));

// I. Introduction
add(section("Introduction"));
add(para("Billions of IoT devices sit on home, industrial and critical-infrastructure networks, and botnets such as Mirai and BASHLITE have shown how quickly such devices can be conscripted [14]. Detecting malicious traffic on the device itself avoids cloud round trips and keeps data local, but IoT hardware offers kilobytes of RAM and a tight energy budget. Deep sequence models (LSTM, BiLSTM, hybrid CNN-RNN and transformers) detect attacks well [2], [7], [9], yet they charge every window the same full inference cost, even though live IoT traffic is overwhelmingly benign and mostly easy to classify."));
add(para("Lightweight IoT detectors usually compress one thing at a time: the inputs (incremental PCA [1]), the architecture (MobileNet-style networks [4]) or the weights (quantization [1]). Early-exit networks such as BranchyNet [15] instead adapt compute to the input, but an uncalibrated early exit can silently miss attacks. This paper combines both ideas and adds a statistical guarantee on what early exit may cost."));
add(para("Our contributions are:", { after: 40 }));
add(bullet("**Cascade architecture.** A PCA-initialised projection (PIP) and a depthwise-separable stem shared by a sub-1K-parameter Sentinel exit and a distilled temporal Analyst exit, in a recurrent (R) and a recurrent-free (T) variant.", 1));
add(bullet("**Risk-controlled gate.** Temperature scaling plus a Clopper–Pearson threshold that bounds the exit-induced miss rate by ε with confidence 1−α, implemented exactly (tie-safe in float32) with an explicit infeasibility rule.", 2));
add(bullet("**Ordered compression chain.** Distillation, structured pruning, INT8 quantization-aware training and post-quantization re-calibration of the gate.", 3));
add(bullet("**Reproducible deep-learning benchmark.** Leakage-safe device-level splits with a disjoint calibration split, eleven deep comparison models (including BiGRU, TCN and ResNet-1D) on one pipeline, MAC counting that includes recurrent layers, and a live dashboard where every model runs on the same replayed windows.", 4));

// II. Related work
add(section("Related Work"));
add(sub("Deep learning for IoT intrusion and malware detection"));
add(para("Recurrent networks are the most common deep detectors for IoT traffic. Wang *et al.* [1] reduce inputs with incremental PCA and apply dynamic quantization to a DNN-BiLSTM, and Jouhari and Guizani [7] propose a lightweight CNN-BiLSTM, which motivates our teacher. Katib *et al.* [11] tune a BiLSTM with metaheuristics under a TinyML objective. Transformer-based detectors reach high accuracy on malware benchmarks [2] but ignore resource cost, and Pai *et al.* [5] show that an L1-regularised MLP (SNIPE) is competitive on IoT-23 at a fraction of the cost. Temporal convolutional networks [17] and residual convolutional networks [30] are strong sequence models that we add to the comparison together with GRUs [29]."));
add(sub("Compression and early exit"));
add(para("Knowledge distillation [18], pruning [21] and integer quantization [23] shrink a model uniformly. Early-exit networks [15] attach classifiers to intermediate layers and stop when an exit is confident, but confidence thresholds are usually set by hand. Modern networks are often miscalibrated [20], so a fixed threshold gives no control over the errors that early exit introduces."));
add(sub("Distribution-free risk control"));
add(para("Conformal and risk-controlling methods choose thresholds on a held-out calibration set so that a risk is bounded under exchangeability [31]. We apply this idea to the exit decision itself: the risk is the rate of malicious windows that the Analyst would have caught but the Sentinel released as benign, bounded with an exact binomial (Clopper–Pearson) upper confidence limit [22]."));

// III. Problem
add(section("Problem Formulation"));
add(para(`A device observes records x_{t} ∈ ℝ^{F} of flow statistics. A window X_{t} = [x_{t−T+1}, …, x_{t}] of T = 10 records receives the label y ∈ {0, 1} of its last record (0 = benign), so it can be classified as soon as that record arrives. We seek a classifier with high macro-F1 and malicious recall that minimises expected compute, subject to a bound on the additional misses introduced by exiting early. Design targets for F = 115 are at most 40K parameters and a 64 KB INT8 model.`));

// IV. Method
add(section("LiteCascade"));
add(figure("fig1_architecture.png", "LiteCascade. Every window passes the PIP layer, the shared stem and the Sentinel; the gate either releases it (early exit) or sends the stem features to the Analyst."));
add(sub("PCA-initialised projection (PIP)"));
add(para("Instead of a fixed PCA pre-processing step [1], the projection is a trainable layer initialised with the top-d principal axes W_{p} and the mean μ of the scaled training records:"));
add(eq("*z*_{t} = *W*_{p} (*x*_{t} − *μ*),   *W*_{p} ∈ ℝ^{d×F}."));
add(para("It is frozen for three warm-up epochs so that randomly initialised layers above it cannot destroy the information-preserving initialisation, and is then fine-tuned with the rest of the network."));
add(sub("Shared stem and Sentinel exit"));
add(para("The stem is one depthwise-separable 1-D convolution over time (kernel 3, d = 32 → c = 24 channels, BatchNorm, ReLU6) [16]. The Sentinel adds one more depthwise-separable block, global average pooling (GAP) and a linear layer:"));
add(eq("*p*_{1} = softmax(*W*_{1} GAP(*H*) / *T*_{1}),"));
add(para("where the temperature T_{1} is fitted afterwards on the calibration split by minimising the negative log-likelihood [20]. It is a buffer, not a trained parameter, and is optimised in log-space so that T_{1} > 0 and predictions are unchanged."));
add(sub("Analyst exit"));
add(para("The Analyst reuses the stem features. It applies two residual inverted-bottleneck depthwise-separable TCN blocks (dilations 1 and 2, expansion 2) [16], [17], a bidirectional GRU with 24 hidden units in variant R (omitted in variant T because recurrent layers are poorly supported by microcontroller runtimes), attention pooling and a linear layer, giving p_{2} = softmax(z_{2}/T_{2})."));
add(sub("Risk-controlled asymmetric gate"));
add(para("Let ŷ_{1} = argmax p_{1}. A window exits at the Sentinel if and only if"));
add(eq("(ŷ_{1} = 0 ∧ *p*_{1,0} ≥ *τ*_{b}) ∨ (ŷ_{1} ≠ 0 ∧ max *p*_{1} ≥ *τ*_{m})."));
add(para("A malicious calibration window counts as an exit-induced miss at threshold τ if it exits benign while the Analyst would have flagged it. With m(τ) such windows among N_{m} malicious calibration windows we select"));
add(eq("*τ*_{b} = min{τ : *U*_{CP}(*m*(τ), *N*_{m}; α) ≤ ε},"));
add(para(`where U_{CP} is the one-sided (1−α) Clopper–Pearson upper bound [22]. Because m(τ) is non-increasing in τ, the admissible thresholds form an interval and this choice equals fixed-sequence testing from τ = ∞ downwards, which controls the family-wise error at α without multiplicity correction [31]. Under exchangeability of calibration and test windows, the exit-induced miss rate is therefore at most ε with probability at least 1−α. Thresholds are placed at the next representable float32 value above a calibration score, so ties cannot add exits. If even m = 0 violates the bound (N_{m} < ln α / ln(1−ε), i.e. ${int(gate.min_N_m)} windows for ε = 0.5% and α = 0.05), benign exits are disabled. The alert threshold τ_{m} is set empirically for a 1% Sentinel false-alarm rate on benign calibration windows; it carries no guarantee and can be disabled so that every alert is verified by the Analyst.`));
add(sub("Training objective"));
add(para("Both exits are trained jointly with deep supervision, distilling the teacher t into the Analyst and the Analyst into the Sentinel [18]:"));
add(eqLine("ℒ = CE(*p*_{1}, *y*) + CE(*p*_{2}, *y*)"));
add(eq("+ γτ^{2} KL(σ(*t*/τ) ‖ σ(*s*_{2}/τ)) + δ KL(*p*_{2} ‖ *p*_{1})."));
add(para("We use γ = 0.5, δ = 0.3, τ = 3 and class-balanced cross-entropy. Teacher logits are computed once for exactly the training windows and travel with them through the shuffled loader. The self-distillation target p_{2} is detached, so its gradient trains only the Sentinel and cannot pull the Analyst towards the weaker exit."));
add(sub("Compression chain and expected cost"));
add(para("After joint training we (i) remove 30% of the Analyst expansion channels and of the Sentinel channels by L1-norm ranking, physically rebuilding the layers, and fine-tune for five epochs; (ii) run three epochs of INT8 quantization-aware training [23] on the PIP layer, stem and all convolution blocks (Conv+BN folded), with dynamic INT8 for the remaining linear and GRU layers; and (iii) re-fit T_{1}, T_{2}, τ_{b} and τ_{m} on the INT8 model, because quantization changes confidences. With Sentinel-path cost C_{1}, Analyst cost C_{2} and exit rate ρ,"));
add(eq("E[*C*] = *C*_{1} + (1 − ρ) *C*_{2},   *S* = (*C*_{1} + *C*_{2}) / E[*C*]."));

// V. Setup
add(section("Experimental Setup"));
add(sub("Data and leakage-safe protocol"));
add(para(`The real N-BaIoT [25], CICIoT2023 [26] and IoT-23 [27] corpora were not available in our environment, so all experiments use a deterministic synthetic benchmark written in the native N-BaIoT file layout and read by the same loader as the real data: ten synthetic devices, 115 features with the original N-BaIoT names, one benign file and four attack files per device (16,000 records). Records are generated from a six-dimensional latent AR(1) process mixed into the features with noise. "Flood" attacks shift the latent mean strongly, "combo" attacks moderately, and "scan" attacks add an alternating period-2 pattern that only a temporal model can see. Attack intensity drifts over time and benign traffic contains busy episodes, so part of each class is hard. **The benchmark exercises the full pipeline; it is not evidence of performance on real traffic.**`));
add(para(`Whole devices are assigned to splits: five for training (${int(D.train.windows)} windows), one for validation (${int(D.val.windows)}), two for calibration (${int(D.calib.windows)}, of which ${int(D.calib.malicious)} malicious) and two for testing (${int(D.test.windows)}). The RobustScaler and PCA are fitted on training records only, and windows never cross a device file. The calibration split is disjoint from the validation split used for model selection.`));
add(sub("Deep-learning comparison models"));
add(para("All models use the same windows, class-balanced cross-entropy, AdamW with a cosine schedule and validation-based model selection. Besides the CNN-BiLSTM teacher (2 convolutions, 2-layer BiLSTM with 128 units, attention), the baselines are: B1 an L1-regularised MLP on the flattened window (SNIPE-style [5]); B3 LSTM; B4 BiLSTM; B5 BiLSTM on 32 PCA components with dynamic INT8 (after [1]); B6 CNN-BiLSTM [7]; B7 depthwise-separable CNN; B8 a 2-layer, 2-head transformer encoder [32]; and three models added in this work: B9 a bidirectional GRU [29], B10 a TCN with three residual dilated causal blocks (dilations 1, 2, 4) [17] and B11 a two-stage residual 1-D CNN [30]."));
add(sub("Metrics and measurement"));
add(para("We report macro-F1 with a 200-sample bootstrap 95% confidence interval, malicious recall, false-positive rate (FPR), parameters, multiply-accumulate operations (MACs), serialized model size, exit rate ρ (raw and re-weighted to 1–5% malicious prevalence) and the exit-induced miss rate. MACs are counted with fvcore plus our own handlers for LSTM, GRU and attention, because fvcore counts recurrent layers as zero. Latency is wall-clock time of PyTorch eager inference at batch size 1 on the development PC (x86 CPU) and is **not** a measurement on IoT hardware. Experiments use seed 42."));

// VI. Results
add(section("Results"));
add(sub("Accuracy versus cost"));
const mainRows = [...N.cards].filter((c) => c.id !== "fixture_R_fp32")
  .sort((a, b) => (a.kind === "cascade" ? 1 : 0) - (b.kind === "cascade" ? 1 : 0) || b.metrics.f1_macro - a.metrics.f1_macro)
  .map((c) => [
    c.name.replace(" (SNIPE)", "").replace("Tiny ", "").replace("LiteCascade", "LiteCascade"),
    f3(c.metrics.f1_macro),
    f3(c.metrics.recall_malicious),
    f3(c.metrics.fpr),
    k(c.metrics.params),
    c.kind === "cascade" ? `${k(c.metrics.macs)}*` : k(c.metrics.macs),
    c.metrics.size_kb.toFixed(1),
  ]);
add(table("Test-split comparison (synthetic benchmark, 2 unseen devices)", ["Model", "F1", "Recall", "FPR", "Params", "MACs", "KB"], mainRows,
  [1460, 560, 600, 560, 620, 640, 600],
  { highlight: (ri) => mainRows[ri][0].startsWith("LiteCascade"),
    note: "* expected MACs E[C] at the measured exit rate. KB = serialized state dict (INT8 where the model is quantized: B5, LiteCascade)." }));
add(para(`Table I and Fig. 2 compare all deep models. The INT8 LiteCascade-R reaches macro-F1 ${f3(lcR.metrics.f1_macro)} with ${int(lcR.metrics.params)} parameters and ${k(lcR.metrics.macs)} expected MACs per window, matching the ${int(teacher.metrics.params)}-parameter teacher (${f3(teacher.metrics.f1_macro)}) at ${(teacher.metrics.macs / lcR.metrics.macs).toFixed(0)}× fewer MACs and a ${(teacher.metrics.size_kb / lcR.metrics.size_kb).toFixed(0)}× smaller file. It is not the most accurate model on this benchmark: ${best.name.replace(" (SNIPE)", "")} (${f3(best.metrics.f1_macro)}), B10 TCN (${f3(card("fixture_b10").metrics.f1_macro)}) and B7 DS-CNN (${f3(card("fixture_b7").metrics.f1_macro)}) score higher, and the bootstrap intervals of most models overlap. The synthetic signal is separable enough that small models suffice, so this benchmark cannot show an accuracy advantage for the cascade; its value here is the controlled reduction of compute. LiteCascade-T, without the BiGRU, needs only ${int(lcT.metrics.params)} parameters and ${lcT.metrics.size_kb.toFixed(1)} KB at macro-F1 ${f3(lcT.metrics.f1_macro)}.`));
add(figure("fig2_pareto.png", "Test macro-F1 versus MACs per window (log scale); bubble area grows with parameter count. LiteCascade (blue) is plotted at its expected cost E[C]. Synthetic benchmark."));

add(sub("Compression chain"));
const stageRows = R.map((s, i) => [
  ["Joint + KD", "Pruned 30%", "INT8, FP32 gate", "INT8, re-calibrated"][i],
  f4(s.test.cascade.f1_macro), f3(s.test.cascade.recall_malicious), pc(s.test.rho_raw), pc(s.test.exit_induced_miss_rate, 2),
  s.profile.size_kb_serialized.toFixed(1), `${s.profile.measured_speedup.toFixed(2)}×`,
]);
add(table("LiteCascade-R through the compression chain (test split)", ["Stage", "F1", "Recall", "ρ", "Miss", "KB", "S"], stageRows,
  [1500, 620, 600, 580, 560, 560, 620],
  { note: `Miss = exit-induced miss rate (bound ε = 0.5%). S = measured speed-up of real cascade inference over always running both exits (batch 1, host CPU, same model).` }));
add(para(`Table II and Fig. 3 follow LiteCascade-R through the chain. Pruning reduces parameters from ${int(R[0].profile.params.total)} to ${int(R[1].profile.params.total)} and the full-path cost from ${k(R[0].profile.macs_full)} to ${k(R[1].profile.macs_full)} MACs without loss of F1 (${f4(R[0].test.cascade.f1_macro)} → ${f4(R[1].test.cascade.f1_macro)}). INT8 quantization-aware training shrinks the serialized model from ${R[1].profile.size_kb_serialized.toFixed(1)} to ${RF.profile.size_kb_serialized.toFixed(1)} KB at F1 ${f4(RF.test.cascade.f1_macro)}. Quantization moved the Sentinel's confidences, and re-calibration changed T_{1} from ${R[2].gate.T1.toFixed(3)} to ${gate.T1.toFixed(3)} and τ_{b} from ${R[2].gate.tau_b.toFixed(4)} to ${gate.tau_b.toFixed(4)}. For variant T, the INT8 model kept the FP32 gate produced a ${pc(T[2].test.exit_induced_miss_rate, 2)} exit-induced miss rate on test, and re-calibration brought it to ${pc(TF.test.exit_induced_miss_rate, 2)}. Both values are below ε, so this small benchmark does not show the guarantee breaking without re-calibration; it only shows that quantization moves the confidences the gate depends on.`));
add(figure("fig4_stages.png", "Serialized size (bars), macro-F1 and exit rate ρ of LiteCascade-R after each compression step (test split, synthetic benchmark)."));

add(sub("Risk-controlled gate"));
const gp = (e) => N.gate_curve.find((p) => Math.abs(p.epsilon - e) < 1e-9);
add(para(`The deployed INT8 LiteCascade-R gate uses ε = 0.5% and α = 0.05. The calibration split holds N_{m} = ${int(gate.N_m)} malicious windows, enough for the bound to be feasible (${int(gate.min_N_m)} needed); it admits at most ${gate.m_max} exit-induced misses on calibration (U_{CP} = ${gate.ucp_calib.toFixed(4)}). Temperature scaling reduced the Sentinel's expected calibration error from ${gate.ece_sentinel_before.toFixed(4)} to ${gate.ece_sentinel_after.toFixed(4)}. On the unseen test devices ${pc(RF.test.rho_raw)} of windows exit at the Sentinel (${pc(RF.test.rho_reweighted["1pct"])} when re-weighted to 1% malicious prevalence), the Analyst runs ${int(RF.test.live_inference.analyst_invocations)} times instead of ${int(RF.test.n)}, and the exit-induced miss rate is ${pc(RF.test.exit_induced_miss_rate, 2)}.`));
add(para(`Fig. 4 sweeps ε using the same calibration outputs. At ε = 0.1% the bound is infeasible, benign exits are disabled and only alerts can exit (ρ = ${pc(gp(0.001).rho)}). From ε = 0.2% the exit rate jumps to ${pc(gp(0.002).rho)} and rises to ${pc(gp(0.05).rho)} at ε = 5%. Across the sweep the measured test miss rate stays below ε (for example ${pc(gp(0.02).exit_induced_miss_rate, 2)} at ε = 2% and ${pc(gp(0.05).exit_induced_miss_rate, 2)} at ε = 5%), and macro-F1 varies only between ${f3(Math.min(...N.gate_curve.map((p) => p.f1_macro)))} and ${f3(Math.max(...N.gate_curve.map((p) => p.f1_macro)))}. The analytic speed-up from (6) at the deployed operating point is ${RF.profile.analytic_speedup_macs.toFixed(2)}× in MACs; the measured wall-clock speed-up of real cascade inference is ${RF.profile.measured_speedup.toFixed(2)}×.`));
add(figure("fig3_gate.png", "Exit rate, macro-F1 and exit-induced miss rate on the test devices as functions of the risk budget ε (α = 0.05); dotted: the bound ε. Shaded: infeasible (benign exit disabled). Dashed vertical: deployed ε = 0.5%."));

add(sub("Ablations"));
const ablRows = N.ablations.map((a) => [
  a.name.replace(" (PIP init + KD + self-KD + temp. scaling + CP gate)", "").replace("(Analyst->Sentinel)", "").replace(" (tau=0.9, no T, no CP bound)", " (τ = 0.9)").replace(" (all benign-exit misses)", "").replace(" (tau_m = never)", "").replace(" (no BiGRU)", "").replace("- ", "− "),
  f4(a.cascade_f1), f3(a.recall), pc(a.rho), pc(a.exit_miss, 2), f3(a.sentinel_f1),
]);
add(table("Ablations before compression (test split)", ["Configuration", "F1", "Recall", "ρ", "Miss", "Sent. F1"], ablRows,
  [1820, 640, 620, 620, 640, 700]));
add(para(`Table III removes one component at a time (FP32, no pruning). Random instead of PCA initialisation of the projection costs the most: F1 falls from ${f4(abl["Full"].cascade_f1)} to ${f4(abl["- PIP initialisation"].cascade_f1)} and recall from ${f3(abl["Full"].recall)} to ${f3(abl["- PIP initialisation"].recall)}. Removing teacher distillation lowers the Analyst's F1 from ${f4(abl["Full"].analyst_f1)} to ${f4(abl["- teacher KD"].analyst_f1)}, and removing self-distillation also lowers cascade F1 (${f4(abl["- self-KD"].cascade_f1)}). Removing temperature scaling changes nothing. That is expected for a binary task: temperature is a monotone transform of p_{1,0}, so a threshold chosen by quantile on the calibration set selects the same exits; temperature still matters for the reported probabilities and their calibration error. A naive fixed gate (τ = 0.9) happens to work on this benchmark (miss ${pc(abl["naive gate"].exit_miss, 2)}) but carries no guarantee. The conservative variant, which counts every malicious benign exit as a miss whether or not the Analyst would have caught it, lowers ρ to ${pc(abl["conservative CP gate"].rho)}; verifying every alert with the Analyst lowers it to ${pc(abl["verify alerts"].rho)}.`));

add(sub("Parameter budget and genuine early exit"));
add(para(`For F = 115 before pruning, the Sentinel exit has 746 parameters (target below 2K), the Analyst exit 12,722 and the whole network 18,092 (target about 20K). The Sentinel path costs ${k(R[0].profile.macs_sentinel_path)} MACs and the full path ${k(R[0].profile.macs_full)}, close to the analytic estimates of 0.06M and 0.18M. Early exit is genuine: in the deployed code the Analyst is executed only on windows the gate did not release, which our tests verify with a forward hook. Live batched inference agrees with offline two-head evaluation on ${pc(RF.test.live_inference.agreement_with_offline, 2)} of test windows for INT8 LiteCascade-R and ${pc(TF.test.live_inference.agreement_with_offline, 2)} for INT8 LiteCascade-T; the small gap comes from dynamic quantization, whose activation scales depend on the contents of each batch.`));

// VII. Discussion
add(section("Limitations and Threats to Validity"));
add(para("(i) All results come from a synthetic benchmark. On it the Sentinel alone is almost as accurate as the Analyst, so the benchmark is easier than real traffic, and small baselines are as accurate as the cascade. (ii) No real-dataset or on-device result is reported; latencies are PyTorch eager on a desktop CPU, where INT8 kernels are slower than FP32, so INT8 here buys size, not speed. TFLite / TFLite-Micro export, energy measurement and microcontroller deployment remain to be done. (iii) The guarantee assumes exchangeable calibration and test windows and bounds only the extra misses caused by early exit, not the Analyst's own errors; it may fail under drift or for unseen attack families. (iv) τ_{m} is an empirical choice. (v) Baselines are re-implementations. (vi) Windows are labelled by their last record, so a window spanning an attack onset is scored as the later class."));

// VIII. Conclusion
add(section("Conclusion"));
add(para(`LiteCascade couples a PCA-initialised projection, a two-exit deep network trained with two-way distillation, a Clopper–Pearson-controlled early-exit gate and an ordered compression chain. On a synthetic N-BaIoT-format benchmark its INT8 variant matches a ${(teacher.metrics.params / lcR.metrics.params).toFixed(0)}× larger teacher with ${lcR.metrics.size_kb.toFixed(0)} KB of weights while most windows skip the Analyst and the measured exit-induced miss rate stays within the chosen budget. The open pipeline, eleven deep comparison models and live dashboard make the next steps concrete: running the same protocol on N-BaIoT, CICIoT2023 and IoT-23; leave-one-family-out and perturbation tests of the guarantee under shift; and measuring latency, memory and energy on Raspberry Pi and microcontroller targets.`));

// References
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
  children: [new TextRun({ text: "References", font: FONT, size: 20, smallCaps: true })] }));
const refs = {
  1: "Z. Wang, H. Chen, S. Yang, X. Luo, D. Li, and J. Wang, “A lightweight intrusion detection method for IoT based on deep learning and dynamic quantization,” PeerJ Comput. Sci., vol. 9, p. e1569, 2023.",
  2: "A. A. Almazroi and N. Ayub, “Deep learning hybridization for improved malware detection in smart Internet of Things,” Sci. Rep., vol. 14, 2024.",
  4: "S. Anand, B. Mitra, S. Dey, A. Rao, R. Dhar, and J. Vaidya, “MALITE: Lightweight malware detection and classification for constrained devices,” arXiv:2309.03294, 2023.",
  5: "V. Pai et al., “Systematic approach for malware detection in IoT devices: Enhancing security and performance,” Int. J. Comput. Intell. Syst., vol. 18, art. 196, 2025.",
  7: "M. Jouhari and M. Guizani, “Lightweight CNN-BiLSTM based intrusion detection systems for resource-constrained IoT devices,” arXiv:2406.02768, 2024.",
  9: "S. U. Qureshi et al., “Systematic review of deep learning solutions for malware detection and forensic analysis in IoT,” J. King Saud Univ. Comput. Inf. Sci., vol. 36, no. 8, p. 102164, 2024.",
  11: "I. Katib et al., “DLTML-RTADPM,” [complete venue, volume, pages and DOI before submission].",
  14: "M. Antonakakis et al., “Understanding the Mirai botnet,” in Proc. 26th USENIX Security Symp., 2017, pp. 1093–1110.",
  15: "S. Teerapittayanon, B. McDanel, and H. T. Kung, “BranchyNet: Fast inference via early exiting from deep neural networks,” in Proc. ICPR, 2016, pp. 2464–2469.",
  16: "M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in Proc. CVPR, 2018, pp. 4510–4520.",
  17: "S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” arXiv:1803.01271, 2018.",
  18: "G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv:1503.02531, 2015.",
  20: "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. ICML, 2017, pp. 1321–1330.",
  21: "S. Han, H. Mao, and W. J. Dally, “Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding,” in Proc. ICLR, 2016.",
  22: "C. J. Clopper and E. S. Pearson, “The use of confidence or fiducial limits illustrated in the case of the binomial,” Biometrika, vol. 26, no. 4, pp. 404–413, 1934.",
  23: "B. Jacob et al., “Quantization and training of neural networks for efficient integer-arithmetic-only inference,” in Proc. CVPR, 2018, pp. 2704–2713.",
  25: "Y. Meidan et al., “N-BaIoT—Network-based detection of IoT botnet attacks using deep autoencoders,” IEEE Pervasive Comput., vol. 17, no. 3, pp. 12–22, 2018.",
  26: "E. C. P. Neto et al., “CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment,” Sensors, vol. 23, no. 13, p. 5941, 2023.",
  27: "S. Garcia, A. Parmisano, and M. J. Erquiaga, “IoT-23: A labeled dataset with malicious and benign IoT network traffic,” Stratosphere Lab, Zenodo, 2020.",
  29: "K. Cho et al., “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in Proc. EMNLP, 2014, pp. 1724–1734.",
  30: "K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. CVPR, 2016, pp. 770–778.",
  31: "A. N. Angelopoulos, S. Bates, E. J. Candès, M. I. Jordan, and L. Lei, “Learn then test: Calibrating predictive algorithms to achieve risk control,” arXiv:2110.01052, 2021.",
  32: "A. Vaswani et al., “Attention is all you need,” in Proc. NeurIPS, 2017, pp. 5998–6008.",
};
const uncited = Object.keys(refs).filter((n) => !CITE_ORDER.includes(n));
if (uncited.length) throw new Error(`references never cited: ${uncited}`);
const missing = CITE_ORDER.filter((n) => !refs[n]);
if (missing.length) throw new Error(`citations without a reference: ${missing}`);
CITE_ORDER.forEach((n, i) => {
  add(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 360, hanging: 360 }, spacing: { after: 30 },
    children: [new TextRun({ text: `[${i + 1}]\t${refs[n]}`, font: FONT, size: 16 })], tabStops: [{ type: TabStopType.LEFT, position: 360 }] }));
});

// ------------------------------------------------------------------ document
const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1440, left: 900, right: 900 } };
const doc = new Document({
  creator: "LiteCascade authors",
  title: "LiteCascade IEEE paper",
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [
    { properties: { page }, children: title },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: 360 } }, children: body },
  ],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "LiteCascade_IEEE.docx");
  fs.writeFileSync(out, buf);
  console.log(`Wrote ${out} (${(buf.length / 1024).toFixed(0)} KB)`);
});
