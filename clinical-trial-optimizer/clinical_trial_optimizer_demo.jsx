import { useState } from "react";

const MARIPOSA_CRITERIA = `Inclusion Criteria:
1. Participant must have at least 1 measurable lesion per RECIST v1.1 not previously irradiated
2. Histologically confirmed locally advanced or metastatic non-squamous NSCLC with EGFR Exon 19del or Exon 21 L858R mutation
3. A participant with brain metastases must have had all lesions treated; local therapy completed ≥14 days prior to randomisation; ≤10mg prednisone daily
4. ECOG performance status 0 or 1
5. Toxicities from prior therapy must have resolved to NCI CTCAE Grade ≤1 or baseline
6. Negative serum pregnancy test at screening (participants of childbearing potential)
7. Progressed on or after osimertinib monotherapy as most recent line of treatment (1L or 2L after 1G/2G EGFR TKI)

Exclusion Criteria:
1. Radiotherapy for palliative NSCLC treatment less than 14 days prior to randomisation
2. Symptomatic or progressive brain metastases
3. History or current evidence of leptomeningeal disease or spinal cord compression not definitively treated
4. Known small cell transformation
5. Medical history of interstitial lung disease (ILD), including drug-induced ILD or radiation pneumonitis
6. Clinically significant cardiovascular disease (DVT/PE within 4 weeks, MI, stroke, unstable angina, ACS)`;

const SOC_ROWS = [
  { type: "Inclusion", criterion: "Progressed on/after osimertinib monotherapy", gap: "none", increase: "N/A", rec: "Already captured by existing I/E criteria" },
  { type: "Suggested Addition", criterion: "New criterion", gap: "major", increase: "12.0%", rec: "Include participants progressed on osimertinib+carboplatin+pemetrexed as 1L for locally advanced/metastatic EGFR Exon 19del or L858R NSCLC" },
];

const ET_ROWS = [
  { criterion: "osimertinib+carboplatin+pemetrexed (L1)", gap: "major", p1: "0%", p2: "12.0%", trend: "⚠ Emerging" },
  { criterion: "amivantamab+carboplatin+pemetrexed (L2)", gap: "emerging", p1: "0%", p2: "7.07%", trend: "⚠ Emerging" },
  { criterion: "carboplatin+pemetrexed+pembrolizumab (L2)", gap: "major", p1: "0%", p2: "11.07%", trend: "↑ Increasing" },
  { criterion: "osimertinib monotherapy (L1+L2)", gap: "none", p1: "5% L1 / 64% L2", p2: "61.4% L1 / 30% L2", trend: "↑ / ↓" },
];

const CI_ROWS = [
  { criterion: "RECIST measurable lesion (not irradiated)", label: "KEEP", conf: 0.78, trials: "NCT05382728, NCT06838273" },
  { criterion: "Histological NSCLC + EGFR Exon 19del/L858R", label: "KEEP", conf: 0.78, trials: "NCT04181060, NCT05382728" },
  { criterion: "ECOG performance status 0 or 1", label: "RELAX", conf: 0.62, trials: "NCT05382728, NCT06838273" },
  { criterion: "Progressed on/after osimertinib monotherapy", label: "KEEP", conf: 0.62, trials: "NCT04181060, NCT02317016" },
  { criterion: "Brain metastases — treated, prednisone ≤10mg", label: "KEEP", conf: 0.35, trials: "NCT05382728" },
  { criterion: "Radiotherapy washout ≥14 days", label: "KEEP", conf: 0.52, trials: "NCT04181060, NCT03040973" },
  { criterion: "No leptomeningeal disease / spinal cord compression", label: "KEEP", conf: 0.45, trials: "NCT05382728" },
  { criterion: "No ILD / radiation pneumonitis history", label: "KEEP", conf: 0.55, trials: "NCT04181060, NCT06838273" },
  { criterion: "Cardiovascular disease exclusion", label: "KEEP", conf: 0.35, trials: "NCT05382728" },
];

const RAG_CRITERIA = [
  { label: "RECIST measurable lesion", p5: 0.80, mrr: 1.00, top: "NCT05382728 (0.033)" },
  { label: "EGFR Exon 19del/L858R", p5: 0.80, mrr: 1.00, top: "NCT04181060 (0.032)" },
  { label: "ECOG status 0 or 1", p5: 0.80, mrr: 1.00, top: "NCT05382728 (0.031)" },
  { label: "Osimertinib progression", p5: 0.60, mrr: 0.50, top: "NCT02317016 (0.032)" },
  { label: "Radiotherapy washout", p5: 0.40, mrr: 0.33, top: "NCT04181060 (0.031)" },
  { label: "No ILD history", p5: 0.40, mrr: 0.50, top: "NCT04181060 (0.032)" },
  { label: "Brain metastases treated", p5: 0.20, mrr: 0.20, top: "NCT05382728 (0.031)" },
  { label: "No leptomeningeal disease", p5: 0.20, mrr: 0.20, top: "NCT05382728 (0.030)" },
  { label: "Cardiovascular exclusion", p5: 0.20, mrr: 0.20, top: "NCT05382728 (0.032)" },
];

const RRF_BARS = [
  { label: "NCT05382728 → RECIST", score: 0.0328 },
  { label: "NCT06838273 → RECIST", score: 0.0315 },
  { label: "NCT04181060 → EGFR mutation", score: 0.0320 },
  { label: "NCT05382728 → ECOG", score: 0.0310 },
  { label: "NCT02317016 → Osimertinib progression", score: 0.0320 },
  { label: "NCT03040973 → Brain metastases", score: 0.0323 },
  { label: "NCT04181060 → ILD exclusion", score: 0.0318 },
];

const C = {
  bg: "#0a0e1a", surface: "#111827", high: "#1a2236", border: "#1e2d45",
  accent: "#00c2ff", green: "#00e5a0", amber: "#ffb347", red: "#ff5a5a",
  text: "#e2e8f0", muted: "#64748b", dim: "#94a3b8",
};

function Badge({ type }) {
  const map = {
    KEEP: ["#00e5a020", "#00e5a0", "KEEP"],
    RELAX: ["#ffb34720", "#ffb347", "RELAX"],
    TIGHTEN: ["#ff5a5a20", "#ff5a5a", "TIGHTEN"],
    major: ["#ff5a5a20", "#ff5a5a", "Major"],
    minor: ["#ffb34720", "#ffb347", "Minor"],
    none: ["#00e5a020", "#00e5a0", "No gap"],
    emerging: ["#00c2ff20", "#00c2ff", "Emerging"],
    "Suggested Addition": ["#00c2ff20", "#00c2ff", "Suggested"],
    Inclusion: ["#00e5a020", "#00e5a0", "Inclusion"],
  };
  const [bg, color, label] = map[type] || ["#1a223640", C.dim, type];
  return (
    <span style={{ background: bg, color, border: `1px solid ${color}40`, borderRadius: 6, padding: "2px 8px", fontSize: 11, fontFamily: "monospace", whiteSpace: "nowrap" }}>
      {label}
    </span>
  );
}

function Bar({ value, max = 1, color = C.accent }) {
  return (
    <div style={{ flex: 1, height: 8, background: C.border, borderRadius: 4, overflow: "hidden" }}>
      <div style={{ width: `${(value / max) * 100}%`, height: "100%", background: color, borderRadius: 4 }} />
    </div>
  );
}

function MetricCard({ value, label }) {
  return (
    <div style={{ background: C.high, border: `1px solid ${C.border}`, borderRadius: 10, padding: "14px 16px", textAlign: "center" }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: C.accent, fontFamily: "monospace" }}>{value}</div>
      <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{label}</div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("input");
  const [criteria, setCriteria] = useState(MARIPOSA_CRITERIA);

  const tabs = [
    { id: "input", label: "I/E Criteria input" },
    { id: "results", label: "Agent results" },
    { id: "rag", label: "RAG pipeline performance" },
  ];

  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.text, fontFamily: "'DM Sans', sans-serif", padding: "0 0 60px" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 4px; }
        textarea { outline: none; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        .fade { animation: fadeIn 0.25s ease; }
      `}</style>

      {/* Header */}
      <div style={{ background: C.high, borderBottom: `1px solid ${C.border}`, padding: "20px 40px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <div style={{ width: 28, height: 28, borderRadius: 7, background: `linear-gradient(135deg, ${C.accent}, ${C.green})`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14 }}>⬡</div>
            <span style={{ fontSize: 16, fontWeight: 700 }}>Clinical Trial Optimizer</span>
          </div>
          <div style={{ fontSize: 12, color: C.muted, fontFamily: "monospace" }}>SOC · ET · CI agents · Advanced Modular RAG</div>
        </div>
        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 8, padding: "5px 12px", fontSize: 11, color: C.muted, fontFamily: "monospace" }}>
          Trial: <span style={{ color: C.accent }}>MARIPOSA-2</span> · NCT04988295
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "28px 40px 0" }}>
        {/* Tabs */}
        <div style={{ display: "flex", borderBottom: `1px solid ${C.border}`, marginBottom: 24 }}>
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)}
              style={{ padding: "9px 20px", fontSize: 13, fontWeight: tab === t.id ? 600 : 400, color: tab === t.id ? C.accent : C.muted, borderBottom: tab === t.id ? `2px solid ${C.accent}` : "2px solid transparent", marginBottom: -1, background: "none", border: "none", cursor: "pointer", fontFamily: "inherit" }}>
              {t.label}
            </button>
          ))}
        </div>

        {/* TAB 1: INPUT */}
        {tab === "input" && (
          <div className="fade">
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20, marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8, fontFamily: "monospace" }}>Trial</div>
              <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>MARIPOSA-2 · NCT04988295</div>
              <div style={{ fontSize: 12, color: C.dim }}>Phase III · Amivantamab + chemotherapy ± lazertinib · EGFR-mutated NSCLC · Post-osimertinib</div>
            </div>
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20, marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8, fontFamily: "monospace" }}>Inclusion / exclusion criteria</div>
              <textarea value={criteria} onChange={e => setCriteria(e.target.value)}
                style={{ width: "100%", minHeight: 200, fontSize: 12, fontFamily: "monospace", padding: 10, background: C.high, border: `1px solid ${C.border}`, borderRadius: 8, color: C.text, resize: "vertical", lineHeight: 1.6 }} />
              <div style={{ display: "flex", gap: 10, marginTop: 12, alignItems: "center" }}>
                <button onClick={() => setTab("results")} style={{ background: `linear-gradient(135deg, ${C.accent}, #0099cc)`, color: "#000", border: "none", borderRadius: 8, padding: "8px 20px", fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>
                  View agent results →
                </button>
                <button onClick={() => setCriteria(MARIPOSA_CRITERIA)} style={{ background: "none", border: `1px solid ${C.border}`, color: C.dim, borderRadius: 8, padding: "8px 14px", fontSize: 13, cursor: "pointer", fontFamily: "inherit" }}>
                  Reset to MARIPOSA-2
                </button>
                <span style={{ fontSize: 11, color: C.muted }}>Pre-loaded with MARIPOSA-2 I/E criteria · edit to customise</span>
              </div>
            </div>
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12, fontFamily: "monospace" }}>Data sources</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
                {[
                  { label: "SOC dataset", val: "1,500 patients", sub: "EGFR+ NSCLC · 2021-2024" },
                  { label: "ET dataset", val: "2 periods", sub: "2017-2020 vs 2021-2024" },
                  { label: "Competing trials", val: "20 trials", sub: "ClinicalTrials.gov API v2" },
                ].map(d => (
                  <div key={d.label} style={{ background: C.high, borderRadius: 8, padding: "12px 14px" }}>
                    <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>{d.label}</div>
                    <div style={{ fontSize: 14, fontWeight: 600, color: C.text }}>{d.val}</div>
                    <div style={{ fontSize: 11, color: C.dim, marginTop: 2 }}>{d.sub}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: RESULTS */}
        {tab === "results" && (
          <div className="fade">
            {/* SOC */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20, marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.green }} />
                <div>
                  <div style={{ fontSize: 14, fontWeight: 600 }}>SOC agent — standard of care gap analysis</div>
                  <div style={{ fontSize: 12, color: C.muted }}>RWD regimen summary · 2021-2024 · EGFR+ NSCLC</div>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
                <div>
                  <div style={{ fontSize: 11, color: C.muted, marginBottom: 6, fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.06em" }}>Prompt (excerpt)</div>
                  <div style={{ background: C.high, border: `1px solid ${C.border}`, borderRadius: 8, padding: 10, fontSize: 11, fontFamily: "monospace", color: C.dim, maxHeight: 160, overflowY: "auto", lineHeight: 1.6 }}>
                    You are a biomedical trial analysis agent. Analyze treatment-related I/E criteria using exact I/E criteria text and pre-processed real-world regimen summary...<br/><br/>
                    STEP 0: Extract disease/biomarker. Select ONLY treatment-related criteria.<br/>
                    STEP 1: Extract regimens from LOT data aligned with trial target line.<br/>
                    STEP 2: Generate gap-based recommendations.<br/><br/>
                    Dataset: osimertinib L1R1 61.4% | osimertinib+carboplatin+pemetrexed L1R2 12.0% | carboplatin+pemetrexed L2R2 25.07%...
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: C.muted, marginBottom: 6, fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.06em" }}>Output summary</div>
                  <div style={{ background: C.high, border: `1px solid ${C.border}`, borderRadius: 8, padding: 10, fontSize: 12, color: C.dim, maxHeight: 160, overflowY: "auto", lineHeight: 1.6 }}>
                    <strong style={{ color: C.text }}>Treatment-related criteria: 1</strong><br/><br/>
                    Gap: osimertinib+carboplatin+pemetrexed (12% of L1, Major, Increasing) is NOT covered by the monotherapy requirement.<br/><br/>
                    Suggested Addition: Include participants who have progressed on or after osimertinib+carboplatin+pemetrexed as first-line therapy for locally advanced or metastatic EGFR Exon 19del or L858R NSCLC.<br/><br/>
                    Primary Recommendation: The monotherapy-only restriction excludes 12% of real-world L1 patients. Consider expanding inclusion to cover combination-treated patients.
                  </div>
                </div>
              </div>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                      {["Type", "Criterion", "Gap type", "Expected increase", "Recommendation"].map(h => (
                        <th key={h} style={{ textAlign: "left", padding: "6px 10px", color: C.muted, fontWeight: 500, whiteSpace: "nowrap" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {SOC_ROWS.map((r, i) => (
                      <tr key={i} style={{ borderBottom: i < SOC_ROWS.length - 1 ? `1px solid ${C.border}` : "none" }}>
                        <td style={{ padding: "8px 10px" }}><Badge type={r.type} /></td>
                        <td style={{ padding: "8px 10px", color: C.dim, maxWidth: 180 }}>{r.criterion}</td>
                        <td style={{ padding: "8px 10px" }}><Badge type={r.gap} /></td>
                        <td style={{ padding: "8px 10px", color: C.text, fontFamily: "monospace" }}>{r.increase}</td>
                        <td style={{ padding: "8px 10px", color: C.dim, maxWidth: 220, fontSize: 11 }}>{r.rec}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* ET */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20, marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.accent }} />
                <div>
                  <div style={{ fontSize: 14, fontWeight: 600 }}>ET agent — evolving treatment trends</div>
                  <div style={{ fontSize: 12, color: C.muted }}>Period 1: 2017-2020 · Period 2: 2021-2024 · EGFR+ NSCLC</div>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
                <div>
                  <div style={{ fontSize: 11, color: C.muted, marginBottom: 6, fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.06em" }}>Prompt (excerpt)</div>
                  <div style={{ background: C.high, border: `1px solid ${C.border}`, borderRadius: 8, padding: 10, fontSize: 11, fontFamily: "monospace", color: C.dim, maxHeight: 160, overflowY: "auto", lineHeight: 1.6 }}>
                    Analyze treatment-related I/E criteria combining Period 1 (2017-2020) and Period 2 (2021-2024)...<br/><br/>
                    STEP 1 filter: Select regimens where Trend="Emerging" OR Gap Type="Major"<br/><br/>
                    osimertinib L1R1 nan→61.4% (Major, Increasing)<br/>
                    osimertinib+carboplatin+pemetrexed L1R2 nan→12.0% (Major, Emerging)<br/>
                    amivantamab+carboplatin+pemetrexed L2R5 nan→7.07% (Emerging)<br/>
                    carboplatin+pemetrexed+pembrolizumab L2R4 nan→11.07% (Major, Increasing)...
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: C.muted, marginBottom: 6, fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.06em" }}>Output summary</div>
                  <div style={{ background: C.high, border: `1px solid ${C.border}`, borderRadius: 8, padding: 10, fontSize: 12, color: C.dim, maxHeight: 160, overflowY: "auto", lineHeight: 1.6 }}>
                    <strong style={{ color: C.text }}>Key temporal signals</strong><br/><br/>
                    Emerging (Period 2 only):<br/>
                    • osi+carboplatin+pemetrexed: 0%→12.0% ⚠<br/>
                    • amivantamab+chemo: 0%→7.07% ⚠<br/>
                    • carboplatin+pem+pembrolizumab: 0%→11.07% ⚠<br/><br/>
                    Declining:<br/>
                    • osimertinib at L2: 64%→30%<br/>
                    • 1G/2G TKIs at L1: gefitinib 28%→8%, erlotinib 24%→6%
                  </div>
                </div>
              </div>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                      {["Criterion", "Gap", "Period 1", "Period 2", "Trend"].map(h => (
                        <th key={h} style={{ textAlign: "left", padding: "6px 10px", color: C.muted, fontWeight: 500 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {ET_ROWS.map((r, i) => (
                      <tr key={i} style={{ borderBottom: i < ET_ROWS.length - 1 ? `1px solid ${C.border}` : "none" }}>
                        <td style={{ padding: "8px 10px", color: C.dim, maxWidth: 220 }}>{r.criterion}</td>
                        <td style={{ padding: "8px 10px" }}><Badge type={r.gap} /></td>
                        <td style={{ padding: "8px 10px", color: C.dim, fontFamily: "monospace", fontSize: 11 }}>{r.p1}</td>
                        <td style={{ padding: "8px 10px", color: C.text, fontFamily: "monospace", fontSize: 11 }}>{r.p2}</td>
                        <td style={{ padding: "8px 10px", color: C.dim, fontFamily: "monospace", fontSize: 11 }}>{r.trend}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* CI */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
                <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.amber }} />
                <div>
                  <div style={{ fontSize: 14, fontWeight: 600 }}>CI agent — competitive intelligence (RAG-powered)</div>
                  <div style={{ fontSize: 12, color: C.muted }}>20 competing trials · PubMedBERT + FAISS + BM25 · RRF fusion · Claude Sonnet reasoning</div>
                </div>
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
                <div>
                  <div style={{ fontSize: 11, color: C.muted, marginBottom: 6, fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.06em" }}>Prompt (excerpt)</div>
                  <div style={{ background: C.high, border: `1px solid ${C.border}`, borderRadius: 8, padding: 10, fontSize: 11, fontFamily: "monospace", color: C.dim, maxHeight: 160, overflowY: "auto", lineHeight: 1.6 }}>
                    You are a clinical trial eligibility criteria expert. Analyze a source criterion against evidence from competing trials and make a structured recommendation. Output only valid JSON.<br/><br/>
                    Source criterion: "ECOG performance status 0 or 1"<br/><br/>
                    Evidence from 5 competing trials:<br/>
                    1. [NCT05382728] — inclusion — "ECOG 0, 1, or 2" — RRF: 0.031<br/>
                    2. [NCT06838273] — inclusion — "ECOG performance status 0-2" — RRF: 0.030<br/>
                    3. [NCT04181060] — inclusion — "ECOG 0-2 required" — RRF: 0.028...
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 11, color: C.muted, marginBottom: 6, fontFamily: "monospace", textTransform: "uppercase", letterSpacing: "0.06em" }}>Output summary</div>
                  <div style={{ background: C.high, border: `1px solid ${C.border}`, borderRadius: 8, padding: 10, fontSize: 12, color: C.dim, maxHeight: 160, overflowY: "auto", lineHeight: 1.6 }}>
                    <strong style={{ color: C.text }}>13 criteria evaluated · 1 RELAX · 12 KEEP</strong><br/><br/>
                    <strong style={{ color: C.amber }}>Key finding — ECOG → RELAX (confidence: 0.62)</strong><br/>
                    Multiple competing Phase II/III trials use ECOG 0-2. NCT05382728 and NCT06838273 both allow ECOG 2, which could expand the eligible population by ~15-20%.<br/><br/>
                    Suggested wording: "ECOG performance status 0, 1, or 2 at screening"<br/><br/>
                    All other criteria: KEEP — aligned with or more rigorous than competing trial standards.
                  </div>
                </div>
              </div>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                      {["Criterion", "Label", "Confidence", "Evidence trials"].map(h => (
                        <th key={h} style={{ textAlign: "left", padding: "6px 10px", color: C.muted, fontWeight: 500 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {CI_ROWS.map((r, i) => (
                      <tr key={i} style={{ borderBottom: i < CI_ROWS.length - 1 ? `1px solid ${C.border}` : "none", background: r.label === "RELAX" ? `${C.amber}10` : "transparent" }}>
                        <td style={{ padding: "8px 10px", color: C.dim, maxWidth: 200 }}>{r.criterion}</td>
                        <td style={{ padding: "8px 10px" }}><Badge type={r.label} /></td>
                        <td style={{ padding: "8px 10px", color: r.conf >= 0.6 ? C.green : C.dim, fontFamily: "monospace" }}>{r.conf.toFixed(2)}</td>
                        <td style={{ padding: "8px 10px", color: C.muted, fontSize: 11, fontFamily: "monospace" }}>{r.trials}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ marginTop: 12, background: `${C.amber}10`, border: `1px solid ${C.amber}30`, borderRadius: 8, padding: "8px 12px", fontSize: 12, color: C.dim, lineHeight: 1.6 }}>
                Key signal: ECOG 0-1 restriction is more conservative than 8 of 20 competing trials which allow ECOG 0-2. Relaxing could increase eligible population by an estimated 15-20%.
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: RAG */}
        {tab === "rag" && (
          <div className="fade">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 16 }}>
              <MetricCard value="85" label="Chunks indexed" />
              <MetricCard value="19" label="Trials represented" />
              <MetricCard value="768" label="Embedding dims" />
              <MetricCard value="RRF k=60" label="Fusion method" />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
              {[
                { title: "Index configuration", rows: [
                  ["Embedding model", "PubMedBERT"],
                  ["Dense index", "FAISS IndexFlatIP"],
                  ["Sparse index", "BM25Okapi"],
                  ["Chunking strategy", "LLM criterion-level"],
                  ["Persistence", "FAISS + BM25 on disk"],
                  ["Total index size", "85 × 768 float32"],
                ]},
                { title: "Chunk breakdown", rows: [
                  ["Total chunks", "85"],
                  ["Inclusion criteria", "51 (60%)"],
                  ["Exclusion criteria", "34 (40%)"],
                  ["Treatment-related", "21 (25%)"],
                  ["Failed to parse", "2 trials (edge case)"],
                  ["Avg chunks/trial", "4.5"],
                ]},
              ].map(card => (
                <div key={card.title} style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>{card.title}</div>
                  {card.rows.map(([k, v]) => (
                    <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: `1px solid ${C.border}10`, fontSize: 12 }}>
                      <span style={{ color: C.muted }}>{k}</span>
                      <span style={{ color: C.text, fontWeight: 500, fontFamily: "monospace" }}>{v}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>

            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20, marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 14 }}>Per-criterion retrieval quality (Precision@5)</div>
              {RAG_CRITERIA.map(r => (
                <div key={r.label} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <div style={{ fontSize: 12, color: C.muted, minWidth: 220, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.label}</div>
                  <Bar value={r.p5} color={r.p5 >= 0.6 ? C.green : r.p5 >= 0.4 ? C.amber : C.muted} />
                  <div style={{ fontSize: 11, color: C.dim, minWidth: 32, textAlign: "right", fontFamily: "monospace" }}>{r.p5.toFixed(2)}</div>
                </div>
              ))}
              <div style={{ marginTop: 12, overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                      {["Criterion", "Precision@5", "MRR", "Top match"].map(h => (
                        <th key={h} style={{ textAlign: "left", padding: "6px 10px", color: C.muted, fontWeight: 500 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {RAG_CRITERIA.map((r, i) => (
                      <tr key={i} style={{ borderBottom: i < RAG_CRITERIA.length - 1 ? `1px solid ${C.border}` : "none" }}>
                        <td style={{ padding: "8px 10px", color: C.dim }}>{r.label}</td>
                        <td style={{ padding: "8px 10px", color: r.p5 >= 0.6 ? C.green : C.dim, fontFamily: "monospace" }}>{r.p5.toFixed(2)}</td>
                        <td style={{ padding: "8px 10px", color: C.dim, fontFamily: "monospace" }}>{r.mrr.toFixed(2)}</td>
                        <td style={{ padding: "8px 10px", color: C.muted, fontSize: 11, fontFamily: "monospace" }}>{r.top}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ marginTop: 12, background: `${C.accent}10`, border: `1px solid ${C.accent}30`, borderRadius: 8, padding: "8px 12px", fontSize: 12, color: C.dim, lineHeight: 1.6 }}>
                Best retrieval on standard disease/measurement criteria (RECIST, EGFR, ECOG). Lower precision on safety exclusions due to heterogeneous phrasing across competing trial protocols.
              </div>
            </div>

            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 14 }}>RRF score distribution — top retrieved chunks</div>
              {RRF_BARS.map(r => (
                <div key={r.label} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <div style={{ fontSize: 12, color: C.muted, minWidth: 260, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.label}</div>
                  <Bar value={r.score} max={0.034} color={C.green} />
                  <div style={{ fontSize: 11, color: C.dim, minWidth: 48, textAlign: "right", fontFamily: "monospace" }}>{r.score.toFixed(4)}</div>
                </div>
              ))}
              <div style={{ marginTop: 12, fontSize: 12, color: C.muted, lineHeight: 1.6 }}>
                RRF score = Σ 1/(60 + rank). Max theoretical ≈ 0.033. Scores cluster near max for semantically clear criteria.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
