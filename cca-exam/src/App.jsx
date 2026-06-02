import { useState, useEffect, useCallback, useRef } from "react";

const EXAM_KEY  = "cca-exam-v6";
const BATCH_KEY = "cca-batch-progress-v6";

const DOMAIN_NAMES = {
  1: "Agentic Architecture & Orchestration",
  2: "Tool Design & MCP Integration",
  3: "Claude Code Configuration & Workflows",
  4: "Prompt Engineering & Structured Output",
  5: "Context Management & Reliability"
};
const DOMAIN_COLORS  = { 1:"#e8855a", 2:"#5a9ee8", 3:"#5ae8a0", 4:"#c85ae8", 5:"#e8c85a" };
const DOMAIN_WEIGHTS = { 1:"27%", 2:"18%", 3:"20%", 4:"20%", 5:"15%" };

const BATCHES = [
  [[1,1],[2,2],[3,1],[4,3],[5,2]],
  [[6,1],[7,4],[8,3],[9,2],[10,4]],
  [[11,1],[12,3],[13,5],[14,4],[15,1]],
  [[16,2],[17,4],[18,5],[19,3],[20,1]],
  [[21,4],[22,5],[23,2],[24,3],[25,1]]
];

const BATCH_SYS = "Return ONLY a raw JSON array. No markdown, no code fences, no explanation. Start with [ end with ].";

function makePrompt(items) {
  const lines = items.map(([id, d]) => `ID ${id}: Domain ${d} (${DOMAIN_NAMES[d]})`).join("\n");
  return `Write exactly ${items.length} CCA-F certification exam questions about building Claude-based AI systems.

Questions:
${lines}

Requirements:
- Generic SaaS/e-commerce/devtools/enterprise scenarios only. No healthcare.
- 4 options each, exactly one correct answer.
- Wrong options must be plausible mistakes, not obviously wrong.
- Test architectural judgment, not API memorisation.
- "explanation": 1-2 sentences why correct answer is right.
- "antipattern_index": index (0-3) of the most dangerous wrong option.
- "antipattern_reason": 1 sentence why it is an anti-pattern (use terms: context bloat, non-idempotent, missing circuit breaker, unbounded retry, prompt injection, context saturation).

Output a JSON array of exactly ${items.length} objects:
[{"id":N,"domain":N,"scenario":"name","question":"text","options":["A","B","C","D"],"answer":N,"explanation":"text","antipattern_index":N,"antipattern_reason":"text"}]`;
}

// ── Calls YOUR proxy, not Anthropic directly ──────────────────────────────────
async function callBatch(items) {
  const res = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 2500,
      system: BATCH_SYS,
      messages: [{ role: "user", content: makePrompt(items) }]
    })
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error?.message || data.error || `HTTP ${res.status}`);
  const raw = (data.content?.find(b => b.type === "text")?.text || "").trim()
    .replace(/^```json\s*/, "").replace(/^```\s*/, "").replace(/\s*```$/, "").trim();
  return JSON.parse(raw);
}

// ── Markdown export ───────────────────────────────────────────────────────────
function buildMarkdown(questions) {
  const ts = new Date().toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });
  const L = ["A","B","C","D"];
  const qSec = questions.map((q, i) => {
    const opts = q.options.map((o, oi) => `  - **${L[oi]}.** ${o}`).join("\n");
    return `### Q${i+1}. [Domain ${q.domain} — ${DOMAIN_NAMES[q.domain]}]\n**Scenario:** ${q.scenario}\n\n${q.question}\n\n${opts}`;
  }).join("\n\n---\n\n");
  const sSec = questions.map((q, i) => {
    const aL = L[q.antipattern_index] || "N/A";
    const aO = q.options[q.antipattern_index] || "N/A";
    return `### Q${i+1}. ${q.question.slice(0,75)}${q.question.length > 75 ? "…" : ""}\n\n**✅ Correct: ${L[q.answer]}** — ${q.options[q.answer]}\n\n${q.explanation}\n\n**⚠️ Anti-Pattern: ${aL}** — ${aO}\n\n${q.antipattern_reason || "N/A"}`;
  }).join("\n\n---\n\n");
  return `# CCA-F Mock Exam — Questions & Solutions\n_Generated: ${ts}_\n\n---\n\n## Part 1 — Questions\n\n${qSec}\n\n---\n\n## Part 2 — Solutions\n\n${sSec}\n`;
}

function downloadMd(questions) {
  const ts   = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const blob = new Blob([buildMarkdown(questions)], { type: "text/markdown" });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement("a"), { href: url, download: `qna_${ts}.md` });
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
const fmt     = s   => `${String(Math.floor(s/60)).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`;
const fmtDate = iso => iso ? new Date(iso).toLocaleString("en-IN",{day:"numeric",month:"short",hour:"2-digit",minute:"2-digit"}) : null;

const GEN_MSGS = [
  "Generating batch 1 of 5...", "Generating batch 2 of 5...",
  "Generating batch 3 of 5...", "Generating batch 4 of 5...",
  "Generating batch 5 of 5...", "Finalising questions..."
];

// ── Styles ────────────────────────────────────────────────────────────────────
const S = {
  shell:        { minHeight:"100vh", background:"#0a0a0f", fontFamily:"Georgia,serif", color:"#e0e0e0", position:"relative", overflowX:"hidden" },
  bg:           { position:"fixed", inset:0, background:"radial-gradient(ellipse at 20% 50%,#1a0a2e,transparent 60%),radial-gradient(ellipse at 80% 20%,#0a1a2e,transparent 60%)", pointerEvents:"none", zIndex:0 },
  center:       { display:"flex", alignItems:"center", justifyContent:"center" },
  card:         { position:"relative", zIndex:1, maxWidth:600, margin:"0 auto", padding:"60px 40px", display:"flex", flexDirection:"column", alignItems:"center", gap:24 },
  badge:        { fontFamily:"monospace", fontSize:11, letterSpacing:4, color:"#e8855a", borderTop:"1px solid #e8855a40", borderBottom:"1px solid #e8855a40", padding:"4px 16px" },
  h1:           { fontSize:36, fontWeight:"normal", textAlign:"center", margin:0, color:"#fff", lineHeight:1.2 },
  sub:          { fontFamily:"monospace", color:"#666", fontSize:13, letterSpacing:2 },
  grid4:        { display:"grid", gridTemplateColumns:"1fr 1fr 1fr 1fr", gap:12, width:"100%" },
  specBox:      { background:"#ffffff08", border:"1px solid #ffffff15", borderRadius:8, padding:"16px 12px", textAlign:"center" },
  specV:        { fontSize:20, fontWeight:"bold", color:"#fff", fontFamily:"monospace" },
  specK:        { fontSize:10, color:"#666", marginTop:4, letterSpacing:1 },
  dList:        { width:"100%", display:"flex", flexDirection:"column", gap:8 },
  dRow:         { display:"flex", alignItems:"center", gap:10 },
  dot:          { width:8, height:8, borderRadius:"50%", flexShrink:0 },
  resumeBox:    { width:"100%", background:"#ffffff08", border:"1px solid #e8c85a30", borderRadius:10, padding:"20px 24px", display:"flex", flexDirection:"column", gap:10 },
  resumeTitle:  { fontFamily:"monospace", fontSize:13, color:"#e8c85a", letterSpacing:1 },
  resumeSub:    { fontSize:12, color:"#666" },
  batchBox:     { width:"100%", background:"#ffffff08", border:"1px solid #5a9ee830", borderRadius:10, padding:"16px 20px", display:"flex", flexDirection:"column", gap:8 },
  batchTitle:   { fontFamily:"monospace", fontSize:12, color:"#5a9ee8", letterSpacing:1 },
  btnPrimary:   { padding:"10px 28px", background:"#e8855a", border:"none", borderRadius:6, color:"#fff", fontSize:14, cursor:"pointer", fontFamily:"monospace" },
  btnGhost:     { padding:"10px 20px", background:"transparent", border:"1px solid #ffffff20", borderRadius:6, color:"#666", fontSize:14, cursor:"pointer", fontFamily:"monospace" },
  btnStart:     { padding:"14px 48px", background:"#e8855a", border:"none", borderRadius:6, color:"#fff", fontSize:15, cursor:"pointer", letterSpacing:1, fontFamily:"monospace" },
  btnBlue:      { padding:"10px 20px", background:"transparent", border:"1px solid #5a9ee860", color:"#5a9ee8", borderRadius:6, cursor:"pointer", fontFamily:"monospace", fontSize:13 },
  btnDl:        { padding:"10px 20px", background:"transparent", border:"1px solid #5a9ee860", color:"#5a9ee8", borderRadius:6, cursor:"pointer", fontFamily:"monospace", fontSize:13 },
  btnDlSm:      { padding:"8px", width:"100%", background:"transparent", border:"1px solid #5a9ee840", color:"#5a9ee8", borderRadius:6, cursor:"pointer", fontFamily:"monospace", fontSize:11 },
  btnSubmit:    { padding:"10px", background:"transparent", border:"1px solid #e8855a60", color:"#e8855a", borderRadius:6, cursor:"pointer", fontFamily:"monospace", fontSize:12 },
  errBox:       { width:"100%", background:"#e85a5a10", border:"1px solid #e85a5a30", borderRadius:8, padding:"14px 18px" },
  spinner:      { width:36, height:36, border:"2px solid #ffffff10", borderTop:"2px solid #e8855a", borderRadius:"50%", animation:"spin .8s linear infinite" },
  examLayout:   { display:"flex", minHeight:"100vh", position:"relative", zIndex:1 },
  sidebar:      { width:220, flexShrink:0, background:"#0d0d14", borderRight:"1px solid #ffffff10", padding:20, display:"flex", flexDirection:"column", gap:14, position:"sticky", top:0, height:"100vh", overflowY:"auto" },
  timerBox:     { textAlign:"center", padding:"12px 0", borderBottom:"1px solid #ffffff10" },
  timerLbl:     { fontSize:10, color:"#555", letterSpacing:2, fontFamily:"monospace" },
  saveInd:      { display:"block", fontSize:10, fontFamily:"monospace", marginTop:6, color:"#555" },
  sbLbl:        { fontSize:10, letterSpacing:2, color:"#555", fontFamily:"monospace" },
  qGridWrap:    { display:"grid", gridTemplateColumns:"repeat(5,1fr)", gap:4 },
  legend:       { display:"flex", flexDirection:"column", gap:6 },
  legendRow:    { display:"flex", alignItems:"center", gap:8, fontSize:10, color:"#555" },
  legendDot:    { width:12, height:12, borderRadius:2, flexShrink:0 },
  qPanel:       { flex:1, padding:"40px 48px", maxWidth:800 },
  qMeta:        { display:"flex", alignItems:"center", gap:10, flexWrap:"wrap", marginBottom:16 },
  dTag:         { fontSize:11, fontFamily:"monospace", padding:"3px 10px", borderRadius:20, letterSpacing:.5 },
  flagBtn:      { marginLeft:"auto", background:"none", border:"none", cursor:"pointer", fontSize:12, fontFamily:"monospace" },
  qNum:         { fontSize:11, color:"#555", fontFamily:"monospace", marginBottom:12, letterSpacing:1 },
  qText:        { fontSize:17, lineHeight:1.7, color:"#e0e0e0", marginBottom:28, borderLeft:"2px solid #e8855a30", paddingLeft:16 },
  optList:      { display:"flex", flexDirection:"column", gap:10 },
  optBtn:       { display:"flex", alignItems:"flex-start", gap:14, padding:"14px 16px", borderRadius:8, cursor:"pointer", textAlign:"left" },
  optLetter:    { width:24, height:24, borderRadius:4, display:"flex", alignItems:"center", justifyContent:"center", fontSize:12, fontFamily:"monospace", flexShrink:0, marginTop:1 },
  navRow:       { display:"flex", alignItems:"center", marginTop:32, gap:12 },
  btnNav:       { padding:"10px 20px", background:"transparent", border:"1px solid #ffffff20", color:"#888", borderRadius:6, cursor:"pointer", fontFamily:"monospace", fontSize:13 },
  btnNext:      { padding:"10px 28px", background:"#e8855a", border:"none", color:"#fff", borderRadius:6, cursor:"pointer", fontFamily:"monospace", fontSize:13 },
  reviewWrap:   { position:"relative", zIndex:1, maxWidth:820, margin:"0 auto", padding:"48px 32px", display:"flex", flexDirection:"column", gap:24 },
  scoreCard:    { background:"#0d0d14", border:"1px solid #ffffff15", borderRadius:12, padding:"40px 32px", textAlign:"center" },
  scoreBig:     { fontSize:72, fontFamily:"monospace", fontWeight:"bold", lineHeight:1, marginTop:8 },
  passBadge:    { display:"inline-block", marginTop:16, padding:"6px 20px", borderRadius:20, fontFamily:"monospace", fontSize:14, letterSpacing:2 },
  domBreak:     { background:"#0d0d14", border:"1px solid #ffffff15", borderRadius:12, padding:"24px 28px", display:"flex", flexDirection:"column", gap:14 },
  secTitle:     { fontSize:11, letterSpacing:3, color:"#555", fontFamily:"monospace", textTransform:"uppercase", marginBottom:4 },
  domResRow:    { display:"flex", alignItems:"center", gap:12 },
  domResName:   { fontSize:13, color:"#bbb", width:260, flexShrink:0 },
  domBar:       { flex:1, height:6, background:"#ffffff10", borderRadius:3, overflow:"hidden" },
  domBarFill:   { height:"100%", borderRadius:3 },
  rQ:           { background:"#0d0d14", borderRadius:10, padding:"20px 24px", display:"flex", flexDirection:"column", gap:10 },
  rOpt:         { padding:"8px 12px", borderRadius:6, fontSize:13, color:"#aaa", lineHeight:1.5 },
  explBtn:      { alignSelf:"flex-start", padding:"6px 14px", background:"transparent", border:"1px solid #ffffff20", color:"#666", borderRadius:4, cursor:"pointer", fontFamily:"monospace", fontSize:11 },
  explBox:      { fontSize:13, lineHeight:1.7, color:"#aaa", background:"#ffffff05", borderRadius:6, padding:"12px 14px", borderLeft:"2px solid #5ae8a060" },
  apBox:        { fontSize:13, lineHeight:1.7, color:"#aaa", background:"#ffffff05", borderRadius:6, padding:"12px 14px", borderLeft:"2px solid #e8c85a60", marginTop:8 },
};

// ── Component ─────────────────────────────────────────────────────────────────
export default function App() {
  const [phase,          setPhase]          = useState("loading");
  const [questions,      setQuestions]      = useState([]);
  const [current,        setCurrent]        = useState(0);
  const [selected,       setSelected]       = useState({});
  const [flagged,        setFlagged]        = useState({});
  const [timeLeft,       setTimeLeft]       = useState(50 * 60);
  const [timerOn,        setTimerOn]        = useState(false);
  const [showExpl,       setShowExpl]       = useState({});
  const [savedAt,        setSavedAt]        = useState(null);
  const [saveStatus,     setSaveStatus]     = useState("idle");
  const [hasSaved,       setHasSaved]       = useState(false);
  const [savedPhase,     setSavedPhase]     = useState("exam");
  const [genStep,        setGenStep]        = useState(0);
  const [genErr,         setGenErr]         = useState(null);
  const [partialBatches, setPartialBatches] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [examR, batchR] = await Promise.all([
          window.localStorage.getItem(EXAM_KEY),
          window.localStorage.getItem(BATCH_KEY)
        ]);
        if (examR) {
          const s = JSON.parse(examR);
          if ((s.phase === "exam" || s.phase === "review") && s.questions?.length >= 20) {
            setHasSaved(true); setSavedAt(s.savedAt); setSavedPhase(s.phase);
          }
        }
        if (batchR) {
          const b = JSON.parse(batchR);
          if (Array.isArray(b.questions) && b.questions.length > 0 && b.completedBatches < BATCHES.length) {
            setPartialBatches(b);
          }
        }
      } catch {}
      setPhase("intro");
    }
    load();
  }, []);

  const persist = useCallback((state) => {
    if (!state.phase || ["intro","generating","loading"].includes(state.phase)) return;
    setSaveStatus("saving");
    try {
      window.localStorage.setItem(EXAM_KEY, JSON.stringify({ ...state, savedAt: new Date().toISOString() }));
      setSaveStatus("saved"); setSavedAt(new Date().toISOString());
      setTimeout(() => setSaveStatus("idle"), 2000);
    } catch { setSaveStatus("error"); setTimeout(() => setSaveStatus("idle"), 3000); }
  }, []);

  useEffect(() => {
    if (phase !== "exam") return;
    const t = setTimeout(() => persist({ phase, current, selected, flagged, timeLeft, questions, savedAt }), 800);
    return () => clearTimeout(t);
  }, [selected, flagged, current]);

  useEffect(() => {
    if (phase === "review") persist({ phase, current, selected, flagged, timeLeft: 0, questions, savedAt });
  }, [phase]);

  useEffect(() => {
    if (!timerOn) return;
    if (timeLeft <= 0) { setTimerOn(false); setPhase("review"); return; }
    const t = setInterval(() => setTimeLeft(s => {
      const n = s - 1;
      if (n % 30 === 0) persist({ phase, current, selected, flagged, timeLeft: n, questions, savedAt });
      return n;
    }), 1000);
    return () => clearInterval(t);
  }, [timerOn, timeLeft, phase]);

  async function generate(resumeFrom = null) {
    setPhase("generating"); setGenErr(null);
    let accQs      = resumeFrom ? [...resumeFrom.questions] : [];
    let startBatch = resumeFrom ? resumeFrom.completedBatches : 0;
    setGenStep(startBatch);

    try {
      for (let i = startBatch; i < BATCHES.length; i++) {
        setGenStep(i);
        let batch;
        try {
          batch = await callBatch(BATCHES[i]);
          if (!Array.isArray(batch)) throw new Error("Invalid response");
        } catch (batchErr) {
          if (accQs.length > 0) {
            const partial = { questions: accQs, completedBatches: i };
            window.localStorage.setItem(BATCH_KEY, JSON.stringify(partial));
            setPartialBatches(partial);
          }
          throw new Error(`Batch ${i+1} failed: ${batchErr.message}`);
        }
        accQs = [...accQs, ...batch];
        const progress = { questions: accQs, completedBatches: i + 1 };
        window.localStorage.setItem(BATCH_KEY, JSON.stringify(progress));
        setPartialBatches(progress);
      }

      setGenStep(5);
      window.localStorage.removeItem(BATCH_KEY);
      setPartialBatches(null);

      const qs = accQs.slice(0, 25).map((q, i) => ({ ...q, id: i + 1 }));
      setQuestions(qs); setPhase("exam"); setCurrent(0);
      setSelected({}); setFlagged({}); setTimeLeft(50 * 60); setTimerOn(true);
    } catch (e) {
      setGenErr(e.message || "Generation failed");
      setPhase("intro");
    }
  }

  function resumeExam() {
    try {
      const raw = window.localStorage.getItem(EXAM_KEY);
      if (raw) {
        const s = JSON.parse(raw);
        setQuestions(s.questions || []); setPhase(s.phase || "exam");
        setCurrent(s.current || 0); setSelected(s.selected || {});
        setFlagged(s.flagged || {}); setTimeLeft(s.timeLeft ?? 50 * 60);
        setSavedAt(s.savedAt); setSavedPhase(s.phase || "exam");
        if (s.phase === "exam") setTimerOn(true);
      }
    } catch { generate(); }
  }

  function clearAll() {
    window.localStorage.removeItem(EXAM_KEY);
    window.localStorage.removeItem(BATCH_KEY);
    setHasSaved(false); setSavedAt(null); setSavedPhase("exam");
    setPartialBatches(null); setShowExpl({}); setTimerOn(false); setPhase("intro");
  }

  const timerColor = timeLeft < 300 ? "#e85a5a" : timeLeft < 600 ? "#e8c85a" : "#5ae8a0";
  const score      = questions.filter(q => selected[q.id] === q.answer).length;
  const pct        = questions.length > 0 ? Math.round((score / questions.length) * 1000) / 10 : 0;
  const scaled     = Math.round(100 + (pct / 100) * 900);
  const passed     = scaled >= 720;
  const domRes     = Object.keys(DOMAIN_NAMES).map(d => {
    const qs = questions.filter(q => q.domain === Number(d));
    const c  = qs.filter(q => selected[q.id] === q.answer).length;
    return { domain: Number(d), correct: c, total: qs.length, pct: qs.length > 0 ? Math.round((c / qs.length) * 100) : 0 };
  });

  const L = ["A","B","C","D"];

  const SaveInd = () => {
    if (saveStatus === "saving") return <span style={S.saveInd}>⟳ saving...</span>;
    if (saveStatus === "saved")  return <span style={{...S.saveInd, color:"#5ae8a0"}}>✓ saved</span>;
    if (saveStatus === "error")  return <span style={{...S.saveInd, color:"#e85a5a"}}>✗ failed</span>;
    if (savedAt) return <span style={S.saveInd}>saved {fmtDate(savedAt)}</span>;
    return null;
  };

  // ── LOADING ──
  if (phase === "loading") return (
    <div style={{...S.shell, ...S.center}}>
      <div style={S.bg}/>
      <span style={{fontFamily:"monospace", color:"#555", fontSize:13, position:"relative", zIndex:1}}>Loading...</span>
    </div>
  );

  // ── GENERATING ──
  if (phase === "generating") return (
    <div style={{...S.shell, ...S.center}}>
      <div style={S.bg}/>
      <div style={{position:"relative", zIndex:1, textAlign:"center", display:"flex", flexDirection:"column", alignItems:"center", gap:20, maxWidth:340}}>
        <div style={S.spinner}/>
        <div style={{fontFamily:"monospace", fontSize:13, color:"#e8855a", letterSpacing:1}}>GENERATING QUESTIONS</div>
        <div style={{fontFamily:"monospace", fontSize:12, color:"#555", minHeight:18}}>{GEN_MSGS[Math.min(genStep, 5)]}</div>
        <div style={{display:"flex", gap:6}}>
          {BATCHES.map((_, i) => (
            <div key={i} style={{width:32, height:4, borderRadius:2,
              background: genStep > i ? "#5ae8a0" : genStep === i ? "#e8855a" : "#ffffff15",
              transition:"background .3s"}}/>
          ))}
        </div>
        <div style={{fontFamily:"monospace", fontSize:11, color:"#444"}}>
          {genStep < BATCHES.length ? `${genStep * 5} / 25 questions saved` : "Finalising..."}
        </div>
      </div>
    </div>
  );

  // ── INTRO ──
  if (phase === "intro") return (
    <div style={S.shell}>
      <div style={S.bg}/>
      <div style={S.card}>
        <div style={S.badge}>MOCK EXAM</div>
        <h1 style={S.h1}>Claude Certified Architect</h1>
        <div style={S.sub}>Foundations · CCA-F</div>
        <div style={S.grid4}>
          {[["Questions","25"],["Duration","50 min"],["Pass Score","720/1000"],["Format","MCQ · 1 correct"]].map(([k,v]) => (
            <div key={k} style={S.specBox}><div style={S.specV}>{v}</div><div style={S.specK}>{k}</div></div>
          ))}
        </div>
        <div style={S.dList}>
          {Object.entries(DOMAIN_NAMES).map(([d, name]) => (
            <div key={d} style={S.dRow}>
              <div style={{...S.dot, background: DOMAIN_COLORS[d]}}/>
              <span style={{fontSize:13, color:"#bbb", flex:1}}>{name}</span>
              <span style={{fontSize:12, fontFamily:"monospace", color:"#666"}}>{DOMAIN_WEIGHTS[d]}</span>
            </div>
          ))}
        </div>

        {genErr && (
          <div style={S.errBox}>
            <div style={{fontFamily:"monospace", fontSize:12, color:"#e85a5a"}}>{genErr}</div>
            {partialBatches && partialBatches.completedBatches > 0 && (
              <div style={{marginTop:10, display:"flex", gap:10, alignItems:"center", flexWrap:"wrap"}}>
                <span style={{fontSize:11, color:"#888"}}>{partialBatches.questions.length} questions saved from {partialBatches.completedBatches} batch{partialBatches.completedBatches > 1 ? "es" : ""}.</span>
                <button style={{...S.btnBlue, padding:"6px 14px", fontSize:12}} onClick={() => generate(partialBatches)}>Resume generation →</button>
              </div>
            )}
          </div>
        )}

        {!genErr && partialBatches && partialBatches.completedBatches > 0 && (
          <div style={S.batchBox}>
            <div style={S.batchTitle}>Incomplete generation found</div>
            <div style={{fontSize:12, color:"#666"}}>{partialBatches.questions.length} questions saved · batch {partialBatches.completedBatches} of {BATCHES.length} complete</div>
            <div style={{display:"flex", gap:10, marginTop:4}}>
              <button style={S.btnBlue} onClick={() => generate(partialBatches)}>Continue generation →</button>
              <button style={S.btnGhost} onClick={() => { window.localStorage.removeItem(BATCH_KEY); setPartialBatches(null); }}>Discard & start fresh</button>
            </div>
          </div>
        )}

        {hasSaved && (
          <div style={S.resumeBox}>
            <div style={S.resumeTitle}>{savedPhase === "review" ? "Completed exam found" : "Saved progress found"}</div>
            <div style={S.resumeSub}>Last saved {fmtDate(savedAt)}</div>
            <div style={{display:"flex", gap:12, marginTop:4}}>
              <button style={S.btnPrimary} onClick={resumeExam}>{savedPhase === "review" ? "View Results →" : "Resume →"}</button>
              <button style={S.btnGhost} onClick={clearAll}>Clear all & start fresh</button>
            </div>
          </div>
        )}

        <div style={{display:"flex", flexDirection:"column", alignItems:"center", gap:8, width:"100%"}}>
          <button style={S.btnStart} onClick={() => generate(null)}>Generate New Exam →</button>
          <div style={{fontFamily:"monospace", fontSize:10, color:"#333", letterSpacing:1}}>5 batches · each saved on completion</div>
        </div>
      </div>
    </div>
  );

  // ── REVIEW ──
  if (phase === "review") return (
    <div style={S.shell}>
      <div style={S.bg}/>
      <div style={S.reviewWrap}>
        <div style={S.scoreCard}>
          <div style={{fontSize:11, letterSpacing:3, color:"#555", fontFamily:"monospace"}}>YOUR SCORE</div>
          <div style={{...S.scoreBig, color: passed ? "#5ae8a0" : "#e85a5a"}}>{scaled}</div>
          <div style={{color:"#555", fontSize:13, fontFamily:"monospace"}}>out of 1000</div>
          <div style={{...S.passBadge, background: passed ? "#5ae8a020" : "#e85a5a20", color: passed ? "#5ae8a0" : "#e85a5a", border:`1px solid ${passed ? "#5ae8a0" : "#e85a5a"}`}}>
            {passed ? "✓ PASS" : "✗ BELOW PASSING (720)"}
          </div>
          <div style={{marginTop:12, color:"#666", fontFamily:"monospace", fontSize:13}}>{score} / {questions.length} correct ({pct}%)</div>
          <div style={{marginTop:16}}><button style={S.btnDl} onClick={() => downloadMd(questions)}>↓ Download Q&A (.md)</button></div>
        </div>

        <div style={S.domBreak}>
          <div style={S.secTitle}>Domain Breakdown</div>
          {domRes.map(r => (
            <div key={r.domain} style={S.domResRow}>
              <div style={{...S.dot, background: DOMAIN_COLORS[r.domain]}}/>
              <div style={S.domResName}>{DOMAIN_NAMES[r.domain]}</div>
              <div style={S.domBar}><div style={{...S.domBarFill, width:`${r.pct}%`, background: DOMAIN_COLORS[r.domain]}}/></div>
              <div style={{fontFamily:"monospace", fontSize:12, color:"#666", width:36, textAlign:"right"}}>{r.correct}/{r.total}</div>
            </div>
          ))}
        </div>

        <div style={S.secTitle}>Question Review</div>
        {questions.map((qq, i) => {
          const ua = selected[qq.id]; const ok = ua === qq.answer; const show = showExpl[qq.id];
          return (
            <div key={qq.id} style={{...S.rQ, borderLeft:`3px solid ${ok ? "#5ae8a0" : "#e85a5a"}`}}>
              <div style={{display:"flex", alignItems:"center", gap:10}}>
                <span style={{fontFamily:"monospace", fontSize:13, fontWeight:"bold", color: ok ? "#5ae8a0" : "#e85a5a"}}>{ok ? "✓" : "✗"} Q{i+1}</span>
                <span style={{...S.dTag, background: DOMAIN_COLORS[qq.domain]+"22", color: DOMAIN_COLORS[qq.domain]}}>{DOMAIN_NAMES[qq.domain]}</span>
              </div>
              <div style={{fontSize:14, lineHeight:1.6, color:"#ccc"}}>{qq.question}</div>
              <div style={{display:"flex", flexDirection:"column", gap:6}}>
                {qq.options.map((opt, oi) => {
                  const isC = oi === qq.answer; const isU = oi === ua && !ok;
                  const isAP = oi === qq.antipattern_index && oi !== qq.answer;
                  return (
                    <div key={oi} style={{...S.rOpt,
                      background: isC ? "#5ae8a015" : isU ? "#e85a5a15" : isAP ? "#e8c85a08" : "transparent",
                      border: `1px solid ${isC ? "#5ae8a040" : isU ? "#e85a5a40" : isAP ? "#e8c85a30" : "#ffffff10"}`}}>
                      <span style={{color: isC ? "#5ae8a0" : isU ? "#e85a5a" : isAP ? "#e8c85a" : "#555", marginRight:6}}>
                        {isC ? "✓" : isU ? "✗" : isAP ? "⚠" : L[oi]}
                      </span>
                      {opt}
                      {isAP && <span style={{fontFamily:"monospace", fontSize:10, color:"#e8c85a", marginLeft:8}}>anti-pattern</span>}
                    </div>
                  );
                })}
              </div>
              <button style={S.explBtn} onClick={() => setShowExpl(s => ({...s, [qq.id]: !s[qq.id]}))}>
                {show ? "Hide" : "Show"} Explanation & Anti-Pattern
              </button>
              {show && (
                <>
                  <div style={S.explBox}>
                    <span style={{color:"#5ae8a0", fontFamily:"monospace", fontSize:11}}>✓ CORRECT — Option {L[qq.answer]}</span>
                    <div style={{marginTop:6}}>{qq.explanation}</div>
                  </div>
                  {qq.antipattern_reason && (
                    <div style={S.apBox}>
                      <span style={{color:"#e8c85a", fontFamily:"monospace", fontSize:11}}>⚠ ANTI-PATTERN — Option {L[qq.antipattern_index]}</span>
                      <div style={{marginTop:6}}>{qq.antipattern_reason}</div>
                    </div>
                  )}
                </>
              )}
            </div>
          );
        })}
        <div style={{display:"flex", gap:12, flexWrap:"wrap"}}>
          <button style={S.btnStart} onClick={() => generate(null)}>Generate New Exam →</button>
          <button style={S.btnDl} onClick={() => downloadMd(questions)}>↓ Download Q&A (.md)</button>
          <button style={S.btnGhost} onClick={clearAll}>Back to Intro</button>
        </div>
      </div>
    </div>
  );

  // ── EXAM ──
  const q = questions[current];
  if (!q) return null;
  return (
    <div style={S.shell}>
      <div style={S.bg}/>
      <div style={S.examLayout}>
        <div style={S.sidebar}>
          <div style={S.timerBox}>
            <div style={S.timerLbl}>TIME LEFT</div>
            <div style={{fontSize:28, fontFamily:"monospace", fontWeight:"bold", marginTop:4, color:timerColor}}>{fmt(timeLeft)}</div>
            <SaveInd/>
          </div>
          <div style={S.sbLbl}>Questions</div>
          <div style={S.qGridWrap}>
            {questions.map((qq, i) => {
              const ans = selected[qq.id]; const iF = flagged[qq.id]; const iC = i === current;
              return (
                <button key={qq.id} onClick={() => setCurrent(i)} style={{width:"100%", aspectRatio:"1", borderRadius:4, cursor:"pointer", fontSize:11, fontFamily:"monospace",
                  background: iC ? "#e8855a" : ans !== undefined ? "#5ae8a040" : "#ffffff10",
                  border: `1px solid ${iC ? "#e8855a" : ans !== undefined ? "#5ae8a0" : "#ffffff20"}`,
                  color: iC ? "#fff" : ans !== undefined ? "#5ae8a0" : "#666",
                  outline: iF ? "2px solid #e8c85a" : "none"}}>{i+1}</button>
              );
            })}
          </div>
          <div style={S.legend}>
            <div style={S.legendRow}><div style={{...S.legendDot, background:"#5ae8a040", border:"1px solid #5ae8a0"}}/> Answered</div>
            <div style={S.legendRow}><div style={{...S.legendDot, background:"#ffffff10", border:"1px solid #ffffff20"}}/> Unanswered</div>
            <div style={S.legendRow}><div style={{...S.legendDot, background:"#ffffff10", outline:"2px solid #e8c85a"}}/> Flagged</div>
          </div>
          <div style={{fontFamily:"monospace", fontSize:11, color:"#666", textAlign:"center"}}>{Object.keys(selected).length} / {questions.length} answered</div>
          <button style={S.btnDlSm} onClick={() => downloadMd(questions)}>↓ Download Q&A (.md)</button>
          <button style={S.btnSubmit} onClick={() => { setTimerOn(false); setPhase("review"); }}>Submit Exam</button>
        </div>

        <div style={S.qPanel}>
          <div style={S.qMeta}>
            <span style={{...S.dTag, background: DOMAIN_COLORS[q.domain]+"22", color: DOMAIN_COLORS[q.domain]}}>{DOMAIN_NAMES[q.domain]}</span>
            <span style={{fontSize:11, color:"#555", fontFamily:"monospace"}}>📋 {q.scenario}</span>
            <button onClick={() => setFlagged(s => ({...s, [q.id]: !s[q.id]}))} style={{...S.flagBtn, color: flagged[q.id] ? "#e8c85a" : "#444"}}>
              {flagged[q.id] ? "🚩 Flagged" : "🏳 Flag"}
            </button>
          </div>
          <div style={S.qNum}>Question {current+1} of {questions.length}</div>
          <div style={S.qText}>{q.question}</div>
          <div style={S.optList}>
            {q.options.map((opt, oi) => {
              const sel = selected[q.id] === oi;
              return (
                <button key={oi} onClick={() => setSelected(s => ({...s, [q.id]: oi}))} style={{...S.optBtn,
                  background: sel ? "#e8855a15" : "#ffffff08",
                  border: `1px solid ${sel ? "#e8855a" : "#ffffff15"}`,
                  color: sel ? "#fff" : "#bbb"}}>
                  <span style={{...S.optLetter, background: sel ? "#e8855a" : "#ffffff15", color: sel ? "#fff" : "#888"}}>{L[oi]}</span>
                  <span style={{fontSize:14, lineHeight:1.6}}>{opt}</span>
                </button>
              );
            })}
          </div>
          <div style={S.navRow}>
            <button style={{...S.btnNav, opacity: current === 0 ? .3 : 1}} disabled={current === 0} onClick={() => setCurrent(c => c-1)}>← Previous</button>
            <div style={{marginLeft:"auto"}}>
              {current < questions.length - 1
                ? <button style={S.btnNext} onClick={() => setCurrent(c => c+1)}>Next →</button>
                : <button style={S.btnNext} onClick={() => { setTimerOn(false); setPhase("review"); }}>Finish & Review</button>
              }
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
