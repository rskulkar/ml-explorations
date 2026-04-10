"""
ci_agent_prompt.py

Competitive Intelligence (CI) agent for clinical trial I/E criteria analysis.

Architecture:
    1. fetch_competing_trials(ie_criteria)  — Python wrapper, calls ClinicalTrials.gov API v2,
                                              returns a pre-processed markdown table string
    2. get_ci_agent_prompt(ie_criteria,
                           competing_trials_md) — prompt function, same contract as
                                                   get_soc_agent_prompt / get_et_agent_prompt

Usage (mirrors notebook pattern):
    competing_trials_md = fetch_competing_trials(ie_criteria)
    ci_prompt = get_ci_agent_prompt(ie_criteria, competing_trials_md)

    response = client.chat.completions.create(
        model="claude-4-sonnet",
        messages=[{"role": "user", "content": ci_prompt}]
    )

Output table columns match SOC / ET agents for downstream concatenation:
    Type | Original Criterion | Gap Type | Queriable | Competing Trial |
    Competitive Position | Explanation | Recommendation
"""

import re
import time
import requests
import pandas as pd


# ---------------------------------------------------------------------------
# ClinicalTrials.gov API v2 wrapper
# ---------------------------------------------------------------------------

CT_API_BASE = "https://clinicaltrials.gov/api/v2/studies"

# Fields we request from the API — keeps payload small
CT_FIELDS = ",".join([
    "NCTId",
    "BriefTitle",
    "Phase",
    "OverallStatus",
    "LeadSponsorName",
    "Condition",
    "InterventionName",
    "EligibilityCriteria",
    "PrimaryOutcomeMeasure",
    "EnrollmentCount",
    "StartDate",
    "CompletionDate",
])


def _extract_search_terms(ie_criteria: str) -> dict:
    """
    Derive ClinicalTrials.gov search parameters from I/E criteria text.

    Extracts:
        condition  — disease + biomarker context (e.g. "EGFR mutant NSCLC")
        intr       — drug or therapy class (e.g. "amivantamab osimertinib")

    Strategy: regex for common oncology biomarker/disease patterns + drug names.
    Deliberately conservative — better to over-fetch and filter than miss trials.
    """
    text = ie_criteria.lower()

    # --- Disease / condition ---
    condition_hints = []

    disease_patterns = [
        r"(non.?small.?cell lung cancer|nsclc)",
        r"(small.?cell lung cancer|sclc)",
        r"(breast cancer)",
        r"(colorectal cancer|crc)",
        r"(ovarian cancer)",
        r"(prostate cancer)",
        r"(bladder cancer|urothelial)",
        r"(pancreatic cancer)",
        r"(hepatocellular carcinoma|hcc)",
        r"(gastric.?cancer|gastroesophageal)",
        r"(melanoma)",
        r"(glioblastoma|glioma)",
        r"(multiple myeloma)",
        r"(lymphoma)",
        r"(leukemia)",
        r"(renal cell carcinoma|rcc)",
    ]
    for pat in disease_patterns:
        m = re.search(pat, text)
        if m:
            condition_hints.append(m.group(1).strip())
            break  # take first match only

    biomarker_patterns = [
        r"(egfr[^\s,;]*(?:exon\s*\d+[^\s,;]*)?)",
        r"(kras[^\s,;]*)",
        r"(alk[^\s,;]*)",
        r"(ros1[^\s,;]*)",
        r"(her2[^\s,;]*|erbb2[^\s,;]*)",
        r"(braf[^\s,;]*)",
        r"(met[^\s,;]*(?:exon\s*\d+[^\s,;]*)?)",
        r"(pd.?l1[^\s,;]*)",
        r"(brca[^\s,;]*)",
        r"(fgfr[^\s,;]*)",
        r"(ntrk[^\s,;]*)",
        r"(ret[^\s,;]*)",
    ]
    for pat in biomarker_patterns:
        m = re.search(pat, text)
        if m:
            condition_hints.append(m.group(1).strip())

    condition_query = " ".join(condition_hints[:3]) if condition_hints else ""

    # --- Intervention / drug ---
    # Named drugs — expand as needed for other indications
    drug_patterns = [
        r"\b(osimertinib|tagrisso)\b",
        r"\b(amivantamab|rybrevant)\b",
        r"\b(lazertinib|lazcluze)\b",
        r"\b(erlotinib|tarceva)\b",
        r"\b(gefitinib|iressa)\b",
        r"\b(afatinib|gilotrif)\b",
        r"\b(dacomitinib)\b",
        r"\b(pembrolizumab|keytruda)\b",
        r"\b(nivolumab|opdivo)\b",
        r"\b(atezolizumab|tecentriq)\b",
        r"\b(durvalumab|imfinzi)\b",
        r"\b(carboplatin)\b",
        r"\b(pemetrexed|alimta)\b",
        r"\b(docetaxel|taxotere)\b",
        r"\b(paclitaxel|taxol)\b",
        r"\b(bevacizumab|avastin)\b",
        r"\b(sotorasib|lumakras)\b",
        r"\b(adagrasib|krazati)\b",
        r"\b(capmatinib|tabrecta)\b",
        r"\b(tepotinib|tepmetko)\b",
        r"\b(crizotinib|xalkori)\b",
        r"\b(alectinib|alecensa)\b",
        r"\b(lorlatinib|lorbrena)\b",
    ]
    drug_hits = []
    for pat in drug_patterns:
        m = re.search(pat, text)
        if m:
            drug_hits.append(m.group(1))

    # Class-level fallback
    if not drug_hits:
        if re.search(r"\btki\b|tyrosine kinase inhibitor", text):
            drug_hits.append("tyrosine kinase inhibitor")
        if re.search(r"\bimmunotherapy\b|checkpoint inhibitor", text):
            drug_hits.append("immunotherapy")
        if re.search(r"\bchemotherapy\b", text):
            drug_hits.append("chemotherapy")

    intr_query = " ".join(drug_hits[:3]) if drug_hits else ""

    return {
        "condition": condition_query,
        "intr": intr_query,
    }


def fetch_competing_trials(
    ie_criteria: str,
    phases: list[str] | None = None,
    statuses: list[str] | None = None,
    page_size: int = 20,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    Fetch competing trials from ClinicalTrials.gov API v2.

    Derives search terms from ie_criteria, queries the API, and returns
    a pre-processed markdown table string ready to pass into get_ci_agent_prompt().

    Args:
        ie_criteria:  Raw I/E criteria text (same input as SOC/ET agents).
        phases:       CT phases to include. Default: Phase 2, Phase 3.
        statuses:     Trial statuses to include. Default: Recruiting,
                      Active not recruiting, Completed.
        page_size:    Max trials to retrieve (default 20).
        max_retries:  Retry count on transient API errors.
        retry_delay:  Seconds between retries.

    Returns:
        Markdown table string with columns:
            NCT ID | Trial Name | Phase | Status | Sponsor |
            Intervention | Primary Endpoint | N | I/E Criteria (truncated)

        Returns an empty-table string if no trials found or API unreachable.
    """
    if phases is None:
        phases = ["PHASE2", "PHASE3"]
    if statuses is None:
        statuses = ["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"]

    terms = _extract_search_terms(ie_criteria)

    params = {
        "format": "json",
        "pageSize": page_size,
        "fields": CT_FIELDS,
        "filter.overallStatus": "|".join(statuses),
        "filter.phase": "|".join(phases),
    }
    if terms["condition"]:
        params["query.cond"] = terms["condition"]
    if terms["intr"]:
        params["query.intr"] = terms["intr"]

    # --- API call with retry ---
    response_json = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(CT_API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            response_json = resp.json()
            break
        except requests.RequestException as exc:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                # Return empty table so prompt still runs gracefully
                return (
                    f"_ClinicalTrials.gov API unavailable after {max_retries} attempts "
                    f"(search terms: condition='{terms['condition']}', "
                    f"intr='{terms['intr']}')._\n\n"
                    "| NCT ID | Trial Name | Phase | Status | Sponsor | "
                    "Intervention | Primary Endpoint | N | I/E Criteria |\n"
                    "|--------|------------|-------|--------|---------|"
                    "--------------|------------------|---|---------------|\n"
                    "| N/A | No data retrieved | — | — | — | — | — | — | — |"
                )

    studies = response_json.get("studies", [])
    if not studies:
        return (
            f"_No competing trials found for: condition='{terms['condition']}', "
            f"intr='{terms['intr']}'._\n\n"
            "| NCT ID | Trial Name | Phase | Status | Sponsor | "
            "Intervention | Primary Endpoint | N | I/E Criteria |\n"
            "|--------|------------|-------|--------|---------|"
            "--------------|------------------|---|---------------|\n"
            "| N/A | No trials matched | — | — | — | — | — | — | — |"
        )

    # --- Parse and format ---
    rows = []
    for study in studies:
        proto = study.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design_mod = proto.get("designModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        arms_mod = proto.get("armsInterventionsModule", {})
        outcomes_mod = proto.get("outcomesModule", {})
        eligibility_mod = proto.get("eligibilityModule", {})

        nct_id = id_mod.get("nctId", "N/A")
        title = id_mod.get("briefTitle", "N/A")
        phase = ", ".join(design_mod.get("phases", [])) or "N/A"
        status = status_mod.get("overallStatus", "N/A")
        sponsor = sponsor_mod.get("leadSponsor", {}).get("name", "N/A")

        interventions = arms_mod.get("interventions", [])
        intr_names = list({
            i.get("name", "") for i in interventions
            if i.get("type", "").upper() in ("DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT")
        })
        intr_str = "; ".join(intr_names[:4]) or "N/A"

        primary_outcomes = outcomes_mod.get("primaryOutcomes", [])
        primary_ep = primary_outcomes[0].get("measure", "N/A") if primary_outcomes else "N/A"

        enrollment = design_mod.get("enrollmentInfo", {}).get("count", "N/A")

        eligibility_raw = eligibility_mod.get("eligibilityCriteria", "N/A")
        # Truncate for context window management — full text passed via separate column if needed
        eligibility_short = (eligibility_raw[:600] + "…") if len(eligibility_raw) > 600 else eligibility_raw
        # Escape pipe characters to avoid breaking markdown table
        eligibility_short = eligibility_short.replace("|", "\\|").replace("\n", " ")
        title_esc = title.replace("|", "\\|")
        intr_esc = intr_str.replace("|", "\\|")
        primary_ep_esc = primary_ep.replace("|", "\\|")

        rows.append({
            "NCT ID": nct_id,
            "Trial Name": title_esc,
            "Phase": phase,
            "Status": status,
            "Sponsor": sponsor,
            "Intervention": intr_esc,
            "Primary Endpoint": primary_ep_esc,
            "N": enrollment,
            "I/E Criteria": eligibility_short,
        })

    df = pd.DataFrame(rows)

    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    data_rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]

    search_note = (
        f"_Search terms used: condition='{terms['condition']}', "
        f"intr='{terms['intr']}' · {len(rows)} trials retrieved from ClinicalTrials.gov API v2_\n\n"
    )

    return search_note + "\n".join([header, separator] + data_rows)


# ---------------------------------------------------------------------------
# CI Agent Prompt
# ---------------------------------------------------------------------------

def get_ci_agent_prompt(ie_criteria: str, competing_trials_md: str) -> str:
    """
    Generate a competitive intelligence (CI) agent prompt for analyzing
    treatment-related I/E criteria against competing clinical trials.

    The CI agent complements the SOC and ET agents:
        - SOC agent: gaps vs current real-world standard of care
        - ET  agent: gaps vs emerging/evolving real-world treatment trends
        - CI  agent: competitive positioning per criterion vs trials in the same space

    Output table columns match SOC / ET agents for downstream concatenation.

    Args:
        ie_criteria (str):          Inclusion/Exclusion criteria text (same input as SOC/ET).
        competing_trials_md (str):  Pre-processed markdown table from fetch_competing_trials().

    Returns:
        str: Formatted CI agent prompt.
    """
    return f"""
You are a biomedical competitive intelligence agent.

Your role:
Analyze treatment-related inclusion and exclusion (I/E) criteria of a clinical trial by comparing
them against competing trials in the same therapeutic space.

Your analysis must be:
- Deterministic and reproducible
- Grounded ONLY in the I/E criteria text and the competing trials data provided
- You must NOT add external clinical knowledge beyond what is explicitly present in the inputs

### INPUTS

**I/E criteria (free text):**
{ie_criteria}

**Competing trials retrieved from ClinicalTrials.gov:**
{competing_trials_md}

Each competing trial is represented with columns:
NCT ID | Trial Name | Phase | Status | Sponsor | Intervention | Primary Endpoint | N | I/E Criteria

---

### STEP 0: ANALYSIS SCOPE

#### DISEASE AND BIOMARKER SETTING
1. Extract exact disease stage from I/E text (e.g., "locally advanced or metastatic EGFR-mutant NSCLC")
2. Extract exact biomarker requirements (e.g., "EGFR exon 19 deletion or L858R mutation")
3. Do not use generic disease terms when specific ones exist in I/E

#### SELECT TREATMENT-RELATED CRITERIA ONLY
Analyze ONLY criteria that mention:
- Medications (by name or description)
- Systemic therapies
- Prior systemic treatment
- Prohibited systemic treatment
- Treatment lines or prior lines of therapy

Ignore criteria related to:
- Biomarker status only (e.g., EGFR mutation status without treatment mention)
- Lab values, performance status, histology, radiotherapy
- Washout periods, concomitant medications, toxicity thresholds, adverse events

A criterion is TREATMENT-RELATED if it:
- Names specific drugs (e.g., "no prior osimertinib")
- Uses therapy-class language (e.g., "any chemotherapy", "TKI")
- Specifies treatment lines (e.g., "previously treated with first-line therapy")
- Excludes/includes specific regimens

A criterion is NOT treatment-related if it ONLY specifies biomarker status, lab values,
performance status, or toxicity thresholds — EXCLUDE entirely.

#### TARGET TREATMENT LINE
Extract trial's target line from I/E: treatment-naive, first-line, second-line, etc.

This section is INTERNAL. Do NOT emit STEP 0 output in the final response.

---

### STEP 1: MAP I/E CRITERIA TO COMPETING TRIALS

For each treatment-related I/E criterion identified in STEP 0:

1. Identify which competing trials address the SAME therapeutic criterion:
   - Same drug/class named, OR
   - Same line of therapy restriction, OR
   - Same inclusion/exclusion intent (e.g., "required prior TKI" vs "prohibited prior TKI")

2. For each matching competing trial, determine:
   - Does the competing trial's criterion ALIGN with this trial's criterion?
   - Is the competing trial MORE RESTRICTIVE? (e.g., requires specific drug; excludes more patients)
   - Is the competing trial LESS RESTRICTIVE? (e.g., broader class inclusion; fewer exclusions)
   - Does no competing trial address this criterion? → WHITE SPACE

3. Assign a Competitive Position for each criterion:
   | Competitive Position | Definition |
   |----------------------|------------|
   | Aligned              | Competing trials use equivalent or near-identical criterion |
   | More Restrictive     | This trial's criterion is narrower than competing trials |
   | Less Restrictive     | This trial's criterion is broader than competing trials |
   | White Space          | No competing trial addresses this criterion |

4. Assign a Gap Type for each criterion using the following rules:
   | Gap Type | Definition |
   |----------|------------|
   | Major    | Competitive position meaningfully disadvantages enrollment vs competitors (More Restrictive with large exclusion impact, or White Space in a high-prevalence treatment area) |
   | Minor    | Limited competitive disadvantage or low-prevalence treatment area |
   | None     | Aligned with competitors or competitively advantageous |

This section is INTERNAL. Do NOT emit STEP 1 output in the final response.

---

### STEP 2: GENERATE COMPETITIVE INTELLIGENCE OUTPUT TABLE

For each treatment-related I/E criterion from STEP 0, generate ONE row.

Rules:
- Only treatment-related criteria (from STEP 0) generate rows
- Criteria excluded in STEP 0 produce ZERO rows under any circumstance
- A single I/E criterion with multiple related clauses generates ONE row
- If a criterion is already optimally positioned vs competitors, mark Gap Type = None

**Output table columns** (must match SOC/ET agent output format exactly):

| Type | Original Criterion | Gap Type | Queriable | Competing Trial | Competitive Position | Explanation | Recommendation |

Column rules:
- **Type:** Inclusion / Exclusion / Suggested Addition: Inclusion / Suggested Addition: Exclusion
- **Original Criterion:** Quote or summarize precisely from I/E text.
  If Suggested Addition: set to "New Criterion"
- **Gap Type:** Major / Minor / None — from STEP 1 assignment
- **Queriable:** Yes if at least one competing trial addresses this criterion; No if White Space across all competing trials
- **Competing Trial:** NCT ID and brief name of the most relevant competing trial.
  If multiple trials are relevant, list up to 2: "NCT##### (Trial A), NCT##### (Trial B)".
  If White Space: "None identified"
- **Competitive Position:** Aligned / More Restrictive / Less Restrictive / White Space
- **Explanation:** Justify using I/E text and competing trial data. Bold key drugs, lines, trial names.
  - State explicitly: "**[Competing Trial NCT#]** [uses/requires/excludes] [criterion]"
  - Compare: "This trial is [more/less] restrictive because [reason]"
  - If White Space: "No competing Phase II/III trial in this space addresses this criterion"
- **Recommendation:** Must be directly usable as eligibility-criteria wording.
  - If Gap Type = None: "Already captured by existing I/E Criteria"
  - For Suggested Addition (broaden): use template →
    "Include participants who have progressed on or after [REGIMEN] administered as [LINE] systemic therapy for [DISEASE STAGE]."
  - For Suggested Addition (restrict): use template →
    "Exclude participants who have received [REGIMEN] as [LINE] systemic therapy for [DISEASE STAGE]."
  - For competitive alignment recommendation: →
    "Consider [broadening/restricting] [CRITERION] to align with [COMPETING TRIAL NCT#], which [uses/requires/excludes] [CRITERION WORDING]."
  - Avoid ambiguous phrases like "most recent line." Use explicit lines from I/E text.

---

### STEP 3: POST-PROCESSING — CONSOLIDATED COMPETITIVE SUGGESTIONS

After table generation:

1. Identify all rows where Type = "Suggested Addition"
2. Group by similar regimen, line, or disease stage
3. For each group:
   - Identify the relevant original treatment-related I/E criterion
   - Record as **Based on Original Criterion**
   - Append consolidated suggestion using OR logic if multiple additions
   - If no relevant original criterion: mark "Based on Original Criterion: New"

4. Output section titled **Updated Treatment-Related Criteria (Competitive Intelligence)** with format:

   **Based on Original Criterion:** [Original I/E criterion text]
   - **Suggested Addition (Gap Type: [type], vs [NCT#]):** [Recommendation verbatim from table]

   **Updated Combined Criterion:** [Natural, directly-usable eligibility text combining original + additions]

   Or if new:
   **New Criterion:**
   - **Suggested Addition (Gap Type: [type], vs [NCT#]):** [Recommendation from table]
   **Updated Combined Criterion:** [Natural, directly-usable eligibility text]

5. Do NOT modify non-treatment-related I/E parts.

---

### STEP 4: COMPETITIVE INTELLIGENCE SUMMARY

After the table and consolidated suggestions, generate a **Competitive Intelligence Summary** section:

- **Disease & Biomarker Context:** [Extracted from I/E text]
- **Competing Trial Landscape:** Number of trials retrieved, phases, top sponsors, and whether the field is crowded or sparse.
- **Competitive Positioning Overview:** Overall assessment — is this trial's I/E criteria more or less restrictive than the field? Which criteria create the largest competitive enrollment disadvantage?
- **White Space Opportunities:** Criteria or patient populations not addressed by any competitor — potential differentiation points.
- **Primary Recommendation:** One concise, actionable takeaway for trial optimization based on competitive positioning. Reference specific competing trials by NCT ID.

---

### OUTPUT FORMAT

Generate the following sections ONLY. Do NOT emit STEP 0, STEP 1, or STEP 2 internal outputs:

1. Markdown output table (from STEP 2)
2. Updated Treatment-Related Criteria — Competitive Intelligence (from STEP 3)
3. Competitive Intelligence Summary (from STEP 4)

Before generating the table, verify your STEP 1 filtering by listing:
- Total Aligned criteria: [X]
- Total More Restrictive criteria: [Y]
- Total Less Restrictive criteria: [Z]
- Total White Space criteria: [W]
- Grand total: [X+Y+Z+W]
"""


# ---------------------------------------------------------------------------
# Example usage (mirrors notebook pattern)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # MARIPOSA-2 I/E criteria (simplified for illustration)
    ie_criteria_example = """
Inclusion Criteria:
1. Histologically or cytologically confirmed locally advanced or metastatic NSCLC
   with EGFR exon 19 deletion or exon 21 L858R mutation
2. Disease progression on or after osimertinib as the most recent line of therapy
3. Prior treatment with a first- or second-generation EGFR tyrosine kinase inhibitor
   is permitted but not required

Exclusion Criteria:
1. Prior treatment with amivantamab or any bispecific antibody targeting EGFR
2. Prior treatment with platinum-based chemotherapy for metastatic disease
3. Prior treatment with lazertinib or any third-generation EGFR TKI other than osimertinib
"""

    print("Fetching competing trials from ClinicalTrials.gov...")
    competing_md = fetch_competing_trials(ie_criteria_example)
    print("\n--- Competing Trials (first 500 chars) ---")
    print(competing_md[:500])

    print("\n--- CI Agent Prompt (first 500 chars) ---")
    ci_prompt = get_ci_agent_prompt(ie_criteria_example, competing_md)
    print(ci_prompt[:500] + "...\n[truncated for display]")
