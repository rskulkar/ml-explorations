"""ClinicalTrials.gov API wrapper for clinical-trial-optimizer.

Provides:
    - fetch_competing_trials: queries the ClinicalTrials.gov v2 REST API for
      interventional trials matching a given indication and phase, and returns
      structured trial data for competitive intelligence analysis.
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
