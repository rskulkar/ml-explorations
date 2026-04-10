"""ClinicalTrials.gov API wrapper for clinical-trial-optimizer.

Provides:
    - fetch_competing_trials: queries the ClinicalTrials.gov v2 REST API for
      interventional trials matching a given indication and phase, and returns
      structured trial data for competitive intelligence analysis.
"""
import re
import json
import time
import requests
import pandas as pd
import anthropic


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


def extract_search_terms_via_llm(ie_criteria: str, api_key: str | None = None) -> dict:
    """
    Use Claude Haiku to extract clean ClinicalTrials.gov search terms from I/E criteria.

    Returns dict with keys: condition, intr
    Falls back to empty strings on any failure so fetch_competing_trials degrades gracefully.
    """
    prompt = """From the clinical trial I/E criteria below, extract two search terms for ClinicalTrials.gov API.

1. condition: disease name + stage + biomarker positivity (max 5 words, no punctuation, use + for biomarker positivity not -)
2. intr: primary drug class or key agent name (max 3 words, no punctuation)

Output ONLY valid JSON with exactly these two keys, nothing else:
{"condition": "...", "intr": "..."}

I/E criteria:
""" + ie_criteria

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.DOTALL).strip()
        terms = json.loads(raw)
        return {
            "condition": str(terms.get("condition", "")).strip(),
            "intr": str(terms.get("intr", "")).strip(),
        }
    except Exception as exc:
        # Degrade gracefully — empty terms will still call the API with no filters
        print(f"Warning: LLM search term extraction failed ({exc}). Using empty terms.")
        return {"condition": "", "intr": ""}


def fetch_competing_trials(
    ie_criteria: str,
    api_key: str | None = None,
    phases: list[str] | None = None,
    statuses: list[str] | None = None,
    page_size: int = 20,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    Fetch competing trials from ClinicalTrials.gov API v2.

    Derives search terms from ie_criteria using Claude Haiku, queries the API, and returns
    a pre-processed markdown table string ready to pass into get_ci_agent_prompt().

    Args:
        ie_criteria:  Raw I/E criteria text (same input as SOC/ET agents).
        api_key:      Optional Anthropic API key for LLM search term extraction.
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

    terms = extract_search_terms_via_llm(ie_criteria, api_key)
    print(f"Search terms extracted: condition='{terms['condition']}', intr='{terms['intr']}'")

    params = {
        "format": "json",
        "pageSize": page_size,
        "fields": CT_FIELDS,
        "filter.overallStatus": ",".join(statuses),
        "filter.phase": ",".join(phases),
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
