#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rank what employment job postings ask for against what the portfolio can prove.

Corpus: 128 live postings read 2026-11-15.
  Djinni  (n=91) primary_keyword="Data Analyst", 8 result pages, analyst-titled only.
  DOU     (n=37) RSS feeds for categories Analyst / Data Science / Big Data /
          Data Engineer / AI-ML across 5 experience buckets, then filtered to the
          Data-BI / Product-Marketing / HR-analytics title families.

Counts are (required, nice-to-have). A term found before a "will be a plus" marker
counts as required; after it, as nice-to-have. The "we offer" tail of each posting
is cut before matching, because benefits blurbs ("education budget", "wellness
budget") otherwise register as finance demand.

Evidence verdicts come from capability_manifest.json plus a grep of the showable
portfolio root. They are deliberately harsh: a capability with no showable
artifact cannot be put in front of an employer, so it does not count as proof.
"""
from __future__ import annotations
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

N_DJINNI, N_DOU = 91, 37
N = N_DJINNI + N_DOU

# (required, plus) per board, measured in the browser against the live postings.
DJINNI = {
    "Reporting": (58, 5), "SQL": (53, 7), "Dashboards": (43, 4), "Power BI": (34, 7),
    "Python": (24, 15), "Stakeholder communication": (29, 5), "Tableau": (26, 7),
    "Finance / accounting": (29, 3), "Statistics / hypothesis": (23, 7),
    "Fintech/payments": (22, 5), "Supply chain": (19, 4), "Data quality": (19, 3),
    "LTV / CAC / ROMI": (20, 2), "Excel": (18, 4), "Forecasting": (19, 1),
    "Retention / churn": (16, 3), "A/B testing": (14, 4), "HR / recruitment": (11, 7),
    "Data modelling": (14, 4), "Data warehouse": (15, 3), "BigQuery": (14, 2),
    "LLM / AI tooling": (10, 5), "API / REST / JSON": (12, 2),
    "Segmentation / clustering": (14, 0), "Cohort analysis": (12, 1),
    "E-commerce": (10, 3), "AWS": (8, 5), "iGaming/gambling": (11, 1),
    "Marketing/media buying": (11, 1), "Looker": (10, 1), "DAX": (10, 1),
    "Snowflake": (8, 2), "pandas/numpy": (8, 2), "ML / predictive": (6, 4),
    "dbt": (6, 4), "Power Query": (7, 2), "Documentation": (8, 1),
    "Data governance / catalog": (8, 1), "GCP": (2, 7), "Google Sheets": (7, 2),
    "GA4 / Google Analytics": (6, 2), "ClickHouse": (8, 0), "ETL/ELT": (3, 4),
    "Databricks": (5, 2), "PostgreSQL": (5, 1), "Azure": (4, 2), "Agile/Scrum": (3, 3),
    "Jira/Confluence": (3, 3), "Risk / fraud": (6, 0), "Amplitude": (2, 4),
    "Requirements gathering": (5, 0), "Synapse/Fabric": (4, 1), "Airflow": (4, 1),
    "Looker Studio": (4, 1), "n8n/Zapier/Make": (2, 3), "AppsFlyer/Adjust": (2, 2),
    "Superset": (4, 0), "R": (1, 3), "Redshift": (4, 0), "CI/CD": (2, 1),
    "Git": (2, 1), "Mixpanel": (1, 2), "Web scraping": (3, 0), "MongoDB": (2, 0),
    "Semantic layer / metrics layer": (2, 0), "VBA": (1, 1), "GTM / Tag Manager": (1, 1),
    "Oracle": (1, 1), "Metabase": (1, 0), "MySQL": (2, 0), "Time series": (1, 1),
    "Firebase": (1, 0), "Grafana": (1, 0), "Docker/K8s": (0, 1), "SSRS/SSAS": (1, 0),
    "PostHog/Segment": (1, 0), "Qlik": (1, 0),
}
DOU = {
    "SQL": (24, 2), "Reporting": (22, 3), "Dashboards": (20, 1), "Python": (12, 7),
    "Power BI": (14, 2), "Stakeholder communication": (15, 1),
    "Statistics / hypothesis": (13, 1), "A/B testing": (9, 1), "Tableau": (10, 0),
    "BigQuery": (6, 3), "Data modelling": (8, 0), "GA4 / Google Analytics": (6, 2),
    "Excel": (5, 2), "Looker Studio": (6, 1), "Segmentation / clustering": (6, 1),
    "Data quality": (7, 0), "GTM / Tag Manager": (5, 2), "Looker": (7, 0),
    "Data warehouse": (5, 2), "API / REST / JSON": (4, 2), "Documentation": (5, 1),
    "pandas/numpy": (5, 1), "ClickHouse": (4, 2), "LLM / AI tooling": (5, 1),
    "dbt": (4, 2), "Forecasting": (5, 0), "Retention / churn": (4, 1),
    "LTV / CAC / ROMI": (4, 1), "ML / predictive": (3, 2), "AWS": (1, 4),
    "Cohort analysis": (3, 1), "Power Query": (4, 0), "ETL/ELT": (4, 0),
    "Snowflake": (3, 0), "Amplitude": (1, 2), "Google Sheets": (2, 0), "DAX": (2, 0),
    "PostgreSQL": (1, 1), "n8n/Zapier/Make": (2, 0), "AppsFlyer/Adjust": (1, 1),
    "PostHog/Segment": (2, 0), "Jira/Confluence": (1, 1), "Databricks": (2, 0),
    "Git": (1, 1), "CI/CD": (2, 0), "Semantic layer / metrics layer": (1, 1),
    "Firebase": (1, 1), "Airflow": (2, 0), "Agile/Scrum": (1, 1),
    "Data governance / catalog": (1, 0), "Metabase": (1, 0), "Mixpanel": (0, 1),
    "MySQL": (0, 1), "Oracle": (1, 0), "Redshift": (1, 0), "Superset": (1, 0),
    "Web scraping": (1, 0), "GCP": (0, 1),
}

# How strong the proof is TODAY. Levels, worst to best:
#   claimed   the resume claims it and there is no artifact behind the claim
#   none      nothing in the manifest and nothing in the portfolio
#   private   a capability exists but has no showable artifact
#   adjacent  manifest confidence "adjacent" - built, never run against the real thing
#   proven    manifest confidence "proven" with a showable artifact in the portfolio
EVIDENCE = {
    "SQL": ("proven", "sql-mysql proven on MySQL/SQLite; sql-multi-dialect is adjacent"),
    "Reporting": ("proven", "five .pbip report builds in the portfolio"),
    "Dashboards": ("proven", "hiring_funnel_demo, ecommerce_funnel, ta_screening, hiring_star_schema"),
    "Power BI": ("proven", "powerbi-report-build, API-first .pbip/TMDL/PBIR"),
    "Python": ("proven", "python-analysis-pipeline, every portfolio build script"),
    "Stakeholder communication": ("proven", "client-reporting-delivery"),
    "Tableau": ("private", "twb audit+edit tooling exists, showable=False, authoring never done"),
    "Statistics / hypothesis": ("none", "no hypothesis test, CI or p-value anywhere in the portfolio"),
    "A/B testing": ("claimed", "resume/build_resume.py:179 lists it; zero artifacts, zero capability"),
    "Data quality": ("proven", "data-quality-audit, scored blind against a sealed key"),
    "Forecasting": ("proven", "hiring-forecast with P80 bands, logged and scored"),
    "Excel": ("proven", "excel-deliverable + workbook audit/rescue"),
    "LTV / CAC / ROMI": ("none", "no unit-economics metric set defined or computed"),
    "Retention / churn": ("adjacent", "funnel-cohort-analysis covers retention shape, not churn modelling"),
    "Cohort analysis": ("proven", "funnel-cohort-analysis"),
    "Data modelling": ("proven", "hiring_star_schema"),
    "Data warehouse": ("adjacent", "feed tables and a star schema, no warehouse engine"),
    "BigQuery": ("claimed", "on the resume; sql-multi-dialect interface exists, never connected"),
    "Segmentation / clustering": ("none", "no clustering or segment-discovery work"),
    "ML / predictive": ("adjacent", "forecast only; no model training, scoring or validation"),
    "dbt": ("none", "nothing"),
    "Airflow": ("none", "own scheduler only"),
    "Snowflake": ("none", "closed commercially in manifest 1.18, no adapter built"),
    "ClickHouse": ("none", "nothing"),
    "Databricks": ("none", "nothing"),
    "Redshift": ("claimed", "on the resume; one mention in the portfolio"),
    "GA4 / Google Analytics": ("adjacent", "ga4_core 71 checks, showable=False"),
    "Amplitude": ("claimed", "on the resume; nothing in the portfolio"),
    "Mixpanel": ("none", "nothing"),
    "GTM / Tag Manager": ("none", "closed as a commercial decision, no tooling"),
    "Looker": ("none", "Looker proper (LookML) is not Looker Studio"),
    "Looker Studio": ("adjacent", "gsheets_core + looker report audit, showable=False"),
    "AWS": ("none", "nothing"), "GCP": ("none", "nothing"), "Azure": ("none", "nothing"),
    "Docker/K8s": ("none", "nothing"),
    "API / REST / JSON": ("proven", "ats-api-pipeline, incremental and verified"),
    "ETL/ELT": ("proven", "connector_core, pipeline-correctness-audit"),
    "Data governance / catalog": ("proven", "metric-governance, metric_dictionary.md"),
    "Semantic layer / metrics layer": ("proven", "metric-governance"),
    "Documentation": ("proven", "DOCUMENTATION.md, MANUAL.md, metric dictionaries"),
    "LLM / AI tooling": ("proven", "nl-qa-over-governed-mart, ask_engine.py"),
    "pandas/numpy": ("proven", "every analysis script"),
    "DAX": ("proven", "build_measures.py across four reports"),
    "Power Query": ("proven", "M in the .pbip models"),
    "Google Sheets": ("adjacent", "gsheets_core, showable=False"),
    "Git": ("proven", "every portfolio repo"),
    "CI/CD": ("none", "no pipeline runs on a push anywhere"),
    "Web scraping": ("none", "open gap web-scraping"),
    "Time series": ("adjacent", "forecast is time-indexed, no ARIMA/Prophet style work"),
}
# Domain or process context, not a transferable capability worth building a project for.
SKIP = {
    "Finance / accounting", "Fintech/payments", "Supply chain", "HR / recruitment",
    "E-commerce", "iGaming/gambling", "Marketing/media buying", "Risk / fraud",
    "Agile/Scrum", "Jira/Confluence", "Requirements gathering", "R", "VBA", "MongoDB",
    "MySQL", "Oracle", "PostgreSQL", "Metabase", "Superset", "Grafana", "Qlik",
    "SSRS/SSAS", "Synapse/Fabric", "n8n/Zapier/Make", "AppsFlyer/Adjust", "Firebase",
    "PostHog/Segment", "Google Sheets", "Excel", "Git",
}
RANK = {"claimed": 0, "none": 1, "private": 2, "adjacent": 3, "proven": 4}


def build_rows():
    rows = []
    for key in sorted(set(DJINNI) | set(DOU)):
        dj_req, dj_plus = DJINNI.get(key, (0, 0))
        dou_req, dou_plus = DOU.get(key, (0, 0))
        req, plus = dj_req + dou_req, dj_plus + dou_plus
        level, why = EVIDENCE.get(key, ("?", "unclassified"))
        rows.append(dict(req=key, dj=dj_req + dj_plus, dou=dou_req + dou_plus,
                         total=req + plus, required=req, plus=plus,
                         pct=round((req + plus) / N * 100), level=level, why=why))
    return rows


def table(rows, title):
    print("\n" + title)
    header = (f"{'requirement':32s} {'all':>4s} {'%':>5s} {'req':>4s} {'plus':>5s} "
              f"{'dj':>3s} {'dou':>4s}  evidence")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['req']:32s} {r['total']:4d} {r['pct']:4d}% {r['required']:4d} "
              f"{r['plus']:5d} {r['dj']:3d} {r['dou']:4d}  {r['level']}")


def main():
    rows = build_rows()
    print(f"corpus: {N} postings  (Djinni {N_DJINNI}, DOU {N_DOU})   read 2026-11-15")
    table(sorted(rows, key=lambda r: -r["total"])[:26],
          "== WHAT THE MARKET ASKS FOR (top 26) ==")

    gaps = [r for r in rows if r["req"] not in SKIP and RANK.get(r["level"], 9) <= 1]
    table(sorted(gaps, key=lambda r: -r["total"]),
          "== GAPS: demanded, nothing to show (level none/claimed) ==")

    weak = [r for r in rows if r["req"] not in SKIP and RANK.get(r["level"], 9) in (2, 3)]
    table(sorted(weak, key=lambda r: -r["total"]),
          "== THIN: built but not showable, or never run for real ==")

    print("\n== THE TOP GAP ==")
    top = sorted(gaps, key=lambda r: -r["total"])[0]
    print(f"{top['req']}: {top['total']}/{N} postings ({top['pct']}%), "
          f"{top['required']} of them as a hard requirement.")
    print(f"evidence: {top['level']} - {top['why']}")
    stat = next(r for r in rows if r["req"] == "Statistics / hypothesis")
    print(f"\npaired with Statistics / hypothesis: {stat['total']}/{N} ({stat['pct']}%), "
          f"{stat['required']} hard. evidence: {stat['level']} - {stat['why']}")


if __name__ == "__main__":
    main()
