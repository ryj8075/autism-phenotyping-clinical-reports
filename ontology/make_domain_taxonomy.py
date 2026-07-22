# -*- coding: utf-8 -*-

import os, csv
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
YAML = os.path.join(HERE, "domains_19.yaml")
OUT_LOCAL = os.path.join(HERE, "domain_taxonomy.csv")
FIGURE_SOURCE_DIR = os.environ.get("REPORT_LLM_FIGURE_SOURCE_DIR")
OUT_FIGSRC = (
    os.path.join(FIGURE_SOURCE_DIR, "domain_taxonomy.csv")
    if FIGURE_SOURCE_DIR else None
)

GROUP_LABEL = {"A": "ASD-core", "B": "Child-psychiatric", "C": "Format / other"}

DISPLAY = {
    "A1": "Social-emotional reciprocity", "A2": "Nonverbal communication",
    "A3": "Relationship play", "A4": "Stereotyped behavior",
    "A5": "Insistence on sameness", "A6": "Restricted interests",
    "A7": "Sensory processing",
    "B1": "Externalizing", "B2": "Internalizing", "B3": "Language skills",
    "B4": "Physiological function", "B5": "Adaptive behavior",
    "B6": "Intelligence/learning", "B7": "Executive function",
    "B8": "Motor skills", "B9": "Family environment",
    "C1": "Test scores", "C2": "Other/general", "C3": "Recommendations",
}

def build_rows():
    d = yaml.safe_load(open(YAML, encoding="utf-8"))
    rows = []
    for cat in d["categories"]:
        for sub in cat.get("subcategories", []):
            for dom in sub.get("domains", []):
                code = dom["code"]
                grp = code[0]
                name = DISPLAY.get(code)
                if name is None:
                    name = dom.get("name_en", dom["id"])
                    print(f"  [warn] no DISPLAY label for {code}; falling back to name_en '{name}'")
                rows.append((code, grp, GROUP_LABEL.get(grp, grp), name))
    return rows

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "group", "group_label", "domain_en"])
        w.writerows(rows)

def main():
    rows = build_rows()
    paths = [OUT_LOCAL]
    if OUT_FIGSRC:
        paths.append(OUT_FIGSRC)
    for path in paths:
        write_csv(path, rows)
        print(f"  wrote {len(rows)} domains -> {path}")
    n = {g: sum(1 for r in rows if r[1] == g) for g in ("A", "B", "C")}
    print(f"  ({n['A']} A / {n['B']} B / {n['C']} C)")

if __name__ == "__main__":
    main()
