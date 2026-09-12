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

GROUP_LABEL = {"CO": "Core ASD domains",
               "AS": "Associated and co-occurring features",
               "RE": "Report elements and other"}

def build_rows():
    d = yaml.safe_load(open(YAML, encoding="utf-8"))
    rows = []
    for cat in d["categories"]:
        for sub in cat.get("subcategories", []):
            for dom in sub.get("domains", []):
                code = dom["code"]
                grp = code[:2]
                name = dom.get("name_en", dom["id"])
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
    n = {g: sum(1 for r in rows if r[1] == g) for g in ("CO", "AS", "RE")}
    print(f"  ({n['CO']} CO / {n['AS']} AS / {n['RE']} RE)")

if __name__ == "__main__":
    main()
