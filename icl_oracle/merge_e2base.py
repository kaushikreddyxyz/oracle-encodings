"""Patch a results dir whose e2b.json borrowed 6-seed baselines from e2.json with the N-seed
baselines from e2base.json (so plot_gt/stats_gt compare like with like)."""
import json, sys
RES = sys.argv[1]
base = json.load(open(f"{RES}/e2base.json"))
eb = json.load(open(f"{RES}/e2b.json"))
eb["dense"]["none"] = base["dense"]["none"]
eb["dense"]["extracted_b2.0"] = base["dense"]["extracted_b2.0"]
json.dump(eb, open(f"{RES}/e2b.json", "w"))
e2 = json.load(open(f"{RES}/e2.json"))
e2["override"]["none_injmap"] = base["override"]["none_injmap"]
e2["dense"]["none"] = base["dense"]["none"]
json.dump(e2, open(f"{RES}/e2.json", "w"))
print("patched", RES, "seeds:", len(base["dense"]["none"]))
