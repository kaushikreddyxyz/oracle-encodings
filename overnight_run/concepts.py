"""
concepts.py — single source of truth for the overnight concept-probes run.

Every stage (labeling, probe training, attribution, geometry) imports this so the
concept definitions never drift. Two regimes per the brief:

  - PRESENCE (cyclic/categorical): each class gets its own binary probe. Labeling is
    surface/lemma match (pseudo-gold) + judge for sense-disambiguation & misspelling
    rescue. STRICT: literal textual presence only, no "evokes".
  - SCALAR (1D): graded rating along a defined scale. Judge allowed interpretive
    latitude; external value used where one exists (digit value for numbers).

Geometry groupings (Step 4) are encoded at the bottom so Tier 1-5 tests are driven
off this file directly.

NB on lexicons: surface forms are matched case-insensitively with word boundaries.
Keep them high-precision; the judge rescues misspellings the matcher misses.
"""

# ----------------------------------------------------------------------------- #
# PRESENCE CONCEPTS  (each class -> one binary probe)
# ----------------------------------------------------------------------------- #
# Each concept: regime, optional `cycle` (Z/n order for geometry), and `classes`:
#   class_name -> {"lexicon": [surface forms], "traps": [confusables to score 0/low]}

MONTHS = {
    "regime": "presence", "cycle": 12,
    "classes": {
        "January":   {"lexicon": ["january", "jan", "jan."],   "traps": ["January Jones (name)", "New Year (evokes)"]},
        "February":  {"lexicon": ["february", "feb", "feb."],  "traps": ["febrile (substring)"]},
        "March":     {"lexicon": ["march", "mar", "mar."],     "traps": ["march = walk/protest (verb/noun)", "Frederic March (name)"]},
        "April":     {"lexicon": ["april", "apr", "apr."],     "traps": ["April (given name)", "April fools (still names month -> ok)"]},
        "May":       {"lexicon": ["may"],                       "traps": ["may = modal verb (VERY common)", "May (name)", "Theresa May"]},
        "June":      {"lexicon": ["june", "jun", "jun."],      "traps": ["June (given name)"]},
        "July":      {"lexicon": ["july", "jul", "jul."],      "traps": []},
        "August":    {"lexicon": ["august", "aug", "aug."],    "traps": ["august = majestic (adj)", "Augustus (name)"]},
        "September": {"lexicon": ["september", "sep", "sept", "sep.", "sept."], "traps": []},
        "October":   {"lexicon": ["october", "oct", "oct."],   "traps": []},
        "November":  {"lexicon": ["november", "nov", "nov."],  "traps": ["Nov. as abbrev only"]},
        "December":  {"lexicon": ["december", "dec", "dec."],  "traps": ["dec = decimal/decrease abbrev"]},
    },
}

DAYS = {
    "regime": "presence", "cycle": 7,
    "classes": {
        "Monday":    {"lexicon": ["monday", "mon", "mon."],            "traps": ["Mon = Monsieur/abbr"]},
        "Tuesday":   {"lexicon": ["tuesday", "tue", "tues", "tue."],   "traps": []},
        "Wednesday": {"lexicon": ["wednesday", "wed", "weds", "wed."], "traps": ["wed = married (verb)", "Wednesday Addams (name)"]},
        "Thursday":  {"lexicon": ["thursday", "thu", "thur", "thurs", "thu."], "traps": []},
        "Friday":    {"lexicon": ["friday", "fri", "fri."],           "traps": ["Friday (Robinson Crusoe name)"]},
        "Saturday":  {"lexicon": ["saturday", "sat", "sat."],         "traps": ["sat = past tense of sit (VERY common)"]},
        "Sunday":    {"lexicon": ["sunday", "sun", "sun."],           "traps": ["sun = the star (VERY common)", "Sunday (name)"]},
    },
}

# Base-10 single numbers 0..10 (fractions quantized to nearest int per brief).
# Presence of the *number*; magnitude (digit value) is the scalar in SCALARS["numbers"].
NUMBERS_10 = {
    "regime": "presence", "cycle": None,
    "classes": {
        str(n): {
            "lexicon": [str(n), _word]  # digit + English word
            , "traps": ["phone numbers / IDs / years (still the number, but flag context)"]
        }
        for n, _word in zip(
            range(0, 11),
            ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
        )
    },
}

# Base-100 buckets {0-9, 10-19, ..., 90-99}. A number token falls in exactly one bucket.
NUMBERS_100 = {
    "regime": "presence", "cycle": None,
    "classes": {
        f"{lo}-{lo+9}": {"lexicon": [], "range": [lo, lo + 9],
                         "traps": ["assign by parsed integer value, not surface string"]}
        for lo in range(0, 100, 10)
    },
}

# 12-hue artist's color wheel (brief's exact order, Z/12).
COLOR_WHEEL = {
    "regime": "presence", "cycle": 12,
    "classes": {
        "Red":           {"lexicon": ["red"],                                   "traps": ["red = communist/debt (fig.)", "Red (name/nickname)"]},
        "Red-Orange":    {"lexicon": ["red-orange", "red orange", "vermilion"], "traps": []},
        "Orange":        {"lexicon": ["orange"],                                 "traps": ["orange = the fruit", "Orange (place/telecom)"]},
        "Yellow-Orange": {"lexicon": ["yellow-orange", "yellow orange", "amber"], "traps": ["amber = name/traffic"]},
        "Yellow":        {"lexicon": ["yellow"],                                 "traps": ["yellow = cowardly (fig.)"]},
        "Yellow-Green":  {"lexicon": ["yellow-green", "yellow green", "chartreuse"], "traps": []},
        "Green":         {"lexicon": ["green"],                                  "traps": ["green = eco/novice (fig.)", "Green (name)"]},
        "Blue-Green":    {"lexicon": ["blue-green", "blue green", "teal", "cyan"], "traps": ["teal/cyan as exact hue only"]},
        "Blue":          {"lexicon": ["blue"],                                   "traps": ["blue = sad (fig.)", "the blues (music)"]},
        "Blue-Purple":   {"lexicon": ["blue-purple", "blue-violet", "blue violet", "indigo"], "traps": []},
        "Purple":        {"lexicon": ["purple", "violet"],                       "traps": ["Violet (name)", "purple prose (fig.)"]},
        "Red-Purple":    {"lexicon": ["red-purple", "red-violet", "magenta"],   "traps": []},
    },
}

SEASONS = {
    "regime": "presence", "cycle": 4,
    "classes": {
        "Spring": {"lexicon": ["spring"],         "traps": ["spring = coil/water source/leap (VERY common)"]},
        "Summer": {"lexicon": ["summer"],          "traps": ["Summer (given name)"]},
        "Autumn": {"lexicon": ["autumn", "fall"],  "traps": ["fall = to drop / autumn (US)", "Autumn (name)"]},
        "Winter": {"lexicon": ["winter"],          "traps": ["Winter (name)"]},
    },
}

# Cardinal (Z/4). Substring traps are severe -> word-boundary match + judge sense-confirm.
DIRECTIONS = {
    "regime": "presence", "cycle": 4,
    "special_prompting": "Single-letter forms (N/E/S/W) and substrings ('north' in 'northern') are heavy traps; require directional sense.",
    "classes": {
        "North": {"lexicon": ["north", "northward", "northern"], "traps": ["North (surname)", "'north of $5M' (fig.)", "substring of 'northern' ok if directional"]},
        "East":  {"lexicon": ["east", "eastward", "eastern"],    "traps": ["East (surname)", "Middle East (region name, not direction)"]},
        "South": {"lexicon": ["south", "southward", "southern"], "traps": ["South (surname)", "the South (US region)"]},
        "West":  {"lexicon": ["west", "westward", "western"],    "traps": ["West (surname e.g. Kanye)", "western = film genre"]},
    },
}

# 8 standard lunar phases. Brief said "12" but there is no canonical 12-phase set;
# using the 8 principal+intermediate phases and noting the discrepancy in report.md.
# `illum` = fractional illumination (Tier-3 scalar that bridges from a Z/8 cycle).
MOON_PHASES = {
    "regime": "presence", "cycle": 8,
    "classes": {
        "New Moon":         {"lexicon": ["new moon"],                         "illum": 0.0,  "traps": []},
        "Waxing Crescent":  {"lexicon": ["waxing crescent"],                  "illum": 0.25, "traps": []},
        "First Quarter":    {"lexicon": ["first quarter", "first-quarter"],   "illum": 0.5,  "traps": ["first quarter = fiscal Q1 (fig.)"]},
        "Waxing Gibbous":   {"lexicon": ["waxing gibbous"],                   "illum": 0.75, "traps": []},
        "Full Moon":        {"lexicon": ["full moon"],                        "illum": 1.0,  "traps": ["full moon = lunacy (fig.) still names it -> ok"]},
        "Waning Gibbous":   {"lexicon": ["waning gibbous"],                   "illum": 0.75, "traps": []},
        "Last Quarter":     {"lexicon": ["last quarter", "third quarter", "last-quarter"], "illum": 0.5, "traps": ["quarter = coin/fiscal"]},
        "Waning Crescent":  {"lexicon": ["waning crescent"],                  "illum": 0.25, "traps": []},
    },
}

PRESENCE_CONCEPTS = {
    "months": MONTHS, "days": DAYS, "numbers10": NUMBERS_10, "numbers100": NUMBERS_100,
    "color_wheel": COLOR_WHEEL, "seasons": SEASONS, "directions": DIRECTIONS,
    "moon_phases": MOON_PHASES,
}

# ----------------------------------------------------------------------------- #
# SCALAR CONCEPTS  (each -> one regression probe)
# ----------------------------------------------------------------------------- #
# scale: [low_pole, high_pole]; anchors: example points; external: rule if a
# ground-truth scalar exists (judge role shrinks to sense-confirmation).

SCALARS = {
    "numbers":      {"regime": "scalar", "scale": ["0", "large"],
                     "external": "digit value of the number token (log-spacing tested in Tier 3)",
                     "anchors": {"low": "zero/one", "mid": "~50", "high": "thousands+"}},
    "costliness":   {"regime": "scalar", "scale": ["cheap", "priceless"],
                     "anchors": {"low": "dirt cheap / free", "mid": "moderately priced", "high": "priceless / astronomical"}},
    "physical_size":{"regime": "scalar", "scale": ["tiny", "huge"],
                     "anchors": {"low": "microscopic / tiny", "mid": "human-scale", "high": "gigantic / colossal"}},
    "europe":       {"regime": "scalar", "scale": ["not-Europe", "Europe"], "spatial_anchor": "europe",
                     "anchors": {"low": "unrelated/other continent", "high": "clearly European place/entity"}},
    "america":      {"regime": "scalar", "scale": ["not-America", "America"], "spatial_anchor": "america",
                     "anchors": {"low": "unrelated/other continent", "high": "clearly American place/entity"}},
    "africa":       {"regime": "scalar", "scale": ["not-Africa", "Africa"], "spatial_anchor": "africa",
                     "anchors": {"low": "unrelated/other continent", "high": "clearly African place/entity"}},
    "indoors":      {"regime": "scalar", "scale": ["not-indoors", "indoors"], "antipode": "outdoors",
                     "anchors": {"low": "open wilderness", "mid": "doorway/porch", "high": "deep interior room"}},
    "outdoors":     {"regime": "scalar", "scale": ["not-outdoors", "outdoors"], "antipode": "indoors",
                     "anchors": {"low": "sealed interior", "mid": "doorway/porch", "high": "open wilderness"}},
    "lovingness":   {"regime": "scalar", "scale": ["despise", "adore"], "antipode_test": "harmfulness",
                     "anchors": {"low": "despise/loathe", "mid": "neutral/indifferent", "high": "adore/cherish"}},
    "duration":     {"regime": "scalar", "scale": ["instantaneous", "eternal"],
                     "anchors": {"low": "instant/momentary", "mid": "hours/days", "high": "eternal/geological"}},
    "harmfulness":  {"regime": "scalar", "scale": ["benign", "catastrophic"], "antipode_test": "lovingness",
                     "anchors": {"low": "harmless/benign", "mid": "risky", "high": "catastrophic/lethal"}},
}

# ----------------------------------------------------------------------------- #
# GEOMETRY GROUPINGS  (Step 4 — drives the hypothesis tests in priority order)
# ----------------------------------------------------------------------------- #
GEOMETRY = {
    "tier1_z12": {  # Z/12 collision study (headline)
        "cycles": ["months", "color_wheel", "moon_phases"],
        "z4_cycles": ["seasons", "directions"],
        "tests": ["planarity", "cyclic_ordering", "angular_uniformity", "principal_angles", "phase_alignment"],
    },
    "tier2_nesting": {
        "season_in_month": {"coarse": "seasons", "fine": "months",
                            "map": {"Winter": ["December", "January", "February"],
                                    "Spring": ["March", "April", "May"],
                                    "Summer": ["June", "July", "August"],
                                    "Autumn": ["September", "October", "November"]}},
        "base10_in_base100": {"coarse": "numbers100", "fine": "numbers10"},
    },
    "tier3_magnitude": {
        "scalars": ["numbers", "costliness", "physical_size", "duration"],
        "bridge": "moon_illumination",  # from MOON_PHASES[*]['illum']
        "tests": ["pairwise_cosine", "shared_axis_pca", "cross_domain_transfer", "linear_vs_log"],
    },
    "tier4_worldmap": {
        "continents": ["europe", "america", "africa"],
        "compass": "directions",
        "tests": ["procrustes_to_latlong", "compass_frame_alignment"],
    },
    "tier5_antipodal": {
        "pairs": [["indoors", "outdoors"], ["lovingness", "harmfulness"]],
        "tests": ["cosine", "is_1d"],
    },
    "tier6_causal_sae": {
        "claims": ["seasons_in_month_plane", "indoor_eq_neg_outdoor", "shared_magnitude_axis"],
        "sae_repo": "google/gemma-scope-9b-pt-res",
        "note": "budget permitting only",
    },
}

# Convenience: flat list of every (concept, class) presence probe id, + scalar probe ids.
def presence_probe_ids():
    ids = []
    for cname, c in PRESENCE_CONCEPTS.items():
        for cls in c["classes"]:
            ids.append(f"{cname}::{cls}")
    return ids

def scalar_probe_ids():
    return [f"scalar::{s}" for s in SCALARS]

def all_probe_ids():
    return presence_probe_ids() + scalar_probe_ids()

if __name__ == "__main__":
    p = presence_probe_ids()
    s = scalar_probe_ids()
    print(f"presence probes: {len(p)}  (concepts: {len(PRESENCE_CONCEPTS)})")
    print(f"scalar   probes: {len(s)}")
    print(f"TOTAL probes per layer: {len(p)+len(s)}")
    print("presence classes per concept:",
          {k: len(v["classes"]) for k, v in PRESENCE_CONCEPTS.items()})
