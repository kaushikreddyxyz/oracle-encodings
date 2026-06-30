#!/usr/bin/env python3
"""Generate bespoke per-concept judge prompts for Step-1 labeling.

Produces, under overnight_run/prompts/:
  - prompts/{concept}/{safe_class}.txt  for every presence class (68)
  - prompts/scalar/{name}.txt           for every scalar (11)
  - prompts/registry.json               prompt_id -> relative path

Rubric contract (reconciles the brief's January & Wednesday worked bars):
  5  literal mention, correct sense
  4  unambiguous misspelling, clearly this class
  2-3 genuinely ambiguous surface hit
  1  the exact word as a PROPER NAME (person/place/brand/title/character)
  0  different-meaning homograph/idiom/figurative use, unrelated substring,
     or text that only EVOKES the concept without naming it
"""
import json
import os
import re
import sys

OVR = "/Users/kaushikreddy/Projects/oracle-encoding-project/oracle-encodings/overnight_run"
sys.path.insert(0, OVR)
import concepts  # noqa: E402

PROMPTS_DIR = os.path.join(OVR, "prompts")

# --------------------------------------------------------------------------- #
# shared blocks
# --------------------------------------------------------------------------- #
PRESENCE_RUBRIC = """SCORING (0-5)
5 - The target word (or a listed abbreviation) appears, correctly spelled, used in its literal sense as this class.
4 - It appears as an unambiguous misspelling/typo, but is clearly this class.
2-3 - A genuinely ambiguous surface hit; you cannot confidently say it is this class.
1 - The exact word appears, but as a PROPER NAME (a person, place, brand, title, or character), not the concept itself.
0 - The concept is NOT literally named in its real sense: a different-meaning homograph or idiom that merely shares the spelling, an unrelated substring, or text that only evokes/associates the concept."""

PRESENCE_OUT = "Respond with the integer score (0-5) as the very first token, optionally followed by a <=5-word reason."
SCALAR_OUT = "Respond with the integer rating (0-5) as the very first token, optionally followed by a <=5-word reason."


def render_presence(cls, fam, definition, inclusion, exclusion, shots):
    L = []
    L.append(f'You are a STRICT presence judge. Reading the text below, decide whether it LITERALLY mentions "{cls}" ({fam}) in its correct sense. Presence must be textual, not merely evoked.')
    L.append("")
    L.append("DEFINITION")
    L.append(definition)
    L.append("")
    L.append("COUNTS AS PRESENT (score 4-5)")
    L.append(inclusion)
    L.append("")
    L.append("DOES NOT COUNT (score 0-1)")
    L.append(exclusion)
    L.append("")
    L.append(PRESENCE_RUBRIC)
    L.append("")
    L.append("EXAMPLES")
    for text, score, reason in shots:
        L.append(f'{score}  "{text}"' + (f'   ({reason})' if reason else ""))
    L.append("")
    L.append(PRESENCE_OUT)
    return "\n".join(L) + "\n"


def render_scalar(scale_label, definition, scale_lines, boundary, shots, external=None):
    L = []
    L.append(f"You are a graded rating judge for the scale: {scale_label}.")
    L.append("")
    L.append("WHAT THIS MEASURES")
    L.append(definition)
    L.append("")
    L.append("RATING SCALE (0-5)")
    L.append(scale_lines)
    if external:
        L.append("")
        L.append("EXTERNAL VALUE")
        L.append(external)
    L.append("")
    L.append("BOUNDARY RULES")
    L.append(boundary)
    L.append("")
    L.append("EXAMPLES")
    for text, score, reason in shots:
        L.append(f'{score}  "{text}"' + (f'   ({reason})' if reason else ""))
    L.append("")
    L.append("Read tone and context (interpretive latitude) but stay grounded in the text; do not invent a rating the text does not support.")
    L.append(SCALAR_OUT)
    return "\n".join(L) + "\n"


def safe(cls):
    return re.sub(r"\s+", "_", cls)


# --------------------------------------------------------------------------- #
# PRESENCE — BESPOKE (hand-written) content: prompt_id -> dict
# --------------------------------------------------------------------------- #
M = "the calendar month"
D = "the day of the week"
C = "the color/hue"
S = "the season"
DIR = "the compass direction"
MOON = "the lunar phase"

PB = {}  # presence bespoke

# ---- MONTHS ----
PB["months::January"] = dict(fam=M,
    definition="The first month of the calendar year. Score it only when the text refers to this calendar month.",
    inclusion="january / Jan / Jan. in any casing; common misspellings such as 'Janurary' or 'Januray' when clearly the month.",
    exclusion="The surname or given name 'January' (e.g. actress January Jones) -> 1. New-Year imagery that never names the month -> 0. The substring 'jan' inside unrelated words ('janitor') -> 0.",
    shots=[
        ("My birthday is in January.", 5, "literal"),
        ("We met back in Jan. 2019.", 5, "abbreviation"),
        ("school starts the second week of january", 5, "lowercase"),
        ("It was a cold Janurary morning.", 4, "misspelling"),
        ("January Jones starred in the film.", 1, "name sense"),
        ("The new janitor mopped the hall.", 0, "substring, unrelated"),
        ("New Year's resolutions never last.", 0, "evokes, never names"),
        ("Q1 results beat expectations.", 0, "evokes quarter, not the month"),
        ("Due date: mid-January.", 5, "names the month"),
    ])
PB["months::February"] = dict(fam=M,
    definition="The second month of the calendar year. Score only references to this calendar month.",
    inclusion="february / Feb / Feb.; the very common misspelling 'Febuary' (missing first r) and 'Februrary' count when clearly the month.",
    exclusion="'febrile' (substring 'feb', means feverish) -> 0. Valentine's or Groundhog-Day imagery that never names the month -> 0.",
    shots=[
        ("Tax forms are due in February.", 5, "literal"),
        ("Signed on Feb. 12.", 5, "abbreviation"),
        ("we moved here last february", 5, "lowercase"),
        ("It happened in Febuary.", 4, "common misspelling"),
        ("A short, grey Februrary.", 4, "misspelling"),
        ("She had a febrile illness.", 0, "substring, means feverish"),
        ("Valentine's chocolates sold out.", 0, "evokes, never names"),
        ("Groundhog Day predictions varied.", 0, "evokes, never names"),
    ])
PB["months::March"] = dict(fam=M,
    definition="The third month of the calendar year. Note 'march' is also the verb/noun meaning to walk in step or a procession, and a surname.",
    inclusion="march / Mar / Mar. when it denotes the month; common misspelling 'Marhc'.",
    exclusion="'march' = to walk in step or a protest march (verb/noun) -> 0. 'time marches on' (idiom) -> 0. The surname 'March' (e.g. Frederic March) -> 1.",
    shots=[
        ("Spring break is in March.", 5, "literal"),
        ("Filed Mar. 3, 2020.", 5, "abbreviation"),
        ("march was unusually warm this year", 5, "month, context"),
        ("We met in Marhc.", 4, "misspelling"),
        ("March.", 3, "bare token, ambiguous"),
        ("The soldiers march at dawn.", 0, "verb, to walk in step"),
        ("Thousands joined the protest march.", 0, "noun, a procession"),
        ("Time marches on.", 0, "idiom"),
        ("Frederic March won an Oscar.", 1, "surname"),
        ("Warmer days are coming.", 0, "evokes, never names"),
    ])
PB["months::April"] = dict(fam=M,
    definition="The fourth month of the calendar year. 'April' is also a common given name.",
    inclusion="april / Apr / Apr.; misspelling 'Apirl'. 'April Fools' still names the month -> 5.",
    exclusion="The given name 'April' as a person/character (e.g. April O'Neil) -> 1. Spring imagery that never names the month -> 0.",
    shots=[
        ("Taxes are due in April.", 5, "literal"),
        ("Born in Apr. 1998.", 5, "abbreviation"),
        ("early april, the cherry blossoms opened", 5, "lowercase"),
        ("April Fools' Day pranks everywhere.", 5, "names the month"),
        ("It rained all Apirl.", 4, "misspelling"),
        ("April was my college roommate.", 1, "given name"),
        ("April O'Neil from the comics.", 1, "character name"),
        ("Spring showers started early.", 0, "evokes, never names"),
    ])
PB["months::May"] = dict(fam=M,
    definition="The fifth month of the calendar year. Because 'may' is also an extremely common modal verb and a surname, score 5 only when it clearly denotes the calendar month.",
    inclusion="may (the month), any casing; 'May Day' (May 1) names the month -> 5.",
    exclusion="The modal verb 'may' ('you may go', 'it may rain', 'May I?') -> 0. The idiom 'come what may' -> 0. The surname 'May' (e.g. Theresa May) -> 1.",
    shots=[
        ("We graduate in May.", 5, "literal"),
        ("The wedding is set for May 2025.", 5, "literal"),
        ("Late May was unusually hot.", 5, "month, context"),
        ("May Day is May 1st.", 5, "the holiday names the month"),
        ("May.", 3, "bare token, ambiguous"),
        ("You may leave now.", 0, "modal verb"),
        ("It may rain tomorrow.", 0, "modal verb"),
        ("Come what may.", 0, "idiom"),
        ("Theresa May resigned.", 1, "surname"),
        ("Spring flowers bloom.", 0, "evokes, never names"),
    ])
PB["months::June"] = dict(fam=M,
    definition="The sixth month of the calendar year. 'June' is also a common given name.",
    inclusion="june / Jun / Jun.; misspelling 'Juen'.",
    exclusion="The given name 'June' as a person (e.g. June Carter, Aunt June) -> 1. Summer imagery that never names the month -> 0.",
    shots=[
        ("Weddings peak in June.", 5, "literal"),
        ("Effective Jun. 1.", 5, "abbreviation"),
        ("mid-june heat wave", 5, "lowercase"),
        ("We leave in Juen.", 4, "misspelling"),
        ("June Carter sang harmony.", 1, "given name"),
        ("Aunt June visited.", 1, "given name"),
        ("Summer break begins soon.", 0, "evokes, never names"),
        ("Born in June 1990.", 5, "literal"),
    ])
PB["months::July"] = dict(fam=M,
    definition="The seventh month of the calendar year.",
    inclusion="july / Jul / Jul.; misspelling 'Jully'.",
    exclusion="'Jul' in the Scandinavian sense (= Yule/Christmas) -> 0. US Independence-Day imagery that never names the month -> 0.",
    shots=[
        ("Fireworks light up July.", 5, "literal"),
        ("Dated Jul. 4, 1776.", 5, "abbreviation"),
        ("early july, the lake warmed up", 5, "lowercase"),
        ("It was a humid Jully night.", 4, "misspelling"),
        ("Jul is the Scandinavian word for Christmas.", 0, "different word, Yule"),
        ("Independence Day cookouts everywhere.", 0, "evokes, never names"),
        ("Born in July 1990.", 5, "literal"),
    ])
PB["months::August"] = dict(fam=M,
    definition="The eighth month of the calendar year. Note 'august' is also an adjective meaning majestic/dignified, and a given name.",
    inclusion="august / Aug / Aug. when it denotes the month; misspelling 'Agust'.",
    exclusion="'august' = majestic/dignified (adjective) -> 0. The name Augustus / Augusta, or a character named August -> 1. Late-summer imagery that never names the month -> 0.",
    shots=[
        ("School resumes in August.", 5, "literal"),
        ("Signed Aug. 9.", 5, "abbreviation"),
        ("mid-august thunderstorms", 5, "lowercase"),
        ("We move in Agust.", 4, "misspelling"),
        ("An august assembly of scholars.", 0, "adjective, majestic"),
        ("Augustus ruled Rome.", 1, "name"),
        ("August is the main character's name.", 1, "given name"),
        ("Late-summer heat lingered.", 0, "evokes, never names"),
    ])
PB["months::September"] = dict(fam=M,
    definition="The ninth month of the calendar year.",
    inclusion="september / Sep / Sept / Sep. / Sept.; misspelling 'Septmber'.",
    exclusion="The Latin root 'sept-' meaning seven (e.g. 'septet') -> 0. Back-to-school imagery that never names the month -> 0.",
    shots=[
        ("School starts in September.", 5, "literal"),
        ("Due Sept. 30.", 5, "abbreviation"),
        ("Filed in Sep. 2019.", 5, "abbreviation"),
        ("It was a crisp Septmber morning.", 4, "misspelling"),
        ("'Sept-' is the Latin root for seven.", 0, "root, different sense"),
        ("late september, leaves turned", 5, "lowercase"),
        ("Back-to-school sales began.", 0, "evokes, never names"),
    ])
PB["months::October"] = dict(fam=M,
    definition="The tenth month of the calendar year.",
    inclusion="october / Oct / Oct.; misspelling 'Ocotber'.",
    exclusion="The root 'oct-' meaning eight (e.g. 'octagon') -> 0. Halloween or pumpkin-spice imagery that never names the month -> 0.",
    shots=[
        ("Halloween falls in October.", 5, "literal"),
        ("Released Oct. 31.", 5, "abbreviation"),
        ("mid-october frost", 5, "lowercase"),
        ("A foggy Ocotber evening.", 4, "misspelling"),
        ("'Oct-' is the prefix for eight.", 0, "root, different sense"),
        ("Pumpkin spice everything.", 0, "evokes, never names"),
        ("october was rainy", 5, "lowercase"),
    ])
PB["months::November"] = dict(fam=M,
    definition="The eleventh month of the calendar year.",
    inclusion="november / Nov / Nov.; misspelling 'Novmber'.",
    exclusion="'nov' inside unrelated words ('novel', 'innovation') -> 0. 'Nov' as the abbreviation of a different word (e.g. 'novice' in a log) -> 0. Thanksgiving imagery that never names the month -> 0.",
    shots=[
        ("Elections are held in November.", 5, "literal"),
        ("Dated Nov. 5.", 5, "abbreviation"),
        ("early november chill", 5, "lowercase"),
        ("A grey Novmber sky.", 4, "misspelling"),
        ("She wrote a novel.", 0, "substring, unrelated"),
        ("'Nov' is short for novice in the logs.", 0, "abbrev of another word"),
        ("Thanksgiving plans took shape.", 0, "evokes, never names"),
    ])
PB["months::December"] = dict(fam=M,
    definition="The twelfth month of the calendar year.",
    inclusion="december / Dec / Dec.; misspelling 'Decmber'.",
    exclusion="'dec' as an abbreviation of a different word (decimal, decrement, declination) -> 0. 'dec' inside unrelated words ('decide') -> 0. Christmas imagery that never names the month -> 0.",
    shots=[
        ("Christmas is in December.", 5, "literal"),
        ("Signed Dec. 24.", 5, "abbreviation"),
        ("late december blizzard", 5, "lowercase"),
        ("A snowy Decmber day.", 4, "misspelling"),
        ("Set dec to 0 in the config.", 0, "abbrev, decrement/decimal"),
        ("She had to decide quickly.", 0, "substring, unrelated"),
        ("Holiday lights went up everywhere.", 0, "evokes, never names"),
    ])

# ---- DAYS ----
PB["days::Monday"] = dict(fam=D,
    definition="The first weekday, Monday. Note 'mon' collides with French 'mon' and is a substring of many words.",
    inclusion="monday / Mon / Mon.; misspelling 'Munday'.",
    exclusion="'mon' inside unrelated words ('money', 'monitor') -> 0. French 'mon' ('mon ami') -> 0. The Mon people/language of Myanmar (proper noun) -> 1. Start-of-week dread imagery that never names the day -> 0.",
    shots=[
        ("See you Monday.", 5, "literal"),
        ("The meeting moved to Mon.", 5, "abbreviation"),
        ("every monday we meet", 5, "lowercase"),
        ("Ugh, it's Munday again.", 4, "misspelling"),
        ("I deposited the money.", 0, "substring, unrelated"),
        ("Bonjour, mon ami.", 0, "French, different word"),
        ("Mon is a language of Myanmar.", 1, "proper noun"),
        ("Start-of-week dread set in.", 0, "evokes, never names"),
    ])
PB["days::Tuesday"] = dict(fam=D,
    definition="The second weekday, Tuesday.",
    inclusion="tuesday / Tue / Tues / Tue.; misspelling 'Teusday'.",
    exclusion="'tue' inside unrelated words ('statue') -> 0. Imagery of 'two days into the week' that never names the day -> 0.",
    shots=[
        ("Tacos every Tuesday.", 5, "literal"),
        ("Rescheduled to Tue.", 5, "abbreviation"),
        ("Class is Tues. morning.", 5, "abbreviation"),
        ("See you Teusday.", 4, "misspelling"),
        ("The statue was unveiled.", 0, "substring, unrelated"),
        ("tuesday's child is full of grace", 5, "lowercase"),
        ("Two days into the week already.", 0, "evokes, never names"),
    ])
PB["days::Wednesday"] = dict(fam=D,
    definition="The midweek day, Wednesday. Note 'wed' is also the verb meaning to marry, and 'Wednesday' is a character name.",
    inclusion="wednesday / Wed / Weds / Wed.; misspellings 'Wensday', 'Wednsday'.",
    exclusion="'wed' = to marry (verb) -> 0. 'Wednesday Addams' (character) -> 1. Midweek-slump imagery that never names the day -> 0.",
    shots=[
        ("See you Wednesday.", 5, "literal"),
        ("Moved to Wed.", 5, "abbreviation"),
        ("Weds. works for me.", 5, "abbreviation"),
        ("It's Wensday already?", 4, "misspelling"),
        ("Meeting Wednsday.", 4, "misspelling"),
        ("Wednesday Addams is iconic.", 1, "character name"),
        ("They wed last summer.", 0, "verb, to marry"),
        ("The midweek slump hit hard.", 0, "evokes, never names"),
    ])
PB["days::Thursday"] = dict(fam=D,
    definition="The fifth weekday, Thursday.",
    inclusion="thursday / Thu / Thur / Thurs / Thu.; misspelling 'Thrusday'.",
    exclusion="A pet/character literally named 'Thursday' (proper name) -> 1. 'Almost the weekend' imagery that never names the day -> 0.",
    shots=[
        ("Happy Thursday!", 5, "literal"),
        ("Due Thu.", 5, "abbreviation"),
        ("Thurs. is fine.", 5, "abbreviation"),
        ("See you Thrusday.", 4, "misspelling"),
        ("thursday night football", 5, "lowercase"),
        ("Thursday is the dog's name.", 1, "pet name"),
        ("Almost the weekend now.", 0, "evokes, never names"),
    ])
PB["days::Friday"] = dict(fam=D,
    definition="The sixth weekday, Friday. Note 'Friday' is also a character/surname.",
    inclusion="friday / Fri / Fri.; misspelling 'Firday'. 'Good Friday' and 'casual Friday' name the day -> 5.",
    exclusion="The character Friday (Robinson Crusoe's 'Man Friday') or the surname Friday (Sgt. Joe Friday) -> 1. 'fri' inside unrelated words ('fried') -> 0. End-of-week relief imagery that never names the day -> 0.",
    shots=[
        ("TGIF - see you Friday.", 5, "literal"),
        ("Shipped Fri.", 5, "abbreviation"),
        ("Good Friday service this week.", 5, "names the day"),
        ("casual friday at work", 5, "lowercase"),
        ("Is it Firday yet?", 4, "misspelling"),
        ("Crusoe named him Friday.", 1, "character name"),
        ("Sergeant Friday, badge 714.", 1, "surname"),
        ("Fried rice for dinner.", 0, "substring, unrelated"),
        ("End-of-week relief washed over me.", 0, "evokes, never names"),
    ])
PB["days::Saturday"] = dict(fam=D,
    definition="The weekend day Saturday. Note 'sat' is also the past tense of 'sit', and 'SAT' is an exam name.",
    inclusion="saturday / Sat / Sat.; misspelling 'Saterday'.",
    exclusion="'sat' = past tense of sit (verb) -> 0. 'SAT' the exam (proper name) -> 1. 'satur' inside 'Saturn' -> 0. Weekend-chores imagery that never names the day -> 0.",
    shots=[
        ("Let's meet Saturday.", 5, "literal"),
        ("Open Sat. 9-5.", 5, "abbreviation"),
        ("saturday morning cartoons", 5, "lowercase"),
        ("Party on Saterday.", 4, "misspelling"),
        ("She sat down quietly.", 0, "verb, past tense of sit"),
        ("He's retaking the SAT.", 1, "exam, proper name"),
        ("Saturn has rings.", 0, "substring, unrelated"),
        ("Weekend chores piled up.", 0, "evokes, never names"),
    ])
PB["days::Sunday"] = dict(fam=D,
    definition="The weekend day Sunday. Note 'sun' is also the star, 'sundae' is a dessert, and 'Sunday' is a surname.",
    inclusion="sunday / Sun / Sun.; misspelling 'Sundey'.",
    exclusion="'sun' = the star -> 0. 'sundae' the dessert -> 0. The surname Sunday (e.g. Billy Sunday) -> 1. Day-of-rest imagery that never names the day -> 0.",
    shots=[
        ("Brunch this Sunday.", 5, "literal"),
        ("Closed Sun.", 5, "abbreviation"),
        ("sunday roast tradition", 5, "lowercase"),
        ("See you Sundey.", 4, "misspelling"),
        ("Sun.", 3, "bare token, ambiguous"),
        ("The sun rose over the hills.", 0, "the star, different word"),
        ("I ordered a hot fudge sundae.", 0, "dessert, different word"),
        ("Billy Sunday preached here.", 1, "surname"),
        ("A day of rest and worship.", 0, "evokes, never names"),
    ])

# ---- COLORS (bespoke: primaries/secondaries + amber + teal) ----
PB["color_wheel::Red"] = dict(fam=C,
    definition="The color red - the hue/pigment/light at the long-wavelength end of the visible spectrum. Score only literal references to the color.",
    inclusion="red, any casing; misspelling 'rred'/'redd'.",
    exclusion="Figurative 'red' = debt ('in the red'), communism ('better dead than red'), or anger ('saw red') -> 0. 'red' inside unrelated words ('hundred', 'bored') -> 0. A nickname/surname Red (e.g. Red Auerbach) -> 1.",
    shots=[
        ("Paint the door red.", 5, "literal"),
        ("She wore a red dress.", 5, "literal"),
        ("the traffic light turned red", 5, "literal"),
        ("A bright rred balloon.", 4, "misspelling"),
        ("We're in the red this quarter.", 0, "figurative, debt"),
        ("Better dead than red.", 0, "figurative, communism"),
        ("He saw red and stormed off.", 0, "figurative, anger"),
        ("Two hundred guests attended.", 0, "substring, unrelated"),
        ("Red Auerbach coached the Celtics.", 1, "nickname"),
    ])
PB["color_wheel::Orange"] = dict(fam=C,
    definition="The color orange - the hue between red and yellow. Distinguish it sharply from the fruit of the same name.",
    inclusion="orange, any casing; misspelling 'oragne'.",
    exclusion="The fruit 'orange' (peeling/juice/eating) -> 0. A place or brand named Orange (Orange County, the carrier Orange, Agent Orange) -> 1.",
    shots=[
        ("She painted the wall orange.", 5, "the color"),
        ("an orange sunset", 5, "the color"),
        ("traffic cones are orange", 5, "the color"),
        ("A bright oragne scarf.", 4, "misspelling"),
        ("I peeled an orange.", 0, "the fruit"),
        ("freshly squeezed orange juice", 0, "the fruit"),
        ("Orange County, California.", 1, "place name"),
        ("Agent Orange was a defoliant.", 1, "proper name"),
    ])
PB["color_wheel::Yellow"] = dict(fam=C,
    definition="The color yellow - the bright hue between orange and green. Score only literal color references.",
    inclusion="yellow, any casing; misspelling 'yelow'.",
    exclusion="Figurative 'yellow' = cowardly ('a yellow streak'), or 'yellow journalism' -> 0. A place/brand with Yellow in its name (the Yellow River, Yellow Pages) -> 1.",
    shots=[
        ("a yellow raincoat", 5, "the color"),
        ("The walls are pale yellow.", 5, "the color"),
        ("yellow daffodils", 5, "the color"),
        ("A yelow taxi.", 4, "misspelling"),
        ("Don't be so yellow.", 0, "figurative, cowardly"),
        ("Yellow journalism sold papers.", 0, "figurative idiom"),
        ("He has a yellow streak.", 0, "figurative, cowardice"),
        ("the Yellow River flooded", 1, "place name"),
    ])
PB["color_wheel::Green"] = dict(fam=C,
    definition="The color green - the hue between yellow and blue. Score only literal color references.",
    inclusion="green, any casing; misspelling 'greeen'.",
    exclusion="Figurative 'green' = inexperienced ('still green'), eco-friendly ('going green'), or envious ('green with envy') -> 0. A 'green' meaning a lawn/common, or the surname Green (Al Green) -> 0/1 (lawn=0, surname=1).",
    shots=[
        ("freshly painted green fence", 5, "the color"),
        ("green leaves in spring", 5, "the color"),
        ("the traffic light went green", 5, "the color"),
        ("A greeen meadow.", 4, "misspelling"),
        ("He's still green at this job.", 0, "figurative, inexperienced"),
        ("going green to save the planet", 0, "figurative, eco"),
        ("green with envy", 0, "figurative idiom"),
        ("They met on the village green.", 0, "a lawn, different sense"),
        ("Al Green sang soul classics.", 1, "surname"),
    ])
PB["color_wheel::Blue"] = dict(fam=C,
    definition="The color blue - the hue between green and violet. Score only literal color references.",
    inclusion="blue, any casing; misspelling 'bleu' (French spelling, clearly the color).",
    exclusion="Figurative 'blue' = sad ('feeling blue'), or the music genre 'the blues', or idioms ('out of the blue', 'blue-collar') -> 0. A name/nickname Blue (a pet, a person) -> 1.",
    shots=[
        ("a clear blue sky", 5, "the color"),
        ("She has blue eyes.", 5, "the color"),
        ("painted the boat blue", 5, "the color"),
        ("the deep bleu sea", 4, "misspelling"),
        ("I'm feeling blue today.", 0, "figurative, sad"),
        ("He plays the blues.", 0, "music genre"),
        ("It came out of the blue.", 0, "idiom"),
        ("blue-collar workers", 0, "idiom, not the color"),
        ("Blue is the dog's name.", 1, "pet name"),
    ])
PB["color_wheel::Purple"] = dict(fam=C,
    definition="The color purple/violet - the hue between blue and red. The lexicon includes 'violet' as a synonym for this hue.",
    inclusion="purple or violet, any casing; misspelling 'purrple'.",
    exclusion="Figurative 'purple prose' or 'born to the purple' -> 0. The given name Violet, or 'Purple Heart' (medal) -> 1. The flower 'violet' or 'ultraviolet' light -> 0 (different sense / substring).",
    shots=[
        ("she wore a purple gown", 5, "the color"),
        ("violet twilight sky", 5, "violet as color"),
        ("purple and gold banners", 5, "the color"),
        ("A purrple curtain.", 4, "misspelling"),
        ("his purple prose annoyed readers", 0, "figurative idiom"),
        ("She picked a violet from the garden.", 0, "the flower, different sense"),
        ("ultraviolet light", 0, "substring, different word"),
        ("Violet is her daughter's name.", 1, "given name"),
        ("He earned a Purple Heart.", 1, "medal, proper name"),
    ])
PB["color_wheel::Yellow-Orange"] = dict(fam=C,
    definition="The tertiary hue yellow-orange, also called amber - between yellow and orange. Score literal color references; 'amber' is also a name, a gemstone material, and a traffic signal.",
    inclusion="yellow-orange / yellow orange / amber when it names the hue (including the amber traffic light); misspelling 'ambr'.",
    exclusion="The given name Amber (a person) -> 1. The fossil-resin material 'amber' (a fly trapped in amber, amber jewelry) -> 0. Autumn-warmth imagery that never names the hue -> 0.",
    shots=[
        ("the leaves turned a yellow-orange", 5, "the hue"),
        ("an amber glow at sunset", 5, "the hue"),
        ("the signal turned amber", 5, "the traffic color"),
        ("yellow orange flames", 5, "the hue"),
        ("An ambr sky.", 4, "misspelling"),
        ("Amber wrote the report.", 1, "given name"),
        ("a fly trapped in amber", 0, "the material, different sense"),
        ("she collected amber jewelry", 0, "the gemstone, different sense"),
    ])
PB["color_wheel::Blue-Green"] = dict(fam=C,
    definition="The tertiary hue blue-green, also called teal or cyan - between blue and green. Score literal color references; 'teal' is also a species of duck.",
    inclusion="blue-green / blue green / teal / cyan when it names the hue; misspelling 'teel'.",
    exclusion="'teal' the duck (a teal flew off the pond) -> 0. A brand/name 'Teal' -> 1. Tropical-water imagery that never names the hue -> 0.",
    shots=[
        ("a blue-green lagoon", 5, "the hue"),
        ("teal accents on the walls", 5, "the hue"),
        ("cyan ink in the printer", 5, "the hue"),
        ("blue green sea glass", 5, "the hue"),
        ("A teel door.", 4, "misspelling of teal"),
        ("a teal flew off the pond", 0, "the duck, different sense"),
        ("Teal is the studio's brand name.", 1, "brand name"),
        ("the tropical water shimmered", 0, "evokes, never names"),
    ])

# ---- SEASONS ----
PB["seasons::Spring"] = dict(fam=S,
    definition="The season spring (between winter and summer). Note 'spring' also means a metal coil, a natural water source, and the verb to leap.",
    inclusion="spring (the season), any casing; 'springtime' and 'spring break' name the season -> 5; misspelling 'Sprng'.",
    exclusion="A mechanical spring (coil), a water spring, or the verb to leap ('sprang') -> 0. A named historical event 'the Arab Spring' -> 1. New-beginnings imagery that never names the season -> 0.",
    shots=[
        ("Flowers bloom in spring.", 5, "the season"),
        ("We met last spring.", 5, "the season"),
        ("springtime in Paris", 5, "names the season"),
        ("spring break plans", 5, "names the season"),
        ("A warm Sprng morning.", 4, "misspelling"),
        ("The mattress spring broke.", 0, "mechanical coil"),
        ("water flows from a mountain spring", 0, "water source"),
        ("he sprang to his feet", 0, "verb, to leap"),
        ("the Arab Spring of 2011", 1, "named event"),
        ("New beginnings everywhere.", 0, "evokes, never names"),
    ])
PB["seasons::Summer"] = dict(fam=S,
    definition="The season summer (the warmest season). 'Summer' is also a common given name.",
    inclusion="summer (the season), any casing; 'Indian summer' names the season -> 5; 'they summer in Maine' (verb) still refers to the season -> 5; misspelling 'sumer'.",
    exclusion="The given name Summer, or the surname Summer (Donna Summer) -> 1. Beach/sunscreen imagery that never names the season -> 0.",
    shots=[
        ("We swim all summer.", 5, "the season"),
        ("a hot summer day", 5, "the season"),
        ("Indian summer warmth", 5, "names the season"),
        ("they summer in Maine", 5, "verb, still the season"),
        ("A long sumer holiday.", 4, "misspelling"),
        ("Summer is my niece's name.", 1, "given name"),
        ("Donna Summer sang disco.", 1, "surname"),
        ("beach vacations and sunscreen", 0, "evokes, never names"),
    ])
PB["seasons::Autumn"] = dict(fam=S,
    definition="The season autumn, also called fall (US English). Note 'fall' is also the very common verb/noun meaning to drop, a decline, or a collapse.",
    inclusion="autumn or fall when denoting the season, any casing; 'fall foliage', 'in the fall' -> 5; misspelling 'Autum'.",
    exclusion="'fall' = to drop (verb), a decline in prices, or a collapse ('the fall of Rome') -> 0. The given name Autumn -> 1. Harvest imagery that never names the season -> 0.",
    shots=[
        ("Leaves change color in autumn.", 5, "the season"),
        ("We started school in the fall.", 5, "fall = the season (US)"),
        ("fall foliage tours", 5, "the season"),
        ("a crisp autumn breeze", 5, "the season"),
        ("A golden Autum afternoon.", 4, "misspelling"),
        ("Be careful not to fall.", 0, "verb, to drop"),
        ("a sharp fall in prices", 0, "noun, a decline"),
        ("the fall of Rome", 0, "collapse, different sense"),
        ("Autumn is her daughter's name.", 1, "given name"),
        ("Harvest time approaches.", 0, "evokes, never names"),
    ])
PB["seasons::Winter"] = dict(fam=S,
    definition="The season winter (the coldest season). 'Winter' is also a given name.",
    inclusion="winter (the season), any casing; 'they winter in Florida' (verb) and 'midwinter' refer to the season -> 5; misspelling 'Wintr'.",
    exclusion="Figurative 'a winter of decline' or 'nuclear winter' -> 0. The given name Winter -> 1. Cold/snow imagery that never names the season -> 0.",
    shots=[
        ("It snows every winter.", 5, "the season"),
        ("a harsh winter storm", 5, "the season"),
        ("they winter in Florida", 5, "verb, still the season"),
        ("midwinter festival", 5, "names the season"),
        ("A cold Wintr night.", 4, "misspelling"),
        ("the economy entered a winter of decline", 0, "figurative"),
        ("a winter of discontent", 0, "figurative idiom"),
        ("Winter is the character's name.", 1, "given name"),
        ("Cold and snow everywhere.", 0, "evokes, never names"),
    ])

# ---- DIRECTIONS ----
PB["directions::North"] = dict(fam=DIR,
    definition="The compass direction north (toward the North Pole), including 'northward' and 'northern' when they describe orientation or the northern part of a region.",
    inclusion="north / northward / northern when directional (head north, the northern slopes); misspelling 'norht'.",
    exclusion="The surname North (Oliver North) -> 1. A lexicalized place name where 'north' is fixed (North Korea, North Dakota) -> 1. Figurative 'north of $5M' (= more than) or 'true north' (= guiding principle) -> 0.",
    shots=[
        ("Head north for two miles.", 5, "direction"),
        ("Geese fly north in spring.", 5, "direction"),
        ("the northern slopes are icy", 5, "directional adjective"),
        ("the compass needle points north", 5, "direction"),
        ("we drove norht for hours", 4, "misspelling"),
        ("Oliver North testified.", 1, "surname"),
        ("North Korea launched a test.", 1, "country name"),
        ("profits came in north of $5M", 0, "figurative, more than"),
        ("true north of his ambitions", 0, "figurative, guiding principle"),
    ])
PB["directions::East"] = dict(fam=DIR,
    definition="The compass direction east (toward sunrise), including 'eastward' and 'eastern' when they describe orientation or the eastern part of a region.",
    inclusion="east / eastward / eastern when directional (drive east, the eastern shore, eastern Europe = the east part); misspelling 'eastwrd'.",
    exclusion="The surname East -> 1. The fixed region name 'the Middle East' (where East does not mean a direction) -> 1. 'east' inside 'Easter' -> 0. Exotic-faraway imagery that never names a direction -> 0.",
    shots=[
        ("Drive east until the river.", 5, "direction"),
        ("the sun rises in the east", 5, "direction"),
        ("the eastern shore", 5, "directional adjective"),
        ("eastern Europe got colder", 5, "the east part"),
        ("they sailed eastwrd", 4, "misspelling"),
        ("tensions in the Middle East", 1, "fixed region name"),
        ("Mr. East signed the form.", 1, "surname"),
        ("Easter egg hunt", 0, "substring, unrelated"),
        ("exotic, faraway lands", 0, "evokes, never names"),
    ])
PB["directions::South"] = dict(fam=DIR,
    definition="The compass direction south (toward the South Pole), including 'southward' and 'southern' when they describe orientation or the southern part of a region.",
    inclusion="south / southward / southern when directional (head south, the southern face, the river flows southward); misspelling 'soth'.",
    exclusion="The surname South -> 1. A fixed region/country name (the South (US region), South Korea, southern hospitality = the cultural South) -> 1. Figurative 'things went south' (= deteriorated) -> 0.",
    shots=[
        ("Migrate south for winter.", 5, "direction"),
        ("the southern face of the mountain", 5, "directional adjective"),
        ("head due south", 5, "direction"),
        ("the river flows southward", 5, "direction"),
        ("we walked soth all day", 4, "misspelling"),
        ("raised in the South", 1, "US region name"),
        ("South Korea's economy grew", 1, "country name"),
        ("things went south fast", 0, "figurative, deteriorated"),
    ])
PB["directions::West"] = dict(fam=DIR,
    definition="The compass direction west (toward sunset), including 'westward' and 'western' when they describe orientation or the western part of a region.",
    inclusion="west / westward / western when directional (travel west, the western ridge, western Europe = the west part); misspelling 'wst'.",
    exclusion="The surname West (Kanye West) -> 1. 'a western' = the film genre -> 0. 'the West' = civilization/geopolitical bloc -> 1. Frontier/cowboy imagery that never names a direction -> 0.",
    shots=[
        ("Travel west toward the coast.", 5, "direction"),
        ("the western ridge", 5, "directional adjective"),
        ("the wagons rolled westward", 5, "direction"),
        ("western Europe got rain", 5, "the west part"),
        ("drive wst on Route 9", 4, "misspelling"),
        ("Kanye West released an album.", 1, "surname"),
        ("we watched an old western", 0, "film genre, different sense"),
        ("the West during the Cold War", 1, "geopolitical bloc"),
        ("frontier and cowboys", 0, "evokes, never names"),
    ])

# ---- MOON PHASES (bespoke: the four with traps) ----
PB["moon_phases::New Moon"] = dict(fam=MOON,
    definition="The new-moon phase, when the Moon is between Earth and Sun and its disk is unlit (0% illumination). Score only when this specific phase is named.",
    inclusion="'new moon' (the phase); misspelling 'new mooon'.",
    exclusion="A different phase named instead (full moon, crescent) -> 0. The book/film title 'New Moon' (Twilight) -> 1. 'moonless night' or 'a new month' that never names the phase -> 0.",
    shots=[
        ("The new moon rises tonight.", 5, "the phase"),
        ("Plant seeds at the new moon.", 5, "the phase"),
        ("tonight's new moon in Aries", 5, "the phase"),
        ("during the new mooon", 4, "misspelling"),
        ("I read the novel New Moon.", 1, "book/film title"),
        ("the full moon was bright", 0, "a different phase"),
        ("waxing crescent appeared", 0, "a different phase"),
        ("a moonless night", 0, "evokes, never names"),
    ])
PB["moon_phases::Full Moon"] = dict(fam=MOON,
    definition="The full-moon phase, when the Moon's disk is fully illuminated (100%). 'Full moon' used for lunacy/werewolf imagery still names the phase -> high.",
    inclusion="'full moon' (the phase), including figurative 'howl at the full moon'; misspelling 'fulll moon'.",
    exclusion="A different phase named instead (crescent, new moon) -> 0. A band/brand literally named 'Full Moon' -> 1. 'the moon was bright' that never says full -> 0.",
    shots=[
        ("A full moon lit the field.", 5, "the phase"),
        ("Werewolves howl at the full moon.", 5, "still names the phase"),
        ("the harvest full moon", 5, "the phase"),
        ("under the fulll moon", 4, "misspelling"),
        ("the band Full Moon played", 1, "proper name"),
        ("a crescent moon hung low", 0, "a different phase"),
        ("the moon was bright", 0, "doesn't say full"),
        ("lunar tides peaked", 0, "evokes, never names"),
    ])
PB["moon_phases::First Quarter"] = dict(fam=MOON,
    definition="The first-quarter phase, when half the Moon's disk is lit and it is waxing (~50% illumination, one week after new moon). Note 'first quarter' also means a fiscal/sports period.",
    inclusion="'first quarter' / 'first-quarter' when naming the lunar phase; misspelling 'first quater'.",
    exclusion="Fiscal 'first-quarter earnings' or a sports 'first quarter' -> 0. A different phase named instead -> 0. 'half the moon was lit' that never names the phase -> 0.",
    shots=[
        ("The moon reached first quarter.", 5, "the phase"),
        ("a first-quarter moon tonight", 5, "the phase"),
        ("waxing toward first quarter", 5, "the phase"),
        ("the first quater moon", 4, "misspelling"),
        ("first-quarter earnings rose", 0, "fiscal Q1"),
        ("first quarter of the game", 0, "sports period"),
        ("the full moon appeared", 0, "a different phase"),
        ("half the moon was lit", 0, "evokes, never names"),
    ])
PB["moon_phases::Last Quarter"] = dict(fam=MOON,
    definition="The last-quarter phase (also called third quarter), when half the Moon's disk is lit and it is waning (~50% illumination, three weeks after new moon). Note 'quarter' also means a coin or a fiscal/sports period.",
    inclusion="'last quarter' / 'last-quarter' / 'third quarter' when naming the lunar phase; misspelling 'last quater'.",
    exclusion="Fiscal 'third-quarter revenue', a sports 'third quarter', or 'a quarter' (coin) -> 0. A different phase named instead -> 0.",
    shots=[
        ("The moon is at last quarter.", 5, "the phase"),
        ("a third-quarter moon tonight", 5, "the phase"),
        ("last-quarter moon before dawn", 5, "the phase"),
        ("the last quater phase", 4, "misspelling"),
        ("third-quarter revenue grew", 0, "fiscal Q3"),
        ("the third quarter ended 21-14", 0, "sports period"),
        ("I found a quarter on the ground", 0, "coin"),
        ("the new moon tonight", 0, "a different phase"),
    ])


# --------------------------------------------------------------------------- #
# PRESENCE — TEMPLATED generators (rote families, still class-specific)
# --------------------------------------------------------------------------- #
def misspell(word):
    """Drop one interior character of the last whitespace-token (clear typo)."""
    parts = word.split()
    w = parts[-1]
    if len(w) >= 5:
        i = len(w) // 2
        w = w[:i] + w[i + 1:]
    else:
        w = w + w[-1]  # double last letter for short words
    parts[-1] = w
    return " ".join(parts)


# ---- color tertiaries (Red-Orange, Yellow-Green, Blue-Purple, Red-Purple) ----
TERTIARY_EXTRA = {
    "Red-Orange": [("vermilion cliffs in Arizona", 1, "place name")],
    "Yellow-Green": [("a glass of green Chartreuse liqueur", 1, "the liqueur/brand")],
    "Blue-Purple": [("the Indigo Girls performed", 1, "band name")],
    "Red-Purple": [],
}


def render_tertiary(cls, lexicon):
    hyph = lexicon[0]                 # e.g. "red-orange"
    space = lexicon[1] if len(lexicon) > 1 and " " in lexicon[1] else hyph.replace("-", " ")
    synonym = lexicon[-1]            # e.g. "vermilion"
    definition = (f"The tertiary hue {cls.lower()}, also called {synonym} - the intermediate color "
                  f"between its two named components on the artist's color wheel. Score only literal references to this hue.")
    inclusion = f"{hyph} / {space} / {synonym} when naming the hue; misspelling '{misspell(synonym)}'."
    exclusion = (f"A neighbouring but distinct color named instead (e.g. pure red or pure orange) -> 0. "
                 f"Any proper name (place/brand/title) that merely contains the word -> 1. "
                 f"Imagery that evokes the shade without naming it -> 0.")
    shots = [
        (f"a {hyph} sunset", 5, "the hue"),
        (f"painted in {synonym}", 5, "the hue"),
        (f"{space} highlights on the canvas", 5, "the hue"),
        (f"a {misspell(synonym)} streak", 4, "misspelling"),
        ("a pure red wall", 0, "a neighbouring color"),
        ("a richly colored sky", 0, "evokes, never names"),
    ]
    shots += TERTIARY_EXTRA.get(cls, [])
    return render_presence(cls, C, definition, inclusion, exclusion, shots)


# ---- templated moon phases (the four crescent/gibbous phases) ----
def render_moon_templated(cls, lexicon):
    phrase = lexicon[0]
    definition = (f"The lunar phase '{phrase}'. Score only when this specific phase is named; other phases, "
                  f"or generic 'the moon', do not count.")
    inclusion = f"'{phrase}' (the phase); misspelling '{misspell(phrase)}'."
    exclusion = ("A different phase named instead (new moon, full moon, a quarter, the opposite waxing/waning phase) -> 0. "
                 "A proper name/title that merely contains the words -> 1. Generic moon imagery that never names this phase -> 0.")
    shots = [
        (f"a {phrase} tonight", 5, "the phase"),
        (f"the moon is a {phrase}", 5, "the phase"),
        (f"watching the {phrase} rise", 5, "the phase"),
        (f"a {misspell(phrase)} appeared", 4, "misspelling"),
        ("the new moon was invisible", 0, "a different phase"),
        ("a full moon rose", 0, "a different phase"),
        ("the moon was simply bright", 0, "doesn't name this phase"),
        ("stargazing under a clear sky", 0, "evokes, never names"),
    ]
    return render_presence(cls, MOON, definition, inclusion, exclusion, shots)


# ---- numbers10 (presence of a single integer 0..10) ----
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
NUM_MIS = {"zero": "zeero", "one": "onne", "two": "twoo", "three": "thre", "four": "fuor",
           "five": "fvie", "six": "sixx", "seven": "sevn", "eight": "eihgt", "nine": "nien", "ten": "tenn"}
NUM_HOMOPHONE = {
    "1": ("won", "She won the race.", "homophone of 'won'"),
    "2": ("to/too", "I'm going to the store too.", "homophone of 'to'/'too'"),
    "4": ("for/fore", "This gift is for you.", "homophone of 'for'"),
    "8": ("ate", "He ate the whole pie.", "homophone of 'ate'"),
}


def render_numbers10(n):
    word = NUM_WORDS[n]
    alt = (n + 3) % 11
    if alt == n:
        alt = (n + 1) % 11
    definition = (f"Presence of the number {n} (the integer value {n}), written as the digit '{n}' or the "
                  f"word '{word}', used as a genuine quantity/count. The token must denote {n} exactly - not a "
                  f"larger number that merely contains the digit.")
    inclusion = f"the digit '{n}' or the word '{word}' (any casing) denoting the value {n}; misspelling '{NUM_MIS[word]}'."
    exclusion = (f"A different number ({alt}, or a multi-digit number such as '{n}{n}' or a year) -> 0. "
                 f"The word '{word}' used in a non-numeric sense (e.g. the pronoun 'one') -> 0. "
                 f"Vague quantity words ('several', 'a few') that never name {n} -> 0.")
    shots = [
        (f"I counted {n} birds", 5, "the number, a quantity"),
        (f"{word} people arrived", 5, "word form, a quantity"),
        (f"turn to page {n}", 5, "the number is present"),
        (f"a {NUM_MIS[word]} of them", 4, "misspelled number word"),
        (f"there were {alt} chairs", 0, f"a different number ({alt})"),
        (f"room number {n}{n}", 0, f"{n}{n} is a different number"),
        (f"recorded in the year {1900 + n}", 0, "a year, not this number"),
        ("several showed up", 0, "evokes a count, names none"),
    ]
    if n == 1:
        shots.insert(5, ("One never knows what will happen.", 0, "pronoun, not the number"))
    hp = NUM_HOMOPHONE.get(str(n))
    if hp:
        shots.append((hp[1], 0, hp[2]))
    fam = f"the number {n}"
    return render_presence(str(n), fam, definition, inclusion, exclusion, shots)


# ---- numbers100 (presence of any integer in a 10-wide bucket) ----
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ONES19 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
          "eighteen", "nineteen"]


def n2w(x):
    if x < 20:
        return ONES19[x]
    t = TENS[x // 10]
    r = x % 10
    return t + ("-" + ONES19[r] if r else "")


def render_numbers100(cls):
    lo, hi = (int(p) for p in cls.split("-"))
    mid = (lo + hi) // 2
    out = lo - 5 if lo >= 10 else hi + 7          # a value clearly outside this bucket
    out2 = hi + 30
    definition = (f"Presence of a number whose integer value falls in the range {cls} (inclusive), written as digits "
                  f"or words, and used as a genuine quantity (a count or amount). The external integer value is "
                  f"authoritative; your only job is to confirm the token is a quantity in {cls}.")
    inclusion = f"any digit or number-word evaluating to {lo}-{hi} used as a quantity (e.g. {lo}, {mid}, '{n2w(mid)}', {hi})."
    exclusion = (f"A number outside {cls} (e.g. {out}, {out2}) -> 0. A phone number, ID/serial code, or year, where "
                 f"the digits are not a counted quantity -> 0. Vague quantity words that never give a number -> 0.")
    shots = [
        (f"about {mid} people attended", 5, f"{mid} is in {cls}"),
        (f"I bought {lo} of them", 5, f"{lo} is in {cls}, a quantity"),
        (f"roughly {n2w(hi)} miles", 5, f"{hi} is in {cls}"),
        (f"there were {out} guests", 0, f"{out} is outside {cls}"),
        (f"a crowd of {out2}", 0, f"{out2} is outside {cls}"),
        (f"call 555-0{mid:02d}", 0, "phone number, not a quantity"),
        ("born in 1850", 0, "a year, not a counted quantity"),
        ("dozens of items, uncounted", 0, "evokes, gives no number"),
    ]
    fam = f"a number in the range {cls}"
    return render_presence(cls, fam, definition, inclusion, exclusion, shots)


# --------------------------------------------------------------------------- #
# SCALARS — bespoke
# --------------------------------------------------------------------------- #
SB = {}  # scalar bespoke -> dict(scale_label, definition, scale_lines, boundary, shots, external)

SB["numbers"] = dict(
    scale_label="numeric magnitude (0 = none ... 5 = astronomically large)",
    definition="Rate the MAGNITUDE of the numeric quantity referred to, on a log-ish scale. The external digit value is authoritative; your job is to confirm the token is a genuine quantity (a count/amount) and to place its size on the scale.",
    scale_lines=("0 - zero / none\n"
                 "1 - small single digits (1-9)\n"
                 "2 - tens (10-99)\n"
                 "3 - hundreds (100-999)\n"
                 "4 - thousands to millions\n"
                 "5 - billions and beyond / astronomically large"),
    external="When a digit value is given, use it directly. If the token is NOT a quantity (a phone number, SSN, room/ID number, year, or rank/ordinal), rate it 0 and note 'not a quantity'. Fractions: use the rounded magnitude.",
    boundary="Quantity vs label is the key call: a counted amount is rated by size; an identifier or year is rated 0.",
    shots=[
        ("there were zero survivors", 0, "zero"),
        ("I bought three apples", 1, "single digit"),
        ("about forty people came", 2, "tens"),
        ("nearly 500 attendees", 3, "hundreds"),
        ("a budget of $2 million", 4, "millions"),
        ("100 billion stars", 5, "astronomical"),
        ("call me at 555-0199", 0, "phone, not a quantity"),
        ("born in 1994", 0, "year, not a magnitude"),
        ("she finished 2nd", 0, "rank, not a quantity"),
        ("a dozen eggs", 2, "about twelve"),
    ])
SB["costliness"] = dict(
    scale_label="costliness (cheap ... priceless)",
    definition="Rate how COSTLY/expensive the thing referred to is, from free/worthless (0) to priceless or astronomically expensive (5). Judge from explicit price, descriptors, and context.",
    scale_lines=("0 - free / worthless / dirt-cheap\n"
                 "1 - very cheap (a few dollars)\n"
                 "2 - inexpensive\n"
                 "3 - moderately priced\n"
                 "4 - expensive / luxury\n"
                 "5 - priceless / astronomically costly"),
    boundary="Mixed signals -> mid. Resolve sarcasm by intent: 'what a bargain at $40,000' is expensive.",
    shots=[
        ("it was completely free", 0, "free"),
        ("worthless junk", 0, "worthless"),
        ("a dollar-store trinket", 1, "very cheap"),
        ("a cheap weekday lunch", 2, "inexpensive"),
        ("a mid-range laptop", 3, "moderate"),
        ("reasonably priced, nothing fancy", 3, "moderate"),
        ("a designer handbag", 4, "luxury"),
        ("a priceless Rembrandt", 5, "priceless"),
        ("the diamond cost millions", 5, "astronomical"),
        ("what a 'bargain' - $40,000", 4, "sarcasm, expensive by intent"),
    ])
SB["physical_size"] = dict(
    scale_label="physical size (tiny ... huge)",
    definition="Rate the PHYSICAL SIZE of the thing referred to, on a log-ish scale from microscopic (0) to colossal (5).",
    scale_lines=("0 - microscopic / atomic\n"
                 "1 - tiny (insect, coin)\n"
                 "2 - small (handheld object)\n"
                 "3 - human-scale (a person, furniture)\n"
                 "4 - large (building, whale)\n"
                 "5 - colossal (mountain, planet)"),
    boundary="Judge the typical real-world size of the referent. Mixed/ambiguous referents -> mid.",
    shots=[
        ("a single atom", 0, "microscopic"),
        ("a tiny ant", 1, "tiny"),
        ("a grain of rice", 1, "tiny"),
        ("a coffee mug", 2, "small handheld"),
        ("a grown man", 3, "human-scale"),
        ("a refrigerator", 3, "human-scale"),
        ("a ten-story building", 4, "large"),
        ("a blue whale", 4, "large"),
        ("an entire mountain range", 5, "colossal"),
        ("the planet Jupiter", 5, "colossal"),
    ])
SB["europe"] = dict(
    scale_label="European-ness (clearly not Europe ... unmistakably Europe)",
    definition="Rate how strongly the text refers to EUROPE - a European country, city, landmark, language, people, or entity - from clearly not-European (0) to unmistakably European (5).",
    scale_lines=("0 - clearly another continent / unrelated\n"
                 "1 - faint or ambiguous link\n"
                 "2-3 - loosely or partly European (diaspora, colonial echo, ambiguous name)\n"
                 "4 - likely European\n"
                 "5 - unmistakably European (Paris, the Alps, the Euro)"),
    boundary="Confirm sense for ambiguous names (Georgia the country borders Europe/Asia; Portuguese could be Portugal or Brazil). Lean mid when genuinely cross-region.",
    shots=[
        ("a road trip across Texas", 0, "American"),
        ("the Sahara at dawn", 0, "African"),
        ("a generic medieval castle", 1, "faint"),
        ("Georgian wine", 2, "borderline Europe/Asia"),
        ("she spoke fluent Portuguese", 3, "Portugal or Brazil"),
        ("a cafe in Lisbon", 4, "likely European"),
        ("the Eiffel Tower in Paris", 5, "unmistakable"),
        ("the Alps and the Euro", 5, "unmistakable"),
        ("the European Parliament voted", 5, "unmistakable"),
    ])
SB["america"] = dict(
    scale_label="American-ness (clearly not the Americas ... unmistakably the Americas)",
    definition="Rate how strongly the text refers to the AMERICAS (North, Central, or South America) - a place, person, or entity - from clearly not-American (0) to unmistakably American (5).",
    scale_lines=("0 - clearly another continent / unrelated\n"
                 "1 - faint or ambiguous link\n"
                 "2-3 - loose or culturally ambiguous\n"
                 "4 - likely American\n"
                 "5 - unmistakably American (New York, the Andes, the US dollar)"),
    boundary="Confirm sense for ambiguous names (Georgia the US state vs the country). Lean mid when cultural rather than geographic.",
    shots=[
        ("the Tokyo subway", 0, "Asian"),
        ("a London pub", 0, "European"),
        ("a generic cowboy hat", 1, "faint"),
        ("Georgia peaches", 2, "US state, ambiguous name"),
        ("she loves baseball and tacos", 3, "loose cultural"),
        ("a diner in Ohio", 4, "likely"),
        ("the Statue of Liberty in NYC", 5, "unmistakable"),
        ("the Amazon rainforest in Brazil", 5, "unmistakable (S. America)"),
        ("the US dollar and the Rockies", 5, "unmistakable"),
    ])
SB["africa"] = dict(
    scale_label="African-ness (clearly not Africa ... unmistakably Africa)",
    definition="Rate how strongly the text refers to AFRICA - a country, city, landmark, people, or entity - from clearly not-African (0) to unmistakably African (5).",
    scale_lines=("0 - clearly another continent / unrelated\n"
                 "1 - faint or ambiguous link\n"
                 "2-3 - loose or culturally ambiguous (diaspora)\n"
                 "4 - likely African\n"
                 "5 - unmistakably African (the Sahara, Nairobi, Kilimanjaro)"),
    boundary="A lone safari-animal cue is weak (zoos exist) -> low. Egypt/Maghreb are African even when framed as Mediterranean -> mid-high.",
    shots=[
        ("a Parisian boulevard", 0, "European"),
        ("downtown Chicago", 0, "American"),
        ("a lion documentary", 1, "faint, could be a zoo"),
        ("Afrobeat music played", 3, "cultural, loose"),
        ("Egyptian and Mediterranean trade", 3, "Egypt is African, cross-region"),
        ("a market in Lagos", 4, "likely"),
        ("Mount Kilimanjaro in Tanzania", 5, "unmistakable"),
        ("the Sahara and the Nile", 5, "unmistakable"),
        ("Nairobi's skyline", 5, "unmistakable"),
    ])
SB["indoors"] = dict(
    scale_label="indoor-ness (fully outdoors ... deep interior)",
    definition="Rate how INDOORS the described setting is - how enclosed within a building or structure - from fully outdoors/open-air (0) to deep interior (5).",
    scale_lines=("0 - open wilderness / fully outdoors\n"
                 "1 - mostly outdoors (yard, field)\n"
                 "2 - transitional, outdoor-adjacent (porch, balcony)\n"
                 "3 - threshold / partly enclosed (doorway, garage, tent)\n"
                 "4 - clearly inside a room\n"
                 "5 - deep interior (windowless inner room, basement vault)"),
    boundary="A doorway/threshold is the midpoint. A covered-but-open structure (porch) is low-mid, not indoors.",
    shots=[
        ("hiking an open ridge", 0, "outdoors"),
        ("paddling across the lake", 0, "outdoors"),
        ("in the backyard garden", 1, "mostly outdoors"),
        ("on the covered porch", 2, "transitional"),
        ("standing in the doorway", 3, "threshold"),
        ("inside a camping tent", 3, "partly enclosed"),
        ("at her office desk", 4, "inside"),
        ("sitting in the living room", 4, "inside"),
        ("deep in a basement vault", 5, "deep interior"),
    ])
SB["outdoors"] = dict(
    scale_label="outdoor-ness (fully indoors ... open wilderness)",
    definition="Rate how OUTDOORS the described setting is - how exposed to open air and outside any building - from fully enclosed indoors (0) to open wilderness (5).",
    scale_lines=("0 - sealed interior\n"
                 "1 - inside with an outdoor view / near an exit\n"
                 "2 - threshold (doorway, open garage)\n"
                 "3 - covered-outdoor (porch, balcony, pavilion)\n"
                 "4 - open but built (yard, street, parking lot)\n"
                 "5 - open wilderness (forest, mountain, sea)"),
    boundary="This is the antipode of indoor-ness. A doorway/threshold is the midpoint.",
    shots=[
        ("in a windowless conference room", 0, "sealed interior"),
        ("in an elevator", 0, "sealed interior"),
        ("by the window looking out", 1, "inside, near exit"),
        ("in the open garage", 2, "threshold"),
        ("on the covered patio", 3, "covered-outdoor"),
        ("kids playing in the yard", 4, "open but built"),
        ("walking down the street", 4, "open but built"),
        ("summiting a mountain", 5, "open wilderness"),
        ("sailing the open sea", 5, "open wilderness"),
    ])
SB["lovingness"] = dict(
    scale_label="lovingness (despise ... adore)",
    definition="Rate the LOVINGNESS expressed toward the subject - from intense hatred/contempt (0) to deep love/adoration (5). Read the tone; resolve sarcasm by intent.",
    scale_lines=("0 - despise / loathe / hatred\n"
                 "1 - dislike\n"
                 "2 - mild negative / cool\n"
                 "3 - neutral / indifferent\n"
                 "4 - fondness / affection\n"
                 "5 - adoration / deep love"),
    boundary="Mixed feelings -> mid. Sarcasm by intent: 'Oh, I just LOVE being ignored' is contempt, not love.",
    shots=[
        ("I absolutely despise him", 0, "hatred"),
        ("Oh, I just LOVE being ignored.", 0, "sarcasm, contempt"),
        ("I'm not a fan of her", 1, "dislike"),
        ("he's kind of annoying", 2, "mild negative"),
        ("it's fine, I guess", 2, "lukewarm"),
        ("I have no strong feelings either way", 3, "neutral"),
        ("I'm really fond of this town", 4, "affection"),
        ("I adore her with all my heart", 5, "adoration"),
        ("my cherished, beloved grandmother", 5, "deep love"),
    ])
SB["duration"] = dict(
    scale_label="duration (instantaneous ... eternal)",
    definition="Rate the DURATION/length of time referred to, on a log-ish scale from instantaneous (0) to eternal/geological (5).",
    scale_lines=("0 - instantaneous (a flash, a moment)\n"
                 "1 - seconds to minutes\n"
                 "2 - hours\n"
                 "3 - days to weeks\n"
                 "4 - years to decades\n"
                 "5 - centuries / eternal / geological"),
    boundary="Rate the span actually described. Vague 'a while' -> mid-low.",
    shots=[
        ("in the blink of an eye", 0, "instant"),
        ("a quick few seconds", 1, "seconds"),
        ("a two-minute wait", 1, "minutes"),
        ("an afternoon nap", 2, "hours"),
        ("a week-long trip", 3, "days-weeks"),
        ("several days of rain", 3, "days"),
        ("a decades-long career", 4, "years-decades"),
        ("for all eternity", 5, "eternal"),
        ("over geological epochs", 5, "geological"),
    ])
SB["harmfulness"] = dict(
    scale_label="harmfulness (benign ... catastrophic)",
    definition="Rate the HARMFULNESS of the thing, action, or event referred to, from completely harmless (0) to catastrophic/lethal (5).",
    scale_lines=("0 - benign / harmless\n"
                 "1 - trivial nuisance\n"
                 "2 - minor harm\n"
                 "3 - moderate / risky (injury, real damage)\n"
                 "4 - severe (grave injury, major damage)\n"
                 "5 - catastrophic / lethal / mass-casualty"),
    boundary="Rate the realistic worst-case harm of the referent as used. Mixed -> mid.",
    shots=[
        ("a soft pillow", 0, "harmless"),
        ("a gentle breeze", 0, "harmless"),
        ("a paper cut", 1, "trivial"),
        ("a mild allergic rash", 2, "minor"),
        ("a sprained ankle", 2, "minor"),
        ("a car crash with injuries", 3, "moderate-severe"),
        ("a building fire", 4, "severe"),
        ("a nuclear meltdown", 5, "catastrophic"),
        ("a deadly pandemic", 5, "catastrophic"),
    ])


# --------------------------------------------------------------------------- #
# BUILD
# --------------------------------------------------------------------------- #
def build():
    registry = {}
    written = 0

    # --- presence ---
    for cname, cfg in concepts.PRESENCE_CONCEPTS.items():
        for cls, meta in cfg["classes"].items():
            pid = f"{cname}::{cls}"
            if pid in PB:
                d = PB[pid]
                text = render_presence(cls, d["fam"], d["definition"], d["inclusion"], d["exclusion"], d["shots"])
            elif cname == "numbers10":
                text = render_numbers10(int(cls))
            elif cname == "numbers100":
                text = render_numbers100(cls)
            elif cname == "color_wheel":
                text = render_tertiary(cls, meta["lexicon"])
            elif cname == "moon_phases":
                text = render_moon_templated(cls, meta["lexicon"])
            else:
                raise SystemExit(f"NO RENDERER for {pid}")
            rel = f"prompts/{cname}/{safe(cls)}.txt"
            path = os.path.join(OVR, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            registry[pid] = rel
            written += 1

    # --- scalar ---
    for name in concepts.SCALARS:
        pid = f"scalar::{name}"
        if name not in SB:
            raise SystemExit(f"NO SCALAR PROMPT for {name}")
        d = SB[name]
        text = render_scalar(d["scale_label"], d["definition"], d["scale_lines"],
                             d["boundary"], d["shots"], d.get("external"))
        rel = f"prompts/scalar/{name}.txt"
        path = os.path.join(OVR, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        registry[pid] = rel
        written += 1

    with open(os.path.join(PROMPTS_DIR, "registry.json"), "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)

    print(f"WROTE {written} prompt files + registry.json ({len(registry)} ids)")
    return registry


if __name__ == "__main__":
    build()
