"""Bespoke per-concept configuration for all 16 concepts (47 cyclic value-rows + 10 scalar rows).

Per concept:
  family            "cyclic" | "scalar"
  n                 number of cyclic values (1 for scalar)
  values            cyclic: ordered list (index = theta_k position); each value:
                      name        canonical display name (also stored as label_value)
                      surface     in-vocabulary slot fillers for templates (synonyms ok)
                      banned      regex word-stems forbidden in the held-out-vocab split
                                  (trigger + morphological variants + synonyms)
                      periphrasis UNIQUE value-pinning paraphrases (NOT associative vibes);
                                  [] => no unique non-lexical form => §5 SKIPPED+flagged for this value
  frames            template frames with a single typed slot "{X}" (cyclic + phrase scalars)
  confusables       hard negatives: a surface form used WITHOUT the concept meaning
                      (substring / homonym traps). Each: {"word", "sentences":[...]}
  rubric/anchors    scalar only: anchored [0,1] exemplars (value, phrase)
  diffuse           scalar only: True => multi key-phrase spans (§6)
  freegen_hint      instruction fragment for the LLM free-gen prompt
"""

# ---------------------------------------------------------------------------
# CYCLIC CONCEPTS
# ---------------------------------------------------------------------------

SEASON = {
    "concept": "season", "family": "cyclic", "n": 4,
    "values": [
        {"name": "Spring", "surface": ["spring", "springtime", "the spring"],
         "banned": ["spring", "springtime", "vernal"],
         "periphrasis": ["the season when flowers first bloom after the cold",
                         "the blossoming season that comes right after winter",
                         "the season of new growth, melting snow and lengthening days"]},
        {"name": "Summer", "surface": ["summer", "summertime", "the summer"],
         "banned": ["summer", "summertime", "estival"],
         "periphrasis": ["the hottest season of the year",
                         "the warmest season, when school is out and days are longest",
                         "the season of beach trips and long sunny days"]},
        {"name": "Fall", "surface": ["fall", "autumn", "the fall", "the autumn"],
         "banned": ["fall", "autumn", "autumnal"],
         "periphrasis": ["the season when leaves turn color and drop from the trees",
                         "the harvest season between summer and winter",
                         "the season of pumpkins and shortening days before the cold"]},
        {"name": "Winter", "surface": ["winter", "wintertime", "the winter"],
         "banned": ["winter", "wintertime", "hibernal"],
         "periphrasis": ["the coldest season of the year",
                         "the season of snow, frost and the shortest days",
                         "the freezing season that comes right after the leaves have fallen"]},
    ],
    "frames": [
        "I really enjoy {X}.", "We went hiking in {X}.", "She got married in {X}.",
        "My favorite season is {X}.", "The festival happens every {X}.",
        "He always travels during {X}.", "Business is slow in {X}.",
        "The garden looks best in {X}.", "They moved to the coast last {X}.",
        "Nothing beats a morning walk in {X}.", "We celebrate the harvest in {X}.",
        "The reservoir fills up during {X}.", "Tourists flock here in {X}.",
        "Classes resume after {X}.", "I prefer the quiet of {X}.",
        "The whole town comes alive in {X}.",
    ],
    "confusables": [
        {"word": "spring", "sentences": [
            "The mattress had a broken spring poking through the fabric.",
            "Clear water bubbled up from a natural spring in the hills.",
            "He watched the cat spring onto the windowsill.",
            "The watch stopped because its mainspring snapped."]},
        {"word": "fall", "sentences": [
            "Be careful not to fall on the icy steps.",
            "The vase began to fall off the edge of the shelf.",
            "Stock prices are expected to fall again tomorrow.",
            "She watched the climber fall a few feet before the rope caught."]},
        {"word": "summer", "sentences": [
            "The architect added a decorative summer beam across the ceiling.",
            "Donna Summer's records still sell well."]},
    ],
    "freegen_hint": ("naturalistic text that entails the season WITHOUT naming it directly, "
                     "via weather, holidays, nature, or activities"),
}

MONTH = {
    "concept": "month", "family": "cyclic", "n": 12,
    "values": [
        {"name": "January", "surface": ["January", "Jan"], "banned": ["january", "jan"],
         "periphrasis": ["the first month of the year", "the month the new year begins in"]},
        {"name": "February", "surface": ["February", "Feb"], "banned": ["february", "feb"],
         "periphrasis": ["the second month of the year", "the shortest month of the year"]},
        {"name": "March", "surface": ["March"], "banned": ["march"],
         "periphrasis": ["the third month of the year", "the month the spring equinox falls in"]},
        {"name": "April", "surface": ["April"], "banned": ["april"],
         "periphrasis": ["the fourth month of the year", "the month right before May"]},
        {"name": "May", "surface": ["May"], "banned": ["may"],
         "periphrasis": ["the fifth month of the year", "the month between April and June"]},
        {"name": "June", "surface": ["June"], "banned": ["june"],
         "periphrasis": ["the sixth month of the year", "the month the summer solstice falls in"]},
        {"name": "July", "surface": ["July"], "banned": ["july"],
         "periphrasis": ["the seventh month of the year", "the month right after June"]},
        {"name": "August", "surface": ["August", "Aug"], "banned": ["august", "aug"],
         "periphrasis": ["the eighth month of the year", "the month between July and September"]},
        {"name": "September", "surface": ["September", "Sept", "Sep"],
         "banned": ["september", "sept", "sep"],
         "periphrasis": ["the ninth month of the year", "the month the autumn equinox falls in"]},
        {"name": "October", "surface": ["October", "Oct"], "banned": ["october", "oct"],
         "periphrasis": ["the tenth month of the year", "the month Halloween falls in"]},
        {"name": "November", "surface": ["November", "Nov"], "banned": ["november", "nov"],
         "periphrasis": ["the eleventh month of the year", "the month right before December"]},
        {"name": "December", "surface": ["December", "Dec"], "banned": ["december", "dec"],
         "periphrasis": ["the twelfth month of the year", "the last month of the year"]},
    ],
    "frames": [
        "I was born in {X}.", "We are getting married in {X}.", "The conference is in {X}.",
        "She starts her new job in {X}.", "Our lease ends in {X}.",
        "The fiscal quarter closed in {X}.", "They visited Rome in {X}.",
        "Sales peaked in {X}.", "The exam is scheduled for {X}.",
        "He retired in {X}.", "The harvest came early in {X}.",
        "We always vacation in {X}.", "The product launches in {X}.",
        "Construction finished in {X}.", "Her birthday is in {X}.",
        "The deadline moved to {X}.",
    ],
    "confusables": [
        {"word": "may", "sentences": [
            "You may leave the room once you finish the test.",
            "She thought it may rain later that afternoon.",
            "I may consider the offer if the terms improve."]},
        {"word": "march", "sentences": [
            "The soldiers began to march across the parade ground.",
            "Protesters plan to march downtown this weekend.",
            "The band will march at halftime."]},
        {"word": "august", "sentences": [
            "He spoke before an august assembly of scholars.",
            "The college has a long and august tradition."]},
        {"word": "april", "sentences": ["April Ludgate rarely smiled at her coworkers."]},
        {"word": "june", "sentences": ["June Carter sang alongside her husband for decades."]},
    ],
    "freegen_hint": ("naturalistic text that entails the month WITHOUT naming it, via holidays, "
                     "weather, seasonal events, or its position in the year"),
}

WEEKDAY = {
    "concept": "weekday", "family": "cyclic", "n": 7,
    "values": [
        {"name": "Monday", "surface": ["Monday", "Mon"], "banned": ["monday", "mon"],
         "periphrasis": ["the first day of the work week", "the day right after Sunday"]},
        {"name": "Tuesday", "surface": ["Tuesday", "Tue", "Tues"], "banned": ["tuesday", "tue", "tues"],
         "periphrasis": ["the day right after Monday", "the second day of the work week"]},
        {"name": "Wednesday", "surface": ["Wednesday", "Wed"], "banned": ["wednesday", "wed"],
         "periphrasis": ["the midpoint day of the work week", "the day between Tuesday and Thursday"]},
        {"name": "Thursday", "surface": ["Thursday", "Thu", "Thurs"], "banned": ["thursday", "thu", "thurs"],
         "periphrasis": ["the day right before Friday", "the fourth day of the work week"]},
        {"name": "Friday", "surface": ["Friday", "Fri"], "banned": ["friday", "fri"],
         "periphrasis": ["the last day of the work week", "the day right before the weekend"]},
        {"name": "Saturday", "surface": ["Saturday", "Sat"], "banned": ["saturday", "sat"],
         "periphrasis": ["the first day of the weekend", "the day right after Friday"]},
        {"name": "Sunday", "surface": ["Sunday", "Sun"], "banned": ["sunday", "sun"],
         "periphrasis": ["the day right before Monday", "the second day of the weekend"]},
    ],
    "frames": [
        "The meeting is on {X}.", "We always order pizza on {X}.", "Her flight leaves on {X}.",
        "Trash gets collected on {X}.", "The store is closed on {X}.",
        "He goes to the gym every {X}.", "Payday lands on {X}.",
        "Choir practice is on {X}.", "The market opens on {X}.",
        "I have a dentist appointment on {X}.", "The deadline is {X}.",
        "They host trivia night on {X}.", "School lets out early on {X}.",
        "The newsletter goes out every {X}.",
    ],
    "confusables": [
        {"word": "sun", "sentences": [
            "The sun rose over the calm ocean that morning.",
            "We sat in the warm sun until noon.",
            "Sunflowers tracked the sun across the sky."]},
        {"word": "wed", "sentences": [
            "They plan to wed next spring in a small chapel.",
            "The couple chose to wed on a quiet beach.",
            "He vowed to wed her before the year was out."]},
        {"word": "sat", "sentences": [
            "She sat quietly at the back of the lecture hall.",
            "The cat sat on the warm windowsill all afternoon."]},
        {"word": "fri", "sentences": ["The chef will fry the plantains until golden."]},
    ],
    "freegen_hint": ("text that entails the weekday WITHOUT naming it, via routines, the weekend, "
                     "the work week, or recurring events"),
}

COLOR_HUE = {
    "concept": "color_hue", "family": "cyclic", "n": 12,
    "values": [
        {"name": "Red", "surface": ["red", "crimson", "scarlet"], "banned": ["red", "crimson", "scarlet"],
         "periphrasis": ["the color of fresh blood", "the color of a ripe tomato"]},
        {"name": "Red-Orange", "surface": ["red-orange", "vermilion"], "banned": ["red-orange", "vermilion"],
         "periphrasis": []},
        {"name": "Orange", "surface": ["orange"], "banned": ["orange"],
         "periphrasis": ["the color of a ripe carrot", "the color of a basketball"]},
        {"name": "Yellow-Orange", "surface": ["yellow-orange", "amber"], "banned": ["yellow-orange", "amber"],
         "periphrasis": []},
        {"name": "Yellow", "surface": ["yellow"], "banned": ["yellow"],
         "periphrasis": ["the color of a ripe banana", "the color of a school bus"]},
        {"name": "Yellow-Green", "surface": ["yellow-green", "chartreuse"], "banned": ["yellow-green", "chartreuse"],
         "periphrasis": []},
        {"name": "Green", "surface": ["green"], "banned": ["green"],
         "periphrasis": ["the color of fresh grass", "the color of an emerald"]},
        {"name": "Blue-Green", "surface": ["blue-green", "teal", "cyan"], "banned": ["blue-green", "teal", "cyan"],
         "periphrasis": []},
        {"name": "Blue", "surface": ["blue"], "banned": ["blue"],
         "periphrasis": ["the color of a clear daytime sky", "the color of a sapphire"]},
        {"name": "Blue-Purple", "surface": ["blue-purple", "indigo", "violet"], "banned": ["blue-purple", "indigo", "violet"],
         "periphrasis": []},
        {"name": "Purple", "surface": ["purple"], "banned": ["purple"],
         "periphrasis": ["the color of a ripe eggplant", "the color of an amethyst"]},
        {"name": "Red-Purple", "surface": ["red-purple", "magenta"], "banned": ["red-purple", "magenta"],
         "periphrasis": []},
    ],
    "frames": [
        "She painted the door {X}.", "He bought a {X} sweater.", "The logo is {X}.",
        "They repainted the fence {X}.", "Her new car is {X}.", "The walls were {X}.",
        "I picked the {X} mug.", "The team's jerseys are {X}.", "We chose {X} tiles.",
        "The sign glowed {X}.", "His tie was {X}.", "The curtains are {X}.",
        "The bicycle is {X}.", "They dyed the fabric {X}.",
    ],
    "confusables": [
        {"word": "blue", "sentences": [
            "She felt blue after the long, gray winter.",
            "He sang an old blues song on the porch."]},
        {"word": "green", "sentences": [
            "The new intern was still green and made rookie mistakes.",
            "He was green with envy over his brother's promotion."]},
        {"word": "orange", "sentences": [
            "She peeled an orange and shared the slices.",
            "Orange County sits south of Los Angeles."]},
        {"word": "red", "sentences": [
            "The startup was bleeding cash and deep in the red.",
            "There was too much red tape to get the permit."]},
    ],
    "freegen_hint": ("text describing something whose characteristic color is the value, WITHOUT "
                     "naming the color word; describe the object instead"),
}

MOON_PHASE = {
    "concept": "moon_phase", "family": "cyclic", "n": 8,
    "values": [
        {"name": "New Moon", "surface": ["new moon"], "banned": ["new moon"],
         "periphrasis": ["the moon when its disc is completely dark and invisible"]},
        {"name": "Waxing Crescent", "surface": ["waxing crescent"], "banned": ["waxing crescent"],
         "periphrasis": ["a thin sliver of moon that is growing larger each night"]},
        {"name": "First Quarter", "surface": ["first quarter", "first-quarter moon"], "banned": ["first quarter"],
         "periphrasis": ["the moon that is exactly half lit and still growing"]},
        {"name": "Waxing Gibbous", "surface": ["waxing gibbous"], "banned": ["waxing gibbous"],
         "periphrasis": ["the moon more than half lit and still growing toward full"]},
        {"name": "Full Moon", "surface": ["full moon"], "banned": ["full moon"],
         "periphrasis": ["the moon when its entire disc is brightly lit"]},
        {"name": "Waning Gibbous", "surface": ["waning gibbous"], "banned": ["waning gibbous"],
         "periphrasis": ["the moon more than half lit but shrinking after being full"]},
        {"name": "Last Quarter", "surface": ["last quarter", "third quarter"], "banned": ["last quarter", "third quarter"],
         "periphrasis": ["the moon that is exactly half lit and shrinking"]},
        {"name": "Waning Crescent", "surface": ["waning crescent"], "banned": ["waning crescent"],
         "periphrasis": ["a thin sliver of moon that is shrinking toward darkness"]},
    ],
    "frames": [
        "The almanac shows a {X} tonight.", "Astronomers noted the {X}.", "We hiked under the {X}.",
        "The calendar marks a {X} this week.", "Tides ran high during the {X}.",
        "The festival aligns with the {X}.", "Photographers gathered for the {X}.",
        "Her ritual follows the {X}.", "The sky showed a {X}.", "We timed the trip to the {X}.",
    ],
    "confusables": [
        {"word": "full", "sentences": [
            "The auditorium was completely full by the time we arrived.",
            "She felt full after the enormous holiday dinner."]},
        {"word": "new", "sentences": [
            "He just bought a brand new car last week.",
            "The startup hired three new engineers."]},
        {"word": "quarter", "sentences": [
            "She paid the toll with a single quarter.",
            "The team rallied in the fourth quarter of the game."]},
        {"word": "crescent", "sentences": [
            "We rented a flat on Royal Crescent in Bath.",
            "He warmed a buttery crescent roll for breakfast."]},
    ],
    "freegen_hint": ("text that entails the moon phase WITHOUT naming it, via how much of the disc "
                     "is lit and whether it is growing or shrinking"),
}

COMPASS = {
    "concept": "compass", "family": "cyclic", "n": 4,
    "values": [
        {"name": "North", "surface": ["north", "northward", "northbound"], "banned": ["north"],
         "periphrasis": ["the direction a compass needle points",
                         "the direction toward the Arctic from the equator"]},
        {"name": "East", "surface": ["east", "eastward", "eastbound"], "banned": ["east"],
         "periphrasis": ["the direction in which the sun rises",
                         "the direction a sundial's shadow points at dawn"]},
        {"name": "South", "surface": ["south", "southward", "southbound"], "banned": ["south"],
         "periphrasis": ["the direction opposite to where a compass needle points",
                         "the direction toward Antarctica from the equator"]},
        {"name": "West", "surface": ["west", "westward", "westbound"], "banned": ["west"],
         "periphrasis": ["the direction in which the sun sets",
                         "the direction opposite to where the sun rises"]},
    ],
    "frames": [
        "The river flows to the {X}.", "Keep driving {X} for ten miles.", "The storm is moving {X}.",
        "Their cabin faces {X}.", "Migrating geese head {X} in autumn.",
        "The trail bends to the {X}.", "We sailed {X} for three days.",
        "The wind shifted to the {X}.", "Her office window looks {X}.",
        "The highway exit is to the {X}.", "Push the herd {X} toward the river.",
        "The army advanced {X}.",
    ],
    "confusables": [  # substring / homonym traps — direction sense ABSENT
        {"word": "east", "sentences": [
            "The children hunted for eggs on Easter morning.",
            "He studies Eastern philosophy at university.",
            "Yeast makes the dough rise overnight."]},
        {"word": "west", "sentences": [
            "They watched an old Western on Saturday night.",
            "Westminster Abbey draws thousands of visitors.",
            "Kanye West released a new album."]},
        {"word": "south", "sentences": [
            "She grew up on classic Southern cooking.",
            "South Park aired its new season last night.",
            "The deal went south after the audit."]},
        {"word": "north", "sentences": [
            "Northampton hosted the regional finals.",
            "He read every Northanger Abbey chapter twice."]},
    ],
    "freegen_hint": ("text that entails the cardinal direction WITHOUT naming it, via sunrise/sunset, "
                     "a compass, or geography. Avoid words like Easter, Western, Southern that only "
                     "share letters."),
}

# ---------------------------------------------------------------------------
# SCALAR CONCEPTS  (continuous magnitude in [0,1], anchored rubric)
# ---------------------------------------------------------------------------

COSTLINESS = {
    "concept": "costliness", "family": "scalar", "n": 1, "diffuse": False,
    "rubric": "0 = free / no monetary cost; 0.5 = an ordinary mid-priced purchase; 1 = extraordinarily expensive luxury.",
    "anchors": [(0.02, "a free walk in the public park"), (0.1, "a cheap cup of coffee"),
                (0.3, "a paperback novel"), (0.5, "a mid-range dinner for two"),
                (0.7, "a new laptop"), (0.9, "a luxury sports car"),
                (0.98, "a private jet")],
    "frames": ["I bought {X} yesterday.", "We were talking about {X}.", "She splurged on {X}.",
               "He saved up for {X}.", "They reviewed {X} online.", "The shop sells {X}."],
    "confusables": [{"word": "cost", "sentences": [
        "She opened a new checking account at the bank.",
        "The accountant filed the quarterly paperwork on time."]}],
    "freegen_hint": "text whose costliness magnitude matches the target value; mark the key phrase",
}

SIZE = {
    "concept": "size", "family": "scalar", "n": 1, "diffuse": False,
    "rubric": "0 = microscopic / tiny; 0.5 = the size of a person or a piece of furniture; 1 = colossal (a mountain, a planet).",
    "anchors": [(0.02, "a single grain of sand"), (0.1, "a coin"), (0.3, "a house cat"),
                (0.5, "a refrigerator"), (0.7, "a school bus"), (0.9, "a skyscraper"),
                (0.98, "an entire mountain range")],
    "frames": ["They photographed {X}.", "He pointed at {X}.", "We measured {X}.",
               "She drew {X}.", "The exhibit featured {X}.", "I tripped over {X}."],
    "confusables": [{"word": "size", "sentences": [
        "The committee debated the policy for hours.",
        "He whistled an old tune while walking home."]}],
    "freegen_hint": "text whose physical size magnitude matches the target value; mark the key phrase",
}

EUROPE = {
    "concept": "europe", "family": "scalar", "n": 1, "diffuse": False,
    "rubric": "0 = nothing European at all; 0.5 = loosely or partly European; 1 = quintessentially, unmistakably European.",
    "anchors": [(0.02, "a kangaroo in the Australian outback"), (0.2, "a melting pot city with global cuisine"),
                (0.5, "a transcontinental country straddling two continents"),
                (0.8, "a medieval cathedral town on the Rhine"),
                (0.97, "the canals of Venice and the cafes of Paris")],
    "frames": ["The documentary was about {X}.", "She wrote an essay on {X}.", "We visited {X}.",
               "He studied {X}.", "The lecture covered {X}.", "Their trip focused on {X}."],
    "confusables": [{"word": "europe", "sentences": [
        "The robot arm welded the chassis in seconds.",
        "She practiced the violin for an hour."]}],
    "freegen_hint": "text whose European-ness magnitude matches the target value; mark the key phrase",
}

AMERICA = {
    "concept": "america", "family": "scalar", "n": 1, "diffuse": False,
    "rubric": "0 = nothing American at all; 0.5 = loosely or partly American; 1 = quintessentially, unmistakably American (the Americas).",
    "anchors": [(0.02, "a Shinto shrine in rural Japan"), (0.2, "a global fast-food chain"),
                (0.5, "a Pacific island with mixed colonial heritage"),
                (0.8, "a baseball game and apple pie on the Fourth of July"),
                (0.97, "the Grand Canyon, Route 66 and a roadside diner")],
    "frames": ["The documentary was about {X}.", "She wrote an essay on {X}.", "We visited {X}.",
               "He studied {X}.", "The lecture covered {X}.", "Their trip focused on {X}."],
    "confusables": [{"word": "america", "sentences": [
        "The chemist titrated the solution carefully.",
        "He repaired the leaky faucet in the kitchen."]}],
    "freegen_hint": "text whose American-ness magnitude matches the target value; mark the key phrase",
}

AFRICA = {
    "concept": "africa", "family": "scalar", "n": 1, "diffuse": False,
    "rubric": "0 = nothing African at all; 0.5 = loosely or partly African; 1 = quintessentially, unmistakably African.",
    "anchors": [(0.02, "a ski lodge in the Swiss Alps"), (0.2, "a wildlife zoo on another continent"),
                (0.5, "a Mediterranean port city in the north of the continent"),
                (0.8, "a safari across the savanna"),
                (0.97, "the Serengeti migration and Kilimanjaro at dawn")],
    "frames": ["The documentary was about {X}.", "She wrote an essay on {X}.", "We visited {X}.",
               "He studied {X}.", "The lecture covered {X}.", "Their trip focused on {X}."],
    "confusables": [{"word": "africa", "sentences": [
        "The pianist rehearsed the sonata twice.",
        "She uploaded the spreadsheet before lunch."]}],
    "freegen_hint": "text whose African-ness magnitude matches the target value; mark the key phrase",
}

INDOORS = {
    "concept": "indoors", "family": "scalar", "n": 1, "diffuse": False,
    "rubric": "0 = entirely outdoors / no interior at all; 0.5 = a sheltered or threshold space; 1 = deep inside an enclosed interior.",
    "anchors": [(0.02, "standing in an open field under the sky"), (0.2, "sitting on a covered porch"),
                (0.5, "working in a glass-walled sunroom"), (0.8, "reading in a cozy living room"),
                (0.97, "locked in a windowless basement vault")],
    "frames": ["The scene took place {X}.", "They spent the afternoon {X}.", "He set the story {X}.",
               "We held the meeting {X}.", "She filmed it {X}.", "The party was {X}."],
    "confusables": [{"word": "indoor", "sentences": [
        "The algorithm sorted the records efficiently.",
        "He tuned the guitar before the show."]}],
    "freegen_hint": "text whose indoor-ness magnitude matches the target value; mark the key phrase",
}

OUTDOORS = {
    "concept": "outdoors", "family": "scalar", "n": 1, "diffuse": False,
    "rubric": "0 = entirely enclosed indoors; 0.5 = a threshold or partly open space; 1 = deep in the open wilderness.",
    "anchors": [(0.02, "sealed inside a windowless office"), (0.2, "standing in an open doorway"),
                (0.5, "on a balcony overlooking the street"), (0.8, "hiking a forest trail"),
                (0.97, "camping alone on a remote mountain ridge")],
    "frames": ["The scene took place {X}.", "They spent the afternoon {X}.", "He set the story {X}.",
               "We held the meeting {X}.", "She filmed it {X}.", "The party was {X}."],
    "confusables": [{"word": "outdoor", "sentences": [
        "The compiler flagged a syntax error on line ten.",
        "She balanced the ledger before closing."]}],
    "freegen_hint": "text whose outdoor-ness magnitude matches the target value; mark the key phrase",
}

LOVINGNESS = {
    "concept": "lovingness", "family": "scalar", "n": 1, "diffuse": True,
    "rubric": "0 = cold, hostile or cruel; 0.5 = neutral or politely indifferent; 1 = overflowing with tenderness and affection.",
    "anchors": [(0.02, "he sneered and slammed the door in her face"),
                (0.2, "she gave a curt, businesslike nod"),
                (0.5, "they exchanged a routine hello in the hallway"),
                (0.8, "she wrapped him in a warm, reassuring hug"),
                (0.97, "he cradled the newborn, whispering how deeply he adored her")],
    "frames": ["In the letter, {X}.", "During the visit, {X}.", "At the reunion, {X}."],
    "confusables": [{"word": "love", "sentences": [
        "The technician calibrated the sensor array.",
        "He parked the truck behind the warehouse."]}],
    "freegen_hint": ("text whose warmth/affection magnitude matches the target value; mark ONE OR MORE "
                     "key phrases that carry the affection"),
}

HARMFULNESS = {
    "concept": "harmfulness", "family": "scalar", "n": 1, "diffuse": True,
    "rubric": "0 = completely safe and benign; 0.5 = mildly risky; 1 = lethally dangerous or destructive.",
    "anchors": [(0.02, "a soft pillow on a child's bed"), (0.2, "a slippery wet floor"),
                (0.5, "a kitchen knife left on the counter"), (0.8, "a live exposed electrical wire"),
                (0.97, "a vial of weaponized nerve agent")],
    "frames": ["The report described {X}.", "Investigators found {X}.", "The label warned about {X}."],
    "confusables": [{"word": "harm", "sentences": [
        "The librarian reshelved the returned books.",
        "She watered the ferns on the windowsill."]}],
    "freegen_hint": ("text whose danger/harm magnitude matches the target value; mark ONE OR MORE "
                     "key phrases that carry the harm"),
}

DURATION = {
    "concept": "duration", "family": "scalar", "n": 1, "diffuse": True,
    "rubric": "0 = instantaneous (a fraction of a second); 0.5 = lasting hours; 1 = lasting many years or longer.",
    "anchors": [(0.02, "a camera flash that lasted a split second"),
                (0.2, "a five-minute coffee break"), (0.5, "an afternoon-long workshop"),
                (0.8, "a months-long expedition"), (0.97, "a decades-long marriage")],
    "frames": ["The record notes {X}.", "Witnesses described {X}.", "The schedule listed {X}."],
    "confusables": [{"word": "long", "sentences": [
        "She painted the bedroom a soft gray.",
        "He sorted the mail into three neat piles."]}],
    "freegen_hint": ("text whose time-duration magnitude matches the target value; mark ONE OR MORE "
                     "key phrases that carry the duration"),
}

CONCEPTS = {c["concept"]: c for c in [
    SEASON, MONTH, WEEKDAY, COLOR_HUE, MOON_PHASE, COMPASS,
    COSTLINESS, SIZE, EUROPE, AMERICA, AFRICA, INDOORS, OUTDOORS,
    LOVINGNESS, HARMFULNESS, DURATION,
]}

CYCLIC = [k for k, v in CONCEPTS.items() if v["family"] == "cyclic"]
SCALAR = [k for k, v in CONCEPTS.items() if v["family"] == "scalar"]


def get(concept):
    return CONCEPTS[concept]
