import json
from itertools import product

# Load data from uploaded files
with open("data/anthropocene.json", "r") as f:
    anthropocene = json.load(f)

with open("data/locations.json", "r") as f:
    locations = json.load(f)

eras = list(anthropocene["eras"].keys())
sectors = anthropocene["sectors"]
sector_defaults = anthropocene["sector_defaults"]
location_types = locations["types"]

# Wilderness: unique per biome
def biome_wilderness_sound(biome):
    prompts = {
        "rainforest": "Dense rainforest ambience: rainfall, insects, distant animal calls, river sounds",
        "temperate_forest": "Temperate forest: rustling leaves, songbirds, wind in trees, distant animal sounds",
        "boreal_forest": "Boreal forest: pine trees creaking, wind, ravens, light snow in winter",
        "deciduous_forest": "Deciduous forest: birds, wind, seasonal leaf fall, insects",
        "swamp": "Swamp: frogs, insects, water movement, distant birds",
        "desert": "Desert: wind over sand, sparse animal calls, distant shifting dunes",
        "mountain": "Mountain: wind, distant avalanches, eagles, occasional rocks tumbling",
        "tropical_island": "Tropical island: waves on shore, palm leaves, tropical birds, distant thunder",
        "river": "River: flowing water, splashes, reeds in wind, riverbank animals",
        "tundra": "Tundra: wind, distant birds, light snow, silence",
        "steppe": "Steppe: tall grasses rustling, insects, distant herds",
        "coastal": "Coastal: breaking waves, seabirds, wind, tidal sounds",
        "plains": "Plains: wind in grasses, crickets, distant animal herds",
        "arctic": "Arctic: howling wind, cracking ice, distant seals, muffled silence",
        "jungle": "Jungle: dense foliage, insects buzzing, distant animal calls",
    }
    return {
        "path": f"environment/wilderness/{biome}_ambience.wav",
        "tags": [biome, "wilderness"],
        "prompt": prompts.get(biome, "Natural wilderness ambience for this biome")
    }

# Human sounds: by era + sector + detail (biome-agnostic, unless biome really matters)
# Grouped prompts to avoid excess duplication
generic_sounds = {
  "airport": "ambient sounds of a busy airport terminal, including rolling luggage, distant PA announcements, soft chatter, and the occasional roar of planes taking off and landing",
  "airstrip": "open-air ambience with the hum of small aircraft engines, wind over flat land, and scattered distant voices and tool clinks from ground crew",
  "algae_farm": "quiet mechanical bubbling and soft flowing water with intermittent wet sloshes and faint hums of pumps in a humid, open structure",
  "amphitheater": "audience murmurs and applause in a semi-open stone or concrete space, with voice echoes and natural reverb from a live performance or speech",
  "amusement_park": "a vibrant mix of mechanical whirs, people laughing and screaming in delight, carnival game bells, and ambient pop music in the distance",
  "apartment": "interior domestic sounds like muffled voices through walls, footsteps on floors above, appliance hums, and occasional distant traffic",
  "arcade": "retro and modern arcade machine loops, button presses, chiptune melodies, and a blend of excited voices and competitive banter",
  "arena": "cheering crowd reverberating in a large enclosed space, echoing announcements, foot thumps, and team chants",
  "bar": "low clink of glassware, background chatter and laughter, muffled music, and stools scraping on hard floors",
  "barracks": "communal living sounds like snoring, distant drills or roll calls, boots on concrete, and locker doors opening",
  "boat": "wood creaking, gentle waves against the hull, seabirds in the distance, and rope or sail adjustments with occasional voices",
  "bonfire": "crackling wood flames, wind rustling through trees or clothes, and murmured conversation or light acoustic music nearby",
  "bowling_alley": "echoes of heavy balls striking pins, electronic scoring systems, and players cheering or groaning",
  "bridge": "wind and distant vehicle hums over a long span, with occasional footsteps or bike tires clattering on metal or wood planks",
  "bus": "idling engine hums, door hisses, occasional conversations, and shifting chassis sounds over bumps",
  "cable_car": "quiet whooshing of suspended travel, with pulleys, swaying cables, and muffled interior chatter or wind outside",
  "cafe": "quiet clinking of cups and saucers, soft conversations, ambient music, and occasional espresso machine hisses and chair movements",
  "caravan": "light footsteps on dirt, low chatter, animal sounds like hoofbeats or camels, and the creak of loaded carts or wagons",
  "carts": "wooden wheels rolling over uneven ground, creaking axles, and occasional shouts or animal sounds depending on use",
  "church": "soft murmuring of prayer, reverberant footsteps, distant organ music, and the occasional echoing bell or cough",
  "cinema": "rustling popcorn, faint whispers, projector whirr, and muffled sound effects or dialogue from a distant film screen",
  "clubhouse": "friendly conversation, wooden floor creaks, shuffling chairs, and occasional background music or laughter",
  "coal_depot": "heavy crunch of coal being shoveled, low rumble of conveyors, and dusty air with sporadic metallic clangs",
  "coal_terminal": "mechanical loading sounds, deep engine hums, echoing container shifts, and wind across wide open spaces",
  "company_town": "clustered daily activity including footsteps on boardwalks, workers calling out, machinery hums, and domestic background sounds",
  "concert": "cheering crowd, reverb-heavy live music, stage movements, and moments of silence followed by applause",
  "delivery_hub": "repetitive rolling of carts, beep of scanners, box impacts, and intercom announcements in a busy warehouse",
  "dog_park": "joyful barking, panting dogs running, distant conversations, rustling grass, and jingling collars",
  "dome": "large enclosed space with pronounced echoes, subtle machinery hums, and ambient footsteps or murmur",
  "drive_in": "low hum of parked vehicles, faint film audio through tinny speakers, and background night insects or breeze",
  "event": "crowd murmurs, speaker or performance sounds, occasional applause, and a general air of anticipation and energy",
  "factory": "steady mechanical rhythms, hissing steam, metallic clanks, and layered machinery operating at varied tempos",
  "farm": "ambient animal calls, rustling crops or grass, distant tools or tractors, and bird calls in open air",
  "festival": "overlapping music, laughing voices, vendors calling out, footsteps on various terrain, and distant fireworks or instruments",
  "floating_habitat": "subtle water sloshes, distant engines or air systems, muffled movement of people, and creaking buoyant structures",
  "floating_market": "lively merchant voices, water splashing against boats, exchanges of goods, and ambient chatter",
  "floating_village": "quiet water movement, household sounds on wood, children playing, and distant oar or paddle sounds",
  "gallery": "soft footsteps on polished floors, quiet murmurs of viewers, ambient hum of lighting, and occasional camera clicks",
  "garden": "birdsong, rustling leaves, flowing water features, and distant human activity or tool use",
  "gated_community": "low ambient traffic, distant sprinklers or yard work, occasional footsteps and passing conversations",
  "hall": "echoing footsteps, soft voices, occasional door creaks or distant claps, and reverberant indoor ambience",
  "herding_camp": "animal sounds like bleating or mooing, gentle wind, human calls or whistles, and rustling grasses or fabric",
  "highway": "steady flow of vehicles passing at varying speeds, occasional honks, and wind rushing past open stretches",
  "hotel": "indistinct lobby murmur, suitcase wheels on hard floor, elevator dings, and quiet ambient music",
  "hydroelectric_dam": "low thunderous rush of water, mechanical hums from turbines, and occasional distant maintenance sounds",
  "hydrofoil": "gentle slicing of water, low whine of propulsion, occasional cabin rattles, and ambient passenger conversation",
  "library": "pages turning, footsteps on carpet or tile, whispers, pen scratching, and distant carts rolling books",
  "light_rail": "quiet electric hum, smooth rail gliding sounds, door beeps, and station announcements",
  "lodge": "wood creaking underfoot, fire crackling, soft murmurs, and distant wildlife or wind outside",
  "maglev_train": "hushed air displacement, smooth gliding vibration, occasional electronic chimes, and quiet passenger space",
  "marina": "light clinking of boat rigging, water lapping, seabirds calling, and soft footsteps on docks",
  "market": "varied vendor voices, crowd murmur, items being shuffled or exchanged, and ambient trade sounds",
  "mill": "constant rotation sounds, grain or wood being processed, belt systems, and wooden creaks or mechanical whines",
  "mine": "pick or machinery echoes, distant rock falls, dripping water, and ventilation system hums",
  "motel": "faint traffic sounds, buzzing neon, TV murmurs through walls, and footsteps on gravel or pavement",
  "museum": "muffled conversations, interactive display beeps, quiet footfalls, and ambient educational audio",
  "music_hall": "orchestral tuning, soft crowd murmurs, stage creaks, and detailed reverb of live instruments",
  "nature_center": "birdsong, insect hums, distant conversation, soft wood creaks, and footsteps on gravel or boardwalk",
  "nursery": "gentle lullaby melodies, soft coos or giggles, toy rattles, and calming ambient tones",
  "ocean waves": "continuous wash and pull of surf on sand or rock, with seagull calls and coastal breeze",
  "office": "keyboard clicks, printer hums, quiet talking, phone rings, and HVAC background",
  "oil_platform": "wind over metal surfaces, low rumble of drilling or machinery, footsteps on steel, and occasional alerts",
  "opera": "powerful live vocal performance, rich orchestration, and audience shifting and applause in a large hall",
  "paper mill": "steady machine rollers, paper tearing and folding, and layered industrial hums",
  "park": "children playing, bird calls, rustling trees, joggers or strollers passing, and dog barks in the distance",
  "pier": "wood creaks over water, gulls overhead, occasional splash or boat rope sound, and footsteps on planks",
  "plantation": "ambient nature sounds, distant labor or tools, wind over tall plants, and creaking wood",
  "playground": "joyful child voices, swings creaking, running footsteps, and soft thuds on sand or rubber",
  "pod": "soft interior hum, gentle echo from small space, air circulation sounds, and occasional movement or breathing",
  "pool": "splashing water, echoing voices, dripping, and whistles or laughter bouncing off hard surfaces",
  "power_plant": "dense mechanical drone, electric buzzes, coolant hissing, and distant control room sounds",
  "ranch": "livestock calls, gate creaks, wind over dry ground, and distant barking or riding sounds",
  "ranch_house": "wood floor creaks, screen door slaps, ambient kitchen activity, and dogs or birds nearby",
  "refinery": "rhythmic metallic noises, hiss of steam or gas, safety beeps, and constant machine drone",
  "research_center": "computer hums, typing, glassware clinks or electronics beeps, with reverberant sterile ambience",
  "research_pod": "compact interior with quiet beeps, subtle breathing or shifting, and controlled environmental hum",
  "research_station": "distant weather sounds, occasional equipment pings or radio static, and ambient foot traffic",
  "resort": "peaceful ambient music, waves or fountains, clinking glasses, and happy relaxed voices",
  "restaurant": "silverware clinks, soft ambient chatter, order calls from kitchen, and background music",
  "river": "flowing water over rocks, occasional splash or wildlife call, and ambient forest or grassland rustle",
  "roller_rink": "rolling skates on smooth surface, upbeat music, echoed laughter and motion",
  "rover": "mechanical servo movements, dusty crunches, sensor beeps, and wind or silence in alien terrain",
  "row_house": "neighboring domestic sounds through walls, light city traffic, steps on stairs, and doors opening or closing",
  "sawmill": "buzzing blades, wood impacts, conveyor hums, and scattered tool or human sounds",
  "school": "student chatter, bell rings, footsteps in halls, paper rustling, and echoing classroom voices",
  "shanty": "wind flapping thin walls, quiet conversation, distant music or domestic sounds, and footsteps on dirt",
  "ship": "creaking wood or steel, distant ocean waves, rope and rigging sounds, and low thrum of engines",
  "shop": "bell at entrance, shelves being stocked, checkout beeps, and quiet customer conversation",
  "shuttle": "mechanical door seals, rumble of ignition or propulsion, comm chatter, and cabin noise",
  "skatepark": "skateboard wheels grinding on rails, occasional falls or cheers, and ambient urban noise",
  "solar_farm": "quiet wind, subtle motorized panel shifts, occasional bird calls, and faint inverter hums",
  "stadium": "cheering fans, announcer echoes, distant music or horns, and team-related sounds",
  "store": "ambient music, people browsing and chatting, checkout sounds, and carts rolling on tile",
  "tavern": "crowded voices, clinking mugs, laughter, and the low hum of music or game dice",
  "tents": "fabric flapping in breeze, muffled outdoor activity, zippers or rope tension sounds, and soft interior rustling",
  "theatre": "audience whispers, stage creaks, live performance audio, and large room reverberation",
  "tower": "wind against structure, metal creaks or echoes, and distant environmental sounds from above",
  "tractor": "low rumble of engine, treads or tires over soil, occasional gear shifts, and surrounding farm noises",
  "traffic": "layered car and truck motion, horns, tires over pavement, and shifting city density",
  "train": "steady rhythmic rail clatter, horn blasts, wind rushes past windows, and ambient passenger noise",
  "train_station": "announcements over PA, footsteps on concrete, rolling luggage, and engine hiss or clang",
  "tram": "smooth rolling hum, stop chimes, boarding footsteps, and open-window or speaker ambience",
  "treehouse": "birds and leaves rustling, wood creaking, faint voices or footsteps, and distant ground activity",
  "tribal_village": "communal voices, drumbeats or singing, tools in use, and ambient forest or grassland life",
  "village": "children playing, animals nearby, water being carried, and domestic activities in shared outdoor space",
  "village_green": "open laughter and play, wind in trees, community voices, and music or instruments in distance",
  "water_taxi": "boat engine hum, slapping water, radio static or voices, and coastal city sounds nearby",
  "watermill": "flowing stream or river, wooden wheel rotations, internal gear sounds, and bird or village ambience",
  "wave_generator": "rhythmic hydraulic hisses, metal creaks, and ocean wave movements blending with mechanical output",
  "wind_farm": "broad wind gusts, slow blade rotations, faint generator hums, and distant rural openness",
  "windmill": "rotor creaks, wind blowing past wooden or metal structures, and intermittent gear sounds inside"
}

# Aggregate minimal set
env_audio = []

# Biome wilderness sounds
for biome in location_types.keys():
    env_audio.append(biome_wilderness_sound(biome))

# Human-created sounds
for tag, prompt in generic_sounds.items():
    tag_humanized = tag.replace("_", " ")
    prompt = f"{tag_humanized} ambience: {prompt}"

    path = f"environment/{tag}.wav"

    env_audio.append({
        "path": path,
        "tags": [tag],
        "prompt": prompt
    })

import datetime
now = datetime.datetime.now().isoformat(timespec="minutes").replace(":", "-")

# Save the generated environment audio JSON
filename = f"layers_{now}.json"

# Save to file
with open(f"services/audio/config/{filename}", "w") as f:
    json.dump(env_audio, f, indent=4)
