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
        "arctic": "Arctic: howling wind, cracking ice, distant seals, muffled silence"
    }
    return {
        "path": f"environment/wilderness/{biome}_ambience.wav",
        "tags": [biome, "wilderness"],
        "prompt": prompts.get(biome, "Natural wilderness ambience for this biome")
    }

# Human sounds: by era + sector + detail (biome-agnostic, unless biome really matters)
# Grouped prompts to avoid excess duplication
generic_sounds = {
    "pre_industrial": {
        "transportation": {
            "hunting trail": "Footsteps on a soft earth trail, leaves crunching, distant animal calls",
            "river, dugout canoe": "Wooden canoe gliding through water, gentle paddling, birdsong",
            "horse trail": "Horses walking on a forest trail, birds and wind in trees",
            "camel caravan": "Camels walking on sand, bells jingling, light wind",
            "river, wood canoe": "Canoe moving on a quiet river, paddles, forest wildlife",
            "canoes": "Canoes gliding in still water, soft splashes, insect sounds",
            "rafts": "Wooden rafts creaking, gentle water, human voices"
        },
        "housing": {
            "traditional dwellings": "Chatter around a traditional dwelling, fire crackling, animal sounds",
            "tribal camp": "Low voices, fire crackling, distant night animals",
            "log cabin": "Wood crackling in a hearth, creaking logs, wind outside",
            "nomadic tents": "Canvas flapping, distant animal bells, wind",
            "oasis camp": "Murmur of camp near oasis, water, date palms, animal brays",
            "mountain village": "Stone footsteps, goats, distant bells, mountain wind",
            "fishing village": "Children playing, gulls, small waves, nets cast in water",
            "stilt houses": "Water slapping stilts, birds, village chatter",
            "floating villages": "Wood creaks, water movement, people talking"
        },
        "industry": {
            "subsistence agriculture": "Tools working earth, farm animals, distant voices",
            "swidden agriculture": "Fire crackling, machetes clearing brush, forest insects",
            "oasis farming": "Digging, water irrigation, animal brays",
            "subsistence fishing": "Casting nets, water splashes, birds",
            "terraced farming": "Raking earth, water trickling, workers talking"
        },
        "business": {
            "village market": "Crowd chatter, bartering, animal sounds, wood stalls",
            "caravan trading stops": "Camels settling, merchants haggling, market bustle",
            "floating markets": "Vendors calling, wooden boats, splashing, market chatter"
        },
        "culture": {
            "tribal meeting ground": "Group singing, drums, fire crackling, night insects",
            "ceremonial gathering site": "Ceremonial chants, group movement, fire sounds",
            "marketplace": "Crowds, traders, distant music, goods being sold"
        }
    },
    "industrial": {
        "transportation": {
            "dirt roads": "Horse carts on dirt roads, wagon wheels, distant blacksmith hammer",
            "railroad": "Steam train approaching, whistle, track clatter",
            "river, steamboat": "Paddle steamer churning, water, passengers chatting",
            "carts": "Wooden carts, creaking wheels, shouts",
            "river, barge": "Barge creaks, water movement, workers loading"
        },
        "housing": {
            "village": "Children playing, distant animals, village chatter",
            "town": "Market bustle, bells, carts rolling on streets",
            "logging camp, clearcut": "Saws cutting, workers shouting, falling trees",
            "mining camp": "Machinery, clanking, distant explosions, worker voices",
            "plantation houses": "Distant field work, wind in trees, footsteps on wood"
        },
        "industry": {
            "factory, pollution": "Machinery thumping, hiss of steam, pulleys and belts",
            "mine, tailings": "Distant blasts, rocks falling, metal on rock",
            "oil extraction, pollution": "Oil pump jacks, metallic clang, leaking fluids",
            "sawmill, clearcut": "Buzz saws, logs rolling, workers' voices",
            "windmill": "Rotating wooden blades, creaks, wind",
            "papermill, clearcut": "Paper machines, water sprays, mechanical clatter"
        },
        "business": {
            "trading post": "Bell rings, doors creak, conversation, goods exchanged",
            "hotel": "Luggage rolling, bellhop, quiet lobby music",
            "train station": "Steam train sounds, crowd chatter, announcements"
        },
        "culture": {
            "church": "Organ playing, congregation murmur, bells",
            "town square": "Market crowd, carts, music, public speeches",
            "community hall": "Gathered voices, wooden floor footsteps, laughter",
            "theatre": "Murmur of audience, musicians tuning",
            "clock tower": "Mechanical ticking, bell chimes, birds"
        }
    },
    "current": {
        "transportation": {
            "highways": "Distant roar of traffic, horns, whooshing cars",
            "superhighway": "Continuous high-speed vehicles, faint rumble, occasional horn",
            "airport": "Jet engines, PA announcements, rolling suitcases",
            "river, bridge": "Traffic crossing, water below, city noises",
            "shipping": "Large ship engines, waves, dock workers calling"
        },
        "housing": {
            "apartments": "Footsteps in hallway, distant elevator ding, city outside",
            "mansion": "Muted voices, garden birds, faint music",
            "suburban housing": "Children playing, lawnmower, neighbors talking",
            "fortified compounds": "Security gates, radios, distant conversations",
            "villa": "Quiet breeze, pool water, distant laughter",
            "rural town": "Dogs barking, passing cars, birds, voices",
            "favelas": "Distant music, voices, everyday life sounds",
            "tourist resort": "Splashing pool, crowd, tropical music, seagulls",
            "modern research station": "Computer hum, wind, radio chatter"
        },
        "industry": {
            "commercial agriculture fields": "Machinery in fields, birds, wind in crops",
            "factory, pollution": "Continuous machines, conveyor beeps, metal clanks",
            "oil refinery, pollution": "Engines, hiss of gas, loud machinery",
            "power plant": "Turbine whir, deep rumble, warning alarms",
            "oil wells": "Thumping oil pump, mechanical grinding, wind",
            "hydroelectric dam, river, lake": "Rushing water, turbines, echoing mechanical hum",
            "large dam and reservoir": "Water rushing, gates opening, machinery",
            "shipping containers": "Metal containers moved, crane beeps, dock traffic",
            "industrial agriculture": "Sprinklers, combine harvesters, distant tractors",
            "tourism, beaches": "Waves, laughter, music, volleyball",
            "fishing": "Engines, nets pulled, seagulls, crew calls",
            "papermill": "Paper machine noise, fans, forklifts moving"
        },
        "business": {
            "shopping mall": "Background music, crowd chatter, shoe footsteps, escalator hum",
            "box stores": "Shopping carts, announcements, cash register beeps",
            "strip mall": "Parking lot traffic, door chimes, low chatter",
            "skyscrapers": "Elevator bells, city hum, HVAC",
            "garbage dump": "Trucks backing up, seagulls, machinery",
            "tourist resort": "Pool splashing, music, drinks being served",
            "tourist facilities": "Tourist chatter, guide announcements"
        },
        "culture": {
            "stadium": "Crowd roars, announcer, music, horns",
            "park": "Birds, people chatting, kids playing, dogs",
            "museum": "Quiet footsteps, murmured conversations, echoing gallery",
            "amusement park": "Ride clanks, music, laughter, excited shouts",
            "pool": "Splashing water, kids, lifeguard whistle",
            "beach": "Waves, seagulls, voices, distant music",
            "boardwalk": "Footsteps on wood, games, carnival music, gulls",
            "sports fields": "Whistle, cheering, play sounds",
            "sailboats": "Wind in sails, ropes creak, waves"
        }
    },
    "future": {
        "transportation": {
            "highways": "Electric vehicles passing, faint hum, wind turbines",
            "river, bridge": "Silent public transit, water below, birds",
            "high-speed train": "Smooth train whoosh, automated announcements"
        },
        "housing": {
            "eco-villages": "Solar panels, soft garden sounds, quiet conversation",
            "co-op apartments": "Community kitchen, laughter, quiet living spaces",
            "sustainable stilt houses": "Waves, wind in eco-construction, laughter"
        },
        "industry": {
            "solar farms": "Faint inverter hum, wind, energy transmission",
            "wind farm": "Rotating turbine blades, wind passing through",
            "organic farming": "Hand tools, birds, soft conversation",
            "recycling center": "Glass sorting, conveyor belts, alarms"
        },
        "business": {
            "market": "Lively community market, vendors, music, sustainable products"
        },
        "culture": {
            "stadium": "Cheering crowds, music, PA announcements",
            "park": "Children playing, green energy hum, birds",
            "museum": "Guided tours, energy efficient lighting, quiet voices",
            "theatre": "Music, applause, green building acoustics",
            "public art sculpture": "Outdoor voices, wind, occasional footsteps",
            "community center": "Voices, kids, shared activities"
        }
    },
    "dystopia": {
        "transportation": {
            "abandoned highways": "Wind over cracked pavement, distant crows, silence",
            "collapsed bridges": "Echoes, dripping water, rubble shifting",
            "derelict airports": "Rusting metal, loose wires clanging, empty space"
        },
        "housing": {
            "favelas": "Quiet, wind through ruins, occasional voices",
            "crumbling apartments": "Dust, debris falling, wind, silence",
            "fortified mansion": "Clanking metal doors, echoing footsteps",
            "military base": "Rattling fences, distant alarm, crows",
            "makeshift shelter": "Flapping tarps, crackling fire, dogs barking",
            "abandoned research station": "Wind whistling, cracking ice, silence"
        },
        "industry": {
            "derelict factories": "Dripping water, distant clang, rats in metal pipes",
            "abandoned refinery": "Echoes, rusting metal, hollow wind",
            "ruined power plant": "Occasional electrical crackle, broken glass",
            "subsistence farming": "Hoe striking earth, distant birds, wind"
        },
        "business": {
            "deserted shopping mall": "Dripping water, distant echo, broken glass",
            "ruined strip mall": "Shifting debris, creaking beams, silence",
            "ruined box stores": "Wind through open doors, creaks",
            "crumbling skyscrapers": "Loose debris, rattling glass, moaning wind"
        },
        "culture": {
            "empty stadium": "Echoing wind, distant birds, silence",
            "overgrown park": "Birds, rustling leaves, broken playground sounds",
            "decaying amusement park": "Wind through rides, creaks, distant animal calls"
        }
    }
}

# Aggregate minimal set
env_audio = []

# Biome wilderness sounds
for biome in location_types.keys():
    env_audio.append(biome_wilderness_sound(biome))

# Human-created sounds
for era, era_data in generic_sounds.items():
    for sector, details in era_data.items():
        for detail, prompt in details.items():
            # Skip empty details (sometimes present in sector_defaults)
            if not detail.strip():
                continue
            # Use sector_defaults and location_types to tag relevant biomes if relevant
            # Most are biome agnostic, except a few sector+biome combos
            tags = [era, sector, detail]
            # Some details are obviously tied to a biome, e.g., "swidden agriculture" => rainforest
            # Otherwise, biome-agnostic
            biome_specific = False
            biome_tag = None
            # Tie sectoral detail to biome if only plausible there
            if "swamp" in detail:
                biome_tag = "swamp"
            elif "desert" in detail or "camel" in detail or "oasis" in detail:
                biome_tag = "desert"
            elif "tundra" in detail or "igloo" in detail or "sled" in detail:
                biome_tag = "tundra"
            elif "steppe" in detail:
                biome_tag = "steppe"
            elif "mountain" in detail:
                biome_tag = "mountain"
            elif "rainforest" in detail or "swidden" in detail:
                biome_tag = "rainforest"
            elif "river" in detail or "canoe" in detail or "barge" in detail:
                biome_tag = "river"
            elif "coastal" in detail or "fishing village" in detail:
                biome_tag = "coastal"
            elif "plains" in detail or "grass" in detail:
                biome_tag = "plains"
            elif "arctic" in detail:
                biome_tag = "arctic"
            if biome_tag:
                tags = [biome_tag] + tags

            # Generate path
            safe_detail = detail
            for char in "!@#$%^&*()[]{};:'\"\\|<>?-=+`~ .,":
                safe_detail = safe_detail.replace(char, "_")
            safe_detail = safe_detail.replace("__", "_")  # Ensure no double underscores
            path = f"environment/{era}/{sector}_{safe_detail}.wav"

            env_audio.append({
                "path": path,
                "tags": tags,
                "prompt": prompt
            })

import datetime
now = datetime.datetime.now().isoformat(timespec="minutes").replace(":", "-")

# Save the generated environment audio JSON
filename = f"layers_{now}.json"

# Save to file
with open(f"services/audio/config/{filename}", "w") as f:
    json.dump(env_audio, f, indent=4)
