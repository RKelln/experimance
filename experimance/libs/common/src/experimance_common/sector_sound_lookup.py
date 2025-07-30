
"""
Sector sound classification lookup for Experimance.

This module provides a mapping from unique sector items (from anthropocene.json)
to canonical sound categories. Use this to associate sector items with sound cues.

Given #file:unique_anthropocene_elements.txt create a python file that contains a 
lookup table mapping many to one, that can be loaded into other modules that helps 
classify each unique sector item that should have an associated sound. 

Leave out sector items that do not have a recognizable or loud sound associated with them or just ambience
(e.g. "forest", "river", "mountain").

Handle pluralization and adjectives that wouldn't strongly affect the sound:
e.g.
"factories", "factory", "canning factory" => "factory"

But do separate items that are distinct in sound:
"train", "railway", "mountain railway" => "train"
"train station", "railway station" => "train station"
"maglev train", "high-speed maglev train", "solar maglev line" => "maglev train"

Items that are abandoned or derelict or otherwise fallen silent should not map to a sound.

(Please include this entire prompt as a comment in the code).
"""


# Only sector items with a recognizable, non-ambient, non-abandoned sound are included.
# Plural/adjective variants are mapped to a canonical sound class.
SECTOR_SOUND_LOOKUP = {
    # --- NATURAL FEATURES ---
    "river": "river",
    "rivers": "river",
    "stream": "river",
    "ocean": "ocean waves",
    "beach": "ocean waves",
    
    # --- INDUSTRY & ENERGY ---
    "factory": "factory",
    "factories": "factory",
    "canning factory": "factory",
    "industrial park": "factory",
    "steelworks": "factory",
    "steel mill": "factory",
    "industrial sawmill": "sawmill",
    "mechanized sawmill": "sawmill",
    "sawmill": "sawmill",
    "steam-powered sawmill": "sawmill",
    "water-powered sawmill": "sawmill",
    "solar-powered timber mill": "sawmill",
    "pulp and paper mill": "paper mill",
    "paper mill by river": "paper mill",
    "grist mill by river": "mill",
    "grist mill by stream": "mill",
    "textile mills": "mill",
    "textile workshop": "mill",
    "sugar mill": "mill",
    "windmill": "windmill",
    "windmills": "windmill",
    "watermill": "watermill",
    "watermills": "watermill",
    "coal mine": "mine",
    "coal mines": "mine",
    "mining operation": "mine",
    "phosphate mine": "mine",
    "copper mine": "mine",
    "quarry": "mine",
    "oil refinery": "refinery",
    "petrochemical plant": "refinery",
    "gasworks": "refinery",
    "power plant": "power_plant",
    "solar power plant": "power_plant",
    "wind farm": "wind_farm",
    "wind farms": "wind_farm",
    "wind turbine": "wind_farm",
    "wind turbine array": "wind_farm",
    "offshore wind farm": "wind_farm",
    "hydroelectric dam": "hydroelectric_dam",
    "offshore oil platform": "oil_platform",
    "oil/gas extraction platform": "oil_platform",
    "coal depot": "coal_depot",
    "coal barge terminal": "coal_terminal",
    "solar farm": "solar_farm",
    "algae farm": "algae_farm",
    "algae biofilter plant": "algae_farm",
    "algae biofuel platform": "algae_farm",
    "wave energy generator": "wave_generator",

    # --- AGRICULTURE & RANCHING ---
    "agribusiness soy plantation": "plantation",
    "soy plantation": "plantation",
    "palm oil plantation": "plantation",
    "large-scale plantation": "plantation",
    "plantation": "plantation",
    "cattle ranch": "ranch",
    "ranch": "ranch",
    "ranch-style house": "ranch_house",
    "ranch-style houses": "ranch_house",
    "cattle herding camp": "herding_camp",
    "sheep herding": "herding_camp",
    "sheep shearing shed": "herding_camp",
    "nomadic tent camp": "herding_camp",
    "oasis camp": "herding_camp",
    "herding camp": "herding_camp",
    "farm": "farm",
    "farms": "farm",
    "farmstead": "farm",

    # --- MARKETS, STORES, & OFFICES ---
    "company store": "store",
    "company store for farmhands": "store",
    "company store for herders": "store",
    "company store for trappers": "store",
    "company store for loggers": "store",
    "company store for boatmen": "store",
    "company store for dockworkers": "store",
    "company store on stilts": "store",
    "general store at crossroads": "store",
    "plantation store": "store",
    "roadside market": "market",
    "farmers' market": "market",
    "village market": "market",
    "river market": "market",
    "pop-up food market": "market",
    "pop-up seafood market": "market",
    "tide market": "market",
    "market square": "market",
    "market hall": "market",
    "thriving market": "market",
    "community market": "market",
    "village barter market": "market",
    "trading camp": "market",
    "trading fair": "market",
    "trading post": "market",
    "trading outpost": "market",
    "rubber trading post": "market",
    "town square": "market",
    "colonial trading post": "market",
    "urban market": "market",
    "bazaar": "market",
    # Offices (for background office/tech sounds)
    "agribusiness headquarters": "office",
    "resource extraction logistics office": "office",
    "newspaper office": "office",
    "real estate office": "office",
    "NGO headquarters": "office",
    "bank": "office",
    "data center": "office",
    "tech campus": "office",
    "innovation hub": "office",
    "remote work hub": "office",
    "circular economy recycling hub": "office",

    # --- TRANSPORTATION: TRAINS, TRAMS, BUSES, ETC. ---
    "railway": "train",
    "railway siding": "train",
    "rail bridge over river": "train",
    "rail line to pier": "train",
    "rail tram": "train",
    "railroad depots": "train",
    "steam locomotive": "train",
    "steam train": "train",
    "train": "train",
    "mountain railway": "train",
    "train station": "train_station",
    "suburban train station": "train_station",
    "railway station": "train_station",
    "maglev train": "maglev_train",
    "high-speed maglev train": "maglev_train",
    "solar maglev line": "maglev_train",
    "autonomous farm vehicles": "tractor",
    "steam tractor route": "tractor",
    "tractor": "tractor",
    "bus stop": "bus",
    "bus stop at grain silo": "bus",
    "bus terminal": "bus",
    "electric bus route": "bus",
    "mass transit": "bus",
    "urban bus rapid transit": "bus",
    "autonomous electric vehicles": "bus",
    "autonomous forest tram": "tram",
    "tram": "tram",
    "solar-powered tram": "tram",
    "solar-powered trams": "tram",
    "solar-powered river tram": "tram",
    "light rail": "light_rail",
    "autonomous cable car": "cable_car",
    "cable car": "cable_car",
    "autonomous hydrofoil": "hydrofoil",
    "autonomous water taxi": "water_taxi",
    "solar-powered water taxi": "water_taxi",
    "autonomous delivery hub": "delivery_hub",
    "autonomous solar rover": "rover",
    "autonomous wind-powered rover": "rover",
    "autonomous marsh shuttle": "shuttle",
    "airport": "airport",
    "airstrip": "airstrip",
    "airstrip on permafrost": "airstrip",
    "private airstrip": "airstrip",
    "cruise ship terminal": "ship",
    "cruise ship dock": "ship",
    "container barge": "ship",
    "container port": "ship",
    "marina": "marina",
    "motorboat dock": "boat",
    "boat rental kiosk": "boat",
    "kayak route": "boat",
    "river ferry": "boat",
    "flat-bottomed barge": "boat",
    "steam riverboat": "boat",
    "steam riverboat dock": "boat",
    "island-hopping raft": "boat",
    "raft": "boat",
    "outrigger canoe": "boat",
    "dugout canoe": "boat",
    "canoe": "boat",
    "fishing boat": "boat",
    "motorized fishing skiff": "boat",
    "sailing outrigger": "boat",
    "mule caravan": "caravan",
    "camel caravan": "caravan",
    "caravan": "caravan",
    "horse caravan": "caravan",
    "horseback caravan": "caravan",
    "carts": "carts",
    "horse-drawn carts": "carts",
    "horse-drawn cart": "carts",

    # --- ROADS, BRIDGES, HIGHWAYS ---
    "highway": "highway",
    "highways": "highway",
    "superhighway": "highway",
    "superhighways": "highway",
    "paved highway": "highway",
    "paved rural highway": "highway",
    "concrete causeway": "highway",
    "coastal highway": "highway",
    "busy roads and highways": "highway",
    "concrete highway bridge": "bridge",
    "collapsed bridge": "bridge",
    "covered bridge over creek": "bridge",
    "covered wooden bridge": "bridge",
    "wooden bridges": "bridge",
    "iron bridges": "bridge",
    "bridge": "bridge",
    "collapsed pier": "pier",
    "wrecked pier": "pier",
    "pier amusement park": "amusement_park",
    "vehicles": "traffic",
    "traffic": "traffic",
    "traffic jam": "traffic",
    "traffic roundabout": "traffic",
    "traffic circle": "traffic",
    "modern vehicles": "traffic",

    # --- DWELLINGS & HOUSING ---
    "apartment block": "apartment",
    "boarding houses": "apartment",
    "micro-apartments": "apartment",
    "co-housing towers": "apartment",
    "luxury condo tower": "apartment",
    "riverside apartment block": "apartment",
    "rental apartments": "apartment",
    "tenements": "apartment",
    "row houses": "row_house",
    "company row house by river": "row_house",
    "company town row house": "row_house",
    "company town": "company_town",
    "company towns": "company_town",
    "company bunkhouse": "barracks",
    "company barracks": "barracks",
    "worker barracks": "barracks",
    "resource worker camp": "barracks",
    "migrant worker camp": "barracks",
    "migrant worker dormitory": "barracks",
    "temporary worker camp for oil/gas crews": "barracks",
    "salt mining camp": "barracks",
    "industrial logging camp": "barracks",
    "logging camp": "barracks",
    "mobile field shelter": "tents",
    "makeshift shelter": "tents",
    "eco-lodge": "lodge",
    "eco-research pod": "research_pod",
    "exclusive arctic luxury lodge for corporate clients": "lodge",
    "luxury lodge": "lodge",
    "luxury cabin development": "lodge",
    "luxury chalet development": "lodge",
    "luxury resort compound": "lodge",
    "beach bungalow resort": "resort",
    "beach motel": "motel",
    "beach shanty": "shanty",
    "cliffside fishing village": "village",
    "village": "village",
    "village green at forest edge": "village_green",
    "tribal village": "tribal_village",
    "tribal camp": "tribal_village",
    "tribal meeting grounds": "tribal_village",
    "tribal meeting ground": "tribal_village",
    "tribal trade post": "tribal_village",
    "floating sea habitat": "floating_habitat",
    "floating solar habitat": "floating_habitat",
    "floating eco-habitat": "floating_habitat",
    "floating village": "floating_village",
    "floating market": "floating_market",
    "wave energy generator": "wave_generator",
    "urban apartments": "apartment",

    # --- FESTIVALS, EVENTS, & RECREATION ---
    "amphitheater": "amphitheater",
    "aurora amphitheater": "amphitheater",
    "coastal amphitheater": "amphitheater",
    "beach festival": "festival",
    "music festival": "festival",
    "festival": "festival",
    "festival tent": "festival",
    "marshland festival": "festival",
    "swamp music festival": "festival",
    "summer beach concert": "concert",
    "drive-in movie field": "drive_in",
    "drive-in theater": "drive_in",
    "outdoor concert stage": "concert",
    "open-air cinema": "cinema",
    "cinema tent": "cinema",
    "theatre": "theatre",
    "opera house": "opera",
    "music hall": "music_hall",
    "plains music hall": "music_hall",
    "coastal music hall": "music_hall",
    "amusement park": "amusement_park",
    "decaying amusement park": "amusement_park",
    "overgrown collapsed amusement park": "amusement_park",
    "pier amusement park": "amusement_park",
    "roller rink": "roller_rink",
    "boardwalk arcade": "arcade",
    "beach bonfire gathering": "bonfire",
    "beachside tiki bar": "bar",
    "cafe": "cafe",
    "riverside tavern": "tavern",
    "inn": "tavern",
    "pub": "tavern",
    "fast food restaurant": "restaurant",
    "surf shop": "shop",
    "shopping mall": "shop",
    "bowling alley": "bowling_alley",
    "skatepark": "skatepark",
    "dog park": "dog_park",
    "playground": "playground",
    "stadium": "stadium",
    "sports arena": "arena",
    "arena": "arena",
    "gated community": "gated_community",
    "gated expat compound": "gated_community",
    "gated desert community": "gated_community",
    "gated community clubhouse": "clubhouse",
    "hotel": "hotel",
    "luxury hotel": "hotel",
    "hotel resort": "hotel",
    "grand hotel": "hotel",

    # --- COMMUNITY, PUBLIC, & CULTURE ---
    "art gallery": "gallery",
    "museum": "museum",
    "public library": "library",
    "public pool": "pool",
    "community pool": "pool",
    "community garden": "garden",
    "community hall": "hall",
    "community rooftop farm": "farm",
    "community sand sculpture event": "event",
    "community treehouse complex": "treehouse",
    "treehouse dwelling": "treehouse",
    "tree nursery": "nursery",
    "public park": "park",
    "city park": "park",
    "urban park": "park",
    "nature center": "nature_center",
    "nature interpretation center": "nature_center",
    "swamp experience center": "nature_center",
    "forest experience center": "nature_center",
    "biodiversity research center": "research_center",
    "biotech research center": "research_center",
    "biotech labs": "research_center",
    "research center": "research_center",
    "research station": "research_station",
    "remote research station": "research_station",
    "mobile research caravan": "research_station",
    "school": "school",
    "schoolhouse": "school",
    "public school": "school",
    "community school": "school",
    "permafrost habitat dome": "dome",
    "dome habitat": "dome",
    "vertical eco-village": "village",
    "vertical eco-aprtments": "apartment",
    "vertical grassland pod": "pod",
    "vertical beach tower": "tower",
    "vertical alpine habitat": "dome",
    "vertical forest village": "village",
    "vertical farms": "farm",
    "vertical farm": "farm",
    "church": "church",
    "cathedral": "church",
    "chapel": "church",
    "temple": "church",
    "monastery": "church",
    
    # Other
    "wartorn": "wartorn",
}


if __name__ == "__main__":
    # list all unique audio tags
    unique_tags = set(SECTOR_SOUND_LOOKUP.values())
    for tag in sorted(unique_tags):
        print(tag)

# airport
# airstrip
# algae_farm
# amphitheater
# amusement_park
# apartment
# arcade
# arena
# bar
# barracks
# boat
# bonfire
# bowling_alley
# bridge
# bus
# cable_car
# cafe
# caravan
# carts
# church
# cinema
# clubhouse
# coal_depot
# coal_terminal
# company_town
# concert
# delivery_hub
# dog_park
# dome
# drive_in
# event
# factory
# farm
# festival
# floating_habitat
# floating_market
# floating_village
# gallery
# garden
# gated_community
# hall
# herding_camp
# highway
# hotel
# hydroelectric_dam
# hydrofoil
# library
# light_rail
# lodge
# maglev_train
# marina
# market
# mill
# mine
# motel
# museum
# music_hall
# nature_center
# nursery
# ocean waves
# office
# oil_platform
# opera
# paper mill
# park
# pier
# plantation
# playground
# pod
# pool
# power_plant
# ranch
# ranch_house
# refinery
# research_center
# research_pod
# research_station
# resort
# restaurant
# river
# roller_rink
# rover
# row_house
# sawmill
# school
# shanty
# ship
# shop
# shuttle
# skatepark
# solar_farm
# stadium
# store
# tavern
# tents
# theatre
# tower
# tractor
# traffic
# train
# train_station
# tram
# treehouse
# tribal_village
# village
# village_green
# water_taxi
# watermill
# wave_generator
# wind_farm
# windmill