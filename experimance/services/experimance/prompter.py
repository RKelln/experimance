import time
import random
import json
import argparse
from pathlib import Path
from typing import Tuple
from functools import reduce
from itertools import product

# Function to load geo-locations from JSON file
def load_geo_locations(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to load human developments from JSON file
def load_human_developments(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to generate child prompts based on base geography and stage
def generate_child_prompt(base_prompt, stage, developments):
    if stage == "" or stage not in developments:
        return base_prompt
    stage_developments = developments[stage]
    development_list = []
    for category, structures in stage_developments.items():
        for structure in structures:
            development_list.append(structure)
    return f"{base_prompt}, including {', '.join(development_list)}."

# Function to generate base prompt for a specific location and type
def generate_base_prompt(base_prompt, location, biome_type, specific_prompt):
    return f"{base_prompt} {location} {specific_prompt}"

# Function to generate development sequence for a given base prompt
def generate_development_sequence(base_prompt, biome_type, developments):
    stages = ["", "pre_industrial", "industrial", "current"]
    return [generate_child_prompt(base_prompt, stage, developments[biome_type]) for stage in stages]


class StringListData():
    def __init__(self, string_or_list=None, random_strategy=None):
        self.random_strategy = random_strategy
        self.strings = string_or_list
        if self.strings is None or self.strings == "":
            self.strings = []
        if not isinstance(self.strings, list):
            self.strings = [self.strings]
        # clean up any extra spaces and commas
        self.strings = [clean_prompt(s) for s in self.strings]
        self.index = 0
        if random_strategy == "shuffle":
            random.shuffle(self.strings)

    def __str__(self):
        return str(self.strings)
    
    def __repr__(self):
        return f"StringListData({self.strings})"

    def __len__(self):
        return len([s for s in self.strings if s != ""])

    def __add__(self, other):
        if not isinstance(other, StringListData):
            return NotImplemented
        if self.random_strategy != other.random_strategy:
            raise ValueError("Cannot add different random_strategy types")
        if len(other.strings) == 0:
            return self
        n = StringListData(self.strings + other.strings, self.random_strategy)
        n.index = self.index # keep index the same
        #print(f"StringListData.__add__:", n, self.strings, other.strings)
        return n

    def get(self):
        if len(self.strings) == 0:
            return ""
        if len(self.strings) == 1:
            return self.strings[0]
        # default is derministic loop
        if self.random_strategy is None:
            s = self.strings[self.index]
            self.index += 1
            if self.index >= len(self.strings):
                self.index = 0
        # else random non-repeating
        elif self.random_strategy == "shuffle":
            if len(self.strings) == 2:
                # just go back and forth between the two options
                s = self.strings[self.index]
                self.index += 1
                if self.index >= len(self.strings):
                    self.index = 0
            else:
                s = self.strings[self.index]
                self.index += 1
                if self.index >= len(self.strings):
                    self.index = 0
                    random.shuffle(self.strings)
                    # avoid repeating ourselves
                    if self.strings[0] == s:
                        # move the first element to the end
                        self.strings = self.strings[1:] + self.strings[:1]
        elif self.random_strategy == "choice":
            s = random.choice(self.strings)
        elif self.random_strategy == "all":
            # return all strings, joined with commas
            s = ", ".join(self.strings)
        # if a function, then run it
        elif callable(self.random_strategy):
            s = self.random_strategy(self.strings)
        return s
    
    def generator(self, loop=False):
        if loop:
            while True:
                yield self.get()
            return
        
        # for one loop
        for _ in range(len(self.strings)):
            yield self.get()

class PromptData():
    # jason data is a key followed by a string or a list of strings
    def __init__(self, data=None, random_strategy=None):
        self.random_strategy = random_strategy
        self.weight = 1.0
        if data is None or data == "":
            self.prompt = StringListData("", random_strategy)
            self.style = StringListData("", random_strategy)
            self.negative = StringListData("", random_strategy)
            self.lora = StringListData("", "all")
        # if a string or list
        elif isinstance(data, (str, list, tuple)):
            # if it is a list of 4 str/lists then assume they are prompt, style, negative, lora
            if isinstance(data, (list, tuple)) and len(data) == 4:
                self.prompt = StringListData(data[0], random_strategy)
                self.style = StringListData(data[1], random_strategy)
                self.negative = StringListData(data[2], random_strategy)
                self.lora = StringListData(data[3], "all")
            else: # assume it is just a prompt
                self.prompt = StringListData(data, random_strategy)
                self.style = StringListData("", random_strategy)
                self.negative = StringListData("", random_strategy)
                self.lora = StringListData("", "all")
        elif isinstance(data, dict): 
            # assume it has this format (wih keys being optional):
            # {"prompt": "prompt text or list", 
            # "style": "style text or list", 
            # "negative": "negative prompt text or list", 
            # "lora": "lora prompt text or list"
            # }
            self.prompt = StringListData(data.get("prompt", ""), random_strategy)
            self.style = StringListData(data.get("style", ""), random_strategy)
            self.negative = StringListData(data.get("negative", ""), random_strategy)
            self.lora = StringListData(data.get("lora", ""), "all")
        else:
            raise ValueError("Invalid json_data format")
        
    def __str__(self):
        return f"{self.prompt} {self.style} {self.lora}"
    
    def __repr__(self):
        return f"PromptData(p:{self.prompt} s:{self.style} n:{self.negative} l:{self.lora})"

    def __len__(self):
        return len(self.prompt) + len(self.style) + len(self.negative) + len(self.lora)

    def __add__(self, other):
        if not isinstance(other, PromptData):
            return NotImplemented
        n = PromptData(random_strategy=self.random_strategy)
        n.prompt = self.prompt + other.prompt
        n.style = self.style + other.style
        n.negative = self.negative + other.negative
        n.lora = self.lora + other.lora
        #print(f"PromptData.__add__:", n)
        return n

    def next_prompt(self):
        if self.weight < 0.1:
            return ""
        if self.weight != 1.0:
            return f"({self.prompt.get()}:{self.weight:.2f})"
        return self.prompt.get()
    
    def next_style(self):
        if self.weight < 0.1:
            return ""
        if self.weight != 1.0:
            return f"({self.prompt.get()}:{self.weight:.2f})"
        return self.style.get()
    
    def next_negative(self):
        if self.weight < 0.1:
            return ""
        if self.weight != 1.0:
            return f"({self.prompt.get()}:{self.weight:.2f})"
        return self.negative.get()
    
    def next_lora(self):
        return self.lora.get().replace(",", "") # remove commas
    
    def all_strings(self) -> Tuple[list, list, list, list]:
        return (self.prompt.strings, self.style.strings, self.negative.strings, self.lora.strings)

    def all_combinations(self) -> list:
        all_strings = self.all_strings()
        # product doesn't work with empty lists, so we replace with a list with empty string
        all_strings = [s if len(s) > 0 else [""] for s in all_strings]
        all_combos = list(product(*all_strings))
        # remove anything that is all empty strings
        all_combos = [p for p in all_combos if not all([s == "" for s in p])]
        if len(all_combos) == 0:
            # return a single so that we can do subsequent operations
            return [PromptData(random_strategy=self.random_strategy)]
        # convert to PromptData
        return [PromptData(p, random_strategy=self.random_strategy) for p in all_combos]

    def generator(self):
        # loop through all the possible combinations
        total = len(self)
        if total == 0:
            yield PromptData() # yield a single empty prompt
            return
        
        for p in list(product(*self.all_strings())):
            yield PromptData({
                                "prompt": p[0],
                                "style": p[1],
                                "negative": p[2],
                                "lora": p[3]
                            }, random_strategy=None)
    
    def set_weight(self, weight):
        if weight < 0:
            weight = 0
        if weight > 2.0:
            weight = 2.0
        self.weight = weight
        
        

def remove_duplicates(s:str|list, delimiter=","):
    # if list then join with delimiter
    # NOTE: this is required so that duplicates work on any internal commas in the list of strings as well
    if isinstance(s, list):
        s = delimiter.join(s)
    # remove duplicates from a string delimited by commas, keep last instance
    string_list = s.split(delimiter)
    # remove spaces
    string_list = [s.strip() for s in string_list if s.strip() != ""]
    string_list = list(dict.fromkeys(string_list))
    return f"{delimiter} ".join(string_list)


def clean_prompt(s:str):
    # remove any extra spaces
    s = s.strip()
    # remove final comma
    s = s[:-1] if s.endswith(",") else s
    return s

# returns a prompt and negative prompt string
def combine_prompts(prompt_data_list:list[PromptData]) -> Tuple[str, str]:
    # create a prompt and negative prompt string using the prompt data list as the ordering
    # prompt data contains: {"prompt": "prompt text", "style": "style text", "negative": "negative prompt text"}
    prompt = []
    style = []
    negative = []
    lora = []
    
    for prompt_data in prompt_data_list:
        prompt.append(prompt_data.next_prompt())
        style.insert(0, prompt_data.next_style()) # style is in reverse order
        negative.append(prompt_data.next_negative())
        lora.append(prompt_data.next_lora())
 
    # combine into string separated by commas
    # remove any duplicates (delimited by commas)
    prompt = remove_duplicates(prompt)
    style = remove_duplicates(style)
    negative = remove_duplicates(negative)
    lora = remove_duplicates(lora)
    
    return clean_prompt(f"{prompt}, {style} {lora}"), clean_prompt(f"{negative}")


def handle_sectors(sector_prompt_json, sectors, strategy):
    # this is complicated, sorry! e.g.:
    # "transportation": ["dirt roads", "river, steamboat", "river, barge"],
    # "housing": ["logging camp"],
    # "industry": ["timber mill"],
    # "business": ["trading post"],
    # "culture": ["church"]
    # we want to combine all the sectors into unique prompts, with one of each sector
    # so we get:
    # ["dirt roads, logging camp, timber mill, trading post, church", 
    #  "river, steamboat, logging camp, timber mill, trading post, church",
    #  "river, barge, logging camp, timber mill, trading post, church"]
    # and so on...

    sector_prompt_combos = list(product(*[sector_prompt_json[sector] for sector in sectors]))
    #print(sector_prompt_combos)
    prompts = []
    for sector_promtpts in sector_prompt_combos:
        prompts.append(clean_prompt(", ".join(sector_promtpts)))
    return PromptData({"prompt": prompts}, strategy)

# Function to save prompts to a file
def save_prompts_to_file(prompts, file_path):
    with open(file_path, 'w') as file:
        for prompt in prompts:
            file.write(f"{prompt}\n")

# Function to generate prompts
def prompter(args):
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist")
    if not Path(data_path / args.locations).exists():
        raise FileNotFoundError(f"Locations file {args.locations} does not exist")
    if not Path(data_path / args.developments).exists():
        raise FileNotFoundError(f"Developments file {args.developments} does not exist")
    geo_data = load_geo_locations(data_path / args.locations)
    human_developments = load_human_developments(data_path / args.developments)

    # set up geo prompts
    biomes = list(geo_data["types"].keys())
    if args.verbose: print("Biomes:", biomes)

    # set up era prompts
    eras = list(human_developments["eras"].keys())
    if args.verbose: print("Eras:", eras)
    sectors = human_developments["sectors"]
    if args.verbose: print("Sectors:", sectors)

    strategy = "shuffle"  # or "choice" or None
    if args.quick_run:
        strategy = None # go through all the prompts in order
    
    era_prompt_data = {}
    for era in eras:
        era_prompt_data[era] = PromptData(human_developments["eras"][era], strategy)
    
    geo_by_era = human_developments["locations"]

    # fill in era defaults
    sector_defaults_by_era = human_developments["sector_defaults"]
    for biome in biomes:
        for era in eras:
            if era not in geo_by_era[biome]:
                geo_by_era[biome][era] = {}
            for sector, default in sector_defaults_by_era[era].items():
                if sector not in geo_by_era[biome][era]:
                    geo_by_era[biome][era][sector] = default
                else: # combine biome specific settings with the era defaults
                    geo_by_era[biome][era][sector] = geo_by_era[biome][era][sector] + default
            #print(f"{biome}: {era}: {geo_by_era[biome][era]}")


    # combine biome and specic location prompts
    biome_prompt_data:dict[str:PromptData] = {}
    biome_era_prompt_data:dict[str:dict[str:PromptData]] = {}
    for biome in biomes:
        # use "style" for "locations"
        data = geo_data["types"][biome]
        if "locations" in data:
            data["style"] = data["locations"]
        biome_prompt_data[biome] = PromptData(data, strategy)
        biome_era_prompt_data[biome] = {}
        for era in eras:
            # combine all the sectors into a single prompt
            if era not in geo_by_era[biome]:
                biome_era_prompt_data[biome][era] = PromptData()
                continue
            sector_prompt_json = geo_by_era[biome][era]
            # combine all the lists into a single list
            biome_era_prompt_data[biome][era] = handle_sectors(sector_prompt_json, sectors, strategy)

    #print("Biome prompts:", biome_prompt_data)
    #print("Biome era prompts:", biome_era_prompt_data)

    all_prompts = []

    base_prompt_data = PromptData(human_developments["base_prompt"], strategy)

    if args.quick_run:
        # go through all possible prompts for biome and era

        for biome in biomes:
            if args.verbose: print(f"Biome: {biome}")
            if args.per_era > 0:
                bps = biome_prompt_data[biome].all_combinations()
                bp = random.choice(bps)

            era_biome_weight = 1.0
            for era in eras:
                if args.verbose: print(f"   Era: {era}")

                if args.per_era > 0:
                    # choose N random prompts per era for each biome
                    beps = biome_era_prompt_data[biome][era].all_combinations()

                    # choose N random prompts per era for each biome
                    for _ in range(args.per_era):    
                        bep = random.choice(beps)
                        # if era is modern then downweight biome specific prompt
                        if era.startswith("industrial"):
                            era_biome_weight -= 0.1
                            bp.set_weight(era_biome_weight)
                        elif era.startswith("modern") or era.startswith("AI"):
                            era_biome_weight -= 0.2
                            bp.set_weight(era_biome_weight)
                        else:
                            bp.set_weight(1.0)
                        
                        prompt, neg = combine_prompts([base_prompt_data, bp, era_prompt_data[era], bep])

                        # format for automatic1111
                        all_prompts.append(f'--prompt "{prompt}" --negative_prompt "{neg}"')

                else:
                    # go through all choices for the biome and era
                    for bep in biome_era_prompt_data[biome][era].all_combinations():
                        #print(f"      {bep}")
                        for bp in biome_prompt_data[biome].all_combinations():
                            #print(f"         {bp}")
                            prompt, neg = combine_prompts([base_prompt_data, bp, era_prompt_data[era], bep])
                            if args.verbose: 
                                print(f"         {prompt}")
                                print(f"         neg: {neg}")

                            # format for automatic1111
                            all_prompts.append(f'--prompt "{prompt}" --negative_prompt "{neg}"')
            all_prompts.append("") # add a blank line between biomes    

        if args.output:
            save_prompts_to_file(all_prompts, args.output)
            print(f"All prompts saved to {args.output}")

    else:
        # generate infinite random prompt
        all_prompts = []
        count = 0
        limit = args.number
        if limit <= 0:
            # estimate total number of prompts
            limit = len(biomes) * len(eras)
            print(f"Total number of prompts: {limit}")
        remaining_biomes = biomes.copy()
        random.shuffle(remaining_biomes)
        while count < limit:
            # random non-repeating biome choice
            biome = remaining_biomes.pop()
            if len(remaining_biomes) == 0:
                remaining_biomes = biomes.copy()
                random.shuffle(remaining_biomes)
            for era in eras:
                if args.verbose: print(f"Biome: {biome}, Era: {era}")
                #print(biome_prompt_data[biome])
                #print(biome_era_prompt_data[biome][era])
                prompt, neg = combine_prompts([base_prompt_data,
                                            biome_prompt_data[biome], 
                                            era_prompt_data[era],
                                            biome_era_prompt_data[biome][era]])
                if args.verbose: 
                    print(prompt)
                    print("neg:", neg)
                    print()
                if args.output:
                    all_prompts.append(f'--prompt "{prompt}" --negative_prompt "{neg}"')
            if limit > 0:
                count += 1

        if args.output:
            save_prompts_to_file(all_prompts, args.output)
            print(f"All prompts saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text prompts for text-to-image AI")
    parser.add_argument('-d','--data-path','--data_path', type=str, default='data/', help="Path containing the JSON files")
    parser.add_argument('--locations', type=str, default='locations.json', help="JSON file containing geo-locations")
    parser.add_argument('--developments', type=str, default='anthropocene.json', help="JSON file containing human developments")
    parser.add_argument('-q','--quick-run','--quick_run', action='store_true', help="Run the program as quickly as possible and save all prompts to a file")
    parser.add_argument('--per_era', type=int, default=1, help="Number of prompts to generate per era in a quick run")
    parser.add_argument('-o','--output', type=str, default='prompts.txt', help="Output file to save the prompts")
    parser.add_argument('--number', type=int, default=0, help="Number of prompts to generate (default: all)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--verbose', action='store_true', help="Print verbose output")
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
    
    prompter(args)
