"""
Prompt generation for Experimance image requests.

This module provides a simplified interface for generating text prompts
by combining base prompts, biome-specific content, era-specific content,
and development sectors. It supports cycling through eras for a locked biome
and switching biomes as needed by the core service.
"""

import json
import random
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from itertools import product

from experimance_common.schemas import Era, Biome


class RandomStrategy(str, Enum):
    """Strategy for selecting from multiple options."""
    DETERMINISTIC = "deterministic"  # Cycle through in order
    SHUFFLE = "shuffle"             # Shuffle and cycle through
    CHOICE = "choice"               # Random choice each time
    REPEAT = "repeat"               # Always return the current option


class PromptComponent:
    """A component that can contribute to prompt generation."""
    strategy: RandomStrategy

    def __init__(self, 
                 prompt: Union[str, List[str]] = "", 
                 style: Union[str, List[str]] = "",
                 negative: Union[str, List[str]] = "",
                 strategy: RandomStrategy = RandomStrategy.SHUFFLE):
        """Initialize a prompt component.
        
        Args:
            prompt: Main prompt text(s)
            style: Style modifier text(s)
            negative: Negative prompt text(s)
            strategy: How to select from multiple options
        """
        self.prompt_options = self._normalize_to_list(prompt)
        self.style_options = self._normalize_to_list(style)
        self.negative_options = self._normalize_to_list(negative)
        self.strategy = strategy
        
        # State for deterministic/shuffle strategies
        self._prompt_index = 0
        self._style_index = 0
        self._negative_index = 0
        
        # Shuffle if needed
        if strategy == RandomStrategy.SHUFFLE or strategy == RandomStrategy.REPEAT:
            random.shuffle(self.prompt_options)
            random.shuffle(self.style_options)
            random.shuffle(self.negative_options)
    
    def _normalize_to_list(self, value: Union[str, List[str]]) -> List[str]:
        """Convert string or list to list, filtering empty strings."""
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        elif isinstance(value, list):
            return [v.strip() for v in value if v.strip()]
        else:
            return []
    
    def _get_next(self, options: List[str], index_attr: str) -> str:
        """Get next item from options based on strategy."""
        if not options:
            return ""
        
        if len(options) == 1:
            return options[0]
        
        if self.strategy == RandomStrategy.CHOICE:
            return random.choice(options)
        
        current_index = getattr(self, index_attr)

        if self.strategy == RandomStrategy.REPEAT:
            next_index = current_index
        else:
            # Advance index
            next_index = (current_index + 1) % len(options)
            setattr(self, index_attr, next_index)
        
            # If we completed a cycle with shuffle, reshuffle and avoid repeats
            if next_index == 0 and self.strategy == RandomStrategy.SHUFFLE and len(options) > 2:
                last_item = options[0]
                random.shuffle(options)
                # Avoid immediate repeat
                if options[0] == last_item:
                    # Swap first with a random other position
                    swap_idx = random.randint(1, len(options) - 1)
                    options[0], options[swap_idx] = options[swap_idx], options[0]

        return options[next_index]
    
    def get_prompt(self) -> str:
        """Get next prompt text."""
        return self._get_next(self.prompt_options, '_prompt_index')
    
    def get_style(self) -> str:
        """Get next style text."""
        return self._get_next(self.style_options, '_style_index')
    
    def get_negative(self) -> str:
        """Get next negative text."""
        return self._get_next(self.negative_options, '_negative_index')


class SectorCombiner:
    """Handles combining development sectors into prompts."""
    
    def __init__(self, sector_data: Dict[str, List[str]], strategy: RandomStrategy = RandomStrategy.SHUFFLE):
        """Initialize with sector data.
        
        Args:
            sector_data: Dict mapping sector names to lists of options
            strategy: How to select combinations
        """
        self.sectors = list(sector_data.keys())
        self.sector_options = {k: [opt for opt in v if opt.strip()] for k, v in sector_data.items()}
        self.strategy = strategy
        
        # Generate all combinations
        self.combinations = self._generate_combinations()
        self._index = 0
        
        if strategy == RandomStrategy.SHUFFLE:
            random.shuffle(self.combinations)
    
    def _generate_combinations(self) -> List[str]:
        """Generate all possible sector combinations."""
        if not self.sector_options or all(not opts for opts in self.sector_options.values()):
            return [""]
        
        # Get non-empty sector options
        valid_sectors = {k: v for k, v in self.sector_options.items() if v}
        
        if not valid_sectors:
            return [""]
        
        # Generate cartesian product of all sector options
        sector_combinations = list(product(*valid_sectors.values()))
        
        # Join each combination into a prompt string
        return [", ".join(combo) for combo in sector_combinations]
    
    def get_next(self) -> str:
        """Get next sector combination."""
        if not self.combinations:
            return ""
        
        if self.strategy == RandomStrategy.CHOICE:
            return random.choice(self.combinations)
        
        # Deterministic or shuffle cycling
        result = self.combinations[self._index]
        self._index = (self._index + 1) % len(self.combinations)
        
        # Reshuffle if we completed a cycle
        if self._index == 0 and self.strategy == RandomStrategy.SHUFFLE and len(self.combinations) > 1:
            last_item = self.combinations[0]
            random.shuffle(self.combinations)
            # Avoid immediate repeat
            if self.combinations[0] == last_item and len(self.combinations) > 2:
                swap_idx = random.randint(1, len(self.combinations) - 1)
                self.combinations[0], self.combinations[swap_idx] = self.combinations[swap_idx], self.combinations[0]
        
        return result


class PromptGenerator:
    """Main prompt generator for Experimance."""
    
    def __init__(self, 
                 data_path: Path,
                 locations_file: str = "locations.json",
                 developments_file: str = "anthropocene.json",
                 strategy: RandomStrategy = RandomStrategy.SHUFFLE):
        """Initialize the prompt generator.
        
        Args:
            data_path: Path to directory containing JSON data files
            locations_file: Name of locations JSON file
            developments_file: Name of developments JSON file
            strategy: Default strategy for randomization
        """
        self.data_path = Path(data_path)
        self.strategy = strategy
        
        # Load data
        self.locations_data = self._load_json(locations_file)
        self.developments_data = self._load_json(developments_file)
        
        # Create base components
        self.base_component = PromptComponent(
            **self.developments_data.get("base_prompt", {}),
            strategy=strategy
        )
        
        # Create era components
        self.era_components = {}
        for era_name, era_data in self.developments_data.get("eras", {}).items():
            # Remove lora field if present
            era_prompt_data = {k: v for k, v in era_data.items() if k != "lora"}
            self.era_components[era_name] = PromptComponent(**era_prompt_data, strategy=strategy)
        
        # Create biome components
        self.biome_components = {}
        for biome_name, biome_data in self.locations_data.get("types", {}).items():
            # Convert "locations" to "style" if present
            if "locations" in biome_data:
                biome_data = dict(biome_data)
                biome_data["style"] = biome_data.pop("locations")
            self.biome_components[biome_name] = PromptComponent(**biome_data, strategy=strategy)
        
        # Prepare sector data
        self._prepare_sector_data()
        
        # Current state
        self.current_biome: Optional[Biome] = None
        self.current_era: Optional[Era] = None
        self._sector_combiners: Dict[Tuple[str, str], SectorCombiner] = {}
    
    def _load_json(self, filename: str) -> dict:
        """Load JSON data file."""
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def _prepare_sector_data(self):
        """Prepare sector combination data for all biome/era pairs."""
        sectors = self.developments_data.get("sectors", [])
        sector_defaults = self.developments_data.get("sector_defaults", {})
        location_sectors = self.developments_data.get("locations", {})
        
        # Fill in defaults for missing combinations
        for biome_name in self.biome_components.keys():
            if biome_name not in location_sectors:
                location_sectors[biome_name] = {}
            
            for era_name in self.era_components.keys():
                if era_name not in location_sectors[biome_name]:
                    location_sectors[biome_name][era_name] = {}
                
                # Fill in sector defaults
                for sector in sectors:
                    if sector not in location_sectors[biome_name][era_name]:
                        default_options = sector_defaults.get(era_name, {}).get(sector, [])
                        location_sectors[biome_name][era_name][sector] = default_options
                    else:
                        # Combine specific with defaults
                        specific = location_sectors[biome_name][era_name][sector]
                        defaults = sector_defaults.get(era_name, {}).get(sector, [])
                        location_sectors[biome_name][era_name][sector] = specific + defaults
        
        self.sector_data = location_sectors
    
    def _get_sector_combiner(self, biome: Biome, era: Era) -> SectorCombiner:
        """Get or create sector combiner for biome/era pair."""
        key = (biome.value, era.value)
        
        if key not in self._sector_combiners:
            # Map schema enum values to JSON data
            biome_key = biome.value
            era_key = era.value
            
            # Handle potential mismatches between schema and JSON
            if biome_key not in self.sector_data:
                # Try alternative mappings or use empty
                sector_options = {}
            elif era_key not in self.sector_data[biome_key]:
                sector_options = {}
            else:
                sector_options = self.sector_data[biome_key][era_key]
            
            self._sector_combiners[key] = SectorCombiner(sector_options, self.strategy)
        
        return self._sector_combiners[key]
    
    def generate_prompt(self, era: Era, biome: Optional[Biome] = None) -> Tuple[str, str]:
        """Generate a prompt for the given era and biome.

        IF biome is not given then uses the previously set biome elements.
        
        Args:
            era: Era enum value
            biome: Optional Biome enum value
            
        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """

        self.current_era = era
        prev_strategy = None # used to indictate it we are repeating the same biome components
        if biome is not None:
            self.current_biome = biome
        else:
            prev_strategy = self.base_component.strategy
            if self.current_biome is None:
                biome = Biome.ARCTIC # set a random default
            else:
                biome = self.current_biome
            
        # Get components - handle potential mismatches
        base_prompt = self.base_component.get_prompt()
        base_style = self.base_component.get_style()
        base_negative = self.base_component.get_negative()
        
        # Era component
        era_key = era.value
        if era_key in self.era_components:
            era_component = self.era_components[era_key]
            era_prompt = era_component.get_prompt()
            era_style = era_component.get_style()
            era_negative = era_component.get_negative()
        else:
            era_prompt = era_style = era_negative = ""
        
        # Biome component
        biome_key = biome.value
        if biome_key in self.biome_components:
            biome_component = self.biome_components[biome_key]
            if prev_strategy is not None:
                biome_component.strategy = RandomStrategy.REPEAT  # Keep same biome components
            biome_prompt = biome_component.get_prompt()
            biome_style = biome_component.get_style()
            biome_negative = biome_component.get_negative()
            if prev_strategy is not None:
                biome_component.strategy = prev_strategy  # Restore original strategy
        
        # Sector combination
        sector_combiner = self._get_sector_combiner(biome, era)
        sector_prompt = sector_combiner.get_next()
        
        # Combine all parts
        positive_parts = [p for p in [base_prompt, biome_prompt, era_prompt, sector_prompt] if p]
        style_parts = [s for s in [base_style, biome_style, era_style] if s]
        negative_parts = [n for n in [base_negative, biome_negative, era_negative] if n]
        
        # Join and clean
        positive_prompt = self._clean_prompt(", ".join(positive_parts + style_parts))
        negative_prompt = self._clean_prompt(", ".join(negative_parts))
        
        return positive_prompt, negative_prompt
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean up prompt text."""
        if not prompt:
            return ""
        
        # Remove duplicates while preserving order
        parts = [p.strip() for p in prompt.split(",") if p.strip()]
        seen = set()
        cleaned_parts = []
        for part in parts:
            if part not in seen:
                seen.add(part)
                cleaned_parts.append(part)
        
        result = ", ".join(cleaned_parts)
        return result.strip().rstrip(",")
    
    def get_available_eras(self) -> List[Era]:
        """Get list of available eras from data."""
        available = []
        for era_name in self.era_components.keys():
            try:
                era = Era(era_name)
                available.append(era)
            except ValueError:
                # Era in JSON but not in schema
                continue
        return available
    
    def get_available_biomes(self) -> List[Biome]:
        """Get list of available biomes from data."""
        available = []
        for biome_name in self.biome_components.keys():
            try:
                biome = Biome(biome_name)
                available.append(biome)
            except ValueError:
                # Biome in JSON but not in schema
                continue
        return available


class PromptManager:
    """Higher-level interface for managing prompts in the core service.
    
    Provides state management, caching, and explicit methods for biome/era changes.
    """
    
    def __init__(self, generator: PromptGenerator, initial_era: Era, initial_biome: Biome):
        """Initialize with a prompt generator and initial state.
        
        Args:
            generator: The underlying PromptGenerator instance
            initial_era: Starting era
            initial_biome: Starting biome
        """
        self.generator = generator
        self.current_era = initial_era
        self.current_biome = initial_biome
        self._cached_prompt: Optional[Tuple[str, str]] = None
        
        # Generate initial prompt
        self._cached_prompt = self.generator.generate_prompt(self.current_era, self.current_biome)
    
    def current_prompt(self) -> Tuple[str, str]:
        """Get the current prompt without advancing state.
        
        Returns:
            Tuple of (positive_prompt, negative_prompt)
        """
        if self._cached_prompt is None:
            self._cached_prompt = self.generator.generate_prompt(self.current_era, self.current_biome)
        return self._cached_prompt
    
    def advance_era(self, new_era: Era) -> Tuple[str, str]:
        """Change to a new era while keeping the same biome.
        
        Args:
            new_era: The era to switch to
            
        Returns:
            Tuple of (positive_prompt, negative_prompt) for the new combination
        """
        if new_era != self.current_era:
            self.current_era = new_era
            self._cached_prompt = self.generator.generate_prompt(self.current_era)
        
        return self.current_prompt()
    
    def switch_biome(self, new_biome: Biome) -> Tuple[str, str]:
        """Change to a new biome while keeping the same era.
        
        Args:
            new_biome: The biome to switch to
            
        Returns:
            Tuple of (positive_prompt, negative_prompt) for the new combination
        """
        if new_biome != self.current_biome:
            self.current_biome = new_biome
            self._cached_prompt = self.generator.generate_prompt(self.current_era, self.current_biome)
        
        return self.current_prompt()
    
    def set_era_and_biome(self, era: Era, biome: Biome) -> Tuple[str, str]:
        """Set both era and biome at once.
        
        Args:
            era: The era to set
            biome: The biome to set
            
        Returns:
            Tuple of (positive_prompt, negative_prompt) for the new combination
        """
        if era != self.current_era or biome != self.current_biome:
            self.current_era = era
            self.current_biome = biome
            self._cached_prompt = self.generator.generate_prompt(self.current_era, self.current_biome)
        
        return self.current_prompt()
    
    def next_variation(self) -> Tuple[str, str]:
        """Generate a new variation for the current era/biome combination.
        
        This advances the internal randomization state to get a different prompt
        for the same era/biome combination.
        
        Returns:
            Tuple of (positive_prompt, negative_prompt) for a new variation
        """
        self._cached_prompt = self.generator.generate_prompt(self.current_era, self.current_biome)
        return self._cached_prompt
    
    def get_available_eras(self) -> List[Era]:
        """Get list of available eras."""
        return self.generator.get_available_eras()
    
    def get_available_biomes(self) -> List[Biome]:
        """Get list of available biomes."""
        return self.generator.get_available_biomes()
    
    def get_state(self) -> Dict[str, str]:
        """Get current state for debugging/logging.
        
        Returns:
            Dict with current era and biome values
        """
        return {
            "era": self.current_era.value,
            "biome": self.current_biome.value
        }


def _demo_cli():
    """Simple CLI for testing the prompt generator."""
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Test prompt generation")
    parser.add_argument("--data-path", default="data/", help="Path to data files")
    parser.add_argument("--locations", default="locations.json", help="Locations file")
    parser.add_argument("--developments", default="anthropocene.json", help="Developments file")
    parser.add_argument("--era", help="Specific era to test")
    parser.add_argument("--biome", help="Specific biome to test")
    parser.add_argument("--count", type=int, default=5, help="Number of prompts to generate")
    parser.add_argument("--output-file", type=str, default=None, help="Output file to write prompts (one per line)")
    parser.add_argument("--strategy", choices=["deterministic", "shuffle", "choice"], 
                       default="shuffle", help="Selection strategy")
    parser.add_argument("--test-manager", action="store_true", 
                       help="Test PromptManager state management")
    args = parser.parse_args()


    try:
        generator = PromptGenerator(
            args.data_path, 
            args.locations, 
            args.developments,
            RandomStrategy(args.strategy)
        )

        available_eras = generator.get_available_eras()
        available_biomes = generator.get_available_biomes()

        print(f"Available eras: {[e.value for e in available_eras]}")
        print(f"Available biomes: {[b.value for b in available_biomes]}")
        print()

        output_lines = []

        def add_prompt_to_output(pos: str, neg: str):
            """Helper to add a prompt to output lines."""
            output_lines.append(f"positive: {pos}")
            if neg and neg != "":
                output_lines.append(f"negative: {neg}")
            output_lines.append("---")

        if args.test_manager:
            # Test PromptManager functionality
            if not available_eras or not available_biomes:
                print("No eras or biomes available for testing")
                return 1

            initial_era = available_eras[0]
            initial_biome = available_biomes[0]

            print(f"Testing PromptManager with initial state: {initial_era.value} + {initial_biome.value}")
            manager = PromptManager(generator, initial_era, initial_biome)

            # Test current_prompt() (should be same each call)
            print("\n1. Testing current_prompt() - should be identical:")
            pos1, neg1 = manager.current_prompt()
            pos2, neg2 = manager.current_prompt()
            print(f"   Call 1: {pos1}")
            print(f"   Call 2: {pos2}")
            print(f"   Identical: {pos1 == pos2}")

            # Test next_variation() - should be different
            print("\n2. Testing next_variation() - should be different:")
            print(f"   Original: {pos1}")
            pos3, neg3 = manager.next_variation()
            print(f"   Variation: {pos3}")
            print(f"   Different: {pos1 != pos3}")

            # Test era change
            if len(available_eras) > 1:
                print(f"\n3. Testing advance_era() to {available_eras[1].value}:")
                for era in available_eras:
                    print(f"   Era: {era.value}")
                    pos4, neg4 = manager.advance_era(era)
                    print(f"   - {pos4}")

            # Test biome change
            if len(available_biomes) > 1:
                print(f"\n4. Testing switch_biome() to {available_biomes[1].value}:")
                pos5, neg5 = manager.switch_biome(available_biomes[1])
                print(f"   New biome prompt: {pos5[:50]}...")
                print(f"   State: {manager.get_state()}")

        elif args.era:
            if args.biome:
                # Test specific combination
                try:
                    era = Era(args.era)
                    biome = Biome(args.biome)
                    print(f"Testing {era.value} + {biome.value}:")

                    for i in range(args.count):
                        pos, neg = generator.generate_prompt(era, biome)
                        print(f"{i+1}. Positive: {pos}")
                        print(f"   Negative: {neg}")
                        print()
                        add_prompt_to_output(pos, neg)
                except ValueError as e:
                    print(f"Error: {e}")
            else: # randomize biome
                era = Era(args.era)
                print(f"Testing {era.value} with random biomes:")
                for i in range(args.count):
                    biome = random.choice(available_biomes)
                    pos, neg = generator.generate_prompt(era, biome)
                    print(f"{i+1}. {era.value} + {biome.value}")
                    print(f"   Positive: {pos}")
                    print(f"   Negative: {neg}")
                    print()
                    add_prompt_to_output(pos, neg)
        elif args.biome:
            # Test specific biome with all eras
            try:
                biome = Biome(args.biome)
                print(f"Testing {biome.value} with all eras:")
                for era in available_eras:
                    pos, neg = generator.generate_prompt(era, biome)
                    print(f"{era.value}: Positive: {pos}")
                    print(f"   Negative: {neg}")
                    print()
                    add_prompt_to_output(pos, neg)
            except ValueError as e:
                print(f"Error: {e}")
        else:
            # Test random combinations
            print(f"Generating {args.count} random prompts:")
            import random
            for i in range(args.count):
                era = random.choice(available_eras)
                biome = random.choice(available_biomes)
                pos, neg = generator.generate_prompt(era, biome)
                print(f"{i+1}. {era.value} + {biome.value}")
                print(f"   Positive: {pos}")
                print(f"   Negative: {neg}")
                print()
                add_prompt_to_output(pos, neg)

        # Write prompts to file if requested
        if args.output_file:
            # remove the last "---" separator if it exists
            if output_lines and output_lines[-1] == "---":
                output_lines.pop()
            try:
                with open(args.output_file, "w", encoding="utf-8") as f:
                    for line in output_lines:
                        f.write(line + "\n")
                print(f"Wrote {len(output_lines)} prompts to {args.output_file}")
            except Exception as e:
                print(f"Error writing to output file: {e}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(_demo_cli())
