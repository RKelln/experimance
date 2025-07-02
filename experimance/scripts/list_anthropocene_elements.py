import json
from pathlib import Path

from experimance_common.constants import DATA_DIR

# Path to the anthropocene.json file
ANTHROPOCENE_PATH = DATA_DIR / "anthropocene.json"

def collect_unique_elements(data):
    unique = set()
    def recurse(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                recurse(v)
        elif isinstance(obj, list):
            for item in obj:
                recurse(item)
        elif isinstance(obj, str):
            # split on commas and strip whitespace
            elements = [e.strip() for e in obj.split(",") if e.strip()]
            for elem in elements:
                if elem:
                    unique.add(elem)
    recurse(data)
    return unique


def main():
    with open(ANTHROPOCENE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    unique_elements = sorted(collect_unique_elements(data))
    print("\n".join(unique_elements))
    # save to a file if needed
    save_file = ANTHROPOCENE_PATH.with_name("unique_anthropocene_elements.txt")
    with open(save_file, "w", encoding="utf-8") as f:
        f.write("\n".join(unique_elements))

    # Cross-check with sector_sound_lookup.py if it exists
    lookup_path = Path(__file__).parent.parent / "libs" / "common" / "sector_sound_lookup.py"
    if lookup_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("sector_sound_lookup", lookup_path)
        if spec is not None and spec.loader is not None:
            lookup_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lookup_mod)
            lookup = getattr(lookup_mod, "SECTOR_SOUND_LOOKUP", {})
            missing = [e for e in unique_elements if e not in lookup]
            if missing:
                print("\n--- Elements NOT in SECTOR_SOUND_LOOKUP ---")
                for m in missing:
                    print(m)
            else:
                print("\nAll elements are present in SECTOR_SOUND_LOOKUP.")
        else:
            print("\nCould not import sector_sound_lookup.py (spec or loader missing).")


if __name__ == "__main__":
    main()
