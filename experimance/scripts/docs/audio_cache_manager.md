# Audio Cache Manager

Manages the audio generation semantic cache: inspection, cleanup, and maintenance.

See `scripts/audio_cache_manager.py`.

## Quick Start

```bash
# Show cache statistics
uv run python scripts/audio_cache_manager.py stats

# List recent cache items
uv run python scripts/audio_cache_manager.py list --limit 10

# Find duplicate items
uv run python scripts/audio_cache_manager.py duplicates

# Clean old cache items (dry run first)
uv run python scripts/audio_cache_manager.py clean --days 30 --dry-run
uv run python scripts/audio_cache_manager.py clean --days 30

# Clear entire cache (with confirmation)
uv run python scripts/audio_cache_manager.py clear --confirm
```

## Commands

### `stats`
Display comprehensive cache statistics:
- Total items and unique prompts
- Cache size and average duration
- CLAP similarity scores
- Age range and missing files

### `list`
Show cache items with metadata:
- `--limit N` — number of items to display (default: 20)
- `--sort` — sort by `timestamp`, `clap_similarity`, `duration`, or `prompt`
- `--oldest-first` — show oldest items first (default: newest first)

```bash
uv run python scripts/audio_cache_manager.py list --limit 5 --sort clap_similarity
```

### `clean`
Remove cache items older than specified days:
- `--days N` — remove items older than N days
- `--dry-run` — show what would be removed without deleting

```bash
# Preview
uv run python scripts/audio_cache_manager.py clean --days 7 --dry-run
# Execute
uv run python scripts/audio_cache_manager.py clean --days 7
```

### `clear`
Remove all cache items and files:
- `--confirm` — required to actually perform the operation
- Without `--confirm`, shows what would be removed

### `duplicates`
Find and optionally remove duplicate cache items:
- Groups items by normalized prompt
- `--remove-duplicates --confirm` — remove all but newest in each group

### `remove-pattern`
Remove items matching a regex pattern:
- `pattern` — regular expression matched against prompts
- `--confirm` — required to actually remove items
- `--dry-run` — preview without deleting

```bash
uv run python scripts/audio_cache_manager.py remove-pattern "test.*" --dry-run
uv run python scripts/audio_cache_manager.py remove-pattern "test.*" --confirm
```

## Cache Directory

Default: `audio_cache/`. Override with `--cache-dir`:

```bash
uv run python scripts/audio_cache_manager.py --cache-dir /path/to/cache stats
```

Common locations:
- Development: `./media/images/generated/audio/audio_cache`
- Production: as configured in service settings

## Safety Features

- **Dry-run mode** — most destructive operations support `--dry-run`
- **Confirmation required** — destructive operations require `--confirm`
- **Detailed reporting** — shows exactly what will be removed and space freed

## Common Workflows

### Daily Maintenance
```bash
uv run python scripts/audio_cache_manager.py stats
uv run python scripts/audio_cache_manager.py clean --days 30
uv run python scripts/audio_cache_manager.py duplicates --remove-duplicates --confirm
```

### Development Cleanup
```bash
# Remove test/debug items
uv run python scripts/audio_cache_manager.py remove-pattern "(test|debug|temp)" --confirm
# Clear a temporary cache dir
uv run python scripts/audio_cache_manager.py --cache-dir ./tmp_cache clear --confirm
```

### Cache Analysis
```bash
# Largest items
uv run python scripts/audio_cache_manager.py list --sort duration --limit 20
# Highest quality items
uv run python scripts/audio_cache_manager.py list --sort clap_similarity --limit 10
# Oldest items
uv run python scripts/audio_cache_manager.py list --oldest-first --limit 10
```
