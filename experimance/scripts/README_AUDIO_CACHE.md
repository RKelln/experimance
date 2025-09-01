# Audio Cache Manager

The `audio_cache_manager.py` script provides comprehensive management of the audio generation cache, including inspection, cleanup, and maintenance operations.

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
Display comprehensive cache statistics including:
- Total items and unique prompts
- Cache size and average duration
- CLAP similarity scores
- Age range and missing files

### `list`
Show cache items with metadata:
- `--limit N`: Number of items to display (default: 20)
- `--sort`: Sort by `timestamp`, `clap_similarity`, `duration`, or `prompt`
- `--oldest-first`: Show oldest items first (default: newest first)

Example:
```bash
uv run python scripts/audio_cache_manager.py list --limit 5 --sort clap_similarity
```

### `clear`
Remove all cache items and files:
- `--confirm`: Required to actually perform the operation
- Without `--confirm`, shows what would be removed

### `clean`
Remove cache items older than specified days:
- `--days N`: Remove items older than N days
- `--dry-run`: Show what would be removed without deleting

Example:
```bash
# See what would be removed
uv run python scripts/audio_cache_manager.py clean --days 7 --dry-run

# Actually remove items older than 7 days
uv run python scripts/audio_cache_manager.py clean --days 7
```

### `duplicates`
Find and optionally remove duplicate cache items:
- Groups items by normalized prompt
- `--remove-duplicates --confirm`: Remove all but newest in each group

### `remove-pattern`
Remove items matching a regex pattern:
- `pattern`: Regular expression to match against prompts
- `--confirm`: Required to actually remove items
- `--dry-run`: Show what would be removed

Example:
```bash
# Remove test items
uv run python scripts/audio_cache_manager.py remove-pattern "test.*" --dry-run
uv run python scripts/audio_cache_manager.py remove-pattern "test.*" --confirm
```

## Cache Directory

By default, the script looks for cache in `audio_cache/`. You can specify a different location:

```bash
uv run python scripts/audio_cache_manager.py --cache-dir /path/to/cache stats
```

Common cache locations:
- Development: `./media/images/generated/audio/audio_cache`
- Production: As configured in your service settings

## Safety Features

- **Dry-run mode**: Most destructive operations support `--dry-run` to preview changes
- **Confirmation required**: Destructive operations require `--confirm` flag
- **Interactive prompts**: Additional confirmation prompts for safety
- **Detailed reporting**: Shows exactly what will be removed and space freed

## Output Format

The script provides colorful, formatted output with:
- 📁 Directory paths
- 🎵 Audio-related information  
- ✅ Success indicators
- ⚠️ Warnings and missing files
- ❌ Errors
- 🔍 Analysis results

## Examples

### Daily Maintenance
```bash
# Check cache health
uv run python scripts/audio_cache_manager.py stats

# Clean items older than 30 days
uv run python scripts/audio_cache_manager.py clean --days 30

# Remove duplicates
uv run python scripts/audio_cache_manager.py duplicates --remove-duplicates --confirm
```

### Development Cleanup
```bash
# Remove test items
uv run python scripts/audio_cache_manager.py remove-pattern "(test|debug|temp)" --confirm

# Clear development cache
uv run python scripts/audio_cache_manager.py --cache-dir ./tmp_cache clear --confirm
```

### Cache Analysis
```bash
# Find largest cache items
uv run python scripts/audio_cache_manager.py list --sort duration --limit 20

# Find highest quality items
uv run python scripts/audio_cache_manager.py list --sort clap_similarity --limit 10

# Find oldest items
uv run python scripts/audio_cache_manager.py list --oldest-first --limit 10
```
