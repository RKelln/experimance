#!/usr/bin/env python3
"""
Audio Cache Management Tool

This script provides comprehensive management of the audio generation cache,
including inspection, cleanup, and maintenance operations.

Usage:
    python scripts/audio_cache_manager.py stats
    python scripts/audio_cache_manager.py list --limit 10 --sort clap_similarity
    python scripts/audio_cache_manager.py clear --confirm
    python scripts/audio_cache_manager.py clean --days 30
    python scripts/audio_cache_manager.py duplicates
    python scripts/audio_cache_manager.py remove-pattern "test.*" --confirm
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.image_server.src.image_server.generators.audio.prompt2audio import AudioSemanticCache


def format_timestamp(timestamp: float) -> str:
    """Format timestamp as human-readable string."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def format_size(size_bytes: float) -> str:
    """Format size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes:.1f} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def print_stats(cache: AudioSemanticCache):
    """Print cache statistics."""
    stats = cache.get_cache_stats()
    
    print("🎵 Audio Cache Statistics")
    print("=" * 50)
    print(f"Total items:           {stats['total_items']}")
    print(f"Unique prompts:        {stats['unique_prompts']}")
    print(f"Total size:            {format_size(stats['total_size_mb'] * 1024 * 1024)}")
    print(f"Average duration:      {stats['average_duration']:.1f}s")
    print(f"Average CLAP score:    {stats['average_clap_similarity']:.3f}")
    
    if stats['missing_files'] > 0:
        print(f"⚠️  Missing files:       {stats['missing_files']}")
    
    if stats['oldest_item']:
        oldest_date = format_timestamp(stats['oldest_item'])
        newest_date = format_timestamp(stats['newest_item'])
        print(f"Oldest item:           {oldest_date}")
        print(f"Newest item:           {newest_date}")
        
        # Calculate age range
        age_days = (stats['newest_item'] - stats['oldest_item']) / 86400
        print(f"Cache age range:       {age_days:.1f} days")


def print_list(cache: AudioSemanticCache, limit: int = 20, sort_by: str = "timestamp", reverse: bool = True):
    """Print list of cache items."""
    items = cache.list_cache_items(limit=limit, sort_by=sort_by, reverse=reverse)
    
    if not items:
        print("No cache items found.")
        return
    
    print(f"🎵 Cache Items (sorted by {sort_by}, {'newest' if reverse else 'oldest'} first)")
    print("=" * 120)
    print(f"{'#':<3} {'Date':<19} {'CLAP':<6} {'Dur':<5} {'Size':<8} {'Prompt'}")
    print("-" * 120)
    
    for i, item in enumerate(items, 1):
        date_str = format_timestamp(item['timestamp'])
        clap_str = f"{item['clap_similarity']:.3f}" if item['clap_similarity'] else "N/A"
        dur_str = f"{item['duration_s']:.1f}s"
        size_str = format_size(item['file_size_kb'] * 1024)
        
        # Show full prompt without truncation
        prompt_str = item['prompt']
        
        status = "✓" if item['file_exists'] else "✗"
        print(f"{i:<3} {date_str} {clap_str:<6} {dur_str:<5} {size_str:<8} {prompt_str}")
        
        if not item['file_exists']:
            print(f"    ⚠️  File missing: {item['path']}")
        
        # Show file path for identification
        print(f"    📁 {item['path']}")
        print()  # Add blank line between entries for readability


def print_grouped_by_prompt(cache: AudioSemanticCache, limit: int = 20):
    """Print cache items grouped by prompt to show multiple generations."""
    if not cache.items:
        print("No cache items found.")
        return
    
    # Group items by normalized prompt
    prompt_groups = {}
    for i, item in enumerate(cache.items):
        norm_prompt = item.prompt_norm
        if norm_prompt not in prompt_groups:
            prompt_groups[norm_prompt] = []
        
        # Add file existence check
        file_exists = Path(item.path).exists()
        file_size = 0
        if file_exists:
            try:
                file_size = Path(item.path).stat().st_size
            except OSError:
                file_exists = False
        
        prompt_groups[norm_prompt].append({
            "index": i,
            "item": item,
            "file_exists": file_exists,
            "file_size_kb": file_size / 1024
        })
    
    # Sort groups by latest timestamp
    sorted_groups = sorted(prompt_groups.items(), 
                          key=lambda x: max(item["item"].timestamp for item in x[1]), 
                          reverse=True)
    
    total_groups = len(sorted_groups)
    total_items = sum(len(group) for _, group in sorted_groups)
    
    print(f"🎵 Cache Items Grouped by Prompt ({total_groups} unique prompts, {total_items} total items)")
    print("=" * 120)
    
    for group_num, (norm_prompt, items) in enumerate(sorted_groups[:limit], 1):
        # Sort items within group by timestamp (newest first)
        items.sort(key=lambda x: x["item"].timestamp, reverse=True)
        
        print(f"\n📝 Group {group_num}: {len(items)} generation{'s' if len(items) > 1 else ''}")
        print(f"Prompt: {items[0]['item'].prompt}")
        print("-" * 120)
        
        for j, item_info in enumerate(items, 1):
            item = item_info["item"]
            date_str = format_timestamp(item.timestamp)
            clap_str = f"{item.clap_similarity:.3f}" if item.clap_similarity else "N/A"
            dur_str = f"{item.duration_s:.1f}s"
            size_str = format_size(item_info['file_size_kb'] * 1024)
            status = "✓" if item_info['file_exists'] else "✗"
            
            print(f"  {j}. {date_str} CLAP:{clap_str} Dur:{dur_str} Size:{size_str} {status}")
            print(f"     📁 {item.path}")
            
            if not item_info['file_exists']:
                print(f"     ⚠️  File missing")
        
        if group_num >= limit:
            remaining = total_groups - limit
            if remaining > 0:
                print(f"\n... and {remaining} more prompt groups")
            break


def print_duplicates(cache: AudioSemanticCache):
    """Print potential duplicate groups."""
    duplicates = cache.find_duplicates()
    
    if not duplicates:
        print("No duplicate groups found.")
        return
    
    print(f"🔍 Found {len(duplicates)} groups with multiple items")
    print("=" * 80)
    
    total_duplicates = sum(len(group) - 1 for group in duplicates)  # -1 because we keep one from each group
    total_size = 0
    
    for i, group in enumerate(duplicates, 1):
        print(f"\nGroup {i}: {len(group)} items for prompt:")
        print(f"'{group[0]['prompt']}'")
        
        for j, item in enumerate(group):
            date_str = format_timestamp(item['timestamp'])
            clap_str = f"{item['clap_similarity']:.3f}" if item['clap_similarity'] else "N/A"
            status = "✓" if item['file_exists'] else "✗"
            
            marker = "📌 KEEP" if j == 0 else "🗑️  DELETE"
            print(f"  {marker} {date_str} CLAP:{clap_str} {status} {Path(item['path']).name}")
            
            if j > 0 and item['file_exists']:  # Count duplicates for deletion
                try:
                    size = Path(item['path']).stat().st_size
                    total_size += size
                except OSError:
                    pass
    
    print(f"\n📊 Summary:")
    print(f"   Potential duplicates to remove: {total_duplicates}")
    print(f"   Potential space savings: {format_size(total_size)}")
    """Print potential duplicate groups."""
    duplicates = cache.find_duplicates()
    
    if not duplicates:
        print("No duplicate groups found.")
        return
    
    print(f"🔍 Found {len(duplicates)} groups with multiple items")
    print("=" * 80)
    
    total_duplicates = sum(len(group) - 1 for group in duplicates)  # -1 because we keep one from each group
    total_size = 0
    
    for i, group in enumerate(duplicates, 1):
        print(f"\nGroup {i}: {len(group)} items for prompt: '{group[0]['prompt'][:60]}{'...' if len(group[0]['prompt']) > 60 else ''}'")
        
        for j, item in enumerate(group):
            date_str = format_timestamp(item['timestamp'])
            clap_str = f"{item['clap_similarity']:.3f}" if item['clap_similarity'] else "N/A"
            status = "✓" if item['file_exists'] else "✗"
            
            marker = "📌 KEEP" if j == 0 else "🗑️  DELETE"
            print(f"  {marker} {date_str} CLAP:{clap_str} {status} {Path(item['path']).name}")
            
            if j > 0 and item['file_exists']:  # Count duplicates for deletion
                try:
                    size = Path(item['path']).stat().st_size
                    total_size += size
                except OSError:
                    pass
    
    print(f"\n📊 Summary:")
    print(f"   Potential duplicates to remove: {total_duplicates}")
    print(f"   Potential space savings: {format_size(total_size)}")


def confirm_action(message: str) -> bool:
    """Ask for user confirmation."""
    response = input(f"{message} (yes/no): ").lower().strip()
    return response in ['yes', 'y']


def main():
    parser = argparse.ArgumentParser(description="Manage audio generation cache")
    parser.add_argument("--cache-dir", default="audio_cache", help="Cache directory path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show cache statistics")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List cache items")
    list_parser.add_argument("--limit", type=int, default=20, help="Number of items to show")
    list_parser.add_argument("--sort", choices=["timestamp", "clap_similarity", "duration", "prompt"], 
                           default="timestamp", help="Sort by field")
    list_parser.add_argument("--oldest-first", action="store_true", help="Show oldest first")
    
    # Grouped command
    grouped_parser = subparsers.add_parser("grouped", help="List items grouped by prompt")
    grouped_parser.add_argument("--limit", type=int, default=20, help="Number of prompt groups to show")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear entire cache")
    clear_parser.add_argument("--confirm", action="store_true", help="Confirm the operation")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Remove old cache items")
    clean_parser.add_argument("--days", type=float, required=True, help="Remove items older than N days")
    clean_parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    
    # Duplicates command
    dup_parser = subparsers.add_parser("duplicates", help="Find and optionally remove duplicates")
    dup_parser.add_argument("--remove-duplicates", action="store_true", help="Remove all but the newest in each group")
    dup_parser.add_argument("--confirm", action="store_true", help="Confirm the removal operation")
    
    # Remove pattern command
    pattern_parser = subparsers.add_parser("remove-pattern", help="Remove items matching a pattern")
    pattern_parser.add_argument("pattern", help="Regex pattern to match against prompts")
    pattern_parser.add_argument("--confirm", action="store_true", help="Confirm the operation")
    pattern_parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize cache
    try:
        cache = AudioSemanticCache(args.cache_dir)
        print(f"📁 Using cache directory: {Path(args.cache_dir).resolve()}")
        print()
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return
    
    # Execute command
    try:
        if args.command == "stats":
            print_stats(cache)
            
        elif args.command == "list":
            print_list(cache, limit=args.limit, sort_by=args.sort, reverse=not args.oldest_first)
            
        elif args.command == "grouped":
            print_grouped_by_prompt(cache, limit=args.limit)
            
        elif args.command == "clear":
            if not args.confirm:
                stats = cache.get_cache_stats()
                print(f"⚠️  This will remove {stats['total_items']} items ({format_size(stats['total_size_mb'] * 1024 * 1024)})")
                if not confirm_action("Are you sure you want to clear the entire cache?"):
                    print("Operation cancelled.")
                    return
            
            result = cache.clear_cache(confirm=True)
            if "error" in result:
                print(f"❌ {result['error']}")
            else:
                print(f"✅ Cleared {result['removed_items']} items, freed {format_size(result['freed_space_mb'] * 1024 * 1024)}")
                
        elif args.command == "clean":
            if args.dry_run:
                # Show what would be removed
                cutoff_time = time.time() - (args.days * 86400)
                old_items = [item for item in cache.items if item.timestamp < cutoff_time]
                
                if not old_items:
                    print(f"No items older than {args.days} days found.")
                else:
                    total_size = 0
                    for item in old_items:
                        if Path(item.path).exists():
                            try:
                                total_size += Path(item.path).stat().st_size
                            except OSError:
                                pass
                    
                    print(f"🔍 Would remove {len(old_items)} items older than {args.days} days")
                    print(f"   Space that would be freed: {format_size(total_size)}")
                    print("\nOldest items that would be removed:")
                    for item in sorted(old_items, key=lambda x: x.timestamp)[:10]:
                        date_str = format_timestamp(item.timestamp)
                        print(f"   {date_str}: {item.prompt[:60]}{'...' if len(item.prompt) > 60 else ''}")
            else:
                result = cache.remove_old_items(args.days)
                if result['removed_items'] == 0:
                    print(f"No items older than {args.days} days found.")
                else:
                    print(f"✅ Removed {result['removed_items']} items, freed {format_size(result['freed_space_mb'] * 1024 * 1024)}")
                    
        elif args.command == "duplicates":
            print_duplicates(cache)
            
            if args.remove_duplicates:
                if not args.confirm:
                    if not confirm_action("Remove all duplicate items (keeping newest in each group)?"):
                        print("Operation cancelled.")
                        return
                
                # Remove duplicates by removing all but the first (newest) in each group
                duplicates = cache.find_duplicates()
                remove_indices = set()
                
                for group in duplicates:
                    # Remove all but the first (newest) item
                    for item_info in group[1:]:
                        remove_indices.add(item_info['index'])
                
                if remove_indices:
                    cache._rebuild_cache_without_indices(remove_indices)
                    print(f"✅ Removed {len(remove_indices)} duplicate items")
                else:
                    print("No duplicates to remove.")
                    
        elif args.command == "remove-pattern":
            if args.dry_run:
                import re
                pattern_re = re.compile(args.pattern, re.IGNORECASE)
                matching_items = [item for item in cache.items if pattern_re.search(item.prompt)]
                
                if not matching_items:
                    print(f"No items match pattern: {args.pattern}")
                else:
                    total_size = 0
                    for item in matching_items:
                        if Path(item.path).exists():
                            try:
                                total_size += Path(item.path).stat().st_size
                            except OSError:
                                pass
                    
                    print(f"🔍 Would remove {len(matching_items)} items matching pattern: {args.pattern}")
                    print(f"   Space that would be freed: {format_size(total_size)}")
                    print("\nMatching items:")
                    for item in matching_items[:10]:
                        date_str = format_timestamp(item.timestamp)
                        print(f"   {date_str}: {item.prompt}")
                    if len(matching_items) > 10:
                        print(f"   ... and {len(matching_items) - 10} more")
            else:
                if not args.confirm:
                    print(f"⚠️  This will remove all items matching pattern: {args.pattern}")
                    if not confirm_action("Are you sure?"):
                        print("Operation cancelled.")
                        return
                
                result = cache.remove_by_prompt_pattern(args.pattern, confirm=True)
                if "error" in result:
                    print(f"❌ {result['error']}")
                elif result['removed_items'] == 0:
                    print(f"No items match pattern: {args.pattern}")
                else:
                    print(f"✅ Removed {result['removed_items']} items matching '{result['pattern']}', freed {format_size(result['freed_space_mb'] * 1024 * 1024)}")
                    
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
