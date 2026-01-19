#!/usr/bin/env python3
"""
Generate visual summary cards from git commits for a specific day.
Creates PNG images with commit information, stats, and changes.
"""

import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os

from experimance_common.config import resolve_path, get_project_services

# Load environment variables from project .env file
try:
    from dotenv import load_dotenv
    env_file = resolve_path(".env", hint="project")
    load_dotenv(env_file)
except ImportError:
    # dotenv not available, that's okay
    pass

class GitDailyCard:
    """Generate visual summary cards from git commits."""
    
    # Card styling
    CARD_WIDTH = 1200
    CARD_PADDING = 40
    LINE_HEIGHT = 30
    TITLE_SIZE = 48
    HEADING_SIZE = 32
    TEXT_SIZE = 24
    SMALL_TEXT_SIZE = 20
    
    # Colors
    TEXT_COLOR = (220, 220, 230)
    HEADING_COLOR = (100, 200, 255)
    ADDED_COLOR = (100, 200, 100)
    REMOVED_COLOR = (255, 100, 100)
    ACCENT_COLOR = (180, 140, 255)
    MUTED_COLOR = (150, 150, 160)
    
    def __init__(self, repo_path: str = ".", bg_color: tuple = None):
        self.repo_path = Path(repo_path).resolve()
        self.bg_color = bg_color or (30, 30, 40, 255)  # Default with alpha
        self._markdown_cache = {}  # Cache for parsed markdown content per date
    
    def _load_markdown_cache(self, markdown_path: str):
        """Load and parse entire markdown file into cache once."""
        md_path = Path(markdown_path)
        if not md_path.exists():
            return
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse all date sections
            import re
            sections = content.split('# Experimance work summary:')
            
            for section in sections[1:]:  # Skip first empty section
                # Extract date from first line
                lines = section.strip().split('\n', 1)
                if not lines:
                    continue
                
                date = lines[0].strip()
                
                # Skip "no commits" entries - don't cache them
                if "No commits on this day" in section:
                    continue
                
                # Extract commit hashes from hidden comment
                match = re.search(r'<!-- commits: ([a-f0-9,]+) -->', section)
                if match:
                    hashes = match.group(1).split(',')
                    self._markdown_cache[date] = hashes
            
            print(f"Loaded {len(self._markdown_cache)} entries from markdown cache")
        except Exception as e:
            print(f"Warning: Could not load markdown cache: {e}")
    
    def _get_existing_commits_from_markdown(self, markdown_path: str, date: str) -> Optional[List[str]]:
        """Get commit hashes for a specific date from cache."""
        # Return cached result if available
        return self._markdown_cache.get(date)
    
    def _wrap_text_to_width(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """Wrap text to fit within max_width pixels using actual font metrics."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            # Get actual pixel width using textbbox
            bbox = font.getbbox(test_line)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]
    
    def _generate_summary(self, commits: List[Dict[str, Any]]) -> Optional[str]:
        """Generate plain-language summary using OpenAI."""
        try:
            from openai import OpenAI
        except ImportError:
            print("Warning: openai package not installed. Run: uv pip install openai")
            return None
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY environment variable not set")
            return None
        
        # Prepare commit messages
        commit_list = "\n".join([
            f"- {commit['message']} ({commit['stats']['files_changed']} files, "
            f"+{commit['stats']['insertions']} -{commit['stats']['deletions']})"
            for commit in commits
        ])
        
        prompt = f"""Write a brief summary (1-2 sentences max) of what was accomplished based on these git commits. Use simple, friendly, non-technical language. Be direct and start with the action (e.g., 'Improved...', 'Fixed...', 'Added...'):

{commit_list}

Focus on high level outcomes, don't just rephrase the commit messages. Avoid technical implementation details or filenames."""
        
        try:
            client = OpenAI(api_key=api_key)
            
            # Try gpt-5-nano first
            try:
                response = client.chat.completions.create(
                    model="gpt-5-nano",
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = response.choices[0].message.content.strip() if response.choices else None
            except Exception as model_error:
                print(f"gpt-5-nano failed ({model_error}), falling back to gpt-4o-mini...")
                # Fallback to gpt-4o-mini
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=150
                )
                summary = response.choices[0].message.content.strip()
            
            if summary:
                print(f"AI summary generated: {summary}")
                return summary
            else:
                print("Warning: AI returned empty summary")
                return None
        except Exception as e:
            print(f"Warning: Failed to generate summary: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def _generate_test_commits(self) -> List[Dict[str, Any]]:
        """Generate fake commits for testing line wrapping and layout."""
        return [
            {
                "hash": "abc1234",
                "message": "This is a very long commit message that should wrap across multiple lines to test the text wrapping functionality and ensure it doesn't overflow the card boundaries when dealing with verbose commit descriptions",
                "author": "Test User",
                "email": "test@example.com",
                "timestamp": "2025-01-07 12:00:00",
                "stats": {"files_changed": 5, "insertions": 123, "deletions": 45},
                "files": [
                    {"status": "Modified", "path": "services/audio/src/experimance_audio/audio_manager.py"},
                    {"status": "Added", "path": "services/core/src/experimance_core/configuration/settings.py"},
                    {"status": "Modified", "path": "libs/common/src/experimance_common/zmq/base_service.py"},
                    {"status": "Deleted", "path": "old/deprecated/legacy_module.py"},
                    {"status": "Modified", "path": "docs/technical_design.md"},
                ]
            },
            {
                "hash": "def5678",
                "message": "Short commit message",
                "author": "Test User",
                "email": "test@example.com",
                "timestamp": "2025-01-07 13:00:00",
                "stats": {"files_changed": 2, "insertions": 10, "deletions": 3},
                "files": [
                    {"status": "Modified", "path": "README.md"},
                    {"status": "Modified", "path": "pyproject.toml"},
                ]
            },
            {
                "hash": "ghi9012",
                "message": "Fix critical bug in the audio playback system that was causing intermittent crashes when handling multiple concurrent audio streams with different sample rates and bit depths",
                "author": "Test User",
                "email": "test@example.com",
                "timestamp": "2025-01-07 14:00:00",
                "stats": {"files_changed": 18, "insertions": 234, "deletions": 156},
                "files": [
                    {"status": "Modified", "path": f"services/test_service_{i}/very/deeply/nested/file_path_number_{i}.py"}
                    for i in range(20)
                ]
            },
        ]
    
    def get_commits_for_date(self, date: str) -> List[Dict[str, Any]]:
        """Get all commits for a specific date using author date (timezone-agnostic)."""
        # Get all commit hashes with author dates
        # Note: --since/--until filter by commit date, not author date
        # So we get all commits and filter by author date manually
        cmd = [
            "git", "-C", str(self.repo_path),
            "log",
            "--all",  # Search all branches
            "--pretty=format:%H %ai",  # Hash and author date (ISO format: YYYY-MM-DD HH:MM:SS +ZZZZ)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Filter commits by author date (using date as-is, ignoring timezone)
        commit_hashes = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            commit_hash, author_date_str = parts
            # Author date format: 2025-06-18 13:28:25 -0400
            # Extract just the date part (first 10 chars: YYYY-MM-DD)
            try:
                commit_date = author_date_str.split()[0]  # Get "2025-06-18" part
                if commit_date == date:
                    commit_hashes.append(commit_hash)
            except (IndexError, ValueError):
                continue
        
        if not commit_hashes or commit_hashes[0] == "":
            return []
        
        commits = []
        for commit_hash in commit_hashes:
            commit_info = self._get_commit_info(commit_hash)
            commits.append(commit_info)
        
        return commits
    
    def _get_commit_info(self, commit_hash: str) -> Dict[str, Any]:
        """Get detailed information about a commit."""
        # Get commit message and metadata
        cmd = [
            "git", "-C", str(self.repo_path),
            "show",
            "--no-patch",
            "--pretty=format:%s%n%b%n---METADATA---%n%an%n%ae%n%ai",
            commit_hash
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.split("---METADATA---")
        message = lines[0].strip()
        metadata = lines[1].strip().split("\n")
        
        author_name = metadata[0]
        author_email = metadata[1]
        timestamp = metadata[2]
        
        # Capitalize first letter of commit message
        if message:
            message = message[0].upper() + message[1:] if len(message) > 0 else message
        
        # Get stats
        stats_cmd = [
            "git", "-C", str(self.repo_path),
            "show",
            "--stat",
            "--pretty=format:",
            commit_hash
        ]
        
        stats_result = subprocess.run(stats_cmd, capture_output=True, text=True)
        stats = self._parse_stats(stats_result.stdout)
        
        # Get changed files
        files_cmd = [
            "git", "-C", str(self.repo_path),
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            commit_hash
        ]
        
        files_result = subprocess.run(files_cmd, capture_output=True, text=True)
        files = self._parse_files(files_result.stdout)
        
        return {
            "hash": commit_hash[:8],
            "message": message,
            "author": author_name,
            "email": author_email,
            "timestamp": timestamp,
            "stats": stats,
            "files": files,
        }
    
    def _parse_stats(self, stats_output: str) -> Dict[str, int]:
        """Parse git stats output."""
        lines = stats_output.strip().split("\n")
        stats = {"files_changed": 0, "insertions": 0, "deletions": 0}
        
        # Last line usually has summary
        for line in reversed(lines):
            if "changed" in line or "insertion" in line or "deletion" in line:
                parts = line.split(",")
                for part in parts:
                    part = part.strip()
                    if "file" in part:
                        stats["files_changed"] = int(part.split()[0])
                    elif "insertion" in part:
                        stats["insertions"] = int(part.split()[0])
                    elif "deletion" in part:
                        stats["deletions"] = int(part.split()[0])
                break
        
        return stats
    
    def _parse_files(self, files_output: str) -> List[Dict[str, str]]:
        """Parse changed files output."""
        files = []
        for line in files_output.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            status = parts[0]
            filepath = parts[1] if len(parts) > 1 else ""
            
            status_map = {
                "A": "Added",
                "M": "Modified",
                "D": "Deleted",
                "R": "Renamed",
                "C": "Copied",
            }
            
            files.append({
                "status": status_map.get(status[0], status),
                "path": filepath
            })
        
        return files
    
    def _get_font(self, size: int, emoji: bool = False) -> ImageFont.FreeTypeFont:
        """Get font, fallback to default if needed."""
        if emoji:
            # For emoji support, use fonts with better Unicode coverage
            emoji_font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
                "/System/Library/Fonts/Apple Color Emoji.ttc",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
            ]
            for font_path in emoji_font_paths:
                if Path(font_path).exists():
                    try:
                        return ImageFont.truetype(font_path, size)
                    except Exception:
                        pass
        
        # Regular monospace fonts
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Monaco.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        ]
        
        for font_path in font_paths:
            if Path(font_path).exists():
                try:
                    return ImageFont.truetype(font_path, size)
                except Exception:
                    pass
        
        return ImageFont.load_default()
    
    def generate_card(self, date: str, output_path: str = None, summarize: bool = False, test_mode: bool = False, markdown_export: str = None) -> Path:
        """Generate a summary card for a specific date."""
        if test_mode:
            commits = self._generate_test_commits()
            date = "TEST"
        else:
            commits = self.get_commits_for_date(date)
        
        # Check if already processed (if markdown export is specified)
        if markdown_export and not test_mode:
            existing_hashes = self._get_existing_commits_from_markdown(markdown_export, date)
            if existing_hashes is not None:
                current_hashes = [c['hash'] for c in commits]
                if existing_hashes == current_hashes:
                    print(f"Skipping {date} - already processed with same commits")
                    return None
                elif len(existing_hashes) > 0:
                    print(f"Regenerating {date} - commits have changed")
        
        if not commits:
            print(f"No commits found for {date}")
            # Add entry to markdown for days with no commits
            if markdown_export:
                self._export_no_commits_to_markdown(date, markdown_export)
            return None
        
        # Generate AI summary if requested
        summary = None
        if summarize:
            if test_mode:
                summary = "This is a test summary with a longer description that demonstrates how the AI-generated summary text wraps across multiple lines when dealing with more verbose explanations of the day's work and accomplishments."
            else:
                print("Generating AI summary...")
                summary = self._generate_summary(commits)
        
        # Calculate card height based on content
        card_height = self._calculate_card_height(commits, summary)
        
        # Create image with or without alpha channel
        mode = "RGBA" if len(self.bg_color) == 4 else "RGB"
        img = Image.new(mode, (self.CARD_WIDTH, card_height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw content
        y_offset = self._draw_card_content(draw, date, commits, summary)
        
        # Save image
        if output_path is None:
            output_path = f"git_summary_{date}.png"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        img.save(output_path)
        print(f"Card generated: {output_path}")
        
        # Export to markdown if requested
        if markdown_export:
            self._export_to_markdown(date, commits, summary, markdown_export)
        
        return output_path
    
    def _export_to_markdown(self, date: str, commits: List[Dict], summary: Optional[str], output_path: str):
        """Export card content to markdown file."""
        md_path = Path(output_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove existing entry for this date if it exists
        if md_path.exists():
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove old entry for this date
            date_marker = f"# Experimance work summary: {date}"
            if date_marker in content:
                sections = content.split('# Experimance work summary:')
                new_sections = []
                for section in sections:
                    # Keep sections that don't match this date
                    if not section.strip().startswith(date):
                        new_sections.append(section)
                
                # Reconstruct without the old entry
                if len(new_sections) > 1:
                    content = '# Experimance work summary:'.join(new_sections)
                else:
                    content = new_sections[0] if new_sections else ""
                
                # Write back
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(content)
        
        # Now append the new entry
        md_lines = []
        md_lines.append(f"# Experimance work summary: {date}\n")
        
        # Overall stats
        total_files = sum(c["stats"]["files_changed"] for c in commits)
        total_insertions = sum(c["stats"]["insertions"] for c in commits)
        total_deletions = sum(c["stats"]["deletions"] for c in commits)
        
        md_lines.append(f"**{len(commits)} commit{'s' if len(commits) != 1 else ''} • "
                       f"{total_files} files • +{total_insertions} -{total_deletions}**\n")
        
        # Add commit hashes for verification (hidden comment)
        commit_hashes = ",".join([c['hash'] for c in commits])
        md_lines.append(f"<!-- commits: {commit_hashes} -->\n")
        
        # AI Summary
        if summary:
            md_lines.append(f"\n{summary}\n\n")
        
        # Commits
        md_lines.append(f"## Commits\n\n")
        for i, commit in enumerate(commits):
            md_lines.append(f"### {commit['message']}\n")
            
            stats = commit["stats"]
            md_lines.append(f"*{stats['files_changed']} files changed, "
                          f"+{stats['insertions']} -{stats['deletions']}*\n")
            
            # Files
            if commit["files"]:
                for file_info in commit["files"]:
                    # Remove experimance/ prefix if present
                    file_path = file_info['path']
                    if file_path.startswith("experimance/"):
                        file_path = file_path[len("experimance/"):]
                    
                    status_symbols = {
                        "Added": "+",
                        "Modified": "~",
                        "Deleted": "-",
                        "Renamed": ">",
                        "Copied": "*",
                    }
                    symbol = status_symbols.get(file_info["status"], "•")
                    md_lines.append(f"- `{symbol}` {file_path}\n")
            
            md_lines.append("\n")
        
        # break up entry for multiple dates
        md_lines.append("\n")
        md_lines.append("---\n")
        md_lines.append("\n")

        # Write to file
        md_path = Path(output_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(md_path, 'a', encoding='utf-8') as f:
            f.writelines(md_lines)
        
        print(f"Markdown exported: {md_path}")
    
    def _export_no_commits_to_markdown(self, date: str, output_path: str):
        """Add entry for date with no commits."""
        md_lines = []
        md_lines.append(f"# Experimance work summary: {date}\n")
        md_lines.append("\n*No commits on this day*\n")
        md_lines.append("\n---\n\n")
        
        md_path = Path(output_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(md_path, 'a', encoding='utf-8') as f:
            f.writelines(md_lines)
    
    def _calculate_card_height(self, commits: List[Dict], summary: Optional[str] = None) -> int:
        """Calculate required card height based on content."""
        height = self.CARD_PADDING  # Top padding
        height += self.TITLE_SIZE + 20  # Title
        
        # AI summary if present
        if summary:
            # Estimate lines - use character-based estimate for height calc
            estimated_chars_per_line = (self.CARD_WIDTH - (self.CARD_PADDING * 2)) // 13
            summary_lines = len(textwrap.wrap(summary, width=estimated_chars_per_line))
            height += summary_lines * self.LINE_HEIGHT + 30
        
        height += 40  # Stats summary section
        
        for commit in commits:
            height += self.HEADING_SIZE + 10  # Commit heading with first line of message
            
            # Additional message lines (wrapped)
            message_lines = textwrap.wrap(commit["message"], width=70)
            if len(message_lines) > 1:
                height += (len(message_lines) - 1) * self.LINE_HEIGHT
            
            # Stats line
            height += self.LINE_HEIGHT + 10
            
            # Files (limit to 15)
            files_to_show = min(len(commit["files"]), 15)
            height += files_to_show * self.SMALL_TEXT_SIZE
            
            if len(commit["files"]) > 15:
                height += self.SMALL_TEXT_SIZE
            
            height += 30  # Spacing between commits
        
        height += self.CARD_PADDING  # Bottom padding to match top
        return max(height, 800)  # Minimum height
    
    def _draw_card_content(self, draw: ImageDraw.Draw, date: str, 
                          commits: List[Dict], summary: Optional[str] = None) -> int:
        """Draw all content on the card."""
        y = self.CARD_PADDING
        
        # Title
        title_font = self._get_font(self.TITLE_SIZE)
        title = f"Experimance work summary: {date}"
        draw.text((self.CARD_PADDING, y), title, 
                 fill=self.HEADING_COLOR, font=title_font)
        y += self.TITLE_SIZE + 20
        
        # AI Summary if present
        if summary:
            text_font = self._get_font(self.TEXT_SIZE)
            # Use actual font metrics for wrapping
            max_width = self.CARD_WIDTH - (self.CARD_PADDING * 2)
            summary_lines = self._wrap_text_to_width(summary, text_font, max_width)
            for line in summary_lines:
                draw.text((self.CARD_PADDING, y), line, 
                         fill=self.TEXT_COLOR, font=text_font)
                y += self.LINE_HEIGHT
            y += 20
        
        # Overall summary
        total_files = sum(c["stats"]["files_changed"] for c in commits)
        total_insertions = sum(c["stats"]["insertions"] for c in commits)
        total_deletions = sum(c["stats"]["deletions"] for c in commits)
        
        heading_font = self._get_font(self.HEADING_SIZE)
        text_font = self._get_font(self.TEXT_SIZE)
        small_font = self._get_font(self.SMALL_TEXT_SIZE)
        
        summary = f"{len(commits)} commit{'s' if len(commits) != 1 else ''} • {total_files} files • "
        draw.text((self.CARD_PADDING, y), summary, 
                 fill=self.MUTED_COLOR, font=text_font)
        
        # Stats with colors
        x_offset = self.CARD_PADDING + draw.textlength(summary, font=text_font)
        draw.text((x_offset, y), f"+{total_insertions}", 
                 fill=self.ADDED_COLOR, font=text_font)
        x_offset += draw.textlength(f"+{total_insertions} ", font=text_font)
        draw.text((x_offset, y), f"-{total_deletions}", 
                 fill=self.REMOVED_COLOR, font=text_font)
        
        y += 40
        
        # Separator line
        draw.line([(self.CARD_PADDING, y), 
                  (self.CARD_WIDTH - self.CARD_PADDING, y)], 
                 fill=self.ACCENT_COLOR, width=2)
        y += 20
        
        # Each commit
        for i, commit in enumerate(commits):
            # Commit message
            # Use actual font metrics for wrapping
            max_width = self.CARD_WIDTH - (self.CARD_PADDING * 2)
            message_lines = self._wrap_text_to_width(commit["message"], text_font, max_width)
            
            # Draw message lines
            for j, line in enumerate(message_lines):
                draw.text((self.CARD_PADDING, y), line, 
                         fill=self.TEXT_COLOR,
                         font=text_font if j == 0 else small_font)
                y += self.LINE_HEIGHT if j == 0 else (self.LINE_HEIGHT - 5)
            
            y += 10
            
            # Stats
            stats = commit["stats"]
            stats_text = f"{stats['files_changed']} files changed, "
            draw.text((self.CARD_PADDING + 20, y), stats_text, 
                     fill=self.MUTED_COLOR, font=small_font)
            
            x_offset = self.CARD_PADDING + 20 + draw.textlength(stats_text, font=small_font)
            draw.text((x_offset, y), f"+{stats['insertions']}", 
                     fill=self.ADDED_COLOR, font=small_font)
            x_offset += draw.textlength(f"+{stats['insertions']} ", font=small_font)
            draw.text((x_offset, y), f"-{stats['deletions']}", 
                     fill=self.REMOVED_COLOR, font=small_font)
            
            y += self.LINE_HEIGHT + 5
            
            # Files (limit to 15)
            files_to_show = commit["files"][:15]
            for file_info in files_to_show:
                status_symbols = {
                    "Added": "+",
                    "Modified": "~",
                    "Deleted": "-",
                    "Renamed": ">",
                    "Copied": "*",
                }
                symbol = status_symbols.get(file_info["status"], "•")
                
                # Remove experimance/ prefix if present
                file_path = file_info['path']
                if file_path.startswith("experimance/"):
                    file_path = file_path[len("experimance/"):]
                
                file_text = f"{symbol} {file_path}"
                # Truncate long paths
                if len(file_text) > 110:
                    file_text = file_text[:107] + "..."
                
                # Color code by status
                symbol_color = self.ADDED_COLOR if symbol == "+" else \
                              self.REMOVED_COLOR if symbol == "-" else \
                              self.MUTED_COLOR
                
                draw.text((self.CARD_PADDING + 30, y), file_text, 
                         fill=symbol_color, font=small_font)
                y += self.SMALL_TEXT_SIZE
            
            if len(commit["files"]) > 15:
                remaining = len(commit["files"]) - 15
                draw.text((self.CARD_PADDING + 30, y), 
                         f"... and {remaining} more files", 
                         fill=self.MUTED_COLOR, font=small_font)
                y += self.SMALL_TEXT_SIZE
            
            # Add spacing only if not the last commit
            if i < len(commits) - 1:
                y += 30  # Spacing between commits
        
        return y


def parse_color(color_str: str) -> tuple:
    """Parse color string to RGB or RGBA tuple."""
    if not color_str:
        return None
    
    color_str = color_str.strip()
    
    # Hex format: #1e1e28 or #1e1e28ff
    if color_str.startswith("#"):
        color_str = color_str[1:]
        if len(color_str) == 6:
            return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
        elif len(color_str) == 8:
            return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4, 6))
    
    # RGB/RGBA format: 30,30,40 or 30,30,40,255
    parts = [int(x.strip()) for x in color_str.split(",")]
    if len(parts) in (3, 4):
        return tuple(parts)
    
    raise ValueError(f"Invalid color format: {color_str}. Use hex (#1e1e28), RGB (30,30,40), or RGBA (30,30,40,255)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual summary cards from git commits"
    )
    parser.add_argument(
        "date",
        nargs="?",
        help="Date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: git_summary_DATE.png)"
    )
    parser.add_argument(
        "-r", "--repo",
        default=".",
        help="Repository path (default: current directory)"
    )
    parser.add_argument(
        "--yesterday",
        action="store_true",
        help="Generate card for yesterday"
    )
    parser.add_argument(
        "--last-week",
        action="store_true",
        help="Generate cards for the last 7 days"
    )
    parser.add_argument(
        "--date-range",
        nargs=2,
        metavar=("START", "END"),
        help="Generate cards for date range (YYYY-MM-DD YYYY-MM-DD)"
    )
    parser.add_argument(
        "--bg-color",
        help="Background color as hex (#1e1e28), RGB (30,30,40), or RGBA (30,30,40,255)"
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Generate AI summary using OpenAI (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate test card with placeholder data for testing layout"
    )
    parser.add_argument(
        "--export-md",
        metavar="PATH",
        help="Export card content to markdown file (appends if processing multiple dates)"
    )
    
    args = parser.parse_args()
    
    # Parse background color if provided
    bg_color = None
    if args.bg_color:
        try:
            bg_color = parse_color(args.bg_color)
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    
    generator = GitDailyCard(args.repo, bg_color=bg_color)
    
    # Load markdown cache once if export-md is specified
    if args.export_md:
        generator._load_markdown_cache(args.export_md)
    
    # Determine dates to process
    dates = []
    
    if args.date_range:
        # Generate cards for date range
        start_date = datetime.strptime(args.date_range[0], "%Y-%m-%d")
        end_date = datetime.strptime(args.date_range[1], "%Y-%m-%d")
        current = start_date
        while current <= end_date:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
    elif args.last_week:
        today = datetime.now()
        dates = [
            (today - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(7)
        ]
    elif args.yesterday:
        yesterday = datetime.now() - timedelta(days=1)
        dates = [yesterday.strftime("%Y-%m-%d")]
    elif args.date:
        dates = [args.date]
    else:
        dates = [datetime.now().strftime("%Y-%m-%d")]
    
    # Generate cards
    for date in dates:
        output = args.output
        if output:
            # Handle output as directory or file
            output_path = Path(output)
            
            # Check if output is a directory (no extension or ends with /)
            if output.endswith('/') or (not output_path.suffix and output_path.is_dir()) or (not output_path.suffix and not output_path.exists()):
                # Treat as directory
                output_path.mkdir(parents=True, exist_ok=True)
                output = str(output_path / f"git_summary_{date}.png")
            elif len(dates) > 1:
                # Multiple dates with file output: add date to filename
                output = str(output_path.parent / f"{output_path.stem}_{date}{output_path.suffix}")
        
        generator.generate_card(date, output, summarize=args.summarize, test_mode=args.test, markdown_export=args.export_md)


if __name__ == "__main__":
    main()
