"""Unified timeline CLI tool to view transcripts and prompts interleaved by timestamp.

This tool extends the transcripts_cli.py pattern to show a complete timeline of:
- Transcript updates (conversations)
- Prompt creation and queuing
- Request events (image generation, completion, etc.)

Usage examples:
  experimance-timeline                    # list recent sessions
  experimance-timeline stream             # follow latest activity (all types)
  experimance-timeline show 3             # show session by index
  experimance-timeline show --session session_20250812_132151
  experimance-timeline follow --session session_20250812_132151
  experimance-timeline show --transcripts-only   # filter to just transcripts
  experimance-timeline show --prompts-only       # filter to just prompts

Environment overrides:
  EXPERIMANCE_TRANSCRIPTS_DIR - path to transcript directory 
  EXPERIMANCE_PROMPTS_DIR - path to prompt directory

Design: Merges JSONL files from both transcript and prompt logs, sorts by timestamp,
and presents a unified timeline view with rich formatting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

# Role styles for different event types
ROLE_STYLES = {
    "system": "bold cyan",
    "user": "bold green", 
    "assistant": "bold magenta",
    "agent": "bold magenta",
    "prompt_generator": "bold yellow",
    "fire_core": "bold blue",
    "tool": "yellow",
    "error": "bold red",
    "unknown": "white",
}

# Event type styles
EVENT_STYLES = {
    "prompt_created": "bold yellow",
    "prompt_queued": "yellow",
    "image_prompt_created": "bold cyan",
    "image_prompt_queued": "cyan",
    "transcript": "green",
    "request_event": "blue",
    "unknown": "white"
}

@dataclass
class TimelineEntry:
    """Unified entry for timeline display."""
    timestamp: float
    datetime_obj: datetime
    session_id: str
    entry_type: str  # "transcript", "prompt", "event"
    source_file: Path
    raw_data: dict
    
    # Display fields
    role: str
    speaker: str
    content: str
    event_type: Optional[str] = None
    request_id: Optional[str] = None
    
    # Prompt-specific fields
    visual_prompt: Optional[str] = None
    audio_prompt: Optional[str] = None
    metadata: Optional[dict] = None

@dataclass
class SessionInfo:
    session_id: str
    start_time: datetime
    end_time: datetime
    transcript_files: List[Path]
    prompt_files: List[Path]
    entry_count: int
    original_index: Optional[int] = None

def _default_dir_candidates(log_type: str) -> list[Path]:
    """Get directory candidates for transcript or prompt logs."""
    candidates: list[Path] = []
    
    if log_type == "transcripts":
        env_var = "EXPERIMANCE_TRANSCRIPTS_DIR"
        default_path = "/var/log/experimance/transcripts"
        fallback_path = "transcripts"
    else:  # prompts
        env_var = "EXPERIMANCE_PROMPTS_DIR"
        default_path = "/var/log/experimance/prompts"
        fallback_path = "prompts"
    
    env_dir = os.environ.get(env_var)
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(Path(default_path))
    candidates.append(Path.cwd() / fallback_path)
    
    return candidates

def discover_directories(explicit_transcripts: Optional[str] = None, 
                        explicit_prompts: Optional[str] = None) -> tuple[Path, Path]:
    """Discover transcript and prompt directories."""
    
    # Transcripts directory
    if explicit_transcripts:
        transcripts_dir = Path(explicit_transcripts).expanduser()
        if not transcripts_dir.is_dir():
            console.print(f"[red]Transcripts directory not found: {transcripts_dir}")
            sys.exit(2)
    else:
        transcripts_dir = None
        for candidate in _default_dir_candidates("transcripts"):
            if candidate.is_dir():
                transcripts_dir = candidate
                break
        if not transcripts_dir:
            console.print("[red]No transcripts directory found. Use --transcripts-path to specify.")
            sys.exit(2)
    
    # Prompts directory  
    if explicit_prompts:
        prompts_dir = Path(explicit_prompts).expanduser()
        if not prompts_dir.is_dir():
            console.print(f"[red]Prompts directory not found: {prompts_dir}")
            sys.exit(2)
    else:
        prompts_dir = None
        for candidate in _default_dir_candidates("prompts"):
            if candidate.is_dir():
                prompts_dir = candidate
                break
        if not prompts_dir:
            console.print("[yellow]No prompts directory found - will only show transcripts")
            prompts_dir = None
    
    return transcripts_dir, prompts_dir

def extract_session_id_from_filename(filename: str) -> Optional[str]:
    """Extract session ID from transcript or prompt filename."""
    # Format: transcript_20250812_132151_session_20250812_132151.jsonl
    # Format: prompts_20250812_132151_session_20250812_132151.jsonl
    if "_session_" in filename:
        return filename.split("_session_")[1].replace(".jsonl", "")
    return None

def find_sessions(transcripts_dir: Path, prompts_dir: Optional[Path]) -> Dict[str, SessionInfo]:
    """Find all sessions by scanning transcript and prompt files."""
    sessions: Dict[str, SessionInfo] = {}
    
    # Scan transcript files
    if transcripts_dir and transcripts_dir.is_dir():
        for file_path in transcripts_dir.glob("transcript_*.jsonl"):
            session_id = extract_session_id_from_filename(file_path.name)
            if session_id:
                if session_id not in sessions:
                    sessions[session_id] = SessionInfo(
                        session_id=session_id,
                        start_time=datetime.min,
                        end_time=datetime.min,
                        transcript_files=[],
                        prompt_files=[],
                        entry_count=0
                    )
                sessions[session_id].transcript_files.append(file_path)
    
    # Scan prompt files
    if prompts_dir and prompts_dir.is_dir():
        for file_path in prompts_dir.glob("prompts_*.jsonl"):
            session_id = extract_session_id_from_filename(file_path.name)
            if session_id:
                if session_id not in sessions:
                    sessions[session_id] = SessionInfo(
                        session_id=session_id,
                        start_time=datetime.min,
                        end_time=datetime.min,
                        transcript_files=[],
                        prompt_files=[],
                        entry_count=0
                    )
                sessions[session_id].prompt_files.append(file_path)
    
    # Calculate time ranges and entry counts for each session
    for session_id, session_info in sessions.items():
        timestamps = []
        entry_count = 0
        
        # Get timestamps from transcript files
        for file_path in session_info.transcript_files:
            try:
                for entry in parse_jsonl_file(file_path):
                    if entry.timestamp:
                        timestamps.append(entry.timestamp)
                        entry_count += 1
            except Exception:
                continue
        
        # Get timestamps from prompt files  
        for file_path in session_info.prompt_files:
            try:
                for entry in parse_jsonl_file(file_path):
                    if entry.timestamp:
                        timestamps.append(entry.timestamp)
                        entry_count += 1
            except Exception:
                continue
        
        if timestamps:
            session_info.start_time = datetime.fromtimestamp(min(timestamps))
            session_info.end_time = datetime.fromtimestamp(max(timestamps))
        session_info.entry_count = entry_count
    
    return sessions

def parse_jsonl_file(file_path: Path) -> Iterator[TimelineEntry]:
    """Parse a JSONL file (transcript or prompt) into TimelineEntry objects."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, raw_line in enumerate(f, 1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                
                try:
                    data = json.loads(raw_line)
                    entry = parse_timeline_entry(data, file_path)
                    if entry:
                        yield entry
                except Exception as e:
                    # Create error entry
                    yield TimelineEntry(
                        timestamp=time.time(),
                        datetime_obj=datetime.now(),
                        session_id="unknown",
                        entry_type="error",
                        source_file=file_path,
                        raw_data={"error": str(e), "line": line_num},
                        role="error",
                        speaker="parser",
                        content=f"Parse error on line {line_num}: {e}"
                    )
    except Exception as e:
        console.print(f"[red]Error reading file {file_path}: {e}")

def parse_timeline_entry(data: dict, source_file: Path) -> Optional[TimelineEntry]:
    """Parse a JSON object into a TimelineEntry."""
    
    # Extract timestamp
    timestamp = None
    for key in ("timestamp", "time", "ts"):
        if key in data:
            timestamp = data[key]
            break
    
    if timestamp is None:
        return None
    
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            timestamp = dt.timestamp()
        else:
            dt = datetime.fromtimestamp(float(timestamp))
    except Exception:
        return None
    
    # Extract basic fields
    session_id = data.get("session_id", "unknown")
    role = data.get("role", "unknown")
    speaker_id = data.get("speaker_id", role)
    speaker_display = data.get("speaker_display_name", speaker_id)
    content = data.get("content", "")
    event_type = data.get("event_type")
    request_id = data.get("request_id")
    
    # Determine entry type
    if "visual_prompt" in data or "audio_prompt" in data:
        entry_type = "prompt"
        # For prompts, create a content summary
        if not content and data.get("visual_prompt"):
            visual = data["visual_prompt"]
            content = f"Visual: {visual[:100]}{'...' if len(visual) > 100 else ''}"
            if data.get("audio_prompt"):
                audio = data["audio_prompt"]
                content += f" | Audio: {audio[:50]}{'...' if len(audio) > 50 else ''}"
    elif event_type and event_type != "transcript":
        entry_type = "event"
    else:
        entry_type = "transcript"
    
    return TimelineEntry(
        timestamp=timestamp,
        datetime_obj=dt,
        session_id=session_id,
        entry_type=entry_type,
        source_file=source_file,
        raw_data=data,
        role=role,
        speaker=speaker_display,
        content=content,
        event_type=event_type,
        request_id=request_id,
        visual_prompt=data.get("visual_prompt"),
        audio_prompt=data.get("audio_prompt"),
        metadata=data.get("metadata")
    )

def load_session_timeline(session_info: SessionInfo, entry_filter: Optional[str] = None) -> List[TimelineEntry]:
    """Load and merge timeline entries for a session."""
    entries = []
    
    # Load transcript entries
    for file_path in session_info.transcript_files:
        for entry in parse_jsonl_file(file_path):
            if not entry_filter or entry.entry_type == entry_filter:
                entries.append(entry)
    
    # Load prompt entries
    for file_path in session_info.prompt_files:
        for entry in parse_jsonl_file(file_path):
            if not entry_filter or entry.entry_type == entry_filter:
                entries.append(entry)
    
    # Sort by timestamp
    entries.sort(key=lambda e: e.timestamp)
    return entries

def render_timeline_entry(entry: TimelineEntry, width: int, full_ts: bool = False) -> Text:
    """Render a timeline entry as Rich Text."""
    
    # Format timestamp
    ts_format = "%Y-%m-%d %H:%M:%S" if full_ts else "%H:%M:%S"
    ts_str = entry.datetime_obj.strftime(ts_format)
    
    # Choose color based on entry type and role
    if entry.entry_type == "prompt":
        style = EVENT_STYLES.get(entry.event_type or "unknown", "yellow")
        type_indicator = "🎨"
    elif entry.entry_type == "event":
        style = EVENT_STYLES.get(entry.event_type or "unknown", "blue")
        type_indicator = "⚙️"
    else:  # transcript
        style = ROLE_STYLES.get(entry.role, "white")
        type_indicator = "💬"
    
    # Build the rendered line
    text = Text(f"{ts_str} {type_indicator} ", style="dim")
    
    # Speaker/role label
    label = entry.speaker[:12] if entry.speaker else entry.role[:12]
    text.append(f"{label:<12}"[:12], style=style)
    text.append(" ")
    
    # Content with optional metadata
    content = entry.content
    if entry.request_id:
        content = f"[{entry.request_id[:8]}] {content}"
    
    # Wrap content if needed
    avail_width = max(width - 25, 20) if width else None
    if avail_width and len(content) > avail_width:
        content = content[:avail_width-3] + "..."
    
    text.append(content, style=None)
    
    return text

def build_sessions_table(sessions: List[SessionInfo], show_details: bool = True) -> Table:
    """Build a table showing available sessions."""
    table = Table(title="Timeline Sessions", box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Index", justify="right", style="bold", no_wrap=True)
    table.add_column("Session ID", overflow="fold")
    table.add_column("Start Time", style="dim")
    table.add_column("Duration", style="dim")
    table.add_column("Entries", justify="right")
    if show_details:
        table.add_column("Files", justify="center")
    
    for idx, session in enumerate(sessions):
        start_str = session.start_time.strftime("%Y-%m-%d %H:%M:%S") if session.start_time != datetime.min else "Unknown"
        
        if session.start_time != datetime.min and session.end_time != datetime.min:
            duration = session.end_time - session.start_time
            duration_str = f"{duration.total_seconds():.0f}s"
        else:
            duration_str = "Unknown"
        
        files_info = ""
        if show_details:
            t_count = len(session.transcript_files)
            p_count = len(session.prompt_files)
            files_info = f"T:{t_count} P:{p_count}"
        
        row = [str(idx), session.session_id, start_str, duration_str, str(session.entry_count)]
        if show_details:
            row.append(files_info)
        
        table.add_row(*row)
    
    if not sessions:
        table.caption = "No sessions found."
    else:
        table.caption = "Use: experimance-timeline show <index> | follow <index>"
    
    return table

def cmd_list(args: argparse.Namespace) -> None:
    """List available timeline sessions."""
    transcripts_dir, prompts_dir = discover_directories(args.transcripts_path, args.prompts_path)
    
    sessions_dict = find_sessions(transcripts_dir, prompts_dir)
    sessions = list(sessions_dict.values())
    
    # Sort by start time (newest first)
    sessions.sort(key=lambda s: s.start_time, reverse=True)
    
    # Add original indices
    for idx, session in enumerate(sessions):
        session.original_index = idx
    
    table = build_sessions_table(sessions)
    console.print(table)

def resolve_session(sessions: Dict[str, SessionInfo], index: Optional[int], 
                   session_id: Optional[str]) -> SessionInfo:
    """Resolve session by index or session_id."""
    if session_id:
        if session_id not in sessions:
            raise SystemExit(f"Session not found: {session_id}")
        return sessions[session_id]
    
    if index is not None:
        sessions_list = list(sessions.values())
        sessions_list.sort(key=lambda s: s.start_time, reverse=True)
        if not (0 <= index < len(sessions_list)):
            raise SystemExit(f"Index out of range (0..{len(sessions_list)-1})")
        return sessions_list[index]
    
    # Default to most recent
    if not sessions:
        raise SystemExit("No sessions available")
    
    sessions_list = list(sessions.values())
    sessions_list.sort(key=lambda s: s.start_time, reverse=True)
    return sessions_list[0]

def cmd_show(args: argparse.Namespace) -> None:
    """Show timeline for a specific session."""
    transcripts_dir, prompts_dir = discover_directories(args.transcripts_path, args.prompts_path)
    
    sessions = find_sessions(transcripts_dir, prompts_dir)
    
    try:
        session = resolve_session(sessions, args.index, args.session)
    except SystemExit as e:
        console.print(f"[red]{e}")
        return
    
    # Determine filter
    entry_filter = None
    if args.transcripts_only:
        entry_filter = "transcript"
    elif args.prompts_only:
        entry_filter = "prompt"
    
    # Load timeline
    entries = load_session_timeline(session, entry_filter)
    
    if not entries:
        console.print("[yellow]No timeline entries found for session.")
        return
    
    # Filter by last N if specified
    if args.last and args.last > 0:
        entries = entries[-args.last:]
    
    width = args.width or console.width
    
    # Header
    console.print(f"[bold]Timeline Session:[/bold] {session.session_id}")
    console.print(f"[dim]Time Range: {session.start_time} → {session.end_time} ({len(entries)} entries)")
    
    if entry_filter:
        console.print(f"[dim]Filter: {entry_filter} only")
    
    console.print()
    
    # Render entries
    for entry in entries:
        rendered = render_timeline_entry(entry, width, full_ts=args.full_ts)
        console.print(rendered)

def cmd_follow(args: argparse.Namespace) -> None:
    """Follow (tail) a specific session."""
    transcripts_dir, prompts_dir = discover_directories(args.transcripts_path, args.prompts_path)
    
    sessions = find_sessions(transcripts_dir, prompts_dir)
    
    try:
        session = resolve_session(sessions, args.index, args.session)
    except SystemExit as e:
        console.print(f"[red]{e}")
        return
    
    console.print(f"[bold]Following Timeline Session:[/bold] {session.session_id} (Ctrl-C to stop)")
    
    # Determine filter
    entry_filter = None
    if args.transcripts_only:
        entry_filter = "transcript"
    elif args.prompts_only:
        entry_filter = "prompt"
    
    width = args.width or console.width
    
    # Track already seen entries by timestamp
    seen_timestamps = set()
    
    # Show existing entries if requested
    if args.from_start:
        initial_entries = load_session_timeline(session, entry_filter)
        for entry in initial_entries:
            seen_timestamps.add(entry.timestamp)
            rendered = render_timeline_entry(entry, width, full_ts=args.full_ts)
            console.print(rendered)
    
    try:
        while True:
            # Reload session files and check for new entries
            new_entries = load_session_timeline(session, entry_filter)
            
            for entry in new_entries:
                if entry.timestamp not in seen_timestamps:
                    seen_timestamps.add(entry.timestamp)
                    rendered = render_timeline_entry(entry, width, full_ts=args.full_ts)
                    console.print(rendered)
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped following.")

def cmd_stream(args: argparse.Namespace) -> None:
    """Stream the latest activity across all sessions."""
    transcripts_dir, prompts_dir = discover_directories(args.transcripts_path, args.prompts_path)
    
    console.print("[bold]Streaming Latest Timeline Activity[/bold] (Ctrl-C to stop)")
    
    width = args.width or console.width
    seen_timestamps = set()
    
    # Determine filter
    entry_filter = None
    if args.transcripts_only:
        entry_filter = "transcript"
    elif args.prompts_only:
        entry_filter = "prompt"
    
    try:
        while True:
            # Rescan for new sessions and entries
            sessions = find_sessions(transcripts_dir, prompts_dir)
            
            # Collect all new entries across all sessions
            new_entries = []
            for session in sessions.values():
                for entry in load_session_timeline(session, entry_filter):
                    if entry.timestamp not in seen_timestamps:
                        new_entries.append(entry)
                        seen_timestamps.add(entry.timestamp)
            
            # Sort by timestamp and display
            new_entries.sort(key=lambda e: e.timestamp)
            for entry in new_entries:
                rendered = render_timeline_entry(entry, width, full_ts=args.full_ts)
                console.print(rendered)
            
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped streaming.")

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    p = argparse.ArgumentParser(
        prog="experimance-timeline",
        description="Unified timeline viewer for transcripts and prompts"
    )
    
    p.add_argument("--transcripts-path", help="Transcripts directory (default auto-detect)")
    p.add_argument("--prompts-path", help="Prompts directory (default auto-detect)")
    p.add_argument("--force-color", action="store_true", help="Force color output")
    
    sub = p.add_subparsers(dest="command")
    
    # List command
    p_list = sub.add_parser("list", help="List available timeline sessions (default)")
    p_list.set_defaults(func=cmd_list)
    
    # Show command
    p_show = sub.add_parser("show", help="Show timeline for a session")
    p_show.add_argument("index", nargs="?", type=int, help="Session index from list")
    p_show.add_argument("--session", help="Explicit session ID")
    p_show.add_argument("--transcripts-only", action="store_true", help="Show only transcript entries")
    p_show.add_argument("--prompts-only", action="store_true", help="Show only prompt entries")
    p_show.add_argument("--last", type=int, help="Show only last N entries")
    p_show.add_argument("--width", type=int, help="Display width")
    p_show.add_argument("--full-ts", action="store_true", help="Show full timestamps")
    p_show.set_defaults(func=cmd_show)
    
    # Follow command
    p_follow = sub.add_parser("follow", help="Follow (tail) a session timeline")
    p_follow.add_argument("index", nargs="?", type=int, help="Session index from list")
    p_follow.add_argument("--session", help="Explicit session ID")
    p_follow.add_argument("--transcripts-only", action="store_true", help="Show only transcript entries")
    p_follow.add_argument("--prompts-only", action="store_true", help="Show only prompt entries")
    p_follow.add_argument("--from-start", action="store_true", help="Show existing entries first")
    p_follow.add_argument("--interval", type=float, default=1.0, help="Polling interval seconds")
    p_follow.add_argument("--width", type=int, help="Display width")
    p_follow.add_argument("--full-ts", action="store_true", help="Show full timestamps")
    p_follow.set_defaults(func=cmd_follow)
    
    # Stream command
    p_stream = sub.add_parser("stream", help="Stream latest activity across all sessions")
    p_stream.add_argument("--transcripts-only", action="store_true", help="Show only transcript entries")
    p_stream.add_argument("--prompts-only", action="store_true", help="Show only prompt entries")
    p_stream.add_argument("--interval", type=float, default=1.0, help="Polling interval seconds")
    p_stream.add_argument("--width", type=int, help="Display width")
    p_stream.add_argument("--full-ts", action="store_true", help="Show full timestamps")
    p_stream.set_defaults(func=cmd_stream)
    
    return p

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    p = build_parser()
    args = p.parse_args(argv)
    
    # Handle force color
    global console
    if getattr(args, "force_color", False):
        console = Console(force_terminal=True, color_system="truecolor")
    
    # Default command is list
    if not args.command:
        args.func = cmd_list
    
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {e}")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
