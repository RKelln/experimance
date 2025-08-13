"""CLI tool to list, view, and follow transcript JSONL files.

Usage examples:
  experimance-transcripts                # list transcripts
  experimance-transcripts show 3         # show transcript by index
  experimance-transcripts show --file transcript_20250812_132151_session_20250812_132151.jsonl
  experimance-transcripts follow         # follow latest transcript (tail -f)
  experimance-transcripts follow 2       # follow specific transcript

Environment overrides:
  EXPERIMANCE_TRANSCRIPTS_DIR - path to transcript directory (default /var/log/experimance/transcripts)

Design notes:
 - Flexible parsing of JSONL lines with unknown schema differences.
 - Rich tables and live updating for follow functionality.
 - Polling based directory watching (lightweight, no extra deps) with --watch / --follow.

TODO: add search/filter, regex, export features.
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
from typing import Iterable, Iterator, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

def _default_dir_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_dir = os.environ.get("EXPERIMANCE_TRANSCRIPTS_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.append(Path("/var/log/experimance/transcripts"))
    candidates.append(Path.cwd() / "transcripts")
    return candidates

console = Console()

ROLE_STYLES = {
    "system": "bold cyan",
    "user": "bold green",
    "assistant": "bold magenta",
    # 'agent' kept separate in case we later want a distinct color; currently mirrors assistant
    "agent": "bold magenta",
    "tool": "yellow",
    "error": "bold red",
    "unknown": "white",
}

@dataclass
class TranscriptInfo:
    path: Path
    size: int
    mtime: float
    ctime: float
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    line_count: Optional[int] = None
    turns: Optional[int] = None  # number of conversation turns (user/assistant by default)
    original_index: Optional[int] = None  # index in the full (unfiltered) sorted list

@dataclass
class ParsedLine:
    raw: str
    data: Optional[dict]
    error: Optional[str]

    @property
    def timestamp(self) -> Optional[datetime]:
        if not self.data:
            return None
        ts = None
        for key in ("timestamp", "time", "ts"):
            if key in self.data and self.data[key] is not None:
                ts = self.data[key]
                break
        if ts is None:
            return None
        # Attempt parsing: numeric epoch or ISO string
        try:
            if isinstance(ts, (int, float)):
                return datetime.fromtimestamp(float(ts))
            if isinstance(ts, str):
                # strip Z
                txt = ts.rstrip("Z")
                # Try isoformat
                try:
                    return datetime.fromisoformat(txt)
                except Exception:
                    # maybe its epoch str
                    return datetime.fromtimestamp(float(txt))
        except Exception:
            return None
        return None

    @property
    def role(self) -> str:
        if not self.data:
            return "invalid"
        # Prefer explicit role, but normalize with heuristics
        base = self.data.get("role") or self.data.get("speaker") or self.data.get("type") or "unknown"
        base_l = str(base).lower()
        t = str(self.data.get("type", "")).lower()
        speaker_id = str(self.data.get("speaker_id", "")).lower()
        message_type = str(self.data.get("message_type", "")).lower()
        if t == "session_start":
            return "system"
        if speaker_id in {"user", "visitor"} or message_type.startswith("user_"):
            return "user"
        if speaker_id in {"agent", "assistant"} or message_type.startswith("agent_") or message_type.startswith("assistant_"):
            # Normalize 'agent' to 'assistant' to reduce style proliferation
            return "assistant"
        if base_l in {"user", "assistant", "system"}:
            return base_l
        return "unknown"

    @property
    def content(self) -> str:
        if not self.data:
            return self.raw.strip()
        for key in ("content", "message", "text", "value"):
            if key in self.data and isinstance(self.data[key], str):
                return self.data[key]
        if self.data.get("type") == "session_start":
            return "Session started"
        # Fallback: join string values
        parts = []
        for k, v in self.data.items():
            if isinstance(v, str) and k not in {"role", "speaker", "type"}:
                parts.append(f"{k}={v}")
        return " | ".join(parts) if parts else json.dumps(self.data, ensure_ascii=False)


def discover_directory(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir():
            return p
        console.print(f"[red]Provided transcripts directory not found: {p}")
        sys.exit(2)
    for candidate in _default_dir_candidates():
        if candidate.is_dir():
            return candidate
    console.print("[red]No transcripts directory found. Use --path to specify.")
    sys.exit(2)


def list_transcripts(dir_path: Path) -> List[TranscriptInfo]:
    entries: List[TranscriptInfo] = []
    for path in sorted(dir_path.glob("transcript_*.jsonl")):
        try:
            stat = path.stat()
            entries.append(
                TranscriptInfo(
                    path=path,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    ctime=stat.st_ctime,
                )
            )
        except OSError:
            continue
    # Sort newest first by mtime
    entries.sort(key=lambda t: t.mtime, reverse=True)
    return entries


def build_list_table(
    infos: List[TranscriptInfo],
    directory: Path | None = None,
    show_turns: bool = True,
    use_original_index: bool = True,
    page: int | None = None,
    page_size: int | None = None,
    total_count: int | None = None,
) -> Table:
    title = "Transcripts"
    if directory is not None:
        title += f" ({directory})"
    if page_size and page is not None:
        count_for_pages = total_count if total_count is not None else len(infos)
        total_pages = max(1, (count_for_pages + page_size - 1) // page_size)
        title += f"  [Page {page+1}/{total_pages}]"
    table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("Index", justify="right", style="bold", no_wrap=True)
    table.add_column("Filename", overflow="fold")
    table.add_column("Modified", style="dim")
    table.add_column("Size", justify="right")
    if show_turns:
        table.add_column("Turns", justify="right")
    for idx, info in enumerate(infos):
        mod = datetime.fromtimestamp(info.mtime).strftime("%Y-%m-%d %H:%M:%S")
        display_index = info.original_index if (use_original_index and info.original_index is not None) else idx
        turns_val = str(info.turns) if (show_turns and info.turns is not None) else ("?" if show_turns else "")
        row = [str(display_index), info.path.name, mod, human_size(info.size)]
        if show_turns:
            row.append(turns_val)
        table.add_row(*row)
    if not infos:
        table.caption = "No transcripts found." + (
            " (directory exists)" if directory and directory.is_dir() else ""
        )
    else:
        table.caption = "Use: experimance-transcripts show <index> | follow <index>"
    return table


def human_size(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num < 1024.0:
            return f"{num:3.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}TB"


def compute_turns(infos: List[TranscriptInfo], all_roles: bool = False) -> None:
    """Compute number of turns for each transcript (mutates infos).

    Definition: A turn is counted for lines whose role is user or assistant unless all_roles is True.
    """
    interesting = {"user", "assistant", "agent"}
    for info in infos:
        try:
            count = 0
            for pl in parse_lines(info.path):
                if all_roles:
                    count += 1
                else:
                    role = pl.role.lower()
                    if role in interesting:
                        count += 1
            info.turns = count
        except Exception:  # noqa: BLE001
            info.turns = None


def apply_turn_filters(infos: List[TranscriptInfo], min_turns: Optional[int], max_turns: Optional[int]) -> List[TranscriptInfo]:
    if min_turns is None and max_turns is None:
        return infos
    filtered: List[TranscriptInfo] = []
    for info in infos:
        t = info.turns if info.turns is not None else -1
        if min_turns is not None and t < min_turns:
            continue
        if max_turns is not None and t > max_turns:
            continue
        filtered.append(info)
    return filtered


def cmd_pick(args: argparse.Namespace) -> None:
    """Interactive transcript picker (aliases: pick, interactive, i).

    Commands inside picker:
      <index>   show transcript
      f<index>  follow transcript (tail)
      r         refresh list
      q         quit picker
    """
    dir_path = discover_directory(args.path)
    console.print("[bold]Interactive transcript picker[/bold] (Ctrl-C to exit)")
    # Pagination state
    current_page = 0
    page_size = args.page_size
    if not page_size or page_size <= 0:
        # auto height: leave some room for header/instructions
        page_size = max(5, console.size.height - 12)
    try:
        while True:
            infos = list_transcripts(dir_path)
            for gi, info in enumerate(infos):
                info.original_index = gi
            compute_turns(infos)
            infos = apply_turn_filters(infos, args.min_turns, args.max_turns)
            total_pages = max(1, (len(infos) + page_size - 1) // page_size)
            if current_page >= total_pages:
                current_page = total_pages - 1
            start = current_page * page_size
            end = start + page_size
            page_slice = infos[start:end]
            console.clear()
            console.print(build_list_table(page_slice, dir_path, show_turns=True, page=current_page, page_size=page_size, total_count=len(infos)))
            console.print("Enter: index (global) | f<index> follow | n/p page | r refresh | q quit")
            inp = input("> ").strip()
            if not inp:
                continue
            if inp.lower() in {"q", "quit", "exit"}:
                return
            if inp.lower() in {"r", "refresh"}:
                continue
            if inp.lower() in {"n", "next"}:
                if current_page < total_pages - 1:
                    current_page += 1
                continue
            if inp.lower() in {"p", "prev", "previous"}:
                if current_page > 0:
                    current_page -= 1
                continue
            follow = False
            if inp.lower().startswith("f"):
                follow = True
                inp = inp[1:]
            if not inp.isdigit():
                console.print("[red]Invalid input.")
                time.sleep(1)
                continue
            idx = int(inp)
            if not (0 <= idx < len(infos)):
                console.print("[red]Index out of range.")
                time.sleep(1)
                continue
            selected_path = infos[idx].path
            if follow:
                try:
                    ns = argparse.Namespace(
                        file=str(selected_path),
                        index=None,
                        roles=None,
                        from_start=False,
                        interval=0.3,
                        width=None,
                        path=None,
                        full_ts=False,
                        color_mode="content",
                        no_color=False,
                    )
                    cmd_follow(ns)
                except KeyboardInterrupt:
                    console.print("\n[dim]Stopped following[/dim]")
            else:
                # Launch persistent interactive transcript view with navigation
                interactive_show(infos, idx, color_mode="content", full_ts=False)
    except KeyboardInterrupt:
        console.print("\n[dim]Picker exited.")


def parse_lines(path: Path) -> Iterator[ParsedLine]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw_strip = raw.rstrip("\n")
            if not raw_strip:
                continue
            try:
                data = json.loads(raw_strip)
                yield ParsedLine(raw=raw_strip, data=data, error=None)
            except Exception as exc:  # noqa: BLE001
                yield ParsedLine(raw=raw_strip, data=None, error=str(exc))


def render_line(pl: ParsedLine, width: int, full_ts: bool = False, color_mode: str = "content") -> Text:
    # Per-line we default to time only; full_ts flag forces full datetime for debugging.
    ts_format = "%Y-%m-%d %H:%M:%S" if full_ts else "%H:%M:%S"
    ts = pl.timestamp.strftime(ts_format) if pl.timestamp else "--:--:--"
    role = pl.role
    style = ROLE_STYLES.get(role, "white")
    content = pl.content
    if pl.error:
        style = "red"
        content = f"[parse error: {pl.error}] {pl.raw}"[: width - 30]
    text = Text(f"{ts} ", style="dim")
    label = role
    if pl.data:
        disp = pl.data.get("speaker_display_name") or pl.data.get("speaker_id")
        if disp:
            label = str(disp)
    text.append(f"{label:<12}"[:12], style=style)
    text.append(" ")
    # Manual wrapping: split content respecting width minus prefix
    avail_width = max(width - 20, 20) if width else None  # rough prefix reserve
    if avail_width and len(content) > avail_width:
        # naive word wrap
        words = content.split()
        line = ""
        first = True
        for w in words:
            if len(line) + 1 + len(w) > avail_width:
                if first:
                    text.append(line, style=style if role in ROLE_STYLES else None)
                    first = False
                else:
                    text.append("\n" + " " * 13 + line, style=style if role in ROLE_STYLES else None)
                line = w
            else:
                line = w if not line else line + " " + w
        if line:
            if first:
                text.append(line, style=style if role in ROLE_STYLES else None)
            else:
                text.append("\n" + " " * 13 + line, style=style if role in ROLE_STYLES else None)
    else:
        text.append(content, style=style if (role in ROLE_STYLES and color_mode in {"content", "line"}) else None)
    # If color_mode == line, recolor whole line (excluding timestamp) by applying style to label already done and optionally to content (handled above).
    return text


def _print_transcript(
    path: Path,
    width: int,
    roles: Optional[List[str]] = None,
    last: Optional[int] = None,
    full_ts: bool = False,
    color_mode: str = "content",
) -> None:
    """Render a transcript similar to cmd_show without exiting afterward."""
    lines = list(parse_lines(path))
    if last and last > 0:
        lines = lines[-last:]
    lines = list(filter_parsed(lines, roles))
    dates = [pl.timestamp.date() for pl in lines if pl.timestamp]
    date_str = "Unknown date"
    if dates:
        first = dates[0]
        lastd = dates[-1]
        date_str = str(first) if first == lastd else f"{first} -> {lastd}"
    console.print(f"[bold]Transcript:[/bold] {path.name}  [dim]{date_str}[/dim]")
    for pl in lines:
        console.print(render_line(pl, width, full_ts=full_ts, color_mode=color_mode))


def interactive_show(
    infos: List[TranscriptInfo],
    start_index: int,
    color_mode: str = "content",
    full_ts: bool = False,
) -> None:
    """Interactive transcript viewer with next/prev navigation.

    Keys:
      n / RightArrow  next transcript
      p / LeftArrow   previous transcript
      r               refresh current transcript
      f               follow (tail) current transcript (Ctrl-C to return)
      q / b           back to picker list
    """
    try:
        import termios  # noqa: PLC0415
        import tty  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        console.print("[red]Raw terminal mode unsupported; showing single transcript[/red]")
        if 0 <= start_index < len(infos):
            _print_transcript(infos[start_index].path, console.width, full_ts=full_ts, color_mode=color_mode)
        return

    current = start_index
    width = console.width

    def read_key() -> str:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == "\x1b":  # escape sequence
                seq = sys.stdin.read(2)  # e.g. [C / [D
                return ch + seq
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    while True:
        if not (0 <= current < len(infos)):
            return
        console.clear()
        console.print(
            f"[reverse] Viewing {current+1}/{len(infos)} (global index {infos[current].original_index}) [/reverse]"
        )
        _print_transcript(
            infos[current].path,
            width,
            roles=None,
            last=None,
            full_ts=full_ts,
            color_mode=color_mode,
        )
        console.print("[dim]Keys: n/→ next | p/← prev | r refresh | f follow | q back[/dim]")
        key = read_key()
        if key in {"q", "b"}:
            return
        if key in {"n", "\x1b[C"}:
            if current < len(infos) - 1:
                current += 1
            continue
        if key in {"p", "\x1b[D"}:
            if current > 0:
                current -= 1
            continue
        if key == "r":
            continue
        if key == "f":
            console.print("\n[bold]Follow mode (Ctrl-C to stop)...[/bold]")
            try:
                ns = argparse.Namespace(
                    file=str(infos[current].path),
                    index=None,
                    roles=None,
                    from_start=False,
                    interval=0.3,
                    width=width,
                    path=None,
                    full_ts=full_ts,
                    color_mode=color_mode,
                    no_color=False,
                )
                cmd_follow(ns)
            except KeyboardInterrupt:
                console.print("[dim]\nExited follow mode[/dim]")
            continue
        # ignore other keys


def cmd_list(args: argparse.Namespace) -> None:
    dir_path = discover_directory(args.path)
    # Determine effective page size (non-positive -> auto height, None -> default 25)
    page_size = args.page_size
    if page_size is None:
        page_size = 25
    elif page_size <= 0:
        # derive from terminal height leaving room for header/title
        try:
            page_size = max(5, console.size.height - 12)
        except Exception:  # noqa: BLE001
            page_size = 25
    page = getattr(args, "page", 0) or 0
    if args.watch:
        refresh = args.refresh
        with Live(console=console, refresh_per_second=min(10, int(1 / refresh) if refresh >= 0.1 else 4)) as live:
            last_files = set()
            while True:
                infos = list_transcripts(dir_path)
                # assign original indices before filtering
                for gi, info in enumerate(infos):
                    info.original_index = gi
                if not args.no_turns:
                    compute_turns(infos, all_roles=args.all_roles)
                    infos = apply_turn_filters(infos, args.min_turns, args.max_turns)
                # Pagination after filtering
                total_pages = max(1, (len(infos) + page_size - 1) // page_size)
                if page >= total_pages:
                    page = total_pages - 1
                start = page * page_size
                end = start + page_size
                page_infos = infos[start:end]
                current_files = {i.path for i in infos}
                table = build_list_table(
                    page_infos,
                    dir_path,
                    show_turns=not args.no_turns,
                    use_original_index=True,
                    page=page,
                    page_size=page_size,
                    total_count=len(infos),
                )
                if new := current_files - last_files:
                    table.caption = f"New: {', '.join(p.name for p in new)}"
                if args.min_turns or args.max_turns:
                    flt = []
                    if args.min_turns:
                        flt.append(f"min={args.min_turns}")
                    if args.max_turns:
                        flt.append(f"max={args.max_turns}")
                    table.caption = (table.caption or "") + f"  Filter({' '.join(flt)})"
                # Add paging hint
                if total_pages > 1:
                    hint = f" --page <0-{total_pages-1}> --page-size {page_size}"
                    table.caption = (table.caption or "") + hint
                live.update(table)
                last_files = current_files
                time.sleep(refresh)
    else:
        infos = list_transcripts(dir_path)
        for gi, info in enumerate(infos):
            info.original_index = gi
        if not args.no_turns:
            compute_turns(infos, all_roles=args.all_roles)
            infos = apply_turn_filters(infos, args.min_turns, args.max_turns)
        # Pagination after filtering
        total_pages = max(1, (len(infos) + page_size - 1) // page_size)
        if page >= total_pages:
            page = total_pages - 1
        start = page * page_size
        end = start + page_size
        page_infos = infos[start:end]
        table = build_list_table(
            page_infos,
            dir_path,
            show_turns=not args.no_turns,
            use_original_index=True,
            page=page,
            page_size=page_size,
            total_count=len(infos),
        )
        if args.min_turns or args.max_turns:
            flt = []
            if args.min_turns:
                flt.append(f"min={args.min_turns}")
            if args.max_turns:
                flt.append(f"max={args.max_turns}")
            table.caption = (table.caption or "") + f" Filter({' '.join(flt)})"
        if total_pages > 1:
            hint = f"  --page <0-{total_pages-1}> --page-size {page_size}"
            table.caption = (table.caption or "") + hint
        console.print(table)


def resolve_target(infos: List[TranscriptInfo], index: Optional[int], file: Optional[str]) -> Path:
    if file:
        p = Path(file)
        if not p.is_file():
            raise SystemExit(f"File not found: {file}")
        return p
    if index is not None:
        if not (0 <= index < len(infos)):
            raise SystemExit(f"Index out of range (0..{len(infos)-1})")
        return infos[index].path
    if not infos:
        raise SystemExit("No transcripts available")
    return infos[0].path  # newest


def filter_parsed(lines: Iterable[ParsedLine], roles: Optional[List[str]]) -> Iterator[ParsedLine]:
    if not roles:
        yield from lines
        return
    role_set = {r.lower() for r in roles}
    for pl in lines:
        if pl.role.lower() in role_set:
            yield pl


def cmd_show(args: argparse.Namespace) -> None:
    dir_path = discover_directory(args.path)
    infos = list_transcripts(dir_path)
    try:
        target = resolve_target(infos, args.index, args.file)
    except SystemExit as e:
        console.print(f"[red]{e}")
        return
    lines = list(parse_lines(target))
    if args.last and args.last > 0:
        lines = lines[-args.last :]
    lines = list(filter_parsed(lines, args.roles))
    if not lines:
        console.print("[yellow]No matching lines.")
        return
    width = args.width or console.width
    # Header with full date(s) and file name
    dates = [pl.timestamp.date() for pl in lines if pl.timestamp]
    date_str = "Unknown date"
    if dates:
        first = dates[0]
        last = dates[-1]
        date_str = str(first) if first == last else f"{first} -> {last}"
    console.print(f"[bold]Transcript:[/bold] {target.name}  [dim]{date_str}[/dim]")
    for pl in lines:
        rendered = render_line(pl, width, full_ts=args.full_ts, color_mode=args.color_mode)
        console.print(rendered)


def tail_file(path: Path, from_start: bool, interval: float) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        if not from_start:
            f.seek(0, os.SEEK_END)
        buf = ""
        while True:
            chunk = f.read()
            if chunk:
                buf += chunk
                while True:
                    if "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        if line:
                            yield line
                    else:
                        break
            else:
                # detect rotation
                try:
                    if not path.exists():
                        return
                except OSError:
                    return
                time.sleep(interval)


def cmd_follow(args: argparse.Namespace) -> None:
    dir_path = discover_directory(args.path)
    infos = list_transcripts(dir_path)
    try:
        target = resolve_target(infos, args.index, args.file)
    except SystemExit as e:
        console.print(f"[red]{e}")
        return
    # Print header once
    console.print(f"[bold]Following:[/bold] {target.name}  (Ctrl-C to stop)")
    width = args.width or console.width
    try:
        for raw in tail_file(target, args.from_start, args.interval):
            try:
                data = json.loads(raw)
                pl = ParsedLine(raw=raw, data=data, error=None)
            except Exception as exc:  # noqa: BLE001
                pl = ParsedLine(raw=raw, data=None, error=str(exc))
            if args.roles and pl.role.lower() not in {r.lower() for r in args.roles}:
                continue
            console.print(render_line(pl, width, full_ts=args.full_ts, color_mode=args.color_mode))
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.")


def cmd_follow_latest(args: argparse.Namespace) -> None:
    """Continuously follow the latest transcript, auto-switching to new transcripts when they appear."""
    dir_path = discover_directory(args.path)
    width = args.width or console.width
    current_path: Optional[Path] = None
    file_handle = None
    buffer = ""
    roles_filter = {r.lower() for r in args.roles} if args.roles else None
    interval = args.interval
    check_every = args.check_new
    last_check = 0.0
    try:
        while True:
            now = time.time()
            # Periodically check for a newer transcript
            if current_path is None or (now - last_check) >= check_every:
                infos = list_transcripts(dir_path)
                if infos:
                    newest = infos[0].path
                    if current_path is None or newest != current_path:
                        # Switch
                        if file_handle:
                            try:
                                file_handle.close()
                            except Exception:  # noqa: BLE001
                                pass
                        current_path = newest
                        try:
                            file_handle = current_path.open("r", encoding="utf-8", errors="replace")
                            if not args.from_start:
                                file_handle.seek(0, os.SEEK_END)
                            buffer = ""
                            # Horizontal separator to delineate transcripts
                            console.print("\n[dim]──────────────────────────────────────────────────────────────────[/dim]")
                            console.print(f"[bold yellow]Switched to:[/bold yellow] {current_path.name}")
                        except Exception as exc:  # noqa: BLE001
                            console.print(f"[red]Failed to open {current_path.name}: {exc}")
                            current_path = None
                            file_handle = None
                last_check = now
            if not file_handle:
                time.sleep(interval)
                continue
            # Read new data
            chunk = file_handle.read()
            if chunk:
                buffer += chunk
                while True:
                    if "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            pl = ParsedLine(raw=line, data=data, error=None)
                        except Exception as exc:  # noqa: BLE001
                            pl = ParsedLine(raw=line, data=None, error=str(exc))
                        if roles_filter and pl.role.lower() not in roles_filter:
                            continue
                        console.print(
                            render_line(
                                pl,
                                width,
                                full_ts=args.full_ts,
                                color_mode=args.color_mode,
                            )
                        )
                    else:
                        break
            else:
                # Detect rotation or deletion of current file earlier than switch interval
                if not current_path.exists():  # type: ignore[union-attr]
                    console.print("[dim]Current transcript disappeared; rechecking...[/dim]")
                    current_path = None
                    if file_handle:
                        try:
                            file_handle.close()
                        except Exception:  # noqa: BLE001
                            pass
                        file_handle = None
                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped stream mode.[/dim]")
    finally:
        if file_handle:
            try:
                file_handle.close()
            except Exception:  # noqa: BLE001
                pass


def cmd_color_test(args: argparse.Namespace) -> None:
    """Print diagnostic information about color capability and sample styles."""
    console.print("[bold underline]Color Diagnostics[/bold underline]")
    console.print(f"Terminal reported color system: [bold]{console.color_system}[/bold]")
    console.print(f"isatty(stdout)={sys.stdout.isatty()}  isatty(stderr)={sys.stderr.isatty()}")
    console.print(f"TERM={os.environ.get('TERM')}")
    console.print("Force color flag active? " + ("yes" if getattr(args, 'force_color', False) else "no"))
    console.print("\n[bold]Role Styles:[/bold]")
    for role, style in ROLE_STYLES.items():
        console.print(f"  {role:<10} -> [{style}]{style} sample text[/]")
    console.print("\n[bold]ANSI 0-15 test:[/bold]")
    for i in range(16):
        console.print(f"[{i}] color index {i}", style=f"color({i})")
    console.print("\n[bold]256-color gradient sample:[/bold]")
    gradient = "".join(f"[color({i})]█" for i in range(16, 256, 8)) + "[/]"
    console.print(gradient)
    console.print("\nIf everything is grey, your terminal emulator or SSH settings may be stripping ANSI; try: ssh -tt, or ensure no `NO_COLOR` env var is set.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="experimance-transcripts", description="Transcript viewer")
    p.add_argument("--force-color", action="store_true", help="Force color even if stdout not a TTY (useful when piping to head)")
    p.add_argument("--path", help="Transcript directory (default auto-detect)")
    sub = p.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List available transcripts (default if no command)")
    p_list.add_argument("--watch", action="store_true", help="Refresh list continuously")
    p_list.add_argument("--refresh", type=float, default=2.0, help="Refresh interval seconds")
    p_list.add_argument("--min-turns", type=int, help="Filter: minimum turns")
    p_list.add_argument("--max-turns", type=int, help="Filter: maximum turns")
    p_list.add_argument("--all-roles", action="store_true", help="Count every line as a turn (instead of user/assistant only)")
    p_list.add_argument("--no-turns", action="store_true", help="Skip turn counting (faster)")
    p_list.add_argument("--page-size", type=int, help="Page size (default 25; 0 = auto height)")
    p_list.add_argument("--page", type=int, default=0, help="Zero-based page number to display")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Show (pretty print) a transcript")
    p_show.add_argument("index", nargs="?", type=int, help="Transcript index (from list)")
    p_show.add_argument("--file", help="Explicit transcript file path")
    p_show.add_argument("--roles", nargs="*", help="Filter roles (space separated)")
    p_show.add_argument("--last", type=int, help="Show only last N lines")
    p_show.add_argument("--width", type=int, help="Wrap width")
    p_show.add_argument("--full-ts", action="store_true", help="Show full datetime in timestamps")
    p_show.add_argument("--color-mode", choices=["content", "line", "label"], default="content", help="Coloring strategy: content (default), line (label+content), label (label only)")
    p_show.add_argument("--no-color", action="store_true", help="Disable role coloring (still uses dim timestamp)")
    p_show.set_defaults(func=cmd_show)

    p_follow = sub.add_parser("follow", help="Tail -f a transcript")
    p_follow.add_argument("index", nargs="?", type=int, help="Transcript index (from list)")
    p_follow.add_argument("--file", help="Explicit transcript file path")
    p_follow.add_argument("--roles", nargs="*", help="Filter roles")
    p_follow.add_argument("--from-start", action="store_true", help="Start at beginning")
    p_follow.add_argument("--interval", type=float, default=0.3, help="Polling interval seconds")
    p_follow.add_argument("--width", type=int, help="Wrap width")
    p_follow.add_argument("--full-ts", action="store_true", help="Show full datetime in timestamps")
    p_follow.add_argument("--color-mode", choices=["content", "line", "label"], default="content")
    p_follow.add_argument("--no-color", action="store_true", help="Disable role coloring")
    p_follow.set_defaults(func=cmd_follow)

    p_follow_latest = sub.add_parser("stream", aliases=["latest"], help="Follow the latest transcript, auto-switching when new ones start")
    p_follow_latest.add_argument("--roles", nargs="*", help="Filter roles")
    p_follow_latest.add_argument("--from-start", action="store_true", help="Start at beginning of the first transcript")
    p_follow_latest.add_argument("--interval", type=float, default=0.3, help="Polling interval seconds for reading lines")
    p_follow_latest.add_argument("--check-new", type=float, default=2.0, help="Interval seconds to check for newer transcript files")
    p_follow_latest.add_argument("--width", type=int, help="Wrap width")
    p_follow_latest.add_argument("--full-ts", action="store_true", help="Show full datetime in timestamps")
    p_follow_latest.add_argument("--color-mode", choices=["content", "line", "label"], default="content")
    p_follow_latest.add_argument("--no-color", action="store_true", help="Disable role coloring")
    p_follow_latest.set_defaults(func=cmd_follow_latest)

    p_pick = sub.add_parser("pick", aliases=["i", "interactive"], help="Interactive picker (choose transcript to show or follow)")
    p_pick.add_argument("--min-turns", type=int, help="Filter: minimum turns")
    p_pick.add_argument("--max-turns", type=int, help="Filter: maximum turns")
    p_pick.add_argument("--all-roles", action="store_true", help="Count every line as a turn")
    p_pick.add_argument("--refresh", type=float, default=3.0, help="Refresh interval seconds")
    p_pick.add_argument("--page-size", type=int, help="Page size (default auto from terminal height; 0 = auto)")
    p_pick.set_defaults(func=cmd_pick)

    p_color = sub.add_parser("color-test", help="Show sample colors and diagnostic info")
    p_color.set_defaults(func=cmd_color_test)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    global console  # noqa: PLW0603
    if getattr(args, "force_color", False):
        console = Console(force_terminal=True, color_system="truecolor")
    # If no-color specified for subcommands we neutralize role styles by setting all to white
    if hasattr(args, "no_color") and args.no_color:
        for k in ROLE_STYLES.keys():
            ROLE_STYLES[k] = "white"
        if getattr(args, "color_mode", None):
            args.color_mode = "label"  # minimal coloring
    # Default color_mode for commands without explicit flag
    if not hasattr(args, "color_mode"):
        args.color_mode = "content"
    if console.color_system is None:
        console.print("[yellow]Warning: Color system not detected (TERM may be 'dumb' or output not a TTY). Use --force-color for ANSI output.[/yellow]")
    if not args.command or args.command == "list":
        # default command is list
        if not args.command:
            args.watch = False
            args.refresh = 2.0
            args.no_turns = False
            args.min_turns = None
            args.max_turns = None
            args.all_roles = False
            args.func = cmd_list
        args.func(args)
        return 0
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
