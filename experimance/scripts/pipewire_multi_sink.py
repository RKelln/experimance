#!/usr/bin/env python3
"""Combine multiple PipeWire sinks into one multi-channel virtual sink.

This script creates a PipeWire virtual sink that routes different channels to different
physical audio devices. For example, you can create a 4-channel virtual sink where:
- Channels 0,1 (FL,FR) go to your laptop speakers
- Channels 2,3 (RL,RR) go to a Bluetooth speaker

This enables multi-speaker setups for audio applications, installations, and testing.

FEATURES:
- Reliable sink discovery using pw-dump JSON output
- Interactive or non-interactive operation
- Proper channel mapping with standard audio channel labels
- Duplicate name detection and conflict prevention
- Destroy sinks by name or ID (handles duplicates)
- Unlink existing connections to prevent audio doubling
- Set created virtual sink as default output device

REQUIREMENTS:
- PipeWire audio system with tools: pw-dump, pw-cli, pw-link, wpctl
- Python 3.7+ with json, subprocess modules (standard library)

BASIC USAGE:
    # Interactive mode - lists sinks and prompts for selection
    python scripts/pipewire_multi_sink.py
    
    # Non-interactive - combine sinks 0,1 into "Virtual-4ch" and set as default
    python scripts/pipewire_multi_sink.py --name "Virtual-4ch" \\
        --select "0,1" --non-interactive --make-default

ADVANCED EXAMPLES:
    # Create 6-channel setup (laptop + bluetooth + USB speakers)
    python scripts/pipewire_multi_sink.py --name "Virtual-6ch" \\
        --select "0,1,2" --make-default --unlink-existing
    
    # Remove existing virtual sinks
    python scripts/pipewire_multi_sink.py --destroy "Virtual-4ch"
    python scripts/pipewire_multi_sink.py --destroy "123"  # by ID
    
    # List available sinks without creating anything
    python scripts/pipewire_multi_sink.py --name "dummy" --destroy "nonexistent"

CHANNEL MAPPING:
The script creates virtual sinks with standard channel positions:
- 1ch: MONO
- 2ch: FL, FR (Front Left, Front Right)  
- 4ch: FL, FR, RL, RR (Front + Rear Left/Right)
- 6ch: FL, FR, RL, RR, SL, SR (Front + Rear + Side Left/Right)
- 8ch: FL, FR, FC, LFE, RL, RR, SL, SR (surround with center/subwoofer)

Selected hardware sinks contribute their channels in order:
- Sink 0 (2ch) → Virtual channels 0,1
- Sink 1 (2ch) → Virtual channels 2,3
- Sink 2 (2ch) → Virtual channels 4,5
- etc.

INTEGRATION WITH TEST SCRIPTS:
After creating a virtual sink, test it with:
    python scripts/test_multi_channel_audio.py -v
    
In the test interface:
- 't 0' plays test tone on first hardware sink
- 't 2' plays test tone on second hardware sink
- 'i 0,1,2,3' enters interactive delay calibration mode

TROUBLESHOOTING:
- If audio doubles: use --unlink-existing to remove conflicting connections
- If sink creation fails: check that the name doesn't already exist
- If no sinks found: ensure PipeWire is running and pw-dump works
- If test script doesn't use virtual sink: verify it's set as default with wpctl

TECHNICAL DETAILS:
1. Uses pw-dump for reliable JSON-based sink discovery
2. Creates null-audio-sink via pw-cli with proper channel mapping
3. Links virtual sink monitor ports to hardware sink playback ports
4. Handles PipeWire's asynchronous node creation with retries
5. Parses pw-link output to manage existing connections

See also: scripts/test_multi_channel_audio.py for testing and calibration."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ----------------------------- Utilities ----------------------------------

def _require(cmd: str) -> None:
    if shutil.which(cmd) is None:
        print(f"Error: '{cmd}' not found in PATH. Install PipeWire tools.")
        sys.exit(1)


def run(cmd: List[str], check: bool = True, capture: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, capture_output=capture, text=text)


def pw_dump() -> List[dict]:
    try:
        out = run(["pw-dump"]).stdout
        return json.loads(out or "[]")
    except Exception:
        return []


# ---------------------------- Data models ----------------------------------

@dataclass
class Sink:
    id: int
    name: str
    description: str
    alias: Optional[str]
    playback_ports: List[dict]  # list of Port dicts

    @property
    def channels(self) -> int:
        return len(self.playback_ports)


# ----------------------------- Discovery -----------------------------------

def discover_sinks(exclude_names: Optional[List[str]] = None) -> List[Sink]:
    exclude_names = exclude_names or []
    data = pw_dump()
    if not data:
        return []

    nodes: Dict[int, dict] = {}
    ports_by_node: Dict[int, List[dict]] = {}

    for obj in data:
        if obj.get("type") == "PipeWire:Interface:Node":
            props = obj.get("info", {}).get("props", {}) or obj.get("props", {})
            if props.get("media.class") == "Audio/Sink":
                nodes[obj["id"]] = obj
        elif obj.get("type") == "PipeWire:Interface:Port":
            # Try multiple ways to get node_id
            node_id = obj.get("info", {}).get("props", {}).get("node.id") or obj.get("props", {}).get("node.id") or obj.get("node.id")
            if node_id is not None:
                try:
                    ports_by_node.setdefault(int(node_id), []).append(obj)
                except Exception:
                    pass

    sinks: List[Sink] = []
    for node_id, node in nodes.items():
        props = node.get("info", {}).get("props", {}) or node.get("props", {})
        name = props.get("node.name", f"node-{node_id}")
        desc = props.get("node.description", name)
        alias = props.get("node.nick") or props.get("device.description") or desc

        if any(n.lower() in name.lower() or n.lower() in desc.lower() for n in exclude_names):
            continue

        # Collect playback ports (direction=in) with audio channel
        pports = []
        for p in ports_by_node.get(node_id, []):
            port_props = p.get("info", {}).get("props", {}) or p.get("props", {})
            port_direction = port_props.get("port.direction") or p.get("port.direction")
            format_dsp = port_props.get("format.dsp") or p.get("format.dsp", "")
            port_name = port_props.get("port.name") or p.get("port.name", "")
            
            if str(port_direction) == "in" and "audio" in str(format_dsp):
                if str(port_name).startswith("playback_"):
                    # Add more properties for sorting
                    audio_channel = port_props.get("audio.channel") or p.get("audio.channel", "")
                    port_alias = port_props.get("port.alias") or p.get("port.alias")
                    
                    pports.append({
                        "port.name": port_name,
                        "audio.channel": audio_channel,
                        "port.alias": port_alias,
                        "port.direction": port_direction,
                        "format.dsp": format_dsp
                    })

        # Sort ports by a stable channel order
        def ch_key(port: dict) -> Tuple[int, str]:
            order = [
                "FL","FR","FC","LFE","RL","RR","SL","SR",
                "FLC","FRC","RLC","RRC","TFL","TFR","TRL","TRR",
            ]
            ch = port.get("audio.channel", "")
            try:
                idx = order.index(ch)
            except ValueError:
                idx = 100
            return (idx, ch)

        pports.sort(key=ch_key)
        sinks.append(Sink(id=int(node_id), name=name, description=desc, alias=alias, playback_ports=pports))

    return [s for s in sinks if s.channels > 0]


# ----------------------------- Creation ------------------------------------

def channel_labels(n: int) -> List[str]:
    # Provide common sensible labels; fallback to AUXi
    presets = {
        1: ["MONO"],
        2: ["FL","FR"],
        4: ["FL","FR","RL","RR"],
        6: ["FL","FR","RL","RR","SL","SR"],
        8: ["FL","FR","FC","LFE","RL","RR","SL","SR"],
    }
    if n in presets:
        return presets[n]
    return [f"AUX{i}" for i in range(n)]


def create_virtual_sink(name: str, total_channels: int, rate: int) -> int:
    positions = " ".join(channel_labels(total_channels))
    props = (
        "{ factory.name=support.null-audio-sink "
        f"node.name=\"{name}\" node.description=\"{name}\" "
        "media.class=Audio/Sink object.linger=true "
        f"audio.channels={total_channels} audio.position=[ {positions} ] audio.rate={rate} }}"
    )
    # Create node
    run(["pw-cli", "create-node", "adapter", props], check=True, capture=False)

    # Find the node id
    time.sleep(0.2)
    for s in discover_sinks():
        if s.name == name or s.description == name:
            return s.id
    raise RuntimeError("Failed to locate created virtual sink")


def set_default_sink(node_id: int) -> None:
    run(["wpctl", "set-default", str(node_id)], check=True, capture=False)


def link_virtual_to_sinks(virtual_name: str, selected: List[Sink]) -> None:
    # Build targets list of port.alias strings in the desired order
    targets: List[str] = []
    for s in selected:
        for p in s.playback_ports:
            alias = p.get("port.alias") or f"{s.description}:{p.get('port.name')}"
            targets.append(alias)

    # For each virtual monitor port, link to the corresponding sink playback port
    for idx, alias in enumerate(targets):
        src = f"{virtual_name}:monitor_{idx}"
        run(["pw-link", src, alias], check=False, capture=False)


def _current_links() -> Dict[str, List[str]]:
    """Parse 'pw-link -l' into a mapping: input_port -> [output_ports]."""
    blocks = run(["pw-link", "-l"]).stdout.split("\n\n")
    mapping: Dict[str, List[str]] = {}
    for b in blocks:
        lines = [l for l in b.splitlines() if l.strip()]
        if not lines:
            continue
        head = lines[0].strip()
        # head format: '<source_port>'
        source = head
        for l in lines[1:]:
            ls = l.strip()
            if ls.startswith("|-> "):
                target = ls[4:].strip()
                mapping.setdefault(target, []).append(source)
            elif ls.startswith("|<- "):
                # reverse arrow variant: target block
                source2 = ls[4:].strip()
                mapping.setdefault(head, []).append(source2)
    return mapping


def unlink_existing_to_selected(selected: List[Sink]) -> None:
    linkmap = _current_links()
    for s in selected:
        for p in s.playback_ports:
            alias = p.get("port.alias") or f"{s.description}:{p.get('port.name')}"
            outputs = linkmap.get(alias, [])
            for outp in outputs:
                # Delete link: outp -> alias
                run(["pw-link", "-d", outp, alias], check=False, capture=False)


def destroy_virtual_by_name_or_id(name_or_id: str) -> int:
    """Destroy virtual sink by name or ID. Returns count of destroyed sinks."""
    sinks = discover_sinks()
    destroyed = 0
    
    # Try as ID first
    try:
        target_id = int(name_or_id)
        for s in sinks:
            if s.id == target_id:
                try:
                    run(["pw-cli", "destroy", str(s.id)], check=True, capture=False)
                    destroyed += 1
                    print(f"Destroyed sink ID {s.id}: {s.description}")
                except subprocess.CalledProcessError:
                    pass
        if destroyed > 0:
            return destroyed
    except ValueError:
        pass
    
    # Try as name - destroy ALL matching names
    for s in sinks:
        if s.name == name_or_id or s.description == name_or_id:
            try:
                run(["pw-cli", "destroy", str(s.id)], check=True, capture=False)
                destroyed += 1
                print(f"Destroyed sink ID {s.id}: {s.description}")
            except subprocess.CalledProcessError:
                pass
    
    return destroyed


# ------------------------------- CLI ---------------------------------------

def main() -> None:
    _require("pw-dump")
    _require("pw-cli")
    _require("pw-link")
    _require("wpctl")

    ap = argparse.ArgumentParser(description="Combine PipeWire sinks into a multi-channel virtual sink")
    ap.add_argument("--name", default="Virtual-Multi", help="Name for the virtual sink")
    ap.add_argument("--rate", type=int, default=48000, help="Sample rate for the virtual sink")
    ap.add_argument("--make-default", action="store_true", help="Set the virtual sink as default output")
    ap.add_argument("--unlink-existing", action="store_true", help="Unlink any existing connections to selected sinks' playback ports before wiring")
    ap.add_argument("--destroy", metavar="NAME_OR_ID", help="Destroy virtual sink(s) by name or ID and exit. Destroys ALL sinks with matching name.")
    ap.add_argument("--non-interactive", action="store_true", help="Do not prompt; use --select indices")
    ap.add_argument("--select", help="Comma-separated sink indices to combine, in order")
    args = ap.parse_args()

    if args.destroy:
        destroyed_count = destroy_virtual_by_name_or_id(args.destroy)
        print(f"Destroyed: {destroyed_count} sink(s)")
        return

    # Discover sinks, exclude any existing virtual with the target name
    sinks = discover_sinks(exclude_names=[args.name])
    if not sinks:
        print("No sinks discovered.")
        return

    # Check for name conflicts
    all_sinks = discover_sinks()  # Don't exclude to check for conflicts
    existing_names = {s.name for s in all_sinks} | {s.description for s in all_sinks}
    if args.name in existing_names:
        print(f"Error: A sink with name '{args.name}' already exists.")
        print("Use --destroy to remove it first, or choose a different name.")
        return

    # List sinks (show duplicates with unique index)
    print("Available PipeWire sinks:")
    for i, s in enumerate(sinks):
        print(f"  [{i}] id={s.id:3d}  {s.description}  ({s.channels} ch)  name={s.name}")

    # Choose selection
    if args.non_interactive and args.select:
        sel_indices = [int(x.strip()) for x in args.select.split(",") if x.strip()]
    else:
        raw = input("Select sink indices to combine (comma-separated, in order): ").strip()
        sel_indices = [int(x.strip()) for x in raw.split(",") if x.strip()]

    selected: List[Sink] = []
    for idx in sel_indices:
        if 0 <= idx < len(sinks):
            selected.append(sinks[idx])
        else:
            print(f"Index out of range: {idx}")
            return

    total_channels = sum(s.channels for s in selected)
    print(f"Creating virtual sink '{args.name}' with {total_channels} channels at {args.rate} Hz ...")
    node_id = create_virtual_sink(args.name, total_channels, args.rate)
    print(f"Created virtual sink id={node_id}")

    print("Linking channels:")
    cursor = 0
    for s in selected:
        chs = s.channels
        print(f"  Virtual [{cursor}..{cursor+chs-1}] -> {s.description} ({chs} ch)")
        cursor += chs
    if args.unlink_existing:
        print("Unlinking existing connections to selected sinks...")
        unlink_existing_to_selected(selected)
    link_virtual_to_sinks(args.name, selected)

    if args.make_default:
        set_default_sink(node_id)
        print("Set as default output.")

    print("Done. Verify with: wpctl status and pw-link -l")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(e.cmd)}\n{e.stderr}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
