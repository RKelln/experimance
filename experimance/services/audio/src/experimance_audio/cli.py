"""
Command-line interface for testing the Experimance audio system.

This module provides a command-line interface for manually sending OSC commands
to SuperCollider, allowing for interactive testing and development.
"""

import argparse
import asyncio
import cmd
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Any, Optional

from .osc_bridge import OscBridge
from .config_loader import AudioConfigLoader

from experimance_common.constants import DEFAULT_PORTS

# Configure logging
logger = logging.getLogger(__name__)


class AudioCli(cmd.Cmd):
    """Interactive CLI for testing the Experimance audio system."""
    
    intro = "Experimance Audio CLI. Type 'help' or '?' to list commands."
    prompt = "audio> "
    
    def __init__(self, osc_host: str = "localhost", 
                 osc_port: int = DEFAULT_PORTS['audio_osc_recv_port'], 
                 config_dir: Optional[str] = None, 
                 sc_script_path: Optional[str] = None, 
                 sclang_path: str = "sclang"):
        """Initialize the audio CLI.
        
        Args:
            osc_host: SuperCollider host address
            osc_port: SuperCollider OSC listening port
            config_dir: Directory containing audio configuration files
            sc_script_path: Path to the SuperCollider script to execute
            sclang_path: Path to the SuperCollider language interpreter executable
        """
        super().__init__()
        self.osc = OscBridge(host=osc_host, port=osc_port)
        self.config = AudioConfigLoader(config_dir=config_dir)
        self.config.load_configs()
        
        # Store SC paths for later use
        self.sc_script_path = sc_script_path
        self.sclang_path = sclang_path
        
        # Track current state
        self.current_biome = "temperate_forest"  # Default biome
        self.current_era = "wilderness"  # Default era
        self.active_tags = set([self.current_biome, self.current_era])
        
        # Track volume levels (0.0 to 1.0)
        self.master_volume = 1.0
        self.environment_volume = 1.0
        self.music_volume = 1.0
        self.sfx_volume = 1.0
    
    def do_start_sc(self, arg):
        """Start SuperCollider with the default script.
        
        Usage: start_sc [script_path]
        Example: start_sc
        """
        script_path = arg.strip() if arg else self.sc_script_path
        
        # Try to find the default script in the sc_scripts directory
        import os
        from pathlib import Path
        
        # Get the directory of the current file
        current_dir = Path(__file__).parent.parent.parent  # Go up from src/experimance_audio to services/audio
        sc_scripts_dir = current_dir / "sc_scripts"

        if not script_path:
            default_script = sc_scripts_dir / "experimance_audio.scd"
            
            if default_script.exists():
                script_path = str(default_script)
            else:
                print("Error: No SuperCollider script path specified and default script not found.")
                print(f"Looked for default script at: {default_script}")
                return
        else: 
            # if just a filename is provided, look in the sc_scripts directory
            script_path = Path(script_path)
            if not script_path.is_absolute():
                # If it's a relative path, assume it's in the sc_scripts directory
                script_path = sc_scripts_dir / script_path
            if not script_path.exists():
                print(f"Error: SuperCollider script not found at {script_path}")
                return
            
        print(f"Starting SuperCollider with script: {script_path}")
        # Start SuperCollider with logging to file (not console)
        self.sc_log_file_path = self.osc.start_supercollider(
            str(script_path), 
            str(self.sclang_path),
            log_to_file=True,
            log_to_console=False
        )
        
        if self.sc_log_file_path:
            print(f"Started SuperCollider (success)")
            print(f"Logs are being written to: {self.sc_log_file_path}")
            print(f"To view logs in real-time, open another terminal and run:")
            print(f"  tail -f {self.sc_log_file_path}")
        else:
            print(f"Started SuperCollider (failed)")
        
    def do_stop_sc(self, arg):
        """Stop the SuperCollider process.
        
        Usage: stop_sc
        """
        success = self.osc.stop_supercollider()
        print(f"Stopped SuperCollider" + (" (success)" if success else " (failed)"))
    
    def do_spacetime(self, arg):
        """Set the current spacetime context (biome and era).
        
        Usage: spacetime <biome> <era>
        Example: spacetime desert pre_industrial
        """
        args = arg.split()
        if len(args) != 2:
            print("Usage: spacetime <biome> <era>")
            return
            
        biome, era = args
        self.current_biome = biome
        self.current_era = era
        
        # Send to SuperCollider
        success = self.osc.send_spacetime(biome, era)
        
        # Update active tags
        self.active_tags.clear()
        self.active_tags.add(biome)
        self.active_tags.add(era)
        
        # Include the default tags
        for tag in self.active_tags:
            self.osc.include_tag(tag)
            
        print(f"Spacetime set to biome={biome}, era={era}" + (" (success)" if success else " (failed)"))
    
    def do_include(self, arg):
        """Include a sound tag in the active set.
        
        Usage: include <tag>
        Example: include birds
        """
        if not arg:
            print("Usage: include <tag>")
            return
            
        tag = arg.strip()
        success = self.osc.include_tag(tag)
        
        if success:
            self.active_tags.add(tag)
            
        print(f"Included tag: {tag}" + (" (success)" if success else " (failed)"))
    
    def do_exclude(self, arg):
        """Exclude a sound tag from the active set.
        
        Usage: exclude <tag>
        Example: exclude birds
        """
        if not arg:
            print("Usage: exclude <tag>")
            return
            
        tag = arg.strip()
        success = self.osc.exclude_tag(tag)
        
        if success and tag in self.active_tags:
            self.active_tags.remove(tag)
            
        print(f"Excluded tag: {tag}" + (" (success)" if success else " (failed)"))
    
    def do_listening(self, arg):
        """Trigger listening UI sound effect.
        
        Usage: listening <start|stop>
        Example: listening start
        """
        if arg not in ["start", "stop"]:
            print("Usage: listening <start|stop>")
            return
            
        start = (arg == "start")
        success = self.osc.listening(start)
        
        print(f"Listening: {arg}" + (" (success)" if success else " (failed)"))
    
    def do_speaking(self, arg):
        """Trigger speaking UI sound effect.
        
        Usage: speaking <start|stop>
        Example: speaking start
        """
        if arg not in ["start", "stop"]:
            print("Usage: speaking <start|stop>")
            return
            
        start = (arg == "start")
        success = self.osc.speaking(start)
        
        print(f"Speaking: {arg}" + (" (success)" if success else " (failed)"))
    
    def do_transition(self, arg):
        """Trigger transition sound effect.
        
        Usage: transition <start|stop>
        Example: transition start
        """
        if arg not in ["start", "stop"]:
            print("Error: Please use 'start' or 'stop'")
            return
            
        start = (arg == "start")
        success = self.osc.transition(start)
        
        print(f"Transition: {arg}" + (" (success)" if success else " (failed)"))
    
    def do_volume(self, arg):
        """Set volume levels.
        
        Usage: volume <type> <level>
        Where: <type> is one of: master, environment, music, sfx
               <level> is a float between 0.0 and 1.0
        
        Examples:
          volume master 0.8
          volume music 0.5
          volume environment 0.7
          volume sfx 0.9
        """
        args = arg.split()
        if len(args) != 2:
            print("Error: Please provide both volume type and level")
            print("Usage: volume <type> <level>")
            return
        
        volume_type, level_str = args
        
        if volume_type not in ["master", "environment", "music", "sfx"]:
            print("Error: Volume type must be one of: master, environment, music, sfx")
            return
        
        try:
            level = float(level_str)
            if not (0.0 <= level <= 1.0):
                print("Error: Volume level must be between 0.0 and 1.0")
                return
        except ValueError:
            print("Error: Volume level must be a float")
            return
        
        # Use the appropriate method based on volume type
        success = False
        if volume_type == "master":
            success = self.osc.set_master_volume(level)
            self.master_volume = level  # Track the volume level
        elif volume_type == "environment":
            success = self.osc.set_environment_volume(level)
            self.environment_volume = level  # Track the volume level
        elif volume_type == "music":
            success = self.osc.set_music_volume(level)
            self.music_volume = level  # Track the volume level
        elif volume_type == "sfx":
            success = self.osc.set_sfx_volume(level)
            self.sfx_volume = level  # Track the volume level
        
        print(f"Set {volume_type} volume to {level}" + (" (success)" if success else " (failed)"))
    
    def help_volume(self):
        """Print detailed help for the volume command."""
        print("\nVolume Control Commands")
        print("=====================")
        print("Usage: volume <type> <level>")
        print("  where <type> is one of:")
        print("    master      - controls overall volume")
        print("    environment - controls environmental sound layers")
        print("    music       - controls music loops")
        print("    sfx         - controls sound effects")
        print("  and <level> is a value between 0.0 and 1.0")
        print("\nExamples:")
        print("  volume master 0.8     # Set master volume to 80%")
        print("  volume music 0.5      # Set music volume to 50%")
        print("  volume environment 0.7 # Set environment volume to 70%")
        print("  volume sfx 0.9        # Set sound effects volume to 90%")
        print("\nNote: Volume changes take effect immediately.")
        print("      Master volume affects all other volumes.")
        print("      The actual volume will be: <specific volume> × <master volume>")
    
    def do_reload(self, arg):
        """Reload audio configurations in SuperCollider.
        
        Usage: reload
        """
        # First reload our own configs
        self.config.load_configs()
        
        # Then tell SuperCollider to reload
        success = self.osc.reload_configs()
        
        print("Reloaded audio configurations" + (" (success)" if success else " (failed)"))
    
    def do_status(self, arg):
        """Show current audio system status.
        
        Usage: status
        """
        print("=== Audio System Status ===")
        print(f"Biome: {self.current_biome}")
        print(f"Era: {self.current_era}")
        print(f"Active tags: {', '.join(sorted(self.active_tags))}")
        print(f"OSC connection: {self.osc.host}:{self.osc.port}")
        
        # Get volume information (we just use the stored values as there's no way to query SuperCollider)
        print("\n--- Volume Levels ---")
        print(f"Master volume: {int(self.master_volume * 100)}%")
        print(f"Environment volume: {int(self.environment_volume * 100)}%")
        print(f"Music volume: {int(self.music_volume * 100)}%")
        print(f"SFX volume: {int(self.sfx_volume * 100)}%")
        print("=========================")
    
    def do_layers(self, arg):
        """Show available audio layers.
        
        Usage: layers
        """
        print("=== Available Audio Layers ===")
        for i, layer in enumerate(self.config.layers):
            tags = ", ".join(layer.get("tags", []))
            path = layer.get("path", "unknown")
            prompt = layer.get("prompt", "")
            print(f"{i+1}. {path} [{tags}]")
            if prompt:
                print(f"   Description: {prompt}")
        print("============================")
    
    def do_triggers(self, arg):
        """Show available sound effect triggers.
        
        Usage: triggers
        """
        print("=== Available Sound Effect Triggers ===")
        for i, trigger in enumerate(self.config.triggers):
            name = trigger.get("trigger", "unknown")
            path = trigger.get("path", "unknown")
            prompt = trigger.get("prompt", "")
            print(f"{i+1}. {name}: {path}")
            if prompt:
                print(f"   Description: {prompt}")
        print("=====================================")
    
    def do_music(self, arg):
        """Show available music loops for each era.
        
        Usage: music [era]
        Example: music pre_industrial
        """
        if arg:
            # Show music for specific era
            era = arg.strip()
            loops = self.config.get_music_loops_for_era(era)
            
            if not loops:
                print(f"No music loops found for era: {era}")
                return
                
            print(f"=== Music Loops for Era: {era} ===")
            for i, loop in enumerate(loops):
                path = loop.get("path", "unknown")
                prompt = loop.get("prompt", "")
                print(f"{i+1}. {path}")
                if prompt:
                    print(f"   Description: {prompt}")
            print("===============================")
        else:
            # Show music for all eras
            print("=== Music Loops by Era ===")
            for era, loops in self.config.music_loops.items():
                print(f"\nEra: {era} ({len(loops)} loops)")
                for i, loop in enumerate(loops):
                    path = loop.get("path", "unknown")
                    print(f"  {i+1}. {path}")
            print("==========================")
    
    def do_view_logs(self, arg):
        """Open the SuperCollider log file with tail -f in a new terminal.
        
        Usage: view_logs [log_file_path]
        Example: view_logs
        """
        log_file = arg.strip() if arg else self.sc_log_file_path
        
        if not log_file:
            print("No log file available. Start SuperCollider first with the 'start_sc' command.")
            return
            
        try:
            # Use the system's default terminal to open a new window with tail
            if os.path.exists(log_file):
                # Try different terminal emulators based on what might be available
                for terminal_cmd in ["gnome-terminal", "xterm", "konsole", "terminal"]:
                    try:
                        # Check if the command exists
                        if subprocess.run(["which", terminal_cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
                            # The command exists, use it
                            subprocess.Popen([terminal_cmd, "--", "tail", "-f", log_file])
                            print(f"Opening log viewer using {terminal_cmd}...")
                            return
                    except Exception:
                        continue
                        
                # If we get here, none of the terminals worked
                print(f"Could not open terminal. To view logs manually, run: tail -f {log_file}")
            else:
                print(f"Log file not found: {log_file}")
        except Exception as e:
            print(f"Error opening log viewer: {e}")
            print(f"To view logs manually, run: tail -f {log_file}")
    
    def do_exit(self, arg):
        """Exit the CLI.
        
        Usage: exit
        """
        print("Exiting...")
        return True
        
    def do_quit(self, arg):
        """Exit the CLI.
        
        Usage: quit
        """
        return self.do_exit(arg)
    
    def do_demo(self, arg):
        """Run a quick demo sequence through different eras.
        
        Usage: demo
        """
        # List of eras to cycle through
        eras = ["wilderness", "pre_industrial", "industrial", "current", "ai_future"]
        biomes = ["temperate_forest", "desert", "coastal", "mountains", "plains"]
        
        print("Starting demo sequence...")
        
        for i, era in enumerate(eras):
            biome = biomes[i % len(biomes)]
            
            print(f"\nChanging to era={era}, biome={biome}")
            self.do_spacetime(f"{biome} {era}")
            
            print("Starting transition effect...")
            self.do_transition("start")
            time.sleep(3)
            self.do_transition("stop")
            
            # Pause between eras
            time.sleep(5)
        
        print("\nDemo sequence completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimance Audio CLI")
    parser.add_argument("--osc-host", type=str, default="localhost", help="SuperCollider host address")
    parser.add_argument("--osc-port", type=int, default=DEFAULT_PORTS['audio_osc_recv_port'], help="SuperCollider OSC port")
    parser.add_argument("--config-dir", type=str, help="Directory containing audio configuration files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--sc-script", type=str, help="Path to SuperCollider script to execute")
    parser.add_argument("--sclang-path", type=str, default="sclang", help="Path to SuperCollider language interpreter executable")

    args = parser.parse_args()
    
    # Set log level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.config_dir and not os.path.isdir(args.config_dir):
        print(f"Config directory does not exist: {args.config_dir}")
        sys.exit(1)

    try:
        cli = AudioCli(
            osc_host=args.osc_host, 
            osc_port=args.osc_port, 
            config_dir=args.config_dir,
            sc_script_path=args.sc_script,
            sclang_path=args.sclang_path)
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, exiting")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1)
