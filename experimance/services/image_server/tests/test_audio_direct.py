#!/usr/bin/env python3
"""
Direct Audio Generation Testing Tool

This tool tests TangoFlux audio generation directly without ZMQ,
useful for debugging and development of the audio generator itself.

Usage:
    uv run python services/image_server/tests/test_audio_direct.py
    uv run python services/image_server/tests/test_audio_direct.py --interactive
    uv run python services/image_server/tests/test_audio_direct.py --prompt "gentle rain" --duration 5
"""

import argparse
import asyncio
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict

# Try to import audio generation capabilities
try:
    from image_server.generators.audio.prompt2audio import Prompt2AudioGenerator
    from image_server.generators.audio.audio_config import Prompt2AudioConfig
    AUDIO_GENERATION_AVAILABLE = True
except ImportError:
    AUDIO_GENERATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("audio_direct_test")

# Define sample audio prompts for environmental sounds
SAMPLE_AUDIO_PROMPTS = {
    "forest_ambience": "gentle forest sounds with birds chirping and rustling leaves",
    "ocean_waves": "peaceful ocean waves lapping against the shore with distant seagulls",
    "rain_on_leaves": "light rain falling on forest leaves with distant thunder",
    "mountain_stream": "babbling brook flowing over rocks in a mountain valley",
    "campfire": "crackling campfire with gentle flames and occasional wood pops",
    "desert_wind": "soft desert wind blowing through sand dunes with distant howling",
    "cave_drips": "water droplets echoing in a deep cave with ambient reverb",
    "grassland_breeze": "gentle wind through tall grass with chirping insects",
    "ice_cave": "subtle ice creaking and wind in a frozen glacial cave",
    "tropical_jungle": "dense jungle with exotic birds, insects, and rustling foliage",
    "arctic_wind": "cold arctic wind with snow blowing and ice shifting",
    "urban_park": "city park ambience with distant traffic and nearby bird songs",
}


async def interactive_mode(debug: bool = False):
    """Run direct audio generation testing in interactive mode."""
    if not AUDIO_GENERATION_AVAILABLE:
        print("❌ Audio generation is not available. Missing dependencies or modules.")
        print("Install with: uv sync --package image-server --extra audio_gen")
        return

    print("\n=== Direct Audio Generation Test ===")
    print("Testing TangoFlux audio generator directly (no ZMQ)...")
    
    # Configuration
    print("\nAudio Configuration:")
    duration_s = int(input("Audio duration in seconds (default=10): ") or "10")
    normalize_audio = input("Apply loudness normalization? (Y/n): ").lower() != 'n'
    
    config = Prompt2AudioConfig(
        duration_s=duration_s,
        normalize_loudness=normalize_audio,
        candidates=2,  # Generate 2 candidates for better quality
        prefetch_in_background=False,  # Disable for testing
    )
    
    # Create generator
    output_dir = Path("/tmp/experimance_audio_direct_test")
    generator = Prompt2AudioGenerator(config, output_dir=str(output_dir))
    
    try:
        await generator.start()
        print(f"✅ Audio generator started. Output directory: {output_dir}")
        
        while True:
            print("\n=== Audio Generation Menu ===")
            
            # Prompt selection
            print("\nSelect an audio prompt:")
            print("  1. Select from predefined audio prompts")
            print("  2. Enter custom audio prompt")
            print("  3. Exit")
            
            prompt_choice = input("Choose option (1-3, default=1): ") or "1"
            
            if prompt_choice == "3":
                break
            
            selected_prompt = ""
            
            if prompt_choice == "1":
                print("\nAvailable audio prompts:")
                audio_prompt_options = list(SAMPLE_AUDIO_PROMPTS.keys())
                for i, name in enumerate(audio_prompt_options):
                    print(f"  {i+1}. {name}: {SAMPLE_AUDIO_PROMPTS[name]}")
                
                try:
                    predefined_choice = int(input(f"Select prompt (1-{len(audio_prompt_options)}): "))
                    if 1 <= predefined_choice <= len(audio_prompt_options):
                        selected_prompt = SAMPLE_AUDIO_PROMPTS[audio_prompt_options[predefined_choice - 1]]
                        print(f"Selected: {selected_prompt}")
                    else:
                        print("Invalid selection, using default")
                        selected_prompt = SAMPLE_AUDIO_PROMPTS["forest_ambience"]
                except ValueError:
                    print("Invalid input, using default")
                    selected_prompt = SAMPLE_AUDIO_PROMPTS["forest_ambience"]
            
            else:
                selected_prompt = input("Enter audio prompt: ").strip()
            
            if not selected_prompt.strip():
                print("Empty prompt, skipping...")
                continue
            
            # Show summary and confirm
            print("\n=== Audio Generation Summary ===")
            print(f"Prompt: {selected_prompt}")
            print(f"Duration: {duration_s} seconds")
            print(f"Normalize: {'Yes' if normalize_audio else 'No'}")
            print(f"Output: {output_dir}")
            
            confirm = input("\nGenerate this audio? (Y/n): ").lower() != 'n'
            if not confirm:
                continue
            
            # Generate audio
            print(f"\n🎵 Generating audio...")
            start_time = time.monotonic()
            
            try:
                audio_path = await generator.generate_audio(
                    selected_prompt,
                    duration_s=duration_s
                )
                
                duration = time.monotonic() - start_time
                print(f"✅ Audio generated in {duration:.1f} seconds")
                print(f"📁 Audio file: {audio_path}")
                
                # Check file info
                audio_file = Path(audio_path)
                if audio_file.exists():
                    file_size = audio_file.stat().st_size
                    print(f"📊 File size: {file_size // 1024}KB")
                    
                    # Try to get audio info
                    try:
                        import soundfile as sf
                        with sf.SoundFile(audio_path) as f:
                            print(f"⏱️  Actual duration: {len(f) / f.samplerate:.1f}s")
                            print(f"🔊 Sample rate: {f.samplerate}Hz")
                            print(f"🎼 Channels: {f.channels}")
                    except ImportError:
                        print("📊 Install soundfile for detailed audio info: uv add soundfile")
                    except Exception as e:
                        print(f"📊 Could not read audio info: {e}")
                else:
                    print("❌ Audio file was not created!")
                    
            except Exception as e:
                duration = time.monotonic() - start_time
                print(f"❌ Audio generation failed after {duration:.1f}s")
                print(f"Error: {e}")
                if debug:
                    print(f"Traceback:\n{traceback.format_exc()}")
            
            # Ask to continue
            if input("\nGenerate another audio? (Y/n): ").lower() == 'n':
                break
                
    except Exception as e:
        print(f"❌ Failed to start audio generator: {e}")
        if debug:
            print(f"Traceback:\n{traceback.format_exc()}")
    finally:
        try:
            await generator.stop()
            print("🛑 Audio generator stopped")
        except Exception as e:
            print(f"Warning: Error stopping generator: {e}")


async def command_line_mode(args):
    """Run direct audio generation in command line mode with arguments."""
    if not AUDIO_GENERATION_AVAILABLE:
        print("❌ Audio generation is not available. Missing dependencies or modules.")
        print("Install with: uv sync --package image-server --extra audio_gen")
        return 1

    # Determine the audio prompt
    audio_prompt = args.prompt
    if not audio_prompt:
        print("Error: --prompt is required for command line mode.")
        return 1

    print(f"\n=== Direct Audio Generation Test ===")
    print(f"Prompt: {audio_prompt}")
    print(f"Duration: {args.duration} seconds")
    
    # Configuration
    config = Prompt2AudioConfig(
        duration_s=args.duration,
        normalize_loudness=True,  # Always normalize for CLI
        candidates=2,  # Generate 2 candidates for better quality
        prefetch_in_background=False,  # Disable for CLI
    )
    
    # Create generator
    output_dir = Path("/tmp/experimance_audio_direct_test")
    generator = Prompt2AudioGenerator(config, output_dir=str(output_dir))
    
    try:
        await generator.start()
        print(f"✅ Audio generator started. Output directory: {output_dir}")
        
        print(f"\n🎵 Generating audio...")
        start_time = time.monotonic()
        
        audio_path = await generator.generate_audio(
            audio_prompt,
            duration_s=args.duration
        )
        
        duration = time.monotonic() - start_time
        print(f"✅ Audio generated in {duration:.1f} seconds")
        print(f"📁 Audio file: {audio_path}")
        
        # Check file info
        audio_file = Path(audio_path)
        if audio_file.exists():
            file_size = audio_file.stat().st_size
            print(f"📊 File size: {file_size // 1024}KB")
            
            # Try to get audio info
            try:
                import soundfile as sf
                with sf.SoundFile(audio_path) as f:
                    print(f"⏱️  Actual duration: {len(f) / f.samplerate:.1f}s")
                    print(f"🔊 Sample rate: {f.samplerate}Hz")
                    print(f"🎼 Channels: {f.channels}")
            except ImportError:
                print("📊 Install soundfile for detailed audio info: uv add soundfile")
            except Exception as e:
                print(f"📊 Could not read audio info: {e}")
        else:
            print("❌ Audio file was not created!")
            
    except Exception as e:
        duration = time.monotonic() - start_time if 'start_time' in locals() else 0
        print(f"❌ Audio generation failed after {duration:.1f}s")
        print(f"Error: {e}")
        
        if args.debug:
            print(f"Traceback:\n{traceback.format_exc()}")
        return 1
    finally:
        try:
            await generator.stop()
            print("🛑 Audio generator stopped")
        except Exception as e:
            print(f"Warning: Error stopping generator: {e}")
    
    return 0


def main():
    """Main entry point for the direct audio generation test tool."""
    parser = argparse.ArgumentParser(description="Direct TangoFlux Audio Generation Test Tool")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode with a menu interface"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Audio prompt for generation"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=10,
        help="Duration of audio in seconds (default: 10)"
    )
    parser.add_argument(
        "--list-prompts", "-l",
        action="store_true",
        help="List available sample audio prompts and exit"
    )
    parser.add_argument(
        "--sample-prompt", "-s",
        type=str,
        help="Use a sample audio prompt by name (use --list-prompts to see available options)"
    )
    parser.add_argument(
        "--debug", "-D",
        action="store_true",
        help="Enable debug logging for more detailed output"
    )
    
    args = parser.parse_args()
    
    # List sample prompts if requested
    if args.list_prompts:
        print("Available sample audio prompts:")
        for name, prompt in SAMPLE_AUDIO_PROMPTS.items():
            print(f"  {name}: {prompt}")
        return 0
    
    # Use a sample prompt if specified
    if args.sample_prompt:
        if args.sample_prompt in SAMPLE_AUDIO_PROMPTS:
            args.prompt = SAMPLE_AUDIO_PROMPTS[args.sample_prompt]
            print(f"Using sample audio prompt: {args.prompt}")
        else:
            print(f"Error: Sample audio prompt '{args.sample_prompt}' not found.")
            print("Use --list-prompts to see available options.")
            return 1

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("audio_direct_test").setLevel(logging.DEBUG)
        print("Debug logging enabled")

    # Check for interactive mode or required parameters
    if args.interactive:
        return asyncio.run(interactive_mode(args.debug))
    elif args.prompt:
        return asyncio.run(command_line_mode(args))
    else:
        print("Error: Must specify either --interactive or --prompt")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
