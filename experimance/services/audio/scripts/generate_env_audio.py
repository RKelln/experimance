"""
Script to generate environmental audio files from a JSON configuration.
Uses the fal.ai API to generate audio based on text prompts.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import aiohttp
import fal_client
import dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("audio_generator")
httpx_logger = logging.getLogger("httpx")
# set httpx logger to warning to avoid debug noise
httpx_logger.setLevel(logging.WARN)

# Load environment variables from .env file
dotenv.load_dotenv()

# Constants
DEFAULT_DURATION = 0  # Default duration in seconds for generated audio
AUDIO_BASE_DIR = Path("services/audio/audio")
CONFIG_DIR = Path("services/audio/config")
LAYERS_CONFIG = CONFIG_DIR / "layers_v2.json"
CONCURRENCY_LIMIT = 3  # Limit concurrent API calls


async def generate_audio(prompt: str, duration: int = DEFAULT_DURATION) -> Optional[str]:
    """
    Generate audio using fal.ai based on a prompt.
    
    Args:
        prompt: Text prompt describing the audio to generate
        duration: Duration in seconds for the generated audio
        
    Returns:
        URL to the generated audio file or None if generation failed
    """
    try:
        args = {
            "text": prompt
        }
        if duration > 0:
            args["duration"] = str(duration)

        logger.info(f"Generating audio for prompt: '{prompt}' with duration {duration}s")
        handler = await fal_client.submit_async(
            "fal-ai/elevenlabs/sound-effects",
            arguments=args,
        )
        result = await handler.get()
        print(f"Result: {result}")
        logger.debug(f"API response: {result}")
        
        if not result or "audio" not in result:
            logger.error(f"Failed to generate audio for prompt: '{prompt}'")
            return None
            
        audio_url = result.get("audio", {}).get("url", None)
        if not audio_url:
            logger.error(f"No audio URL returned for prompt: '{prompt}'")
            return None
            
        logger.info(f"Audio generated successfully: {audio_url}")
        return audio_url
    except Exception as e:
        logger.error(f"Error generating audio for prompt '{prompt}': {e}")
        return None


async def download_file(url: str, destination: str) -> bool:
    """
    Download a file from a URL and save it to the specified path.
    
    Args:
        url: The URL of the file to download
        destination_path: The path where the file will be saved
        
    Returns:
        True if the download was successful, False otherwise
    """
    # set the destination_path extension to match the URL
    url_extension = Path(url).suffix
    destination_path = Path(destination).with_suffix(url_extension)
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Download the file
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download file from {url}: {response.status}")
                    return False
                
                with open(destination_path, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
        
        logger.info(f"Downloaded file successfully to {destination_path}")
        return True
    except Exception as e:
        logger.error(f"Error downloading file from {url} to {destination_path}: {e}")
        return False


async def process_audio_entry(entry: Dict[str, Any], dry_run: bool = False, duration: int = DEFAULT_DURATION) -> None:
    """
    Process a single audio entry from the configuration file.
    
    Args:
        entry: A dictionary containing path, tags, and prompt information
        dry_run: If True, simulate the process without actually generating or downloading files
        duration: Duration in seconds for the generated audio
    """
    # Skip entries without paths or prompts
    if "path" not in entry or "prompt" not in entry:
        logger.warning(f"Skipping entry - missing path or prompt: {entry}")
        return
    
    path = entry["path"]
    prompt = entry["prompt"]
    prompt = prompt.strip()
    prompt = f"high-quality, professionally recorded {prompt}. Seamless loop, no silence at start or end."
    
    # Construct the full path to the audio file
    full_path = AUDIO_BASE_DIR / path
    
    # Skip if the file already exists
    if full_path.exists():
        logger.info(f"File already exists, skipping: {full_path}")
        return
    
    if dry_run:
        logger.info(f"[DRY RUN] Would generate audio for '{path}' with prompt: '{prompt}'")
        return
    
    # Generate the audio
    audio_url = await generate_audio(prompt, duration)
    if not audio_url:
        logger.error(f"Failed to generate audio for {path}")
        return
    
    # Download the audio file
    success = await download_file(audio_url, str(full_path))
    if success:
        logger.info(f"Successfully generated and saved audio for {path}")
    else:
        logger.error(f"Failed to download audio for {path}")


async def process_audio_entries(entries: List[Dict[str, Any]], dry_run: bool = False, duration: int = DEFAULT_DURATION) -> None:
    """
    Process multiple audio entries with concurrency control.
    
    Args:
        entries: List of entries from the configuration file
        dry_run: If True, simulate the process without actually generating or downloading files
        duration: Duration in seconds for the generated audio
    """
    # Use a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async def process_with_limit(entry):
        async with semaphore:
            await process_audio_entry(entry, dry_run, duration)
    
    # Create tasks for all entries
    tasks = [process_with_limit(entry) for entry in entries]
    
    # Process all tasks
    await asyncio.gather(*tasks)


async def main(dry_run: bool = False, duration: int = DEFAULT_DURATION, total: int = 0) -> None:
    """
    Main function to process the audio configuration file.
    
    Args:
        dry_run: If True, simulate the process without actually generating or downloading files
        duration: Duration in seconds for the generated audio
        total: Total number of audio files to generate (0 for all)
    """
    try:
        # Check if the config file exists
        if not LAYERS_CONFIG.exists():
            logger.error(f"Config file not found: {LAYERS_CONFIG}")
            return
        
        # Read and parse the JSON file
        with open(LAYERS_CONFIG, 'r') as f:
            content = f.read()
            # Remove any comment lines (which are not valid JSON)
            content_lines = [line for line in content.splitlines() if not line.strip().startswith('//')]
            content = '\n'.join(content_lines)
            
            # Parse the JSON
            config = json.loads(content)
        
        logger.info(f"Found {len(config)} entries in the configuration file")
        
        if dry_run:
            logger.info("Running in DRY RUN mode - no files will be generated or downloaded")
        
        if duration > 0:
            logger.info(f"Using audio duration: {duration} seconds")
        
        # go through the config and ensure each entry has a path and prompt
        # and remove those without and those who already exist
        if not isinstance(config, list):
            logger.error("Configuration file is not a list of entries")
            return
        
        valid_entries = []
        for entry in config:
            if not isinstance(entry, dict):
                logger.warning(f"Skipping invalid entry: {entry}")
                continue
            
            # Ensure each entry has a path and prompt
            if "path" not in entry or "prompt" not in entry:
                logger.warning(f"Skipping entry without path or prompt: {entry}")
                continue
            
            # Ensure the path is a string
            if not isinstance(entry["path"], str):
                logger.warning(f"Skipping entry with non-string path: {entry}")
                continue
            
            # Ensure the prompt is a string
            if not isinstance(entry["prompt"], str):
                logger.warning(f"Skipping entry with non-string prompt: {entry}")
                continue

            # check if the file already exists
            full_path = AUDIO_BASE_DIR / entry["path"]
            if full_path.exists():
                logger.info(f"File already exists, skipping: {full_path}")
                continue
            valid_entries.append(entry)

        logger.info(f"Filtered down to {len(valid_entries)} valid entries")

        # If total is specified, limit the entries to that number
        if total > 0:
            valid_entries = valid_entries[:total]
            logger.info(f"Limiting to {total} entries")

        # Process all entries
        await process_audio_entries(valid_entries, dry_run, duration)
        
        logger.info("Audio generation complete")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate audio files based on text prompts from a JSON configuration."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the process without actually generating or downloading files"
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=DEFAULT_DURATION,
        help=f"Duration in seconds for generated audio (default: {DEFAULT_DURATION})"
    )
    parser.add_argument(
        "-t", "--total",
        type=int,
        default=0,
        help="Total number of audio files to generate (0 for all)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    asyncio.run(main(dry_run=args.dry_run, duration=args.duration, total=args.total))