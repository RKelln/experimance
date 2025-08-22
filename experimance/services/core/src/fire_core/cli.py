#!/usr/bin/env python3
"""
CLI utility for testing the Fire Core Service.

This utility allows users to test the Fire Core service by sending:
1. Story transcripts (StoryHeard messages) for full pipeline testing
2. Direct prompts (debug-only mode) for prompt-to-image testing
3. Transcript conversations (sequential agent/user messages via TranscriptUpdate)
4. Individual transcript updates (single TranscriptUpdate messages)

$ uv run -m fire_core.cli
"""

import asyncio
import argparse
import base64
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

from experimance_common.constants import DEFAULT_PORTS, ZMQ_TCP_BIND_PREFIX, ZMQ_TCP_CONNECT_PREFIX
from experimance_common.schemas import MessageType, StoryHeard, TranscriptUpdate # type: ignore
from experimance_common.zmq.components import PublisherComponent
from experimance_common.zmq.config import PublisherConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fire_core_cli")

# Sample story transcripts for testing
SAMPLE_STORIES = {
    "forest_discovery": """
    I remember walking through the ancient forest near my grandmother's cabin. The towering pines 
    created a cathedral of green, with shafts of golden sunlight breaking through the canopy. 
    There was a small clearing where wildflowers grew in abundance - lupines, Indian paintbrush, 
    and mountain asters. A crystal-clear stream wound through the rocks, and I could hear the 
    distant call of a loon from the lake beyond.
    """,
    
    "desert_sunset": """
    The vast Sonoran desert stretched endlessly before me, painted in hues of rust and gold. 
    Ancient saguaro cacti stood like sentinels against the crimson sky, their arms reaching 
    toward the heavens. As the sun dipped below the horizon, the temperature began to drop, 
    and I could hear the haunting call of coyotes in the distance. The stars emerged one by 
    one, brilliant against the clear desert sky.
    """,
    
    "mountain_lake": """
    High in the Rocky Mountains, I discovered a pristine alpine lake nestled between granite 
    peaks. The water was so clear I could see trout swimming in the depths. Snow-capped summits 
    reflected perfectly in the still surface, creating a mirror image of the sky. Alpine 
    wildflowers carpeted the shoreline - purple lupines, yellow glacier lilies, and delicate 
    mountain forget-me-nots.
    """,
    
    "coastal_storm": """
    The Pacific coast was wild that day, with massive waves crashing against the rocky cliffs. 
    Seabirds wheeled overhead, riding the powerful winds. The salt spray reached high into the 
    air, creating rainbows in the mist. Ancient redwoods grew right to the cliff's edge, their 
    gnarled roots holding fast against the constant erosion. In the tide pools, I found sea 
    anemones and hermit crabs going about their ancient rhythms.
    """,
    
    "prairie_spring": """
    The Great Plains came alive in spring, with endless waves of grass rippling in the wind 
    like a green ocean. Scattered oak groves dotted the landscape, and wildflowers bloomed 
    in every color imaginable - purple coneflowers, orange butterfly weed, and golden black-eyed 
    Susans. Hawks circled overhead, and in the distance, I could see a herd of bison grazing 
    peacefully in the morning light.
    """
}

# Sample direct prompts for testing
SAMPLE_PROMPTS = {
    "mountain_panorama": "Majestic mountain range with snow-capped peaks, alpine lakes, and evergreen forests stretching to the horizon, golden hour lighting, highly detailed panoramic view",
    
    "forest_canopy": "Dense old-growth forest with towering trees, filtered sunlight through the canopy, moss-covered logs, ferns and wildflowers on the forest floor, atmospheric perspective",
    
    "desert_landscape": "Vast desert landscape with saguaro cacti, red rock formations, mesa in the distance, dramatic clouds, warm desert light, wide panoramic composition",
    
    "coastal_scene": "Rocky coastline with crashing waves, sea stacks, seabirds, dramatic cliffs, tide pools, stormy sky, panoramic ocean view",
    
    "prairie_vista": "Rolling grasslands with scattered oak trees, wildflowers, big sky with dramatic clouds, wind patterns visible in the grass, endless horizon"
}

# Sample transcript conversations for testing
SAMPLE_TRANSCRIPT_CONVERSATIONS = {
    "forest_memories": [
        {"speaker_id": "user", "speaker_name": "Visitor", "content": "I've been thinking about my childhood memories of forests."},
        {"speaker_id": "agent", "speaker_name": "Fire Spirit", "content": "Tell me more about those forest memories. What do you remember most vividly?"},
        {"speaker_id": "user", "speaker_name": "Visitor", "content": "There was this incredible old-growth forest near my grandmother's cabin. The trees were so tall, like ancient pillars reaching up to heaven."},
        {"speaker_id": "agent", "speaker_name": "Fire Spirit", "content": "Those ancient trees hold so much wisdom. Can you describe what the forest felt like to you?"},
        {"speaker_id": "user", "speaker_name": "Visitor", "content": "It felt sacred, you know? The sunlight would filter through in these golden shafts, and there were wildflowers everywhere - lupines, paintbrush, mountain asters."},
        {"speaker_id": "user", "speaker_name": "Visitor", "content": "And there was this crystal-clear stream winding through the rocks. Sometimes I could hear loons calling from the lake beyond."},
    ],
    
    "desert_journey": [
        {"speaker_id": "user", "speaker_name": "Traveler", "content": "I once traveled through the Sonoran Desert. It was unlike anything I'd ever experienced."},
        {"speaker_id": "agent", "speaker_name": "Fire Spirit", "content": "The desert has its own unique beauty. What struck you most about that journey?"},
        {"speaker_id": "user", "speaker_name": "Traveler", "content": "The vastness, first of all. Miles and miles of rust-colored earth stretching to the horizon."},
        {"speaker_id": "user", "speaker_name": "Traveler", "content": "And these ancient saguaro cacti everywhere, like sentinels with their arms reaching up to the crimson sky."},
        {"speaker_id": "agent", "speaker_name": "Fire Spirit", "content": "The saguaros are ancient wisdom keepers. How did the desert make you feel?"},
        {"speaker_id": "user", "speaker_name": "Traveler", "content": "At sunset, when the sky turned all these incredible colors, I felt so small but also connected to something eternal."},
        {"speaker_id": "user", "speaker_name": "Traveler", "content": "At night, the stars were brilliant - you could see the Milky Way stretching across the entire sky."},
    ],
    
    "mountain_reflection": [
        {"speaker_id": "user", "speaker_name": "Hiker", "content": "I found this hidden alpine lake high in the Rockies last summer."},
        {"speaker_id": "agent", "speaker_name": "Fire Spirit", "content": "Mountain lakes hold special magic. Tell me about this discovery."},
        {"speaker_id": "user", "speaker_name": "Hiker", "content": "It was nestled between these granite peaks, the water so clear I could see trout swimming deep below."},
        {"speaker_id": "user", "speaker_name": "Hiker", "content": "The snow-capped summits reflected perfectly in the still surface, like a mirror image of the sky itself."},
        {"speaker_id": "agent", "speaker_name": "Fire Spirit", "content": "That reflection speaks to the connection between earth and sky. What else did you notice?"},
        {"speaker_id": "user", "speaker_name": "Hiker", "content": "Alpine wildflowers everywhere along the shoreline - purple lupines, yellow glacier lilies, tiny mountain forget-me-nots."},
        {"speaker_id": "user", "speaker_name": "Hiker", "content": "The silence was profound. Just the gentle lapping of water and the distant call of a hawk."},
    ]
}


class FireCoreClient:
    """Simple client for sending messages to the Fire Core Service."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # Publishers to send messages to fire core
        self.agent_publisher: Optional[PublisherComponent] = None  # For stories (agent channel)
        self.updates_publisher: Optional[PublisherComponent] = None  # For debug prompts (updates channel)

    async def start(self):
        """Start the ZMQ components for communication."""
        
        # Publisher for agent channel (stories) - bind since we're acting as the agent
        agent_config = PublisherConfig(
            address=ZMQ_TCP_BIND_PREFIX,
            port=DEFAULT_PORTS['agent'],  # 5557 - agent channel
            bind=True  # CLI binds on agent channel (acting as agent service)
        )
        self.agent_publisher = PublisherComponent(agent_config)
        await self.agent_publisher.start()
        
        # Publisher for updates channel (debug prompts) - connect to where core binds
        updates_config = PublisherConfig(
            address=ZMQ_TCP_CONNECT_PREFIX, 
            port=DEFAULT_PORTS['updates'],  # 5556 - updates channel  
            bind=False  # Connect to core's bound updates channel
        )
        self.updates_publisher = PublisherComponent(updates_config)
        await self.updates_publisher.start()
        
        logger.info(f"CLI client started:")
        logger.info(f"  Agent channel (stories): {ZMQ_TCP_BIND_PREFIX}:{DEFAULT_PORTS['agent']} (binding)")
        logger.info(f"  Updates channel (prompts): {ZMQ_TCP_CONNECT_PREFIX}:{DEFAULT_PORTS['updates']} (connecting)")

        await asyncio.sleep(2.0)  # Allow connections to establish

    async def stop(self):
        """Stop all ZMQ components."""
        if self.agent_publisher:
            await self.agent_publisher.stop()
        if self.updates_publisher:
            await self.updates_publisher.stop()
        logger.info("Client stopped")

    async def send_story(self, story_content: str) -> str:
        """Send a StoryHeard message via the agent channel."""
        request_id = str(uuid.uuid4())
        
        story_message = StoryHeard(
            request_id=request_id,
            content=story_content,
            timestamp=str(time.time())  # Convert to string as expected by schema
        )
        
        logger.info(f"Sending story via agent channel ({len(story_content)} chars)")
        assert self.agent_publisher is not None, "Agent publisher not initialized"
        await self.agent_publisher.publish(story_message, MessageType.STORY_HEARD)
        
        return request_id

    async def send_debug_prompt(self, prompt: str) -> str:
        """Send a debug prompt via the updates channel."""
        request_id = str(uuid.uuid4())
        
        # Create a simple debug message for the updates channel
        debug_message = {
            "type": "prompt",
            "request_id": request_id,
            "prompt": prompt,
            "timestamp": str(time.time())  # Keep as string for consistency
        }
        
        logger.info(f"Sending debug prompt via updates channel: {prompt}")
        assert self.updates_publisher is not None, "Updates publisher not initialized"
        await self.updates_publisher.publish(debug_message, "prompt")
        
        return request_id

    async def send_transcript_update(self, content: str, speaker_id: str, speaker_display_name: Optional[str] = None, 
                                   session_id: Optional[str] = None, is_partial: bool = False) -> str:
        """Send a TranscriptUpdate message via the agent channel."""
        request_id = str(uuid.uuid4())
        
        transcript_message = TranscriptUpdate(
            request_id=request_id,
            content=content,
            speaker_id=speaker_id,
            speaker_display_name=speaker_display_name or speaker_id,
            session_id=session_id or f"cli_session_{int(time.time())}",
            timestamp=str(time.time()),
            is_partial=is_partial
        )
        
        logger.info(f"Sending transcript update via agent channel: [{speaker_id}] '{content}'")
        assert self.agent_publisher is not None, "Agent publisher not initialized"
        await self.agent_publisher.publish(transcript_message, MessageType.TRANSCRIPT_UPDATE)
        
        return request_id

    async def send_conversation(self, conversation: List[Dict[str, str]], session_id: Optional[str] = None, 
                              delay_between_messages: float = 2.0) -> List[str]:
        """Send a full conversation as a series of transcript updates."""
        if session_id is None:
            session_id = f"cli_conversation_{int(time.time())}"
        
        request_ids = []
        logger.info(f"Starting conversation with {len(conversation)} messages (session: {session_id})")
        
        for i, msg in enumerate(conversation):
            request_id = await self.send_transcript_update(
                content=msg["content"],
                speaker_id=msg["speaker_id"], 
                speaker_display_name=msg.get("speaker_name", msg["speaker_id"]),
                session_id=session_id
            )
            request_ids.append(request_id)
            
            # Add delay between messages to simulate realistic conversation flow
            if i < len(conversation) - 1:  # Don't wait after the last message
                logger.info(f"Waiting {delay_between_messages}s before next message...")
                await asyncio.sleep(delay_between_messages)
        
        logger.info(f"Conversation complete - sent {len(request_ids)} transcript updates")
        return request_ids


async def interactive_mode(debug: bool = False):
    """Run the client in interactive mode with a menu-based interface."""
    print("\n=== Fire Core Test Client ===\n")
    print("ZMQ Configuration:")
    print(f"  Agent channel (stories): tcp://localhost:{DEFAULT_PORTS['agent']}")
    print(f"  Updates channel (prompts): tcp://localhost:{DEFAULT_PORTS['updates']}")

    # Create and start client
    client = FireCoreClient(debug=debug)
    await client.start()

    try:
        while True:
            print("\n=== Fire Core Testing Menu ===")
            print("  1. Send story transcript (full pipeline via agent channel)")
            print("  2. Send direct prompt (debug mode via updates channel)")  
            print("  3. Send transcript conversation (sequential agent/user messages)")
            print("  4. Send single transcript update")
            print("  5. Send custom story")
            print("  6. Send custom prompt")
            print("  0. Exit")

            choice = input("\nChoose option (0-6): ").strip()

            if choice == "0":
                break
            elif choice == "1":
                # Send story transcript
                print("\nAvailable story transcripts:")
                story_options = list(SAMPLE_STORIES.keys())
                for i, name in enumerate(story_options):
                    print(f"  {i+1}. {name.replace('_', ' ').title()}")
                
                story_choice = input(f"Choose story (1-{len(story_options)}): ").strip()
                try:
                    story_index = int(story_choice) - 1
                    if 0 <= story_index < len(story_options):
                        story_name = story_options[story_index]
                        story_content = SAMPLE_STORIES[story_name].strip()
                        
                        print(f"\nSending story: {story_name}")
                        print(f"Content preview: {story_content[:100]}...")
                        
                        request_id = await client.send_story(story_content)
                        print(f"Story sent via agent channel with ID: {request_id}")
                        print("Check the fire_core service logs to see processing results.")
                    else:
                        print("Invalid story selection.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            elif choice == "2":
                # Send direct prompt
                print("\nAvailable direct prompts:")
                prompt_options = list(SAMPLE_PROMPTS.keys())
                for i, name in enumerate(prompt_options):
                    print(f"  {i+1}. {name.replace('_', ' ').title()}")
                
                prompt_choice = input(f"Choose prompt (1-{len(prompt_options)}): ").strip()
                try:
                    prompt_index = int(prompt_choice) - 1
                    if 0 <= prompt_index < len(prompt_options):
                        prompt_name = prompt_options[prompt_index]
                        prompt_content = SAMPLE_PROMPTS[prompt_name]
                        
                        print(f"\nSending prompt: {prompt_name}")
                        print(f"Prompt: {prompt_content}")
                        
                        request_id = await client.send_debug_prompt(prompt_content)
                        print(f"Prompt sent via updates channel with ID: {request_id}")
                        print("Check the fire_core service logs to see processing results.")
                    else:
                        print("Invalid prompt selection.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            elif choice == "3":
                # Send transcript conversation
                print("\nAvailable transcript conversations:")
                conversation_options = list(SAMPLE_TRANSCRIPT_CONVERSATIONS.keys())
                for i, name in enumerate(conversation_options):
                    print(f"  {i+1}. {name.replace('_', ' ').title()}")
                
                conv_choice = input(f"Choose conversation (1-{len(conversation_options)}): ").strip()
                try:
                    conv_index = int(conv_choice) - 1
                    if 0 <= conv_index < len(conversation_options):
                        conv_name = conversation_options[conv_index]
                        conversation = SAMPLE_TRANSCRIPT_CONVERSATIONS[conv_name]
                        
                        print(f"\nSending conversation: {conv_name}")
                        print(f"Will send {len(conversation)} transcript messages sequentially...")
                        
                        # Ask for delay between messages
                        delay_input = input("Delay between messages in seconds (default 2.0): ").strip()
                        try:
                            delay = float(delay_input) if delay_input else 2.0
                        except ValueError:
                            delay = 2.0
                            
                        request_ids = await client.send_conversation(conversation, delay_between_messages=delay)
                        print(f"Conversation sent with {len(request_ids)} transcript updates")
                        print("Check the fire_core service logs to see processing results.")
                    else:
                        print("Invalid conversation selection.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            elif choice == "4":
                # Send single transcript update
                speaker_id = input("\nEnter speaker ID (e.g., 'user', 'agent'): ").strip()
                if not speaker_id:
                    speaker_id = "user"
                
                speaker_name = input(f"Enter speaker display name (default: {speaker_id}): ").strip()
                if not speaker_name:
                    speaker_name = speaker_id
                
                content = input("Enter transcript content: ").strip()
                if content:
                    request_id = await client.send_transcript_update(content, speaker_id, speaker_name)
                    print(f"Transcript update sent with ID: {request_id}")
                    print("Check the fire_core service logs to see processing results.")
                else:
                    print("No content entered.")

            elif choice == "5":
                # Send custom story
                print("\nEnter your custom story (press Ctrl+D when finished):")
                lines = []
                try:
                    while True:
                        line = input()
                        lines.append(line)
                except EOFError:
                    pass
                
                story_content = '\n'.join(lines).strip()
                if story_content:
                    request_id = await client.send_story(story_content)
                    print(f"\nCustom story sent via agent channel with ID: {request_id}")
                    print("Check the fire_core service logs to see processing results.")
                else:
                    print("No story content entered.")

            elif choice == "6":
                # Send custom prompt
                prompt_content = input("\nEnter your custom prompt: ").strip()
                if prompt_content:
                    request_id = await client.send_debug_prompt(prompt_content)
                    print(f"Custom prompt sent via updates channel with ID: {request_id}")
                    print("Check the fire_core service logs to see processing results.")
                else:
                    print("No prompt content entered.")

            else:
                print("Invalid option. Please choose 0-6.")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        await client.stop()


async def command_line_mode(args):
    """Run the client in command line mode with arguments."""
    client = FireCoreClient(debug=args.debug)
    await client.start()

    try:
        if args.story:
            # Send story transcript
            if args.story in SAMPLE_STORIES:
                story_content = SAMPLE_STORIES[args.story].strip()
            else:
                story_content = args.story
            
            request_id = await client.send_story(story_content)
            print(f"Story sent via agent channel with ID: {request_id}")
            print("Check the fire_core service logs to see processing results.")
        
        elif args.prompt:
            # Send direct prompt
            if args.prompt in SAMPLE_PROMPTS:
                prompt_content = SAMPLE_PROMPTS[args.prompt]
            else:
                prompt_content = args.prompt
            
            request_id = await client.send_debug_prompt(prompt_content)
            print(f"Prompt sent via updates channel with ID: {request_id}")
            print("Check the fire_core service logs to see processing results.")
        
        elif args.conversation:
            # Send transcript conversation
            if args.conversation in SAMPLE_TRANSCRIPT_CONVERSATIONS:
                conversation = SAMPLE_TRANSCRIPT_CONVERSATIONS[args.conversation]
                delay = args.delay if args.delay is not None else 2.0
                
                print(f"Sending conversation: {args.conversation}")
                print(f"Will send {len(conversation)} transcript messages with {delay}s delays...")
                
                request_ids = await client.send_conversation(conversation, delay_between_messages=delay)
                print(f"Conversation sent with {len(request_ids)} transcript updates")
                print("Check the fire_core service logs to see processing results.")
            else:
                print(f"Unknown conversation: {args.conversation}")
                print("Available conversations:", list(SAMPLE_TRANSCRIPT_CONVERSATIONS.keys()))
        
        elif args.transcript:
            # Send single transcript update
            speaker_id = args.speaker_id or "user"
            speaker_name = args.speaker_name or speaker_id
            
            request_id = await client.send_transcript_update(args.transcript, speaker_id, speaker_name)
            print(f"Transcript update sent with ID: {request_id}")
            print("Check the fire_core service logs to see processing results.")

    finally:
        await client.stop()


def main():
    """Main entry point for the CLI utility."""
    parser = argparse.ArgumentParser(description="Test client for the Fire Core Service")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode with a menu interface"
    )
    parser.add_argument(
        "--story", "-s",
        type=str,
        help="Story transcript to send (use story name or full text)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Direct prompt to send (use prompt name or full text)"
    )
    parser.add_argument(
        "--conversation", "-c",
        type=str,
        help="Transcript conversation to send (use conversation name from samples)"
    )
    parser.add_argument(
        "--transcript", "-t",
        type=str,
        help="Single transcript message to send"
    )
    parser.add_argument(
        "--speaker-id",
        type=str,
        help="Speaker ID for transcript messages (default: 'user')"
    )
    parser.add_argument(
        "--speaker-name",
        type=str,
        help="Speaker display name for transcript messages (default: same as speaker-id)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        help="Delay between messages in conversations (default: 2.0 seconds)"
    )
    parser.add_argument(
        "--debug", "-D",
        action="store_true",
        help="Enable debug logging for more detailed output"
    )
    parser.add_argument(
        "--list-stories",
        action="store_true",
        help="List available sample stories and exit"
    )
    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available sample prompts and exit"
    )
    parser.add_argument(
        "--list-conversations",
        action="store_true",
        help="List available sample transcript conversations and exit"
    )
    
    args = parser.parse_args()
    
    # List sample stories if requested
    if args.list_stories:
        print("Available sample stories:")
        for name, story in SAMPLE_STORIES.items():
            print(f"  {name}: {story[:100].strip()}...")
        return
    
    # List sample prompts if requested
    if args.list_prompts:
        print("Available sample prompts:")
        for name, prompt in SAMPLE_PROMPTS.items():
            print(f"  {name}: {prompt}")
        return
    
    # List sample conversations if requested
    if args.list_conversations:
        print("Available sample transcript conversations:")
        for name, conversation in SAMPLE_TRANSCRIPT_CONVERSATIONS.items():
            print(f"  {name}: {len(conversation)} messages")
            for i, msg in enumerate(conversation[:3]):  # Show first 3 messages as preview
                speaker = msg.get('speaker_name', msg['speaker_id'])
                content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
                print(f"    {i+1}. {speaker}: {content}")
            if len(conversation) > 3:
                print(f"    ... and {len(conversation) - 3} more messages")
        return

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("fire_core_cli").setLevel(logging.DEBUG)
        print("Debug logging enabled")

    # Check for interactive mode or required parameters
    if args.interactive:
        asyncio.run(interactive_mode(args.debug))
    elif args.story or args.prompt or args.conversation or args.transcript:
        asyncio.run(command_line_mode(args))
    else:
        print("Error: Either --interactive mode, --story, --prompt, --conversation, or --transcript must be specified.")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    # $ uv run -m fire_core.cli -i
    sys.exit(main())