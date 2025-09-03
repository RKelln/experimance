#!/usr/bin/env python3
"""
CLI utility for testing the Fire Core Service.

This utility allows users to test the Fire Core service by sending:
1. Story transcripts (StoryHeard messages) for full pipeline testing
2. Direct prompts (debug-only mode) for prompt-to-image testing
3. Transcript conversations (sequential agent/user messages via TranscriptUpdate)
4. Individual transcript updates (single TranscriptUpdate messages)
5. Audio generation and playback testing (AudioRenderRequest/AudioReady)
6. MediaPrompt testing (combined visual + audio prompt generation)

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

# Import Fire-specific schemas for audio testing
AudioRenderRequest = None
AudioReady = None
try:
    import sys
    from pathlib import Path
    # Add projects path for fire schemas
    project_path = Path(__file__).parent.parent.parent.parent.parent / "projects" / "fire"
    if project_path.exists():
        sys.path.insert(0, str(project_path))
    from schemas import AudioRenderRequest, AudioReady  # Fire-specific schemas
except ImportError:
    pass  # Will log warning after logger is set up

# Import local audio testing components
AudioManager = None
MediaPrompt = None
LLMPromptBuilder = None
InsufficientContentException = None
UnchangedContentException = None
try:
    from audio_manager import AudioManager
    from config import MediaPrompt
    from llm_prompt_builder import LLMPromptBuilder, InsufficientContentException, UnchangedContentException
except ImportError:
    pass  # Will log warning after logger is set up

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fire_core_cli")

# Log warnings for failed imports
if AudioRenderRequest is None:
    logger.warning("Could not import Fire schemas - audio render requests will not work")
if MediaPrompt is None:
    logger.warning("Could not import MediaPrompt - media prompt generation will not work")

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
    ],
    
    "session_change_test": [
        {"speaker_id": "user", "speaker_name": "Visitor", "content": "I'd like to tell you about my forest memories.", "session_id": "session_1"},
        {"speaker_id": "agent", "speaker_name": "Fire Spirit", "content": "Please, tell me about those memories.", "session_id": "session_1"},
        {"speaker_id": "user", "speaker_name": "Visitor", "content": "There were these tall pines...", "session_id": "session_1"},
        # Session change happens here - should trigger display clear
        {"speaker_id": "user", "speaker_name": "New Visitor", "content": "Actually, I have a different story about deserts.", "session_id": "session_2"},
        {"speaker_id": "agent", "speaker_name": "Fire Spirit", "content": "Tell me about the desert.", "session_id": "session_2"},
        {"speaker_id": "user", "speaker_name": "New Visitor", "content": "The vast Sonoran desert was incredible...", "session_id": "session_2"},
    ]
}

# Sample audio prompts for testing
SAMPLE_AUDIO_PROMPTS = {
    "forest_ambience": "gentle forest sounds with rustling leaves and distant birds",
    "ocean_waves": "ocean waves crashing against rocky shore with seagull calls",
    "mountain_stream": "babbling mountain stream with wind through pine trees",
    "rain_on_leaves": "soft rain on leaves with distant thunder rumbles",
    "desert_wind": "gentle desert wind with sand shifting and distant howling",
    "campfire_night": "crackling campfire with gentle wind through pine trees",
    "urban_ambience": "quiet city street ambience with distant traffic and footsteps",
    "cave_echoes": "water droplets echoing in deep cave with subtle reverb",
    "beach_calm": "gentle waves lapping against sandy shore with distant seagulls", 
    "prairie_wind": "wind through tall grassland with distant bird calls"
}

# Sample media prompts (visual + audio combined)
SAMPLE_MEDIA_PROMPTS = {
    "forest_cathedral": {
        "visual": "ancient cathedral of towering pine trees with golden sunlight filtering through canopy, moss-covered forest floor, wildflowers in clearing",
        "audio": "gentle forest sounds with rustling leaves and distant bird songs"
    },
    "desert_vista": {
        "visual": "vast Sonoran desert at sunset with saguaro cacti silhouetted against crimson sky, rugged mountains in distance",
        "audio": "gentle desert wind with sand shifting and distant coyote calls"
    },
    "mountain_lake": {
        "visual": "pristine alpine lake reflecting snow-capped peaks, granite boulders, alpine wildflowers along shoreline",
        "audio": "gentle wind over mountain water with distant eagle cries"
    },
    "coastal_storm": {
        "visual": "dramatic Pacific coastline with massive waves crashing against rocky cliffs, sea spray creating rainbows",
        "audio": "powerful ocean waves crashing against rocks with wind and seabird calls"
    }
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
        
        logger.info(f"Sending transcript update via agent channel: [{speaker_id}] '{content}' (session: {transcript_message.session_id})")
        assert self.agent_publisher is not None, "Agent publisher not initialized"
        await self.agent_publisher.publish(transcript_message, MessageType.TRANSCRIPT_UPDATE)
        
        return request_id

    async def send_conversation(self, conversation: List[Dict[str, str]], session_id: Optional[str] = None, 
                              delay_between_messages: float = 2.0) -> List[str]:
        """Send a full conversation as a series of transcript updates.
        
        Each message in the conversation can optionally include a 'session_id' field.
        If present, that session_id will be used for the message. If not present,
        the session_id parameter (or generated default) will be used.
        
        This allows testing session changes within a conversation.
        """
        if session_id is None:
            session_id = f"cli_conversation_{int(time.time())}"
        
        request_ids = []
        logger.info(f"Starting conversation with {len(conversation)} messages (default session: {session_id})")
        
        for i, msg in enumerate(conversation):
            # Use per-message session_id if provided, otherwise use conversation default
            msg_session_id = msg.get("session_id", session_id)
            
            request_id = await self.send_transcript_update(
                content=msg["content"],
                speaker_id=msg["speaker_id"], 
                speaker_display_name=msg.get("speaker_name", msg["speaker_id"]),
                session_id=msg_session_id
            )
            request_ids.append(request_id)
            
            # Add delay between messages to simulate realistic conversation flow
            if i < len(conversation) - 1:  # Don't wait after the last message
                logger.info(f"Waiting {delay_between_messages}s before next message...")
                await asyncio.sleep(delay_between_messages)
        
        logger.info(f"Conversation complete - sent {len(request_ids)} transcript updates")
        return request_ids

    async def send_audio_render_request(self, audio_prompt: str, duration_s: Optional[int] = None,
                                      generator: Optional[str] = None) -> str:
        """Send an AudioRenderRequest message to test audio generation."""
        if AudioRenderRequest is None:
            logger.error("AudioRenderRequest not available - Fire schemas not imported")
            return ""
        
        request_id = str(uuid.uuid4())
        
        audio_request = AudioRenderRequest(
            request_id=request_id,
            generator=generator,
            prompt=audio_prompt,
            duration_s=duration_s
        )
        
        logger.info(f"Sending audio render request: '{audio_prompt}' (duration: {duration_s}s)")
        assert self.agent_publisher is not None, "Agent publisher not initialized"
        await self.agent_publisher.publish(audio_request, "AudioRenderRequest")
        
        return request_id

    async def send_audio_ready(self, audio_file_path: str, request_id: Optional[str] = None) -> str:
        """Send an AudioReady message to Fire Core for audio playback testing."""
        if AudioReady is None:
            logger.error("AudioReady not available - Fire schemas not imported")
            return ""
        
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Convert file path to file:// URL if it's a local file
        if os.path.exists(audio_file_path):
            audio_url = f"file://{os.path.abspath(audio_file_path)}"
        else:
            audio_url = audio_file_path  # Assume it's already a URL
        
        audio_ready = AudioReady(
            request_id=request_id,
            uri=audio_url,
            prompt="Direct audio testing via CLI",
            duration_s=None,  # Will be determined by Fire Core
            is_loop=True,
            metadata={"source": "cli_testing"}
        )
        
        logger.info(f"Sending AudioReady message via updates channel: {audio_url}")
        assert self.updates_publisher is not None, "Updates publisher not initialized"
        await self.updates_publisher.publish(audio_ready, "audio_ready")
        
        return request_id

    async def test_full_audio_workflow(self, story_content: str, audio_file_path: str) -> bool:
        """
        Test the complete audio workflow:
        1. Send a story transcript to Fire Core
        2. Listen for AudioRenderRequest from Fire Core 
        3. Respond with AudioReady containing a real audio file
        4. Fire Core should then play the audio
        """
        if AudioRenderRequest is None or AudioReady is None:
            logger.error("Audio schemas not available - Fire schemas not imported")
            return False
            
        # Import ZMQ components for listening/responding
        import zmq
        import zmq.asyncio
        
        context = zmq.asyncio.Context()
        subscriber = None
        publisher = None
        
        try:
            # Set up audio response listener
            subscriber = context.socket(zmq.SUB)
            subscriber.connect(f"tcp://localhost:{DEFAULT_PORTS.get('audio_requests', 5560)}")
            subscriber.setsockopt(zmq.SUBSCRIBE, b"AudioRenderRequest")
            
            publisher = context.socket(zmq.PUB)
            publisher.bind(f"tcp://*:{DEFAULT_PORTS.get('audio_responses', 5561)}")
            
            # Give sockets time to connect
            await asyncio.sleep(0.5)
            
            logger.info("🎬 Starting full audio workflow test")
            logger.info("   1. Sending story transcript to Fire Core...")
            
            # Step 1: Send story transcript
            story_id = await self.send_story(story_content)
            logger.info(f"   Story sent with ID: {story_id}")
            
            logger.info("   2. Listening for AudioRenderRequest from Fire Core...")
            
            # Step 2: Listen for AudioRenderRequest (with timeout)
            timeout_seconds = 30
            end_time = asyncio.get_event_loop().time() + timeout_seconds
            
            while asyncio.get_event_loop().time() < end_time:
                try:
                    topic, message_data = await subscriber.recv_multipart(zmq.NOBLOCK)
                    
                    if topic == b"AudioRenderRequest":
                        message_dict = json.loads(message_data.decode('utf-8'))
                        request_id = message_dict.get("request_id", "unknown")
                        prompt = message_dict.get("prompt", "")
                        
                        logger.info(f"   ✅ Received AudioRenderRequest!")
                        logger.info(f"      Request ID: {request_id}")
                        logger.info(f"      Audio prompt: {prompt}")
                        
                        # Step 3: Respond with real audio file
                        logger.info(f"   3. Sending AudioReady with real audio file...")
                        
                        # Convert file path to file:// URL if it's a local file
                        if os.path.exists(audio_file_path):
                            audio_url = f"file://{os.path.abspath(audio_file_path)}"
                        else:
                            audio_url = audio_file_path
                        
                        audio_ready = AudioReady(
                            request_id=request_id,
                            url=audio_url,
                            success=True,
                            error_message=None
                        )
                        
                        await publisher.send_multipart([
                            b"AudioReady",
                            audio_ready.model_dump_json().encode('utf-8')
                        ])
                        
                        logger.info(f"   ✅ Sent AudioReady: {audio_url}")
                        logger.info("   4. Fire Core should now play the audio!")
                        logger.info("      Check Fire Core logs to see if audio playback started.")
                        
                        return True
                        
                except zmq.Again:
                    await asyncio.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error processing audio request: {e}")
                    return False
            
            logger.warning(f"   ⚠️  No AudioRenderRequest received within {timeout_seconds} seconds")
            logger.info("      This might mean:")
            logger.info("      - Fire Core is not running")
            logger.info("      - Fire Core is not configured to generate audio from transcripts")
            logger.info("      - The story content didn't trigger audio generation")
            
            return False
            
        except Exception as e:
            logger.error(f"Audio workflow test error: {e}")
            return False
        finally:
            # Clean up sockets
            try:
                if subscriber:
                    subscriber.close()
                if publisher:
                    publisher.close()
                if context:
                    context.term()
            except Exception:
                pass

    async def test_media_prompt_generation(self, story_content: str) -> Optional[Dict[str, str]]:
        """Test MediaPrompt generation using LLMPromptBuilder."""
        if LLMPromptBuilder is None or MediaPrompt is None:
            logger.error("LLMPromptBuilder or MediaPrompt not available")
            return None
        
        # This is a simplified test - in real usage, you'd need an LLM instance
        logger.info("Testing MediaPrompt generation (mock response)")
        
        # Mock a MediaPrompt response for testing
        mock_prompt = MediaPrompt(
            visual_prompt="ancient forest cathedral with towering pines, golden sunlight filtering through canopy, moss-covered logs, wildflowers",
            visual_negative_prompt="modern objects, people, buildings",
            audio_prompt="gentle forest sounds with rustling leaves and distant bird songs"
        )
        
        result = {
            "visual_prompt": mock_prompt.visual_prompt,
            "visual_negative_prompt": mock_prompt.visual_negative_prompt or "",
            "audio_prompt": mock_prompt.audio_prompt or ""
        }
        
        logger.info(f"Generated MediaPrompt:")
        logger.info(f"  Visual: {result['visual_prompt']}")
        logger.info(f"  Visual Negative: {result['visual_negative_prompt']}")
        logger.info(f"  Audio: {result['audio_prompt']}")
        
        return result


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
            print("\n=== Audio Testing ===")
            print("  7. Send audio render request")
            print("  8. Test audio playback (send AudioReady with real audio file)")
            print("  9. Test full audio workflow (story → AudioRenderRequest → AudioReady)")
            if MediaPrompt is not None:
                print(" 10. Test media prompt generation")
                print(" 11. Test combined media workflow")
            print("  0. Exit")

            choice = input("\nChoose option (0-11): ").strip()

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

            elif choice == "7":
                # Send audio render request
                print("\nAvailable sample audio prompts:")
                audio_options = list(SAMPLE_AUDIO_PROMPTS.keys())
                for i, name in enumerate(audio_options):
                    print(f"  {i+1}. {name.replace('_', ' ').title()}: {SAMPLE_AUDIO_PROMPTS[name]}")
                
                audio_choice = input(f"Choose audio prompt (1-{len(audio_options)}) or enter custom: ").strip()
                try:
                    if audio_choice.isdigit():
                        audio_index = int(audio_choice) - 1
                        if 0 <= audio_index < len(audio_options):
                            audio_name = audio_options[audio_index]
                            audio_prompt = SAMPLE_AUDIO_PROMPTS[audio_name]
                        else:
                            print("Invalid audio prompt selection.")
                            continue
                    else:
                        audio_prompt = audio_choice
                        
                    duration_input = input("Duration in seconds (default: auto): ").strip()
                    duration_s = None
                    if duration_input:
                        try:
                            duration_s = int(duration_input)
                        except ValueError:
                            print("Invalid duration, using auto")
                    
                    request_id = await client.send_audio_render_request(audio_prompt, duration_s)
                    if request_id:
                        print(f"Audio render request sent with ID: {request_id}")
                        print("Check the fire_core and image_server logs for audio generation results.")
                except Exception as e:
                    print(f"Error sending audio render request: {e}")

            elif choice == "8":
                # Test audio playback (send AudioReady with real audio file)
                print("\n🎵 Test Audio Playback with Fire Core")
                print("This sends an AudioReady message to Fire Core with a real audio file.")
                print()
                
                # Look for audio files in media directory
                media_dir = Path(__file__).parent.parent.parent.parent.parent / "media" / "audio"
                print(f"Looking for audio files in: {media_dir}")
                
                audio_files = []
                if media_dir.exists():
                    audio_files = list(media_dir.glob("*.wav")) + list(media_dir.glob("*.mp3"))
                
                if audio_files:
                    print("Available audio files:")
                    for i, file in enumerate(audio_files):
                        print(f"  {i+1}. {file.name}")
                    print(f"  {len(audio_files)+1}. Enter custom path")
                    
                    file_choice = input(f"Choose audio file (1-{len(audio_files)+1}): ").strip()
                    
                    if file_choice.isdigit():
                        file_index = int(file_choice) - 1
                        if 0 <= file_index < len(audio_files):
                            audio_file_path = str(audio_files[file_index])
                        elif file_index == len(audio_files):
                            audio_file_path = input("Enter audio file path: ").strip()
                        else:
                            print("Invalid file selection.")
                            continue
                    else:
                        audio_file_path = file_choice
                else:
                    print("No audio files found in media directory.")
                    audio_file_path = input("Enter audio file path: ").strip()
                
                if audio_file_path and (Path(audio_file_path).exists() or audio_file_path.startswith(('http://', 'https://', 'file://'))):
                    request_id = await client.send_audio_ready(audio_file_path)
                    if request_id:
                        print(f"✅ AudioReady sent with ID: {request_id}")
                        print(f"🎵 Audio file: {audio_file_path}")
                        print("Fire Core should now play this audio!")
                        print("Check Fire Core logs to see if playback started.")
                    else:
                        print("❌ Failed to send AudioReady message.")
                else:
                    print("❌ Invalid or missing audio file path.")

            elif choice == "9":
                # Test full audio workflow
                print("\n🔄 Test Full Audio Workflow")
                print("This tests the complete audio pipeline:")
                print("  1. Send story transcript to Fire Core")
                print("  2. Fire Core generates AudioRenderRequest")  
                print("  3. We respond with AudioReady containing real audio")
                print("  4. Fire Core plays the audio")
                print()
                
                # Choose story
                print("Available sample stories:")
                story_options = list(SAMPLE_STORIES.keys())
                for i, name in enumerate(story_options):
                    print(f"  {i+1}. {name.replace('_', ' ').title()}")
                
                story_choice = input(f"Choose story (1-{len(story_options)}) or enter custom: ").strip()
                
                if story_choice.isdigit():
                    story_index = int(story_choice) - 1
                    if 0 <= story_index < len(story_options):
                        story_name = story_options[story_index]
                        story_content = SAMPLE_STORIES[story_name].strip()
                    else:
                        print("Invalid story selection.")
                        continue
                else:
                    story_content = story_choice if story_choice else SAMPLE_STORIES['forest_discovery'].strip()
                
                # Choose audio file to respond with
                media_dir = Path(__file__).parent.parent.parent.parent.parent / "media" / "audio"
                audio_files = []
                if media_dir.exists():
                    audio_files = list(media_dir.glob("*.wav")) + list(media_dir.glob("*.mp3"))
                
                if audio_files:
                    print("\nChoose audio file to use in AudioReady response:")
                    for i, file in enumerate(audio_files):
                        print(f"  {i+1}. {file.name}")
                    print(f"  {len(audio_files)+1}. Enter custom path")
                    
                    file_choice = input(f"Choose audio file (1-{len(audio_files)+1}): ").strip()
                    
                    if file_choice.isdigit():
                        file_index = int(file_choice) - 1
                        if 0 <= file_index < len(audio_files):
                            audio_file_path = str(audio_files[file_index])
                        elif file_index == len(audio_files):
                            audio_file_path = input("Enter audio file path: ").strip()
                        else:
                            print("Invalid file selection.")
                            continue
                    else:
                        audio_file_path = file_choice
                else:
                    audio_file_path = input("Enter audio file path for AudioReady response: ").strip()
                
                if audio_file_path and (Path(audio_file_path).exists() or audio_file_path.startswith(('http://', 'https://', 'file://'))):
                    print(f"\n🚀 Starting full audio workflow test...")
                    success = await client.test_full_audio_workflow(story_content, audio_file_path)
                    if success:
                        print("✅ Full audio workflow test completed successfully!")
                    else:
                        print("❌ Full audio workflow test failed.")
                else:
                    print("❌ Invalid or missing audio file path.")

            elif choice == "10":
                # Test media prompt generation
                print("\nTesting MediaPrompt generation:")
                print("Available sample stories:")
                story_options = list(SAMPLE_STORIES.keys())
                for i, name in enumerate(story_options):
                    print(f"  {i+1}. {name.replace('_', ' ').title()}")
                
                story_choice = input(f"Choose story (1-{len(story_options)}) or enter custom: ").strip()
                if story_choice.isdigit():
                    story_index = int(story_choice) - 1
                    if 0 <= story_index < len(story_options):
                        story_name = story_options[story_index]
                        story_content = SAMPLE_STORIES[story_name].strip()
                    else:
                        print("Invalid story selection.")
                        continue
                else:
                    story_content = story_choice
                
                result = await client.test_media_prompt_generation(story_content)
                if result:
                    print("MediaPrompt generation test completed!")
                else:
                    print("MediaPrompt generation test failed.")

            elif choice == "11":
                # Test combined media workflow
                print("\nTesting combined media workflow (mock):")
                print("Available sample media prompts:")
                media_options = list(SAMPLE_MEDIA_PROMPTS.keys())
                for i, name in enumerate(media_options):
                    print(f"  {i+1}. {name.replace('_', ' ').title()}")
                
                media_choice = input(f"Choose media prompt (1-{len(media_options)}): ").strip()
                try:
                    media_index = int(media_choice) - 1
                    if 0 <= media_index < len(media_options):
                        media_name = media_options[media_index]
                        media_data = SAMPLE_MEDIA_PROMPTS[media_name]
                        
                        print(f"\nTesting combined workflow with: {media_name}")
                        print(f"Visual prompt: {media_data['visual']}")
                        print(f"Audio prompt: {media_data['audio']}")
                        
                        # Send visual prompt as debug prompt
                        visual_request_id = await client.send_debug_prompt(media_data['visual'])
                        print(f"Visual prompt sent with ID: {visual_request_id}")
                        
                        # Send audio render request  
                        audio_request_id = await client.send_audio_render_request(media_data['audio'])
                        if audio_request_id:
                            print(f"Audio prompt sent with ID: {audio_request_id}")
                        
                        print("Combined media workflow test completed!")
                        print("Check the logs to see both visual and audio processing.")
                    else:
                        print("Invalid media prompt selection.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            else:
                print("Invalid option. Please choose 0-10.")

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

        elif args.audio_prompt:
            # Send audio render request
            if args.audio_prompt in SAMPLE_AUDIO_PROMPTS:
                audio_prompt = SAMPLE_AUDIO_PROMPTS[args.audio_prompt]
                print(f"Using sample audio prompt: {args.audio_prompt}")
            else:
                audio_prompt = args.audio_prompt
                
            request_id = await client.send_audio_render_request(
                audio_prompt, 
                duration_s=args.audio_duration
            )
            if request_id:
                print(f"Audio render request sent with ID: {request_id}")
                print("Check the fire_core and image_server logs for audio generation results.")
            else:
                print("Failed to send audio render request")
        
        elif args.test_audio_playback:
            # Test audio playback by sending AudioReady
            audio_file_path = args.test_audio_playback
            
            print(f"� Testing Fire Core audio playback with: {audio_file_path}")
            request_id = await client.send_audio_ready(audio_file_path)
            if request_id:
                print(f"✅ AudioReady sent with ID: {request_id}")
                print("Fire Core should now play this audio!")
                print("Check Fire Core logs to see if playback started.")
            else:
                print("❌ Failed to send AudioReady message.")
        
        elif args.test_audio_workflow:
            # Test full audio workflow
            story_content = args.test_audio_workflow[0]
            audio_file_path = args.test_audio_workflow[1]
            
            print(f"� Testing full audio workflow")
            print(f"Story: {story_content}")
            print(f"Audio file: {audio_file_path}")
            
            success = await client.test_full_audio_workflow(story_content, audio_file_path)
            if success:
                print("✅ Full audio workflow test completed successfully!")
            else:
                print("❌ Full audio workflow test failed.")

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
    parser.add_argument(
        "--audio-prompt", "-a",
        type=str,
        help="Audio prompt to send for audio generation"
    )
    parser.add_argument(
        "--audio-duration",
        type=int,
        help="Duration in seconds for audio generation (default: auto)"
    )
    parser.add_argument(
        "--test-audio-playback",
        type=str,
        help="Test Fire Core audio playback by sending AudioReady with specified audio file"
    )
    parser.add_argument(
        "--test-audio-workflow", 
        nargs=2,
        metavar=("STORY", "AUDIO_FILE"),
        help="Test full audio workflow: send story, intercept AudioRenderRequest, respond with AudioReady"
    )
    parser.add_argument(
        "--list-audio-prompts",
        action="store_true",
        help="List available sample audio prompts and exit"
    )
    parser.add_argument(
        "--list-media-prompts",
        action="store_true",
        help="List available sample media prompts and exit"
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

    # List sample audio prompts if requested
    if args.list_audio_prompts:
        print("Available sample audio prompts:")
        for name, prompt in SAMPLE_AUDIO_PROMPTS.items():
            print(f"  {name}: {prompt}")
        return
    
    # List sample media prompts if requested
    if args.list_media_prompts:
        print("Available sample media prompts:")
        for name, data in SAMPLE_MEDIA_PROMPTS.items():
            print(f"  {name}:")
            print(f"    Visual: {data['visual']}")
            print(f"    Audio: {data['audio']}")
        return

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("fire_core_cli").setLevel(logging.DEBUG)
        print("Debug logging enabled")

    # Check for interactive mode or required parameters
    if args.interactive:
        asyncio.run(interactive_mode(args.debug))
    elif args.story or args.prompt or args.conversation or args.transcript or args.audio_prompt or args.test_audio_playback or args.test_audio_workflow:
        asyncio.run(command_line_mode(args))
    else:
        print("Error: Either --interactive mode, --story, --prompt, --conversation, --transcript, --audio-prompt, --test-audio-playback, or --test-audio-workflow must be specified.")
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    # $ uv run -m fire_core.cli -i
    sys.exit(main())