# Pipecat Backend for Experimance Agent

This directory contains the Pipecat backend implementation for the Experimance agent service. The Pipecat backend provides local audio processing with speech-to-text, LLM conversation, and text-to-speech capabilities running in a single process.

## Features

- **Local Audio Processing**: Uses PyAudio for real-time microphone and speaker access
- **Voice Activity Detection**: Silero VAD for detecting when users start/stop speaking
- **Speech-to-Text**: Whisper models (tiny to large) for transcription
- **Conversational AI**: OpenAI GPT models for natural language understanding and generation  
- **Text-to-Speech**: ElevenLabs for high-quality voice synthesis
- **Tool Integration**: Support for function calling and tool registration
- **Event System**: Real-time events for speech detection, transcription, and responses
- **ZMQ Integration**: Can publish events to other Experimance services

## Requirements

### System Dependencies

On Linux, you'll need audio development headers:

```bash
# Ubuntu/Debian
sudo apt install portaudio19-dev

# Fedora/CentOS
sudo yum install portaudio-devel
```

### Python Dependencies

The backend requires:
- `pipecat-ai[openai,whisper,elevenlabs]` - Core Pipecat framework
- `pyaudio` - Audio capture and playback
- `python-dotenv` - Environment variable management

### API Keys

You'll need API keys for:
- **OpenAI**: For GPT language models
- **ElevenLabs**: For text-to-speech synthesis

Set these as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"
```

## Quick Start

### Basic Usage

```python
from experimance_agent.backends import create_pipecat_backend, AgentBackendEvent

# Create backend with default settings
backend = create_pipecat_backend()

# Add event callbacks
async def on_transcription(event, data):
    print(f"User said: {data['transcription']}")

async def on_response(event, data):
    print(f"Agent responded: {data['response']}")

backend.add_event_callback(AgentBackendEvent.TRANSCRIPTION_RECEIVED, on_transcription)
backend.add_event_callback(AgentBackendEvent.RESPONSE_GENERATED, on_response)

# Start the backend
await backend.start()
await backend.connect()

# Backend will now listen for speech and respond
```


## Configuration

### PipecatConfig Options

```python
from experimance_agent.backends import PipecatBackend

config = {
    # Audio settings
    "audio_in_sample_rate": 16000,
    "audio_out_sample_rate": 16000, 
    "audio_in_enabled": True,
    "audio_out_enabled": True,
    "vad_enabled": True,
    
    # STT settings
    "whisper_model": "tiny",  # tiny, base, small, medium, large
    
    # LLM settings  
    "openai_api_key": "your-key",
    "openai_model": "gpt-4o-mini",
    
    # TTS settings
    "elevenlabs_api_key": "your-key", 
    "elevenlabs_voice_id": "EXAVITQu4vr4xnSDxMaL",  # Bella voice
    
    # System prompt
    "system_prompt": "You are a helpful AI assistant..."
}

backend = PipecatBackend(config)
```

### Performance Tuning

For different use cases:

**Fast Response (Low Latency)**:
```python
config = {
    "whisper_model": "tiny",
    "openai_model": "gpt-4o-mini", 
    "audio_in_sample_rate": 16000,
    "audio_out_sample_rate": 24000,
}
```

**High Quality**:
```python
config = {
    "whisper_model": "small",
    "openai_model": "gpt-4o",
    "audio_out_sample_rate": 44100,
}
```

**Balanced**:
```python
config = {
    "whisper_model": "base",
    "openai_model": "gpt-4o-mini",
    "audio_in_sample_rate": 16000,
    "audio_out_sample_rate": 24000,
}
```

## Tool Integration

Register custom tools that the agent can call:

```python
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"The weather in {location} is sunny and 75°F"

def control_lights(brightness: int, color: str = "white") -> str:
    """Control installation lighting."""
    return f"Set lights to {brightness}% brightness, {color} color"

# Register tools
backend.register_tool("get_weather", get_weather, "Get weather information")
backend.register_tool("control_lights", control_lights, "Control installation lights")
```

The LLM will automatically detect when to call these tools based on the conversation.

## Events

The backend emits events for integration with other services:

- `CONNECTED`: Backend connected and ready
- `DISCONNECTED`: Backend disconnected  
- `SPEECH_DETECTED`: User started speaking
- `SPEECH_ENDED`: User stopped speaking
- `TRANSCRIPTION_RECEIVED`: Speech-to-text completed
- `RESPONSE_GENERATED`: Agent response generated
- `TOOL_CALLED`: Agent called a registered tool
- `ERROR`: Error occurred

```python
async def handle_event(event, data):
    if event == AgentBackendEvent.TRANSCRIPTION_RECEIVED:
        # Send to other Experimance services via ZMQ
        await publish_to_core_service(data['transcription'])

backend.add_event_callback(AgentBackendEvent.TRANSCRIPTION_RECEIVED, handle_event)
```

## Integration with Experimance Services

The Pipecat backend integrates with the broader Experimance system:

```python
import zmq.asyncio
from experimance_common.zmq.services import ZMQPublisher

# Create ZMQ publisher for events
zmq_publisher = ZMQPublisher(port=5555)

async def forward_transcription(event, data):
    """Forward user speech to other services."""
    await zmq_publisher.publish("user_speech", {
        "text": data['transcription'],
        "timestamp": data['turn'].timestamp,
        "confidence": data.get('confidence', 1.0)
    })

async def forward_response(event, data):
    """Forward agent responses to other services.""" 
    await zmq_publisher.publish("agent_response", {
        "text": data['response'],
        "timestamp": data['turn'].timestamp
    })

backend.add_event_callback(AgentBackendEvent.TRANSCRIPTION_RECEIVED, forward_transcription)
backend.add_event_callback(AgentBackendEvent.RESPONSE_GENERATED, forward_response)
```

## Troubleshooting

### Audio Issues

**No microphone detected**:
```bash
# Check audio devices
python -c "import pyaudio; p=pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)}') for i in range(p.get_device_count())]"
```

**Permission denied**:
```bash
# Add user to audio group (Linux)
sudo usermod -a -G audio $USER
# Then log out and back in
```

**Sample rate mismatch**:
- Ensure your microphone supports the configured sample rate (typically 16kHz or 44.1kHz)
- Update `audio_in_sample_rate` in config to match your device

### Performance Issues

**High latency**:
- Use smaller Whisper model (`tiny` vs `large`)
- Reduce audio buffer sizes
- Use faster OpenAI model (`gpt-4o-mini` vs `gpt-4o`)

**Poor transcription quality**:
- Use larger Whisper model (`small` or `base`)
- Ensure good microphone quality and positioning
- Reduce background noise

**Choppy audio output**:
- Increase audio output buffer size
- Check CPU usage during synthesis
- Ensure stable network connection to ElevenLabs

### API Issues

**OpenAI rate limits**:
- Use `gpt-4o-mini` for lower costs
- Implement request queuing if needed
- Monitor usage in OpenAI dashboard

**ElevenLabs quota exceeded**:
- Monitor character usage
- Consider switching to alternative TTS if needed
- Implement fallback TTS service

## Architecture

The Pipecat backend follows this pipeline structure:

```
Microphone → VAD → Whisper STT → OpenAI LLM → ElevenLabs TTS → Speakers
     ↓              ↓                ↓              ↓
   Events       Transcription    Tool Calls    Response Events
     ↓              ↓                ↓              ↓  
   ZMQ Pub      Core Service    Tool Registry   Display Service
```

Key components:
- **LocalAudioTransport**: Manages microphone and speaker I/O
- **SileroVADAnalyzer**: Detects voice activity to trigger transcription
- **WhisperSTTService**: Converts speech to text
- **OpenAILLMService**: Processes conversation and generates responses
- **ElevenLabsTTSService**: Converts text responses to speech
- **Event Processors**: Capture pipeline events and forward to Experimance services

The entire pipeline runs in a single process with async/await for efficient concurrent processing.
