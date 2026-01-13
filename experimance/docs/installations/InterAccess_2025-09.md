# Feed the Fires installation documentation

*InterAccess, Toronto ON*  
September 10 - October 11, 2025 (Tuesday-Saturday, 11am-6pm)

_git tag:_ `Feed-the-Fires-v1.0`

## Exhibition Context

**Presented by the Sôhkêpayin Collective**
`
A transformative installation resulting from a unique collaboration during a trip to Japan, organized by the Ainu. Five Indigenous creatives from the Five Directions came together in the East to explore the intersection of artificial intelligence and environmental stewardship. Through sharing knowledge and traditions related to fire, the participants sparked discussions on how technology can deepen our connection with nature.

## Equipment List

### Ubuntu Linux Machine (ia360.local)
- **IA 360 Projection system**
  - 2x A100 Nvidia graphics cards
      1) Image prompt to panorama images (base image + 3 tiles) by Juggernaut XI Lightning by RunDiffusion (SDXL custom model).
      2) Audio prompt to audio by TangoFlux.
  - runs core software components (core, image_server, display, health services)
  - controls Matter smart plug via chip-tool
  

### macOS Machine (FireProjects-Mac-mini.local / iamini)
- **[Mac mini](https://www.apple.com/ca/mac-mini/)**:
  - runs voice agent and health monitoring
  - runs TouchDesigner for fire projection
  - handles microphone input and speaker output
- **[Yealink SP92 Conference Speaker and Microphone](https://www.amazon.ca/Yealink-SP92-Cancellation-Full-Duplex-Speakerphone/dp/B0F28HTJP3)**:
  - for voice chat (with built-in echo cancellation)
- **[Handheld wireless microphone]()**:
  - additional microphone option for audience interaction

### Central Firepit
- **Custom 4'x4' firepit structure**:
  - fabrication by [Cultivate Collective](https://www.8kollective.com/)
  - contains projector for fire surface projection
  - frosted acrylic top surface
  - LED lights and "coals" for flickering light effects
  - water vapour machine with fans for billowing "smoke" effect
- **4 wooden benches**:
  - surrounding firepit as invitation to sit
- **TouchDesigner system** by [Kyle Duffield](https://www.kyleduffield.com/):
  - controls fire projection, LEDs, and smoke effects

### 360 Projection system
- **4x [1920x1080 projectors]()**:
  - 360° wall projection (4 walls of rectangular room)

### Other
- **Ethernet switch with PoE**
  - connects two computers and Reolink camera (over PoE) and to the Internet
- **[Reolink Camera](https://reolink.com/)** (192.168.2.229):
  - presence detection for audience
- **[TP-Link Kasa Smart Plug](https://www.tp-link.com/ca/home-networking/smart-plug/)** (Matter device ID: 110):
  - installation remote power control
  - NOTE: never got fully working

## Services

### Cloud Services
- **[Assembly AI](https://www.assemblyai.com/)**:
  - Speech-to-text service (Universal Streaming)
- **[Cartesia](https://cartesia.ai/)**:
  - Text-to-speech service (Sonic v2)
  - Voice ID: bf0a246a-8642-498a-9950-80c35e9276b5 (fire spirit voice)
- **[OpenAI](https://openai.com/)**:
  - Chat agent (GPT-4o) - plays the role of fire spirit
  - Prompt-to-prompt conversion for image and audio generation
- **Tailscale**:
  - for remote admin access to both machines
- **ntfy.sh**:
  - push notifications for monitoring
  - topic: fire-alerts

## Software

### Operating Systems
- **Ubuntu 24.04** (ia360.local):
  - Linux machine running core services
- **macOS** (FireProjects-Mac-mini.local):
  - Mac machine running agent services

### Experimance Software
- **Git branch**: `fire`
- **Git tag**: `Feed-the-Fires-v1.0`
- **Project**: `fire`

### Python Environment
```
Python: 3.11+
uv: 0.8.0+
Pipecat: AI conversation framework
```

---

## Installation References

For complete setup procedures, see:
- **Multi-machine deployment**: [`infra/docs/multi_machine_deployment.md`](../../infra/docs/multi_machine_deployment.md) - Distributed service setup, SSH config
- **Gallery automation**: [`projects/fire/README.md`](../../projects/fire/README.md) - Gallery hour scheduling, coordinated control
- **Matter device setup**: [`docs/smart_plug_matter_control.md`](../smart_plug_matter_control.md) - Smart plug pairing and control

---

## System State Snapshot

### Distributed Architecture

Four micro-services cooperate via ZeroMQ event bus, distributed across two machines:

| Service            | Machine | Role                                                 | Key Messages                    |
| ------------------ | ------- | ---------------------------------------------------- | ------------------------------- |
| **`fire_core`**    | Ubuntu  | Builds prompts, orchestrates rendering, routes media | `RenderRequest`, `DisplayMedia` |
| **`image_server`** | Ubuntu  | Generates images (fast pass + 6 tiles)               | `ImageReady`, `AudioReady`      |
| **`display`**      | Ubuntu  | Assembles & renders panorama, shader squashes × 6    | Receives `DisplayMedia`         |
| **`fire_agent`**   | macOS   | Collects audience stories via voice                  | `TranscriptUpdate` → Core       |
| **`health`**       | Both    | Health monitoring and notifications                  | —                               |

### Data Flow Timeline

1. **0 s** – Story finishes → half-length strip request sent to image_server
2. **~1 s** – Strip arrives, displayed blurred on screen
3. **≤5 s** – Three high-resolution tiles fade in, mirrored to reach full panoramic length

---

## Deployment Configuration

### Ubuntu Machine (ia360.local)

**Services**: core, image_server, display, health  
**User**: experimance  
**Mode**: prod  
**Matter Controller**: Yes (controls TP-Link smart plug)

```bash
# Deploy and start Ubuntu services
sudo ./infra/scripts/deploy.sh fire install prod
sudo ./infra/scripts/deploy.sh fire start

# Verify services
sudo systemctl status "*@fire"

# Check health
curl http://localhost:8080/health
```

### macOS Machine (iamini / FireProjects-Mac-mini.local)

**Services**: agent, health  
**User**: fireproject  
**Mode**: prod

```bash
# Deploy and start macOS services  
sudo ./infra/scripts/deploy.sh fire install prod
sudo ./infra/scripts/deploy.sh fire start

# Verify services
sudo launchctl list | grep fire
```

### Gallery Control System

For coordinated multi-machine control:

```bash
# Interactive gallery control (from any machine)
python infra/scripts/ia_gallery.py

# Command-line controls
python infra/scripts/ia_gallery.py --start     # Start all services on both machines
python infra/scripts/ia_gallery.py --stop      # Stop all services on both machines
python infra/scripts/ia_gallery.py --status    # Check service status on both machines
```

### Gallery Hour Automation

Automatic scheduling for gallery hours (Tuesday-Saturday, 10:55 AM - 6:05 PM):

```bash
# Setup automatic gallery hours (macOS only)
./infra/scripts/launchd_scheduler.sh fire setup-schedule gallery
```

---

## Image Generation Technical Details

### Model & API
- **Model**: [Juggernaut XI Lightning by RunDiffusion](https://www.rundiffusion.com/juggernaut-xi) (SDXL custom model)
- **Format**: 360° equirectangular panorama
- **Resolution**: 1920×340 (strip), 1344×768 (tiles)

### Generation Workflow
1. **Fast pass**: Half-length strip (blurred placeholder)
2. **High-resolution tiles**: 6 tiles generated in parallel
3. **Display assembly**: Tiles fade in as they arrive, mirrored to complete panorama

### Prompt Engineering
- Base system prompt guides image generation for dream-like memories
- Location extraction from story transcript
- Prompts emphasize: 360° equirectangular view, dream-like, impressionistic, cinematic, muted colors, soft edges

---

## Audio Generation

- **Service**: TangoFlux
- **Purpose**: Ambient environmental audio based on story context
- **Integration**: Generated alongside images, routed through fire_core

---

## Projection Setup

### Wall Projection
- **4 projectors** arranged for 360° coverage
- **Resolution**: 1920×1080 per projector
- **Content**: Panoramic images from AI generation
- **Rendering**: experimance_display service with shader effects

### Firepit Projection
- **TouchDesigner system** (Kyle Duffield)
- **Content**: Dynamic fire visualization, reactive to audience interaction
- **LED integration**: Synchronized with projection for realistic fire effect
- **Smoke effects**: Water vapor machine triggered on audience arrival

---

## Network Configuration

### Machine Hostnames
- Ubuntu: `ia360.local` (SSH: `ia360`)
- macOS: `FireProjects-Mac-mini.local` (SSH: `iamini`)

### Service Ports
- Health monitoring: 8080 (both machines)
- ZeroMQ: See `experimance_common.constants.DEFAULT_PORTS`

### Camera
- Reolink: 192.168.2.229
- Credentials in `.env` file

---

## Lessons Learned

- Need simple way for gallery staff to control all the installation, made some progress with CLI menu but failed to get everything automated using Matter plug
  - Staff needs ability to override automation, etc
- Handheld mic worked reasonably well to isolate noise duringthe opening but no one could hear what the speaker was saying
  - Maybe need to display the transcript of what is said? (Optionally or when noisy?)
- In general the speech to text transcription seemed poor?
- The blur effect and smooth transitions worked between images, knowing when to update the image was less successful
- Some people never really engaged or told a story and didn't discover the image generation


---

## Credits

### Sôhkêpayin Collective

**Indigenous Ainu Host**: Mayunkiki (Mai Hachiya)

**Indigenous Led Contributors**: Susan Blight, Rheanne Chartrand, Dr. Desiree Hernandez Ibinarriaga, Dr. James Oliver, Howard Munroe, Jason Baerg

**Allied Contributors**: Fran Rawlings, Calla Lee, Ziyan Hossain

**Japanese Contributors**: Kanoko Tamura, Tomoko Momiyama

**Technical and Creative Contributors**:
- Ryan Kelln (AI integration: custom Python software)
- [Kyle Duffield](https://www.kyleduffield.com/) (fire projection & LEDs: TouchDesigner)
- [Cultivate Collective](https://www.8kollective.com/): Mike Dunn & Robbie Foti (firepit fabrication, water vapour machine)

**Photography**: [Liam Mackenzie](https://www.liammackenzie.com/), [Benjamin Lappalainen](https://www.blap64.xyz/)

---

## Additional Resources

- **Project website**: [https://www.ryankelln.com/project/feed-the-fires/](https://www.ryankelln.com/project/feed-the-fires/)
- **Source code**: [https://github.com/RKelln/experimance/tree/fire](https://github.com/RKelln/experimance/tree/fire)
- **Venue**: [InterAccess](https://www.interaccess.org/)
