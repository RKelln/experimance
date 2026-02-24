# Experimance

Experimance is an interactive installation with body, perception, and voice. Visitors gather around a shallow field of white sand and sculpt its surface with their hands. An overhead depth camera measures the terrain, and custom software translates these changes into a living projected landscape that reads like satellite imagery, but refuses stable realism. The work’s “body” is the sand; its “perception” is the sensing and interpretation pipeline; its “voice” (a speech-to-speech chatbot) addresses the room as the installation itself. The piece knows itself, how and why it was made, and its self-critique is looped back as authored proto-selfhood.

The projected world follows a fixed, cyclical arc: it always begins as complete wilderness. Over time it moves through eras of human technology and development as roads, industry, extraction scars, and computational motifs gradually enter the image. The trajectory may reach a utopian or dystopian phase, then turns into collapse that cycles back to wilderness again. The audience interactions with the sand set the tempo and drift: gentle, smaller gestures slow the system’s acceleration and keep the world within livable trajectories longer.

This interaction rule is the piece’s core argument. Experimance is about experimentation without consent and without regard for consequence; first as environmental transformation (climate catastrophe as civilization-scale feedback), and second as a metaphor for the accelerating experimentation of AI. In both domains, new capabilities are often exploited in ways that concentrate wealth and power in a small minority, which can then obscure risks, delay accountability, and externalize harm. The installation makes that structure felt in the body while the satellite viewpoint imparts the false omniscience of spy satellites and video games. Speaking with it is first playful, then uncanny: an experiment in intelligence, our future’s unintended inheritance. The awe and horror of what we build.


# Use of AI

Images are generated based on depth maps of the sand topology fed to the Juggernaut XL Lightning text-to-image model, a derivative of Stable Diffusion XL that has been refined through two custom LoRAs: one trained by Ryan Kelln on his generated images, and the other on drone photography by Lord Jia. The prompts themselves are AI-generated based on era of human development and biome. Environmental audio is sourced from Stable Audio, while the installation features conversational AI capabilities for speech-to-speech interaction. AI coding assistance from Anthropic Claude and OpenAI GPT has supported the development process. Refer to the Software section below for additional details.


# Software

Custom written Python and Supercollider scripts by Ryan Kelln.

AI and other software services used (subject to change based on availability and improvements):
- Assembly AI: Speech-to-text service (Universal Streaming)
- Cartesia: Text-to-speech service (Sonic v2)
- OpenAI: Chat agent (GPT-4o)
- Vast.ai: Cloud compute for AI generated images
- Tailscale: for remote admin access to both machines
- ntfy.sh: push notifications for monitoring


# Hardware

Compute (Single machine):
* Mini PC (Linux): Runs all software components (control, vision, audio, display).
  * Supplied by artist. (e.g., Beelink SER8)
* Note: images generated on Vast.ai, unless local GPU available.

Vision & Projection:
* Mounting Rig: Desk mount arm holding sensors and projector above the bowl.
  * Supplied by artist.
* Depth Camera: Intel Realsense D415 (reads sand topology).
  * Supplied by artist.
* Projector: Pico/Mini laser projector (e.g., AAXA M8) projecting down into the bowl.
  * Supplied by artist.
* Optional Projector: Preferred 4k projector pointed at ceiling for animation loop.
  * Supplied by venue.
* Webcam: Used for presence detection (detecting when someone sits down).
  * Supplied by artist.

Audio:
* Conference Puck: Yealink SP92 (or similar) for microphone input and voice output.
  * Supplied by artist.
* Environmental Speakers: Small USB speakers (mounted under/near bowl) for tactile/local sound.
  * Supplied by artist.
* External Audio: Venue provides surround speakers for room-filling ambient music/soundscapes. Artist provides audio interface (USB or 3.5mm with ground loop isolation) to connect.

Physical Interface:
* Small table: approx 22-25" high, sturdy
  * Supplied by artist or venue (negotiable).
* Ceramic Bowl: ~11" Diameter.
  * Supplied by artist.
* Sand: Ethically sourced. Specific grain/type for projection and tactile feel.
  * Supplied by artist.
* Fabric: Black velvet or similar to cover table/mask cables.
  * Supplied by artist.