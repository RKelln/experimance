## **FEED THE FIRES — Technical Rider**

### **2.1 Overview**

Feed the Fires is a voice-interactive installation. A “fire spirit” voice agent listens to visitor stories, generates responsive visuals (panoramic projections) and ambient audio, and drives a central firepit projection \+ lighting \+ smoke effects.

### **2.2 Venue requirements (general)**

* **Lighting:** dark / controlled lighting strongly preferred.  
* **Sound:** conversational; audience must hear the fire spirit clearly.  
* **Access:** wheelchair-accessible path to seating circle; clear egress maintained.  
* **Network:** internet required for cloud services and remote administration.

### **2.3 Physical layout (Full 360 configuration)**

* **Central firepit:** 4’×4’ custom structure with frosted acrylic top, internal projector, LEDs/coals, water vapor \+ fans (“smoke”).  
* **Seating:** 4 benches around firepit.  
* **Projection:** 4+ projectors for 360° wall coverage (1+ per wall in rectangular room).

### **2.4 Equipment (what you had at InterAccess)**

**Compute (two-machine, distributed):**

* Linux PC (custom control software, and image generation if GPU are available).  
  * Supplied by venue or rented.  
* macOS machine (voice agent, TouchDesigner for fire).  
  * Supplied by artist.

**Audio I/O:**

* Conference speaker/mic used for voice chat (with echo cancellation).  
  * Supplied  
* Optional handheld wireless mic (for openings or other situations where background noise is considerable).

**Presence sensing:**

* Camera-based presence detection (Reolink). Needs to be mounted on the ceiling or 8’ or so high on the wall.

### **2.5 System Operation and Support (Artist-managed)**

Feed the Fires is operated and maintained by the artist. The venue does not need to install or administer software. The artist will be **onsite for installation/commissioning** and available for **remote support** during the run.

**Operational model**

* **Onsite commissioning:** Artist installs, configures, calibrates, and tests the complete system.  
* **Run-of-show:** Venue staff operate via simple start/stop/restart procedures (provided), with no software administration required.  
* **Remote debugging:** Artist provides remote monitoring and troubleshooting; venue provides basic access to power/network as specified below.

**What the venue must provide (IT / access)**

* **Internet:** Stable connection for the show computer(s) (wired Ethernet preferred). If Wi-Fi only, venue must provide a dedicated SSID with stable coverage at the installation location.  
* **Firewall/filters:** Outbound access for the show computer must not be blocked by aggressive captive portals or strict egress rules. (If restrictions exist, venue IT to coordinate with artist in advance.)  
* **Remote support path (choose one):**  
  * Venue provides a temporary VPN account or remote support method approved by venue IT, **or**  
  * *(preferred)* Artist provides a secure remote method (e.g., Tailscale/SSH) that venue IT approves during install.  
* **No inbound access required** unless specifically requested by venue IT.

**Reliability and fail-safe behavior**

* The system is designed to recover from common faults via **restart/reset** procedures.  
* In the event of internet interruption, the installation will enter a defined degraded mode (to be specified by artist: e.g., ambient loop / local fallback / pause state) until connectivity returns.  
* Venue staff are never asked to edit configuration files, run command-line tools, or manage accounts.

**Escalation \+ response**

* The artist provides a **single point of contact** for technical support (phone \+ email).  
* Remote support hours: 9am-5pm EST. Emergency contact available for show-critical failures.  
* Venue agrees to allow basic actions requested by the artist (restart computer, power-cycle specific devices, check connections).

**Data, privacy, and consent (high-level)**

* This work involves visitor speech input (microphone) used to generate the live experience.  
* The artist will provide venue-ready signage describing: what is captured (audio/transcript), whether anything is stored, and how to opt out.  
* Venue staff are not responsible for managing data systems.

### **2.6 Power**

(To fill with your measured numbers—don’t guess in the final.)

* Full 360 version: 4 projectors \+ compute \+ firepit projector \+ LEDs \+ vapor machine \+ audio.  
* Dedicated circuits recommended; specify minimum once you confirm.

### **2.7 Install / strike**

* **All variations:**  
  * Firepit placement  
  * Calibration (lighting, audio and camera tuning)  
  * Network testing  
  * Camera installation (requires lift/ladder)  
* **Full 360:**   
  * Projection alignment (may require lift/ladder)  
* **Compact:**  
  * Projection alignment (may require lift/ladder)

### **2.8 Operations (gallery staff)**

**Start of day**

1. Turn on power bar labeled “FEED THE FIRES — MAIN”  
2. Press “START” on the control interface  
3. Confirm: audio voice is audible, projections active, firepit responsive

**During day**

* If audio is too quiet: adjust “VOICE VOLUME” slider  
* If the system seems stuck: press “RESET SESSION”

**If something breaks (3-step protocol)**

1. Press “RESTART EXPERIENCE”  
2. If not fixed: reboot show computer (instructions)  
3. If not fixed: call/text artist (contact)

### **2.9 Accessibility \+ privacy**

Because this is voice \+ story capture, you need a short policy section:

* Transcripts of the audience and AI speaking are captured to be used only for debugging and improving the piece itself.  
* Audio and text are sent to AI cloud services.  
* Consent signage provided by artist.  
* Firepit activates by visual presence and can be experienced without talking, although no projected images will be generated.

### 

### **3.0 Shipping**

* Firepit structure (crated)  
  * Size, weight  
* Internal projector  
  * Size, weight  
* Control computer(s) \+ interfaces  
  * Size, weight  
* Vapor system  
  * Size, weight  
* Essential cables  
* Printed quickstart \+ signage templates

# RESPONSIBILITIES MATRIX

| Core components |  |
| :---- | :---- |
| Firepit structure (4’×4’) | Artist |
| Firepit internal projector | Artist |
| LEDs/coals \+ control electronics | Artist |
| Smoke/vapor system \+ fans | Artist |
| Consumable **distilled** water | Venue |
| macOS mini PC inside firepit | Artist |
| Linux desktop | Venue / Either |
|  |  |
| **Projection / display** |  |
| Projectors (x4+ for full 360\) | Venue / Rental |
| Projector stands / mounts (is needed) | Venue |
| HDMI and ethernet cabling | Artist |
| Cabling spares | Venue |
|  |  |
| **Audio** |  |
| PA / powered speakers | Venue |
| Conference speaker-mic | Artist |
| Handheld mic and audio interface | Venue / Rental |
|  |  |
| **Networking** |  |
| Internet (hardline preferred) | Venue |
|  |  |
| **Power / rigging** |  |
| Power distribution (venue standard) | Venue |
| Gaffer tape / cable ramps | Venue |
| Ladder/lift (if needed) | Venue |
|  |  |
| **Soft goods / environment** |  |
| Seating / benches | Venue |
| Wood chips | Venue |
|  |  |
| **Staffing** |  |
| Venue tech (install) | Venue |
| Artist (install / remote) | Artist |

