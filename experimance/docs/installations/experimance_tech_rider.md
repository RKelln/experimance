## **EXPERIMANCE — Technical Rider**

### **2.1 Overview**

Experimance is an intimate, tabletop interactive installation. Visitors sit at a table containing a bowl of sand. An AI agent converses with them, listening to their words and observing the shape of the sand, which visitors can sculpt with their hands. Responsive visuals are projected directly onto the sand surface, and the system generates an ambient audio-visual soundscape.

### **2.2 Venue requirements (general)**

* **Lighting:** Dark / controlled lighting is **critical**. The projection onto the sand needs darkness to be visible and effective.
* **Sound:** Quiet environment; the installation uses voice interaction, so background noise should be minimized.
* **Access:** Wheelchair-accessible path to the table.
* **Network:** Internet required for cloud services (AI/Image generation) and remote administration.
* **Maintenance:** Access to a vacuum cleaner /hand vaccuum for sand cleanup during install/strike is recommended.

### **2.3 Physical layout**

* **Installation Footprint:** Approx 6’ x 6’ area minimum (for table + chairs + movement).
* **Furniture:** 
    * One small table (approx 22"-25" height). Steps need to be taken to dampen vibration if the floor is bouncy.
    * Seating for 1-2 participants.
* **Hardware Mounting:** A camera/projector rig is clamped to the table edge or stands on the table.

### **2.4 Equipment**

**Compute (Single machine):**
* **Mini PC (Linux):** Runs all software components (control, vision, audio, display).
  * *Supplied by artist.* (e.g., Beelink SER8)

**Vision & Projection:**
* **Mounting Rig:** Desk mount arm holding sensors and projector above the bowl.
  * *Supplied by artist.*
* **Depth Camera:** Intel Realsense D415 (reads sand topology).
  * *Supplied by artist.*
* **Projector:** Pico/Mini laser projector (e.g., AAXA M8) projecting down into the bowl.
  * *Supplied by artist.*
* **Optional Projector**: (Preferred 4k) projector pointed at ceiling for animation loop.
  * Supplied by venue.
* **Webcam:** Used for presence detection (detecting when someone sits down).
  * *Supplied by artist.*

**Audio:**
* **Conference Puck:** Yealink SP92 (or similar) for microphone input and voice output.
  * *Supplied by artist.*
* **Environmental Speakers:** Small USB speakers (mounted under/near bowl) for tactile/local sound.
  * *Supplied by artist.*
* **External Audio:** Venue provides surround speakers for room-filling ambient music/soundscapes. Artist provides audio interface (USB or 3.5mm with ground loop isolation) to connect.

**Physical Interface:**
* **Small table:** approx 22-25" high, sturdy
  * *Supplied by artist* or venue (negotiable).
* **Ceramic Bowl:** ~11" Diameter.
  * *Supplied by artist.*
* **Sand:** Ethically sourced. Specific grain/type for projection and tactile feel.
  * *Supplied by artist.*
* **Fabric:** Black velvet or similar to cover table/mask cables.
  * *Supplied by artist.*

### **2.5 System Operation and Support (Artist-managed)**

Experimance is operated and maintained by the artist. The artist will be **onsite for installation/commissioning** and available for **remote support** during the run.

**Operational model**

* **Onsite commissioning:** Artist installs, calibrates (sand mapping), and tests the system.
* **Run-of-show:** Venue staff operate via simple power-on/power-off or start/stop procedures, with no software administration required.
* **Remote debugging:** Artist provides remote monitoring and troubleshooting.

**What the venue must provide (IT / access)**

* **Internet:** Stable connection (Wired Ethernet preferred). If Wi-Fi only, venue must provide a dedicated SSID with stable coverage.
* **Firewall/filters:** Outbound access required for show computer (AI APIs, Remote Access).
* **Remote support path:**
  * *(preferred)* Artist provides secure remote access (Tailscale/SSH).

**Reliability and fail-safe behavior**

* System is designed to auto-start on power-up.
* In event of internet loss, system enters a degraded mode (looping visuals with no new generations or conversation) until connectivity returns.

**Data, privacy, and consent**

* This work simulates a conversation.
* **Audio is captured** and sent to cloud providers (e.g. OpenAI/AssemblyAI) for processing but is generally **not stored** long-term by the artist (unless for debugging, specified in signage).
* Artist provides signage templates regarding data usage.

### **2.6 Power**

* **Total Draw:** Low power implementation. < 500W total.
* **Requirements:** One standard 15A circuit is sufficient for the entire installation (PC, Projector, Speakers).

### **2.7 Install / strike**

* **Setup Time:** ~4-6 hours.
* **Tasks:**
  * Table setup & draping.
  * Rig mounting & alignment.
  * Sand pouring & calibration (mapping projector to sand surface).
  * Audio test / room tuning.
  * Network config.

### **2.8 Operations (gallery staff)**

**Start of day**
1. Turn on the "Experimance - Main" power bar (or plug in).
2. Wait approx 2-3 minutes for boot & auto-start.
3. Confirm projector is on (may need manual power button if not CEC compliant) and aligned with bowl.
4. Confirm audio "hello" or ambient sound.

**During day**
* If sand spills: lightly brush back into bowl (brush provided).
* If system unresponsive: Press physical "Reset" button (if equipped) or power cycle power bar.

**End of day**
1. Turn off power bar. (System handles safe shutdown or is tolerant of power cut).

### **3.0 Shipping / Transport**

* The entire installation (excluding table if venue-provided) fits in one standard Pelican case or large carry-on suitcase.
* **Weight:** < 40 lbs (excluding heavy table).

---

# RESPONSIBILITIES MATRIX

| Component                      | Responsible Party              |
| :----------------------------- | :----------------------------- |
| **Core Hardware**              |                                |
| Mini PC (Control Unit)         | Artist                         |
| Realsense Depth Camera         | Artist                         |
| Mini Laser Projector           | Artist                         |
| Mounting Rig/Arm               | Artist                         |
|                                |                                |
| **Physical Setup**             |                                |
| Side Table                     | **Venue / Negotiable**         |
| Chairs / Seating               | **Venue**                      |
| Ceramic Bowl & Sand            | Artist                         |
| Black Cloth / Draping          | Artist                         |
|                                |                                |
| **Audio**                      |                                |
| Microphone/Speaker Puck        | Artist                         |
| Audio Interface / DI           | Artist / Negotiable            |
| Local effects speakers         | Artist                         |
| Room fill speakers             | **Venue**                      |
|                                |                                |
| **Infrastructure**             |                                |
| Internet (Ethernet preferred)  | **Venue**                      |
| Power (1x 15A outlet)          | **Venue**                      |
| Extension cords / Power strips | **Venue** (Artist brings some) |
|                                |                                |
| **Staffing**                   |                                |
| Installation / Tuning          | Artist                         |
| Daily Operations               | **Venue**                      |
