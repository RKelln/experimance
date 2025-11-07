# Smart Plug Matter Control System

This document covers the setup, configuration, and operation of the TP-Link Kasa smart plug system for remote power control of the Fire installation at IA Gallery.

## Overview

The smart plug system provides remote power control for the installation using the Matter protocol. This allows gallery staff to:
- Turn the installation power on/off remotely
- Schedule automatic power control for gallery hours
- Monitor power status of the installation

**Hardware:** TP-Link Kasa smart Wi-Fi plug with Matter support
**Protocol:** Matter (Thread/Wi-Fi)
**Control Software:** chip-tool (Matter controller) + custom Python scheduler

## System Components

### 1. Smart Plug Device
- **Model:** TP-Link Kasa smart Wi-Fi plug with Matter support
- **Device ID:** 110 (configured in system)
- **Vendor ID:** 0x1391 (TP-Link)
- **Product ID:** 0x0101
- **Network:** Connects to gallery Wi-Fi, communicates via Matter protocol

### 2. Matter Controller (ia360 machine)
- **Software:** chip-tool (Matter controller application)
- **Installation:** Ubuntu snap package
- **Function:** Sends Matter commands to smart plug
- **Location:** Ubuntu machine at gallery

### 3. Gallery Control Interface
- **Script:** `infra/scripts/ia_gallery.py`
- **Interface:** Interactive menu + command-line flags
- **Features:** Manual control, status monitoring, smart startup/shutdown

### 4. Automated Scheduler
- **Script:** `scripts/matter_scheduler.py`
- **Service:** Systemd user service (auto-start)
- **Configuration:** `projects/fire/matter_schedule.toml`
- **Function:** Automatic power control for gallery hours

## Initial Setup (Off-Site Testing)

### 1. Smart Plug Initial Pairing

Before bringing the smart plug to the gallery, pair it with your Matter controller:

```bash
# Install chip-tool (Ubuntu/Linux only)
sudo snap install chip-tool

# Put smart plug in pairing mode (hold button until LED blinks)
# Then commission it with pairing code
chip-tool pairing code 110 <pairing code> --bypass-attestation-verifier
```

**Note:** The pairing code is stored in `projects/fire/.env` as `MATTER_SMART_PLUG_PAIRING_CODE`.

### 2. Test Basic Control

```bash
# Turn ON
chip-tool onoff on 110 1

# Turn OFF  
chip-tool onoff off 110 1

# Check status
chip-tool onoff read on-off 110 1
```

### 3. Verify Network Requirements

Ensure your test network supports:
- **Wi-Fi:** 2.4GHz or 5GHz (smart plug connects to Wi-Fi)
- **Matter:** IPv6 support recommended
- **Firewall:** Allow Matter protocol traffic (UDP port 5540)

## Gallery Installation Setup

### 1. Network Configuration

**At IA Gallery:**
1. Connect smart plug to gallery Wi-Fi network
2. Ensure gallery network supports Matter protocol
3. Test connectivity from ia360 machine to smart plug

### 2. Re-Pairing at Gallery (if needed)

If the smart plug doesn't work after moving to gallery Wi-Fi:

```bash
# SSH to ia360 machine
ssh experimance@ia360.local

# Navigate to project directory
cd /path/to/experimance

# Reset and re-pair the device
chip-tool pairing code 110 <pairing code> --bypass-attestation-verifier
```

### 3. Automated Installation

Use the gallery control script for complete setup:

```bash
# SSH to ia360 machine
ssh experimance@ia360.local

# Run complete installation (includes chip-tool, pairing, scheduler)
python3 infra/scripts/ia_gallery.py --install
```

This will:
- Install chip-tool via snap
- Auto-pair the smart plug using stored pairing code
- Set up the automated scheduler for gallery hours
- Install systemd service for auto-start control interface

### 4. Manual Installation Steps

If automated installation fails, follow manual steps:

```bash
# 1. Install chip-tool
python3 infra/scripts/ia_gallery.py --install-chip-tool

# 2. Set up Matter devices
python3 infra/scripts/ia_gallery.py --setup-matter

# 3. Set up gallery hour scheduling
python3 infra/scripts/ia_gallery.py --setup-gallery

# 4. Install systemd service
python3 infra/scripts/ia_gallery.py --install
```

## Gallery Operation

### Daily Gallery Control Interface

Gallery staff can use the interactive menu for daily operations:

```bash
# SSH to ia360 or run locally on ia360
python3 infra/scripts/ia_gallery.py
```

**Menu Options:**
- **7. Turn installation power ON** - Manually turn on power
- **8. Turn installation power OFF** - Manually turn off power
- **5. Smart startup** - Turn on power → Start services
- **6. Smart shutdown** - Stop services → Turn off power
- **9-11. Scheduler control** - Manage automatic gallery hours

### Command Line Control

For quick control without menu:

```bash
# Power control
python3 infra/scripts/ia_gallery.py --plug-on      # Turn ON
python3 infra/scripts/ia_gallery.py --plug-off     # Turn OFF
python3 infra/scripts/ia_gallery.py --plug-toggle  # Toggle

# Smart control (services + power)
python3 infra/scripts/ia_gallery.py --smart-start  # Power on → Services
python3 infra/scripts/ia_gallery.py --smart-stop   # Services → Power off

# Scheduler control
python3 infra/scripts/ia_gallery.py --scheduler-start  # Start auto-scheduler
python3 infra/scripts/ia_gallery.py --scheduler-stop   # Stop auto-scheduler
python3 infra/scripts/ia_gallery.py --scheduler-status # Check status
```

### Gallery Hours Automation

The system automatically controls power based on gallery hours:

**Schedule (Tuesday-Saturday):**
- **Tuesday-Friday:** 10:55 AM ON → 6:05 PM OFF
- **Saturday:** 10:55 AM ON → 9:05 PM OFF
- **Sunday-Monday:** OFF (gallery closed)

**Configuration:** `projects/fire/matter_schedule.toml`

```toml
# Gallery hours - Tuesday through Friday
[[schedules]]
name = "Gallery Open - Weekdays"
cron = "55 10 * * 2-5"  # 10:55 AM, Tue-Fri
command = "on"
device_id = 110

[[schedules]]
name = "Gallery Close - Weekdays"  
cron = "5 18 * * 2-5"   # 6:05 PM, Tue-Fri
command = "off"
device_id = 110

# Saturday extended hours
[[schedules]]
name = "Gallery Close - Saturday"
cron = "5 21 * * 6"     # 9:05 PM, Saturday
command = "off"
device_id = 110
```

## Configuration Files

### 1. Deployment Configuration
**File:** `projects/fire/deployment.toml`

```toml
[machines.ubuntu]
matter_controller = true  # Designates this machine as Matter controller
matter_devices = [
    { 
        id = 110, 
        type = "smart_plug", 
        name = "Installation Power",
        description = "TP-Link Kasa smart plug for installation power control"
    }
]
matter_setup = { install_chip_tool = true, auto_pair = true }
```

### 2. Environment Variables  
**File:** `projects/fire/.env`

```bash
# Matter device pairing codes and configuration (keep these secure)
MATTER_SMART_PLUG_PAIRING_CODE=<pairing code>
MATTER_BYPASS_ATTESTATION=true
MATTER_COMMISSIONING_TIMEOUT=120
```

### 3. Schedule Configuration
**File:** `projects/fire/matter_schedule.toml`

Contains cron-style scheduling for automatic power control during gallery hours.

## Troubleshooting

### Smart Plug Not Responding

1. **Check network connectivity:**
   ```bash
   # Test basic network from ia360
   ping 8.8.8.8
   
   # Check if chip-tool is installed
   chip-tool
   ```

2. **Check device status:**
   ```bash
   # Read current state
   chip-tool onoff read on-off 110 1
   
   # If this fails, device may need re-pairing
   ```

3. **Re-pair device:**
   ```bash
   # Put smart plug in pairing mode (hold button)
   chip-tool pairing code 110 <pairing code> --bypass-attestation-verifier
   ```

### Scheduler Not Working

1. **Check scheduler service:**
   ```bash
   systemctl --user status matter-scheduler
   journalctl --user -u matter-scheduler -f
   ```

2. **Test manual scheduling:**
   ```bash
   python3 scripts/matter_scheduler.py
   ```

3. **Verify schedule configuration:**
   ```bash
   cat projects/fire/matter_schedule.toml
   ```

### Network Issues

1. **Check Wi-Fi connection of smart plug:**
   - LED should be solid (not blinking)
   - May need to reconnect to gallery Wi-Fi

2. **Check Matter protocol support:**
   - Gallery network must support IPv6
   - Firewall may block Matter traffic (UDP 5540)

3. **Test from different machine:**
   ```bash
   # Try chip-tool commands from laptop on same network
   chip-tool onoff read on-off 110 1
   ```

### Gallery Control Interface Issues

1. **SSH connection problems:**
   ```bash
   python3 infra/scripts/ia_gallery.py --test
   ```

2. **Service status issues:**
   ```bash
   python3 infra/scripts/ia_gallery.py --status
   ```

3. **Matter controller detection:**
   - Ensure you're running on ia360 machine
   - Check `deployment.toml` has `matter_controller = true`

## Security Considerations

1. **Pairing Code Security:**
   - Pairing code stored in `.env` file (not in git)
   - Code only needed during initial pairing
   - Change default pairing code if possible

2. **Network Security:**
   - Smart plug connects to gallery Wi-Fi
   - Matter protocol uses encryption
   - Ensure gallery network is secure

3. **Access Control:**
   - Only ia360 machine can control smart plug
   - SSH access to ia360 required for remote control
   - Gallery staff need SSH keys for remote access

## Installation Physical Setup

1. **Power Connection:**
   - Plug smart plug into wall outlet near installation
   - Connect installation power cable to smart plug
   - Ensure sufficient power rating for installation

2. **Network Setup:**
   - Smart plug connects to gallery Wi-Fi automatically
   - No additional network cables needed
   - Ensure good Wi-Fi signal strength at installation location

3. **Testing:**
   - Test power on/off from control interface
   - Verify installation starts/stops correctly
   - Test automated scheduling over 24-hour period

## Maintenance

### Regular Checks
- Monthly: Test manual power control
- Weekly: Check scheduler logs
- Daily: Verify installation starts/stops on schedule

### Updates
- Smart plug firmware: Updates automatically via Wi-Fi
- chip-tool: Update via `sudo snap refresh chip-tool`
- Control scripts: Update via git pull

### Backup
- Configuration files backed up in git repository
- Pairing codes stored securely in `.env` file
- Gallery schedule backed up in `matter_schedule.toml`

## Related Documentation

- **Gallery Hours Setup:** See `matter_schedule.toml` configuration
- **Service Management:** See `libs/common/README_SERVICE.md`
- **Multi-Machine Deployment:** See `technical_design.md`
- **SSH Configuration:** See gallery control script help output