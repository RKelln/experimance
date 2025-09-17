#!/usr/bin/env python3
"""
IA Gallery Control Script for Fire Project
Controls Fire project services across Ubuntu (ia360) and macOS (iamini) machines.
Uses SSH shortcuts defined in ~/.ssh/config for remote control.

This script coordinates with launchd_scheduler.sh for gallery hour automation.
Recent improvements (Sept 2025): Fixed launchd_scheduler.sh bugs that prevented
TouchDesigner and other services from properly scheduling during gallery hours.

SSH Config Setup Required:
~/.ssh/config should contain:
    Host iamini
        HostName FireProjects-Mac-mini.local
        User fireproject
        IdentityFile ~/.ssh/ia_fire

    Host ia360
        HostName ia360.local
        User experimance
        IdentityFile ~/.ssh/ia_fire

Usage:
    python3 ia_gallery.py                  # Run interactive menu
    python3 ia_gallery.py --install        # Install as systemd service (auto-start)
    python3 ia_gallery.py --uninstall      # Remove systemd service

Manual service controls:
# Start the service now (for testing)
systemctl --user start ia-gallery.service

# Stop the service  
systemctl --user stop ia-gallery.service

# Check service status
systemctl --user status ia-gallery.service

# Remove completely
python3 infra/scripts/ia_gallery.py --uninstall
"""

import subprocess
import sys
import time
import os
from pathlib import Path

try:
    import tomllib
except ImportError:
    # Python < 3.11 fallback
    try:
        import tomli as tomllib
    except ImportError:
        print("Warning: tomllib/tomli not available, using hardcoded machine config")
        tomllib = None

def load_env_file(env_path):
    """Load environment variables from .env file"""
    env_vars = {}
    try:
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('"\'')
                        env_vars[key.strip()] = value
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")
    return env_vars

# Configuration
SCRIPT_DIR = "infra/scripts"
MATTER_SCHEDULER_SCRIPT = "scripts/manage_matter_scheduler.sh"

def get_current_hostname():
    """Get current hostname to determine if we're running locally or remotely"""
    try:
        hostname = subprocess.run("hostname", shell=True, capture_output=True, text=True).stdout.strip()
        return hostname.lower()
    except:
        return ""

def is_running_on_ia360():
    """Check if we're currently running on the ia360 machine"""
    hostname = get_current_hostname()
    return "ia360" in hostname

def is_matter_controller():
    """Check if current machine is designated as a Matter controller in deployment config"""
    try:
        deployment_config = load_deployment_config()
        if not deployment_config:
            return False
            
        current_hostname = get_current_hostname()
        machines = deployment_config.get("machines", {})
        
        for machine_key, machine_info in machines.items():
            # Check if this machine matches current hostname
            machine_hostname = machine_info.get("hostname", "").lower()
            ssh_hostname = machine_info.get("ssh_hostname", machine_key).lower()
            
            # Match against actual hostname, configured hostname, or ssh_hostname
            if (current_hostname in machine_hostname or 
                machine_hostname in current_hostname or
                current_hostname == ssh_hostname or
                machine_key.lower() == current_hostname):
                
                return machine_info.get("matter_controller", False)
        
        return False
    except Exception as e:
        print(f"Warning: Could not determine Matter controller status: {e}")
        return False

# Machine configurations - dynamically set local/remote based on where script runs
def get_machine_config():
    """Get machine configuration with local/remote detection and deployment config"""
    running_on_ia360 = is_running_on_ia360()
    
    # Try to load deployment configuration
    deployment_config = load_deployment_config()
    
    if deployment_config:
        # Use deployment configuration to build machine config
        machines = {}
        
        for machine_key, machine_info in deployment_config.get("machines", {}).items():
            ssh_hostname = machine_info.get("ssh_hostname", machine_key)
            hostname = machine_info.get("hostname", "")
            platform = machine_info.get("platform", "linux")
            
            # Determine if this machine is local
            is_local = False
            if platform == "linux" and running_on_ia360:
                is_local = True
            
            machines[machine_key] = {
                "ssh_host": ssh_hostname,
                "platform": platform,
                "project_dir": get_project_dir_for_machine(machine_info),
                "services": machine_info.get("services", []),
                "local": is_local,
                "has_matter": machine_info.get("matter_controller", False),
                "matter_devices": machine_info.get("matter_devices", [])
            }
        
        return machines
    
    # Fallback to hardcoded configuration if deployment config not available
    return {
        "ia360": {
            "ssh_host": "ia360",  # SSH config shortcut
            "platform": "linux",
            "project_dir": "/opt/experimance",
            "services": ["core", "image_server", "display", "health"],
            "local": running_on_ia360,  # Local if we're running on ia360, remote otherwise
            "has_matter": True,  # This machine has Matter device support
            "matter_devices": [{"id": 110, "type": "smart_plug", "name": "Installation Power"}]
        },
        "iamini": {
            "ssh_host": "iamini",  # SSH config shortcut  
            "platform": "macos",
            "project_dir": "/Users/fireproject/Documents/experimance/experimance",
            "services": ["agent", "health"],
            "local": False,  # Always remote
            "has_matter": False,  # This machine doesn't have Matter device support
            "matter_devices": []
        }
    }

def load_deployment_config():
    """Load deployment configuration from TOML file"""
    if not tomllib:
        return None
    
    # Look for deployment.toml in the project directory
    script_dir = Path(__file__).parent.parent.parent  # Go up from infra/scripts/
    deployment_file = script_dir / "projects" / "fire" / "deployment.toml"
    
    try:
        if deployment_file.exists():
            with open(deployment_file, 'rb') as f:
                return tomllib.load(f)
    except Exception as e:
        print(f"Warning: Could not load deployment config: {e}")
    
    return None

def get_project_dir_for_machine(machine_info):
    """Get the project directory for a machine based on platform"""
    platform = machine_info.get("platform", "linux")
    
    if platform == "linux":
        return "/opt/experimance"
    elif platform == "macos":
        return "/Users/fireproject/Documents/experimance/experimance"
    else:
        return "/opt/experimance"  # Default fallback

# Get machine configuration
MACHINES = get_machine_config()

def run_local_command(command, show_output=True):
    """Execute command locally"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            if show_output:
                print(f"✓ LOCAL: Command executed")
                if result.stdout.strip():
                    print(f"  {result.stdout.strip()}")
        else:
            print(f"✗ LOCAL: Failed - {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ LOCAL: Command timed out")
        return False
    except Exception as e:
        print(f"✗ LOCAL: Error - {e}")
        return False
    return True

def run_ssh_command(ssh_host, command, show_output=True):
    """Execute command on remote machine via SSH"""
    ssh_cmd = f"ssh {ssh_host} '{command}'"
    
    try:
        result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            if show_output:
                print(f"✓ {ssh_host}: Command executed")
                if result.stdout.strip():
                    print(f"  {result.stdout.strip()}")
        else:
            print(f"✗ {ssh_host}: Failed - {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ {ssh_host}: Command timed out")
        return False
    except Exception as e:
        print(f"✗ {ssh_host}: Error - {e}")
        return False
    return True

def run_command_on_machine(config, command):
    """Run a command on a machine (local or remote based on config)"""
    if config["local"]:
        return run_local_command(command)
    else:
        return run_ssh_command(config["ssh_host"], command)

def get_platform_command(platform, project_dir, action, project="fire"):
    """Get the appropriate command for a platform and action"""
    if platform == "linux":
        # Ubuntu commands (using deploy.sh for proper systemd target management)
        commands = {
            "start": f"cd {project_dir} && sudo ./infra/scripts/deploy.sh {project} start",
            "stop": f"cd {project_dir} && sudo ./infra/scripts/deploy.sh {project} stop",
            "status_systemd": f"cd {project_dir} && sudo ./infra/scripts/deploy.sh {project} status",
            "status_processes": 'ps aux | grep -E "(uv run -m|scripts/dev)" | grep -v grep | grep -E "(fire_|experimance_)" || echo "No processes running"'
        }
    elif platform == "macos":
        # macOS commands (use manual-unload instead of manual-stop since services auto-restart with KeepAlive)
        commands = {
            "start": f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh {project} manual-start",
            "stop": f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh {project} manual-unload",
            "status": f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh {project} show-schedule",
            "setup_gallery": f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh {project} setup-schedule gallery",
            "remove_gallery": f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh {project} remove-schedule"
        }
    else:
        raise ValueError(f"Unknown platform: {platform}")
    
    return commands.get(action)

def start_services():
    """Start Fire project services across all machines"""
    print("Starting Fire project services...")
    
    for machine_name, config in MACHINES.items():
        services = config["services"]
        platform = config["platform"]
        project_dir = config["project_dir"]
        
        print(f"\n📍 Starting services on {machine_name} ({platform}): {', '.join(services)}")
        
        command = get_platform_command(platform, project_dir, "start")
        run_command_on_machine(config, command)

def stop_services():
    """Stop Fire project services across all machines"""
    print("🛑 Stopping Fire project services...")
    
    for machine_name, config in MACHINES.items():
        platform = config["platform"]
        project_dir = config["project_dir"]
        
        print(f"\n📍 Stopping services on {machine_name} ({platform})")
        
        command = get_platform_command(platform, project_dir, "stop")
        run_command_on_machine(config, command)

def restart_services():
    """Restart all Fire project services with delay"""
    print("🔄 Restarting Fire project services...")
    stop_services()
    print("\n⏳ Waiting 5 seconds for clean shutdown...")
    time.sleep(5)
    start_services()

def reset_services():
    """Reset services (Ubuntu audio reset + restart)"""
    print("🔧 Resetting Fire project services (with audio reset)...")
    
    for machine_name, config in MACHINES.items():
        platform = config["platform"]
        ssh_host = config["ssh_host"]
        project_dir = config["project_dir"]
        
        print(f"\n📍 Resetting services on {machine_name} ({platform})")
        
        if config["local"]:
            # Local Ubuntu machine - use reset.sh (includes audio reset)
            command = f"cd {project_dir} && {SCRIPT_DIR}/reset.sh --project fire"
            run_local_command(command)
        else:
            # Remote macOS machine - manual stop + start (no reset.sh equivalent)
            print(f"  Stopping services on {machine_name}...")
            stop_cmd = f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh fire manual-unload"
            run_ssh_command(ssh_host, stop_cmd, show_output=False)
            
            print(f"  Waiting 3 seconds...")
            time.sleep(3)
            
            print(f"  Starting services on {machine_name}...")
            start_cmd = f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh fire manual-start"
            run_ssh_command(ssh_host, start_cmd, show_output=False)

def show_status():
    """Show service status across all machines"""
    print("📊 Fire Project Service Status:")
    
    for machine_name, config in MACHINES.items():
        expected_services = config["services"]
        platform = config["platform"]
        project_dir = config["project_dir"]
        
        print(f"\n📍 {machine_name.upper()} ({platform}) - Expected: {', '.join(expected_services)}")
        
        if platform == "linux":
            # Ubuntu - check systemd and processes
            systemd_cmd = get_platform_command(platform, project_dir, "status_systemd")
            process_cmd = get_platform_command(platform, project_dir, "status_processes")
            
            run_command_on_machine(config, systemd_cmd)
            run_command_on_machine(config, process_cmd)
        else:
            # macOS - use launchd status
            status_cmd = get_platform_command(platform, project_dir, "status")
            run_command_on_machine(config, status_cmd)

def test_connections():
    """Test SSH connections to all machines"""
    print("🔧 Testing connections...")
    
    all_good = True
    for machine_name, config in MACHINES.items():
        if config["local"]:
            print(f"Testing {machine_name} (local)... ✓")
            continue
            
        ssh_host = config["ssh_host"]
        print(f"Testing {ssh_host}...", end=" ")
        
        if run_ssh_command(ssh_host, "echo 'Connection OK'", show_output=False):
            print("✓")
        else:
            print("✗")
            all_good = False
    
    if all_good:
        print("🎉 All connections working!")
    else:
        print("❌ Some connections failed. Check SSH setup.")
    
    return all_good

def setup_gallery_schedule():
    """Set up gallery hour scheduling on macOS machine"""
    print("⏰ Setting up gallery hour scheduling...")
    
    for machine_name, config in MACHINES.items():
        platform = config["platform"]
        project_dir = config["project_dir"]
        
        if platform == "macos":
            print(f"\n📍 Setting up gallery schedule on {machine_name}")
            command = get_platform_command(platform, project_dir, "setup_gallery")
            run_command_on_machine(config, command)
        else:
            print(f"📍 {machine_name} (Linux) - Gallery scheduling handled by systemd (see deploy.sh)")

def remove_gallery_schedule():
    """Remove gallery hour scheduling from macOS machine"""
    print("⏰ Removing gallery hour scheduling...")
    
    for machine_name, config in MACHINES.items():
        platform = config["platform"]
        project_dir = config["project_dir"]
        
        if platform == "macos":
            print(f"\n📍 Removing gallery schedule from {machine_name}")
            command = get_platform_command(platform, project_dir, "remove_gallery")
            run_command_on_machine(config, command)
        else:
            print(f"📍 {machine_name} (Linux) - Gallery scheduling handled by systemd")

def matter_control(action, device_id=None):
    """Control Matter devices (smart plugs, etc.)"""
    print(f"🔌 Matter Device Control: {action}")
    
    # Find machine with Matter support
    matter_machine = None
    matter_devices = []
    
    for machine_name, config in MACHINES.items():
        if config.get("has_matter", False):
            matter_machine = (machine_name, config)
            matter_devices = config.get("matter_devices", [])
            break
    
    if not matter_machine:
        print("❌ No machines configured with Matter support")
        return False
    
    machine_name, config = matter_machine
    project_dir = config["project_dir"]
    
    # If no device_id specified, use the first smart plug found
    if device_id is None:
        smart_plugs = [dev for dev in matter_devices if dev.get("type") == "smart_plug"]
        if smart_plugs:
            device_id = smart_plugs[0]["id"]
            device_name = smart_plugs[0].get("name", f"Device {device_id}")
            print(f"📍 Using device: {device_name} (ID: {device_id}) via {machine_name}")
        else:
            device_id = 110  # Fallback default
            print(f"📍 No smart plugs configured, using default ID: {device_id} via {machine_name}")
    else:
        # Find device info if available
        device_info = next((dev for dev in matter_devices if dev["id"] == device_id), None)
        if device_info:
            device_name = device_info.get("name", f"Device {device_id}")
            print(f"📍 Controlling device: {device_name} (ID: {device_id}) via {machine_name}")
        else:
            print(f"📍 Controlling device ID: {device_id} via {machine_name}")
    
    # Map actions to chip-tool commands
    commands = {
        "on": f"cd {project_dir} && chip-tool onoff on {device_id} 1",
        "off": f"cd {project_dir} && chip-tool onoff off {device_id} 1",
        "toggle": f"cd {project_dir} && chip-tool onoff toggle {device_id} 1"
    }
    
    if action not in commands:
        print(f"❌ Unknown action: {action}. Available: {list(commands.keys())}")
        return False
    
    command = commands[action]
    return run_command_on_machine(config, command)

def matter_scheduler_control(action):
    """Control the Matter device scheduler service"""
    print(f"⏰ Matter Scheduler: {action}")
    
    # Find machine with Matter support
    matter_machine = None
    for machine_name, config in MACHINES.items():
        if config.get("has_matter", False):
            matter_machine = (machine_name, config)
            break
    
    if not matter_machine:
        print("❌ No machines configured with Matter support")
        return False
    
    machine_name, config = matter_machine
    project_dir = config["project_dir"]
    
    print(f"📍 Managing Matter scheduler on {machine_name}")
    
    # Map actions to scheduler script commands
    commands = {
        "start": f"cd {project_dir} && ./{MATTER_SCHEDULER_SCRIPT} start",
        "stop": f"cd {project_dir} && ./{MATTER_SCHEDULER_SCRIPT} stop",
        "restart": f"cd {project_dir} && ./{MATTER_SCHEDULER_SCRIPT} restart",
        "status": f"cd {project_dir} && ./{MATTER_SCHEDULER_SCRIPT} status",
        "test": f"cd {project_dir} && ./{MATTER_SCHEDULER_SCRIPT} test-config"
    }
    
    if action not in commands:
        print(f"❌ Unknown scheduler action: {action}. Available: {list(commands.keys())}")
        return False
    
    command = commands[action]
    return run_command_on_machine(config, command)

def smart_startup():
    """Smart startup sequence: Turn on power first, then start services"""
    print("🚀 Smart Gallery Startup - Power + Services")
    
    # Step 1: Turn on smart plugs
    print("\n🔌 Step 1: Turning on smart plug power...")
    if not matter_control("on"):
        print("⚠️  Smart plug control failed, continuing with services anyway...")
    else:
        print("✅ Smart plug powered on")
        print("⏳ Waiting 10 seconds for devices to power up...")
        time.sleep(10)
    
    # Step 2: Start all services
    print("\n🎭 Step 2: Starting Fire project services...")
    start_services()
    
    # Step 3: Start scheduler if not already running
    print("\n⏰ Step 3: Ensuring Matter scheduler is running...")
    matter_scheduler_control("start")
    
    print("\n🎉 Smart startup complete!")

def smart_shutdown():
    """Smart shutdown sequence: Stop services first, then turn off power"""
    print("🛑 Smart Gallery Shutdown - Services + Power")
    
    # Step 1: Stop all services
    print("\n🎭 Step 1: Stopping Fire project services...")
    stop_services()
    
    # Step 2: Wait for clean shutdown
    print("\n⏳ Step 2: Waiting 15 seconds for clean shutdown...")
    time.sleep(15)
    
    # Step 3: Turn off smart plugs
    print("\n🔌 Step 3: Turning off smart plug power...")
    if not matter_control("off"):
        print("⚠️  Smart plug control failed - devices may still be powered")
    else:
        print("✅ Smart plug powered off")
    
    print("\n🌙 Smart shutdown complete!")

def install_chip_tool():
    """Install chip-tool on the local machine"""
    print("🔧 Installing chip-tool...")
    
    # Check if already installed
    result = subprocess.run("which chip-tool", shell=True, capture_output=True)
    if result.returncode == 0:
        print("✅ chip-tool is already installed")
        return True
    
    try:
        # Install via snap
        print("📦 Installing chip-tool via snap...")
        result = subprocess.run("sudo snap install chip-tool", shell=True, check=True)
        print("✅ chip-tool installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install chip-tool: {e}")
        return False

def pair_matter_device(device_id, pairing_code, bypass_attestation=True):
    """Pair a Matter device using chip-tool"""
    print(f"🔗 Pairing Matter device {device_id} with code {pairing_code}...")
    
    # Build pairing command
    cmd = f"chip-tool pairing code {device_id} {pairing_code}"
    if bypass_attestation:
        cmd += " --bypass-attestation-verifier true"
    
    try:
        print(f"Executing: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0 and "Device commissioning completed with success" in result.stdout:
            print("✅ Matter device paired successfully!")
            return True
        else:
            print(f"❌ Matter device pairing failed:")
            print(f"Exit code: {result.returncode}")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Matter device pairing timed out")
        return False
    except Exception as e:
        print(f"❌ Error during pairing: {e}")
        return False

def setup_matter_devices():
    """Set up Matter devices based on deployment configuration"""
    print("🔌 Setting up Matter devices...")
    
    # Load environment variables for pairing codes
    script_dir = Path(__file__).parent.parent.parent
    env_file = script_dir / "projects" / "fire" / ".env"
    env_vars = load_env_file(env_file)
    
    # Find machine with Matter support
    matter_machine = None
    for machine_name, config in MACHINES.items():
        if config.get("has_matter", False) and config.get("local", False):
            matter_machine = (machine_name, config)
            break
    
    if not matter_machine:
        print("⚠️  No local machine configured with Matter support - skipping device setup")
        return True
    
    machine_name, config = matter_machine
    matter_setup = config.get("matter_setup", {})
    matter_devices = config.get("matter_devices", [])
    
    # Install chip-tool if configured
    if matter_setup.get("install_chip_tool", False):
        if not install_chip_tool():
            return False
    
    # Auto-pair devices if configured
    if matter_setup.get("auto_pair", False):
        for device in matter_devices:
            device_id = device["id"]
            device_name = device.get("name", f"Device {device_id}")
            
            # Get pairing code from environment
            pairing_code = None
            if device.get("type") == "smart_plug":
                pairing_code = env_vars.get("MATTER_SMART_PLUG_PAIRING_CODE")
            
            if not pairing_code:
                print(f"⚠️  No pairing code found for {device_name} - skipping auto-pairing")
                print(f"   Add MATTER_SMART_PLUG_PAIRING_CODE to {env_file}")
                continue
            
            bypass_attestation = env_vars.get("MATTER_BYPASS_ATTESTATION", "true").lower() == "true"
            
            print(f"\n🔗 Setting up {device_name} (ID: {device_id})")
            if not pair_matter_device(device_id, pairing_code, bypass_attestation):
                print(f"⚠️  Failed to pair {device_name} - continuing with other devices")
                continue
            
            # Test the device
            print(f"🧪 Testing {device_name}...")
            if matter_control("toggle", device_id):
                print(f"✅ {device_name} is working correctly!")
            else:
                print(f"⚠️  {device_name} paired but test failed")
    
    return True

def install_systemd_service():
    """Install ia_gallery.py as a systemd user service for auto-start"""
    
    # Only allow installation on machines that can run services locally
    if not is_matter_controller():
        print("❌ Systemd service installation only supported on machines designated as Matter controllers")
        print("Check deployment.toml to ensure this machine has 'matter_controller = true'")
        return False
    
    print("🔧 Installing IA Gallery Control as systemd service...")
    
    # Step 1: Set up Matter devices if configured
    print("\n🔌 Step 1: Setting up Matter devices...")
    if not setup_matter_devices():
        print("⚠️  Matter device setup had issues, but continuing with service installation...")
    
    # Step 2: Install and start Matter scheduler
    print("\n⏰ Step 2: Setting up Matter scheduler...")
    try:
        script_dir = Path(__file__).parent.parent.parent
        result = subprocess.run(
            f"cd {script_dir} && ./{MATTER_SCHEDULER_SCRIPT} install",
            shell=True, capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print("✅ Matter scheduler installed successfully")
            # Start the scheduler
            matter_scheduler_control("start")
        else:
            print(f"⚠️  Matter scheduler installation failed: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Error setting up Matter scheduler: {e}")
    
    # Step 3: Create systemd service
    print("\n🖥️  Step 3: Creating gallery control service...")
    
    # Get the absolute path to this script
    script_path = os.path.abspath(__file__)
    
    # Create systemd user directory
    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)
    
    # Create service file content
    service_content = f"""[Unit]
Description=IA Gallery Fire Project Control Terminal Kiosk
Wants=graphical-session.target network-online.target
After=graphical-session.target network-online.target
Requisite=graphical-session.target

[Service]
Type=simple
# Inherit session environment (DISPLAY / Wayland, DBus):
PassEnvironment=DISPLAY WAYLAND_DISPLAY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS
# Set environment explicitly
Environment="DISPLAY=:0"
# Launch with a wrapper script approach
ExecStart=/bin/bash -c 'exec gnome-terminal --full-screen --hide-menubar -- bash -lc "while true; do python3 /opt/experimance/infra/scripts/ia_gallery.py; echo Gallery control exited. Restarting in 3 seconds...; sleep 3; done"'
Restart=on-failure
RestartSec=10
# Keep logs per-instance (handy for debugging)
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=graphical-session.target
"""
    
    # Write service file
    service_file = systemd_dir / "ia-gallery.service"
    try:
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print(f"✓ Created service file: {service_file}")
        
        # Reload systemd and enable service
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "ia-gallery.service"], check=True)
        
        print("✓ Service enabled for auto-start")
        print("")
        print("🎉 Installation complete!")
        print("")
        print("The IA Gallery Control system now includes:")
        print("• Gallery control terminal (auto-starts on login)")
        print("• Matter device control (smart plugs)")
        print("• Automated scheduling for gallery hours")
        print("• Coordinated startup/shutdown sequences")
        print("")
        print("Manual controls:")
        print("  systemctl --user start ia-gallery.service    # Start now")
        print("  systemctl --user stop ia-gallery.service     # Stop service")
        print("  systemctl --user status ia-gallery.service   # Check status")
        print("")
        print("To test: Log out and log back in, or run:")
        print("  systemctl --user start ia-gallery.service")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install service: {e}")
        return False
    except Exception as e:
        print(f"✗ Error creating service file: {e}")
        return False
    
    return True

def uninstall_systemd_service():
    """Remove the systemd service"""
    
    # Only allow uninstallation on machines that have the service
    if not is_matter_controller():
        print("❌ Systemd service uninstallation only supported on machines designated as Matter controllers")
        print("Check deployment.toml to ensure this machine has 'matter_controller = true'")
        return False
    
    print("🗑️  Removing IA Gallery Control systemd service...")
    
    service_file = Path.home() / ".config" / "systemd" / "user" / "ia-gallery.service"
    
    try:
        # Stop and disable service
        subprocess.run(["systemctl", "--user", "stop", "ia-gallery.service"], check=False)
        subprocess.run(["systemctl", "--user", "disable", "ia-gallery.service"], check=False)
        
        # Remove service file
        if service_file.exists():
            service_file.unlink()
            print(f"✓ Removed service file: {service_file}")
        
        # Reload systemd
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        
        print("✓ Service removed successfully")
        print("The script will no longer auto-start on login.")
        
    except Exception as e:
        print(f"✗ Error removing service: {e}")
        return False
    
    return True

def main_menu():
    """Display main menu and handle user input"""
    # Show current running mode
    running_on_ia360 = is_running_on_ia360()
    hostname = get_current_hostname()
    
    if running_on_ia360:
        mode_info = f"Running locally on ia360 ({hostname})"
    else:
        mode_info = f"Running remotely from {hostname} - controlling both machines via SSH"
    
    # Test connections on startup
    if not test_connections():
        print("\n❌ Connection issues detected. Please check SSH setup.")
        print("\nRequired SSH config:")
        print("~/.ssh/config should contain:")
        print("  Host iamini")
        print("    HostName FireProjects-Mac-mini.local")
        print("    User fireproject")
        print("    IdentityFile ~/.ssh/ia_fire")
        if not running_on_ia360:
            print("  Host ia360")
            print("    HostName ia360.local")
            print("    User experimance")
            print("    IdentityFile ~/.ssh/ia_fire")
        return
    
    while True:
        print("\n" + "="*70)
        print("🔥 IA GALLERY Feed the Fires PROJECT CONTROL")
        print(f"Mode: {mode_info}")
        print("Managing: ia360 (Ubuntu) + iamini (macOS)")
        print("="*70)
        print("SERVICES:")
        print("1. Start services")
        print("2. Stop services") 
        print("3. Restart services")
        print("4. Show status")
        print("")
        print("SMART CONTROL (Services + Power):")
        print("5. Smart startup (Power on → Services)")
        print("6. Smart shutdown (Services → Power off)")
        print("")
        print("MATTER DEVICES:")
        print("7. Turn smart plug ON")
        print("8. Turn smart plug OFF")
        print("9. Toggle smart plug")
        print("")
        print("SCHEDULER:")
        print("11. Start auto-scheduler (gallery hours)")
        print("12. Stop auto-scheduler")
        print("13. Scheduler status")
        print("")
        print("MATTER SETUP:")
        print("14. Install chip-tool")
        print("15. Setup Matter devices (pair & test)")
        print("")
        print("TOOLS:")
        print("0. Test network connections")
        print("")
        print("💡 This interface runs continuously. Use Ctrl+C to exit if needed.")
        print("-"*70)
        
        try:
            choice = input("Enter choice: ").strip()
            
            if choice == "1":
                start_services()
            elif choice == "2":
                stop_services()
            elif choice == "3":
                restart_services()
            elif choice == "4":
                show_status()
            elif choice == "5":
                smart_startup()
            elif choice == "6":
                smart_shutdown()
            elif choice == "7":
                matter_control("on")
            elif choice == "8":
                matter_control("off")
            elif choice == "9":
                matter_control("toggle")
            elif choice == "11":
                matter_scheduler_control("start")
            elif choice == "12":
                matter_scheduler_control("stop")
            elif choice == "13":
                matter_scheduler_control("status")
            elif choice == "14":
                if is_matter_controller():
                    install_chip_tool()
                else:
                    print("❌ chip-tool installation only supported on machines designated as Matter controllers")
            elif choice == "15":
                if is_matter_controller():
                    setup_matter_devices()
                else:
                    print("❌ Matter device setup only supported on machines designated as Matter controllers")
            elif choice == "0":
                test_connections()
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--install":
            if install_systemd_service():
                sys.exit(0)
            else:
                sys.exit(1)
        elif arg == "--uninstall":
            if uninstall_systemd_service():
                sys.exit(0)
            else:
                sys.exit(1)
        elif arg == "--start":
            print("🚀 Starting services via command line...")
            start_services()
            sys.exit(0)
        elif arg == "--stop":
            print("🛑 Stopping services via command line...")
            stop_services()
            sys.exit(0)
        elif arg == "--restart":
            print("🔄 Restarting services via command line...")
            restart_services()
            sys.exit(0)
        elif arg == "--status":
            print("📊 Checking service status via command line...")
            show_status()
            sys.exit(0)
        elif arg == "--smart-start":
            print("🚀 Smart startup via command line...")
            smart_startup()
            sys.exit(0)
        elif arg == "--smart-stop":
            print("🛑 Smart shutdown via command line...")
            smart_shutdown()
            sys.exit(0)
        elif arg == "--plug-on":
            print("🔌 Turning smart plug ON via command line...")
            if matter_control("on"):
                sys.exit(0)
            else:
                sys.exit(1)
        elif arg == "--plug-off":
            print("🔌 Turning smart plug OFF via command line...")
            if matter_control("off"):
                sys.exit(0)
            else:
                sys.exit(1)
        elif arg == "--plug-toggle":
            print("🔌 Toggling smart plug via command line...")
            if matter_control("toggle"):
                sys.exit(0)
            else:
                sys.exit(1)
        elif arg == "--scheduler-start":
            print("⏰ Starting Matter scheduler via command line...")
            if matter_scheduler_control("start"):
                sys.exit(0)
            else:
                sys.exit(1)
        elif arg == "--scheduler-stop":
            print("⏰ Stopping Matter scheduler via command line...")
            if matter_scheduler_control("stop"):
                sys.exit(0)
            else:
                sys.exit(1)
        elif arg == "--scheduler-status":
            print("⏰ Checking Matter scheduler status via command line...")
            if matter_scheduler_control("status"):
                sys.exit(0)
            else:
                sys.exit(1)
        elif arg == "--setup-matter":
            print("🔌 Setting up Matter devices via command line...")
            if is_matter_controller():
                if setup_matter_devices():
                    sys.exit(0)
                else:
                    sys.exit(1)
            else:
                print("❌ Matter setup only supported on machines designated as Matter controllers")
                sys.exit(1)
        elif arg == "--install-chip-tool":
            print("🔧 Installing chip-tool via command line...")
            if is_matter_controller():
                if install_chip_tool():
                    sys.exit(0)
                else:
                    sys.exit(1)
            else:
                print("❌ chip-tool installation only supported on machines designated as Matter controllers")
                sys.exit(1)
        elif arg == "--setup-gallery":
            print("⏰ Setting up gallery hours via command line...")
            setup_gallery_schedule()
            sys.exit(0)
        elif arg == "--remove-gallery":
            print("⏰ Removing gallery hours via command line...")
            remove_gallery_schedule()
            sys.exit(0)
        elif arg == "--test":
            print("🔧 Testing connections via command line...")
            if test_connections():
                sys.exit(0)
            else:
                sys.exit(1)
        elif arg in ["--help", "-h"]:
            print("IA Gallery Control Script")
            print("")
            print("Usage:")
            print("  python3 ia_gallery.py                    # Run interactive menu")
            print("  python3 ia_gallery.py --install          # Install as systemd service")
            print("  python3 ia_gallery.py --uninstall        # Remove systemd service")
            print("")
            print("Service Control:")
            print("  python3 ia_gallery.py --start            # Start all services")
            print("  python3 ia_gallery.py --stop             # Stop all services")
            print("  python3 ia_gallery.py --restart          # Restart all services")
            print("  python3 ia_gallery.py --status           # Show service status")
            print("")
            print("Smart Control (Services + Power):")
            print("  python3 ia_gallery.py --smart-start      # Power on → Services")
            print("  python3 ia_gallery.py --smart-stop       # Services → Power off")
            print("")
            print("Matter Device Control:")
            print("  python3 ia_gallery.py --plug-on          # Turn smart plug ON")
            print("  python3 ia_gallery.py --plug-off         # Turn smart plug OFF")
            print("  python3 ia_gallery.py --plug-toggle      # Toggle smart plug")
            print("")
            print("Matter Scheduler Control:")
            print("  python3 ia_gallery.py --scheduler-start  # Start auto-scheduler")
            print("  python3 ia_gallery.py --scheduler-stop   # Stop auto-scheduler")
            print("  python3 ia_gallery.py --scheduler-status # Check scheduler status")
            print("")
            print("Matter Device Setup:")
            print("  python3 ia_gallery.py --setup-matter     # Set up Matter devices (pairs, tests)")
            print("  python3 ia_gallery.py --install-chip-tool # Install chip-tool only")
            print("")
            print("Gallery Hours:")
            print("  python3 ia_gallery.py --setup-gallery    # Enable gallery hour scheduling")
            print("  python3 ia_gallery.py --remove-gallery   # Disable gallery hour scheduling")
            print("")
            print("Testing:")
            print("  python3 ia_gallery.py --test             # Test SSH connections")
            print("  python3 ia_gallery.py --help             # Show this help")
            print("")
            print("Full Installation Process:")
            print("  python3 ia_gallery.py --install          # Complete setup (service + Matter + scheduler)")
            print("")
            print("The --install command will:")
            print("• Install chip-tool for Matter device control")
            print("• Auto-pair Matter devices using codes from .env file")
            print("• Set up automated scheduling for gallery hours")
            print("• Install systemd service for auto-start gallery control terminal")
            print("")
            print("Configuration Files:")
            print("• projects/fire/deployment.toml - Machine and device definitions")
            print("• projects/fire/.env - Pairing codes and sensitive settings")
            print("• projects/fire/matter_schedule.toml - Gallery hour schedules")
            sys.exit(0)
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
            sys.exit(1)
    
    # Normal interactive mode
    print("🎭 IA Gallery Control Script Starting...")
    
    # Show current running mode
    hostname = get_current_hostname()
    if is_running_on_ia360():
        print(f"📍 Running locally on ia360 ({hostname})")
        print("💡 Tip: Use --install to set up auto-start kiosk mode")
    else:
        print(f"📍 Running remotely from {hostname}")
        print("🌐 Remote admin mode - controlling both machines via SSH")
        print("💡 Note: --install/--uninstall only work when running locally on ia360")
    
    print("\n📋 Gallery Setup:")
    for machine_name, config in MACHINES.items():
        services_str = ", ".join(config["services"])
        platform = config["platform"].title()
        
        matter_info = ""
        if config.get("has_matter", False):
            matter_devices = config.get("matter_devices", [])
            if matter_devices:
                device_names = [f"ID {dev['id']}" for dev in matter_devices]
                matter_info = f" + Matter devices ({', '.join(device_names)})"
            else:
                matter_info = " + Matter controller"
        
        print(f"• {platform} ({machine_name}): {services_str}{matter_info}")
    
    print("• SSH shortcuts: " + ", ".join(config["ssh_host"] for config in MACHINES.values()))
    
    # Show auto-scheduler info if Matter devices are configured
    matter_machine = next((config for config in MACHINES.values() if config.get("has_matter")), None)
    if matter_machine:
        print("• Auto-scheduler: Gallery hours Tues-Sat (10:55 AM - 6:05/9:05 PM)")
    
    main_menu()
