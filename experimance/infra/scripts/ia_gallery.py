#!/usr/bin/env python3
"""
IA Gallery Control Script for Fire Project
Controls Fire project services across Ubuntu (ia360) and macOS (iamini) machines.
Uses SSH shortcuts defined in ~/.ssh/config for remote control.

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
    python3 ia_gallery.py              # Run interactive menu
    python3 ia_gallery.py --install    # Install as systemd service (auto-start)
    python3 ia_gallery.py --uninstall  # Remove systemd service

    
Manual service controls
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

# Configuration
SCRIPT_DIR = "infra/scripts"

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

# Machine configurations - dynamically set local/remote based on where script runs
def get_machine_config():
    """Get machine configuration with local/remote detection"""
    running_on_ia360 = is_running_on_ia360()
    
    return {
        "ia360": {
            "ssh_host": "ia360",  # SSH config shortcut
            "platform": "linux",
            "project_dir": "/home/experimance/experimance",
            "services": ["core", "image_server", "display", "health"],
            "local": running_on_ia360  # Local if we're running on ia360, remote otherwise
        },
        "iamini": {
            "ssh_host": "iamini",  # SSH config shortcut  
            "platform": "macos",
            "project_dir": "/Users/fireproject/Documents/experimance/experimance",
            "services": ["agent", "health"],
            "local": False  # Always remote
        }
    }

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
        # Ubuntu commands (using system services, not --user)
        commands = {
            "start": f"cd {project_dir} && sudo {SCRIPT_DIR}/startup.sh --project {project}",
            "stop": f"cd {project_dir} && sudo {SCRIPT_DIR}/shutdown.sh --project {project}",
            "emergency_stop": f"cd {project_dir} && sudo {SCRIPT_DIR}/shutdown.sh --project {project}",
            "status_systemd": "sudo systemctl status 'experimance@fire.target' --no-pager -l || echo 'Target not active'",
            "status_processes": 'ps aux | grep -E "(uv run -m|scripts/dev)" | grep -v grep | grep -E "(fire_|experimance_)" || echo "No processes running"'
        }
    elif platform == "macos":
        # macOS commands
        commands = {
            "start": f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh {project} manual-start",
            "stop": f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh {project} manual-stop",
            "emergency_stop": f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh {project} manual-unload",
            "status": f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh {project} show-schedule",
            "setup_gallery": f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh {project} setup-schedule gallery",
            "remove_gallery": f"cd {project_dir} && {SCRIPT_DIR}/launchd_scheduler.sh {project} remove-schedule"
        }
    else:
        raise ValueError(f"Unknown platform: {platform}")
    
    return commands.get(action)

def start_services():
    """Start Fire project services across all machines"""
    print("� Starting Fire project services...")
    
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

def emergency_stop():
    """Emergency stop - completely unload services (no auto-restart)"""
    print("🚨 EMERGENCY STOP - Completely shutting down all services")
    
    for machine_name, config in MACHINES.items():
        platform = config["platform"]
        project_dir = config["project_dir"]
        
        print(f"\n📍 Emergency stop on {machine_name} ({platform})")
        
        command = get_platform_command(platform, project_dir, "emergency_stop")
        run_command_on_machine(config, command)

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

def install_systemd_service():
    """Install ia_gallery.py as a systemd user service for auto-start"""
    
    # Only allow installation on ia360
    if not is_running_on_ia360():
        print("❌ Systemd service installation only supported when running locally on ia360")
        print("SSH to ia360 and run the install there, or copy the script to ia360 first.")
        return False
    
    print("🔧 Installing IA Gallery Control as systemd service...")
    
    # Get the absolute path to this script
    script_path = os.path.abspath(__file__)
    
    # Create systemd user directory
    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)
    
    # Create service file content
    service_content = f"""[Unit]
Description=IA Gallery Fire Project Control Terminal Kiosk
Wants=graphical-session.target
After=graphical-session.target
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
# Inherit session environment (DISPLAY / Wayland, DBus):
PassEnvironment=DISPLAY WAYLAND_DISPLAY XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS
# Launch GNOME Terminal full-screen with restart loop for resilience:
ExecStart=/usr/bin/gnome-terminal --full-screen --hide-menubar \\
  -- bash -lc 'while true; do python3 {script_path}; echo "Gallery control exited. Restarting in 3 seconds..."; sleep 3; done'
Restart=always
RestartSec=5
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
        print("The IA Gallery Control terminal will now:")
        print("• Auto-open when you log in")
        print("• Restart automatically if it crashes")
        print("• Open in a dedicated terminal window")
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
    
    # Only allow uninstallation on ia360
    if not is_running_on_ia360():
        print("❌ Systemd service uninstallation only supported when running locally on ia360")
        print("SSH to ia360 and run the uninstall there.")
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
        print("🔥 IA GALLERY FIRE PROJECT CONTROL")
        print(f"Mode: {mode_info}")
        print("Managing: ia360 (Ubuntu) + iamini (macOS)")
        print("="*70)
        print("1. Start all services")
        print("2. Stop all services") 
        print("3. Restart all services")
        print("4. Show service status")
        print("5. Emergency stop (complete shutdown)")
        print("")
        print("6. Set up gallery hours (macOS auto-schedule)")
        print("7. Remove gallery hours (macOS always-on)")
        print("")
        print("8. Test connections")
        print("0. Exit")
        print("-"*70)
        
        try:
            choice = input("Enter choice (0-8): ").strip()
            
            if choice == "1":
                start_services()
            elif choice == "2":
                stop_services()
            elif choice == "3":
                restart_services()
            elif choice == "4":
                show_status()
            elif choice == "5":
                confirm = input("⚠️  EMERGENCY STOP - Are you sure? (yes/no): ").strip().lower()
                if confirm == "yes":
                    emergency_stop()
                else:
                    print("Cancelled.")
            elif choice == "6":
                setup_gallery_schedule()
            elif choice == "7":
                remove_gallery_schedule()
            elif choice == "8":
                test_connections()
            elif choice == "0":
                print("Goodbye! 👋")
                break
            else:
                print("Invalid choice. Please enter 0-8.")
                
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
        elif arg == "--emergency-stop":
            print("🚨 Emergency stop via command line...")
            emergency_stop()
            sys.exit(0)
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
            print("  python3 ia_gallery.py --emergency-stop   # Emergency shutdown")
            print("")
            print("Gallery Hours:")
            print("  python3 ia_gallery.py --setup-gallery    # Enable gallery hour scheduling")
            print("  python3 ia_gallery.py --remove-gallery   # Disable gallery hour scheduling")
            print("")
            print("Testing:")
            print("  python3 ia_gallery.py --test             # Test SSH connections")
            print("  python3 ia_gallery.py --help             # Show this help")
            print("")
            print("The systemd service will auto-open the gallery control terminal")
            print("when you log in to the Ubuntu machine.")
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
    print("• Ubuntu (ia360): Core, Image Server, Display, Health")
    print("• macOS (iamini): Agent, Health")
    print("• SSH shortcuts: ia360, iamini")
    
    main_menu()
