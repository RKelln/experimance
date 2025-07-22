#!/usr/bin/env python3
"""
CLI tool for managing Vast.ai instances for Experimance Image Generation.

Usage:
    python scripts/vastai_manager_cli.py <command> [options]

Commands:
    list           List all Vast.ai instances
    search         Search for suitable GPU offers
    provision      Create or find an Experimance instance
    fix            Fix an instance using SCP provisioning (auto-detects active instance)
    update         Update server code on active instance and restart service
    ssh            Show SSH command for an instance (auto-detects active instance)
    endpoint       Show model server endpoint for an instance (auto-detects active instance)
    stop           Stop an instance (auto-detects active instance)
    start          Start an instance (auto-detects active instance)
    restart        Restart an instance (auto-detects active instance)
    destroy        Destroy an instance (auto-detects active instance)
    health         Check health of the model server (auto-detects active instance)
    test-markers   Check for test provisioning markers (auto-detects active instance)

Note: Commands that take an instance_id will automatically use the first running 
Experimance instance if no ID is provided.
"""
import argparse
import logging
import json
import subprocess
from typing import Optional

import requests
import json
from typing import Optional
from image_server.generators.vastai.vastai_manager import VastAIManager

from image_server.generators.vastai.vastai_manager import VastAIManager


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_active_instance_id(manager: VastAIManager) -> Optional[int]:
    """Get the ID of the first running Experimance instance."""
    try:
        instances = manager.find_experimance_instances()
        if instances:
            instance_id = instances[0]["id"]
            print(f"Using active Experimance instance: {instance_id}")
            return instance_id
        else:
            print("❌ No running Experimance instances found")
            return None
    except Exception as e:
        print(f"❌ Error finding active instance: {e}")
        return None


def list_instances(manager: VastAIManager, args: argparse.Namespace):
    raw  = getattr(args, 'raw', False)
    instances = manager.show_instances(raw=raw)
    if isinstance(instances, str):
        # If it's a string with \n characters, print it directly to interpret newlines
        print(instances)
    else:
        # If it's a JSON object, pretty print it
        print(json.dumps(instances, indent=2))


def search_offers(manager: VastAIManager, args: argparse.Namespace):
    offers = manager.search_offers(
        min_gpu_ram=args.min_gpu_ram,
        max_price=args.max_price,
        dlperf=args.dlperf,
    )
    if not offers:
        print("No offers found matching criteria")
        return

    # Calculate scores and prepare table
    cheapest_price = min(o.get('dph_total', float('inf')) for o in offers)
    price_threshold = cheapest_price * 1.25
    rows = []
    for offer in offers:
        gpu_ram = offer.get('gpu_ram', 0) / 1024  # MB to GB
        score = manager._calculate_offer_score(offer, cheapest_price, price_threshold)
        rows.append([
            offer.get('id', ''),
            f"{score:.1f}",
            offer.get('gpu_name', ''),
            f"{offer.get('dph_total', 0):.3f}",
            f"{offer.get('dlperf', 0):.1f}",
            f"{offer.get('dlperf_per_dphtotal', 0):.1f}",
            f"{offer.get('reliability2', 0):.3f}",
            offer.get('geolocation', ''),
            offer.get('verified', False),
            f"{offer.get('inet_down', 0):.0f}",
            f"{gpu_ram:.1f}",
        ])
    headers = [
        "ID", "Score", "GPU", "Price($/hr)", "DLPerf", "DLPerf/$",
        "Reliability", "Location", "Verified", "Down(Mbps)", "RAM(GB)"
    ]
    # Determine column widths
    col_widths = []
    for idx, header in enumerate(headers):
        max_len = len(header)
        for row in rows:
            cell_len = len(str(row[idx]))
            if cell_len > max_len:
                max_len = cell_len
        col_widths.append(max_len)

    # Print header
    header_row = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    sep_row = "-+-".join('-' * col_widths[i] for i in range(len(headers)))
    print(header_row)
    print(sep_row)
    # Print rows
    for row in rows:
        line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(line)


def provision_instance(manager: VastAIManager, args: argparse.Namespace):
    # Create a new manager with custom provisioning script if provided
    if hasattr(args, 'provision_script') and args.provision_script:
        manager = VastAIManager(provisioning_script_url=args.provision_script)
    
    # Check if there's already an existing instance
    existing_instances = manager.find_experimance_instances()
    
    if existing_instances:
        # Found existing instance - provision it
        instance_id = existing_instances[0]["id"]
        print(f"Found existing Experimance instance {instance_id}, provisioning it...")
        
        # Wait for it to be ready first
        if manager.wait_for_instance_ready(instance_id):
            # Wait for SSH to be ready, then provision
            if manager.wait_for_ssh_ready(instance_id, timeout=180):
                success = manager.provision_instance_via_scp(
                    instance_id,
                    verbose=getattr(args, 'verbose', False)
                )
                if success:
                    endpoint = manager.get_model_server_endpoint(instance_id)
                    if endpoint:
                        print(f"✅ Instance {instance_id} provisioned successfully!")
                        print(f"Instance ready at {endpoint.url} (ID: {endpoint.instance_id})")
                        print(manager.get_ssh_command(endpoint.instance_id))
                    else:
                        print(f"✅ Instance {instance_id} provisioned successfully!")
                        print("⚠️  Could not get model server endpoint, but provisioning completed")
                else:
                    print(f"❌ Failed to provision existing instance {instance_id}")
            else:
                print(f"❌ SSH not ready for existing instance {instance_id}")
        else:
            print(f"❌ Existing instance {instance_id} is not ready")
    else:
        # No existing instance - create and provision a new one
        print("No existing Experimance instances found, creating new one...")
        
        # Disable SCP provisioning if we have a custom provisioning script
        # to test if PROVISIONING_SCRIPT environment variable works
        disable_scp = hasattr(args, 'provision_script') and args.provision_script
        if disable_scp:
            print("🚫 Disabling SCP provisioning to test PROVISIONING_SCRIPT environment variable")
        
        endpoint = manager.find_or_create_instance(
            create_if_none=True,  # Always create since we confirmed none exist
            wait_for_ready=not args.no_wait,
            provision_existing=False,  # Not needed since it's a new instance
            disable_scp_provisioning=disable_scp
        )
        if endpoint:
            print(f"✅ New instance created successfully!")
            print(f"Instance ready at {endpoint.url} (ID: {endpoint.instance_id})")
            print(manager.get_ssh_command(endpoint.instance_id))
            
            if disable_scp:
                print("📋 Test if provisioning script ran by checking for markers...")
                test_provisioning_simple(manager, endpoint.instance_id)
        else:
            print("❌ Failed to create new instance")


def update_instance(manager: VastAIManager, args: argparse.Namespace):
    """Update server code on an instance by uploading .py files and restarting service."""
    instance_id = getattr(args, 'instance_id', None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    
    print(f"Updating server code on instance {instance_id}...")
    
    # Set up more verbose logging for this operation
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    success = manager.update_server_code(
        instance_id, 
        timeout=args.timeout,
        verbose=args.verbose
    )
    if success:
        print(f"✅ Instance {instance_id} updated successfully!")
        print("🔄 Server code updated and service restarted")
        
        # Try to check health to confirm it's working
        endpoint = manager.get_model_server_endpoint(instance_id)
        if endpoint:
            print(f"🌐 Model server should be available at: {endpoint.url}")
            print("💡 You can check health with: python scripts/vastai_cli.py health")
        else:
            print("⚠️  Could not get model server endpoint, but update completed")
    else:
        print(f"❌ Failed to update instance {instance_id}")
        print("💡 Try running with SSH manually to debug:")
        ssh_cmd = manager.get_ssh_command(instance_id)
        if ssh_cmd:
            print(f"   {ssh_cmd}")


def fix_instance(manager: VastAIManager, args: argparse.Namespace):
    """Fix an instance by running SCP provisioning."""
    instance_id = getattr(args, 'instance_id', None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    
    print(f"Fixing instance {instance_id} using SCP provisioning...")
    
    # Set up more verbose logging for this operation
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    success = manager.provision_instance_via_scp(
        instance_id, 
        timeout=args.timeout,
        verbose=args.verbose
    )
    if success:
        print(f"✅ Instance {instance_id} fixed successfully!")
        
        # Try to get the endpoint to confirm it's working
        endpoint = manager.get_model_server_endpoint(instance_id)
        if endpoint:
            print(f"🌐 Model server should be available at: {endpoint.url}")
        else:
            print("⚠️  Could not get model server endpoint, but provisioning completed")
    else:
        print(f"❌ Failed to fix instance {instance_id}")
        print("💡 Try running with SSH manually to debug:")
        ssh_cmd = manager.get_ssh_command(instance_id)
        if ssh_cmd:
            print(f"   {ssh_cmd}")
            print("   Then run: chmod +x /workspace/vast_provisioning.sh && /workspace/vast_provisioning.sh")


def ssh_command(manager: VastAIManager, args: argparse.Namespace):
    instance_id = getattr(args, 'instance_id', None)
    debug = getattr(args, 'debug', False)
    
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    
    if debug:
        # Show all available SSH methods for debugging
        methods = manager.get_ssh_methods(instance_id)
        print(f"SSH methods for instance {instance_id}:")
        
        if methods['direct']:
            print(f"  Direct:  {methods['direct']}")
            _cleanup_ssh_host_keys(methods['direct'])
        else:
            print("  Direct:  Not available")
            
        if methods['proxy']:
            print(f"  Proxy:   {methods['proxy']}")
            _cleanup_ssh_host_keys(methods['proxy'])
        else:
            print("  Proxy:   Not available")
    else:
        # Show preferred SSH method
        cmd = manager.get_ssh_command(instance_id)
        if cmd:
            # Clean up old host keys before showing SSH command
            _cleanup_ssh_host_keys(cmd)
            print(cmd)
        else:
            print("SSH command not available for instance", instance_id)


def _cleanup_ssh_host_keys(ssh_cmd: str):
    """
    Remove old SSH host keys for the host/port combination to prevent 
    'remote host identification has changed' warnings.
    
    Args:
        ssh_cmd: SSH command string like "ssh -p 40730 root@70.77.113.32"
    """
    try:
        import subprocess
        import os
        
        # Parse SSH command to extract host and port
        parts = ssh_cmd.split()
        port = None
        host = None
        
        for i, part in enumerate(parts):
            if part == "-p" and i + 1 < len(parts):
                port = parts[i + 1]
            elif "@" in part:
                host = part.split("@")[1]
        
        if host and port:
            # Format the host entry as SSH stores it: [host]:port
            host_entry = f"[{host}]:{port}"
            
            # Check if we have a known_hosts file
            known_hosts_path = os.path.expanduser("~/.ssh/known_hosts")
            if os.path.exists(known_hosts_path):
                # Remove old entries for this host:port combination
                cleanup_cmd = ["ssh-keygen", "-f", known_hosts_path, "-R", host_entry]
                result = subprocess.run(cleanup_cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and "updated" in result.stdout:
                    print(f"🧹 Cleaned up old SSH host keys for {host_entry}")
                # Don't show anything if no cleanup was needed
                    
    except Exception as e:
        # Don't fail the SSH command if cleanup fails
        print(f"⚠️  Note: Could not clean up SSH host keys: {e}")


def endpoint_info(manager: VastAIManager, args: argparse.Namespace):
    instance_id = getattr(args, 'instance_id', None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    
    endpoint = manager.get_model_server_endpoint(instance_id)
    if endpoint:
        print(json.dumps(vars(endpoint), indent=2))
    else:
        print("Endpoint not available for instance", instance_id)


def stop_instance(manager: VastAIManager, args: argparse.Namespace):
    instance_id = getattr(args, 'instance_id', None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    
    result = manager.stop_instance(instance_id)
    print(json.dumps(result, indent=2))


def start_instance(manager: VastAIManager, args: argparse.Namespace):
    instance_id = getattr(args, 'instance_id', None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    
    result = manager.start_instance(instance_id)
    print(json.dumps(result, indent=2))


def restart_instance(manager: VastAIManager, args: argparse.Namespace):
    instance_id = getattr(args, 'instance_id', None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    
    result = manager.restart_instance(instance_id)
    print(json.dumps(result, indent=2))


def destroy_instance(manager: VastAIManager, args: argparse.Namespace):
    instance_id = getattr(args, 'instance_id', None)
    if instance_id is None:
        # If no specific instance ID provided, destroy ALL instances
        all_instances = manager.show_instances(raw=True)
        if not all_instances:
            print("No instances found to destroy")
            return
        
        print(f"Found {len(all_instances)} instance(s) to destroy:")
        for instance in all_instances:
            inst_id = instance.get("id")
            status = instance.get("actual_status", "unknown")
            image = instance.get("image", "unknown")
            print(f"  - Instance {inst_id}: {status} ({image})")
        
        # Destroy all instances
        destroyed_count = 0
        for instance in all_instances:
            inst_id = instance.get("id")
            if inst_id is not None:
                try:
                    result = manager.destroy_instance(inst_id)
                    print(f"✅ Destroyed instance {inst_id}")
                    destroyed_count += 1
                except Exception as e:
                    print(f"❌ Failed to destroy instance {inst_id}: {e}")
            else:
                print(f"❌ Instance missing ID: {instance}")
        
        print(f"Destroyed {destroyed_count}/{len(all_instances)} instances")
        return
    
    # Destroy specific instance by ID
    result = manager.destroy_instance(instance_id)
    print(json.dumps(result, indent=2))


def test_provisioning_simple(manager: VastAIManager, instance_id: int):
    """Simple test if provisioning markers exist on an instance."""
    import time
    
    print(f"Testing provisioning markers on instance {instance_id}...")
    print("⏳ Waiting 30 seconds for provisioning script to complete...")
    time.sleep(30)
    
    # Get SSH connection info
    ssh_cmd = manager.get_ssh_command(instance_id)
    if not ssh_cmd:
        print("❌ Could not get SSH command for instance")
        return
    
    # Parse SSH command
    try:
        parts = ssh_cmd.split()
        port = parts[2]  # After -p
        host_and_user = parts[3]  # root@HOST
        host = host_and_user.split('@')[1]
        user = host_and_user.split('@')[0]
    except (IndexError, ValueError) as e:
        print(f"❌ Failed to parse SSH command '{ssh_cmd}': {e}")
        return
    
    # Test SSH connectivity first
    test_ssh_cmd = [
        "ssh", "-p", port,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", 
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        f"{user}@{host}",
        "echo 'SSH OK'"
    ]
    
    try:
        result = subprocess.run(test_ssh_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            print(f"❌ SSH connection failed: {result.stderr.strip()}")
            return
        print("✅ SSH connection successful")
    except Exception as e:
        print(f"❌ SSH test failed: {e}")
        return
    
    # Check for test provisioning markers
    test_markers_cmd = [
        "ssh", "-p", port,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR", 
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        f"{user}@{host}",
        "ls -la /TEST_PROVISIONING_* 2>/dev/null || echo 'NO_MARKERS'"
    ]
    
    try:
        result = subprocess.run(test_markers_cmd, capture_output=True, text=True, timeout=15)
        output = result.stdout.strip()
        
        if "NO_MARKERS" in output:
            print("❌ No test provisioning markers found")
            print("   This suggests PROVISIONING_SCRIPT environment variable did not run")
        else:
            print("✅ Test provisioning markers found:")
            print(f"   {output}")
            print("   This suggests PROVISIONING_SCRIPT environment variable worked!")
            
    except Exception as e:
        print(f"❌ Failed to check markers: {e}")
    
    # Check if test web server is running
    endpoint = manager.get_model_server_endpoint(instance_id)
    if endpoint:
        test_url = f"{endpoint.url}/test"
        print(f"\n🌐 Testing if test web server is running at {test_url}...")
        try:
            import requests
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                print("✅ Test web server is running!")
                print(f"   Response: {response.text.strip()}")
            else:
                print(f"❌ Test web server returned HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to connect to test web server: {e}")
    else:
        print("❌ Could not get instance endpoint for web server test")


def test_provisioning_markers(manager: VastAIManager, args: argparse.Namespace):
    """Test if provisioning markers exist on an instance to verify if provisioning script ran."""
    instance_id = getattr(args, 'instance_id', None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    
    print(f"Testing provisioning markers on instance {instance_id}...")
    
    # Get SSH connection info
    ssh_cmd = manager.get_ssh_command(instance_id)
    if not ssh_cmd:
        print("❌ Could not get SSH command for instance")
        return
    
    # Parse SSH command
    try:
        parts = ssh_cmd.split()
        port = parts[2]
        host_and_user = parts[3]
        host = host_and_user.split('@')[1]
        user = host_and_user.split('@')[0]
    except (IndexError, ValueError) as e:
        print(f"❌ Failed to parse SSH command: {e}")
        return
    
    marker_locations = [
        '/workspace/TEST_PROVISIONING_SUCCESS.txt',
        '/tmp/TEST_PROVISIONING_SUCCESS.txt',
        '/root/TEST_PROVISIONING_SUCCESS.txt',
        '/tmp/test_provisioning.log'
    ]
    
    results = {}
    
    for location in marker_locations:
        try:
            import subprocess
            check_cmd = [
                "ssh", "-p", port,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", 
                "-o", "LogLevel=ERROR",
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",
                f"{user}@{host}",
                f"test -f {location} && echo EXISTS || echo MISSING"
            ]
            
            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                exists = "EXISTS" in result.stdout.strip()
                results[location] = exists
            else:
                results[location] = False
                
        except Exception as e:
            print(f"❌ Error checking {location}: {e}")
            results[location] = False
    
    # Display results
    print("Marker file status:")
    for location, exists in results.items():
        status = "✅ EXISTS" if exists else "❌ MISSING"
        print(f"  {location}: {status}")
    
    if any(results.values()):
        print("\n🎉 Provisioning script appears to have run successfully!")
        
        # Try to show the test endpoint if it exists
        endpoint = manager.get_model_server_endpoint(instance_id)
        if endpoint:
            print(f"💡 You can also check the test web page at: http://{endpoint.public_ip}:8000/test")
    else:
        print("\n❌ No provisioning markers found - script may not have executed.")
        print("💡 Try running: python scripts/vastai_cli.py fix")


def health_check(manager: VastAIManager, args: argparse.Namespace):
    instance_id = getattr(args, 'instance_id', None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    
    endpoint = manager.get_model_server_endpoint(instance_id)
    if not endpoint:
        print("Endpoint not found for instance", instance_id)
        return
    try:
        url = f"{endpoint.url}/healthcheck"
        resp = requests.get(url, timeout=10)
        print(f"Healthcheck {resp.status_code}: {resp.text}")
    except requests.RequestException as e:
        print(f"Healthcheck request failed: {e}")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Manage Vast.ai instances for Experimance image_server",
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # list
    p_list = subparsers.add_parser('list', help='List all instances')
    p_list.add_argument('--raw', action='store_true', help='Show json output instead of pretty print')

    # search
    p_search = subparsers.add_parser('search', help='Search for GPU offers')
    p_search.add_argument('--min-gpu-ram', type=int, default=16, help='Minimum GPU RAM (GB)')
    p_search.add_argument('--max-price', type=float, default=0.5, help='Max price ($/hr)')
    p_search.add_argument('--dlperf', type=float, default=32.0, help='Minimum DLPerf score')

    # provision
    p_prov = subparsers.add_parser('provision', help='Find or create an instance (always provisions)')
    p_prov.add_argument('--no-wait', action='store_true', dest='no_wait', help='Do not wait for instance ready')
    p_prov.add_argument('--verbose', action='store_true', help='Show provisioning script output in real-time')
    p_prov.add_argument('--provision-script', type=str, metavar='URL', help='Custom provisioning script URL to use')

    # fix
    p_fix = subparsers.add_parser('fix', help='Fix an instance using SCP provisioning')
    p_fix.add_argument('instance_id', type=int, nargs='?', help='Instance ID to fix (uses active instance if not provided)')
    p_fix.add_argument('--timeout', type=int, default=300, help='Timeout for provisioning (seconds)')
    p_fix.add_argument('--debug', action='store_true', help='Enable debug logging')
    p_fix.add_argument('--verbose', action='store_true', help='Show provisioning script output in real-time')

    # update
    p_update = subparsers.add_parser('update', help='Update server code on instance and restart service')
    p_update.add_argument('instance_id', type=int, nargs='?', help='Instance ID to update (uses active instance if not provided)')
    p_update.add_argument('--timeout', type=int, default=120, help='Timeout for update operations (seconds)')
    p_update.add_argument('--debug', action='store_true', help='Enable debug logging')
    p_update.add_argument('--verbose', action='store_true', help='Show update output in real-time')

    # ssh
    p_ssh = subparsers.add_parser('ssh', help='Get SSH command for an instance')
    p_ssh.add_argument('instance_id', type=int, nargs='?', help='Instance ID (uses active instance if not provided)')
    p_ssh.add_argument('--debug', action='store_true', help='Show all available SSH methods (direct and proxy)')

    # endpoint
    p_ep = subparsers.add_parser('endpoint', help='Get model server endpoint for an instance')
    p_ep.add_argument('instance_id', type=int, nargs='?', help='Instance ID (uses active instance if not provided)')

    # stop/start/restart/destroy/health/test-markers
    for cmd in ('stop', 'start', 'restart', 'destroy', 'health', 'test-markers'):
        p = subparsers.add_parser(cmd, help=f'{cmd.capitalize()} an instance')
        p.add_argument('instance_id', type=int, nargs='?', help='Instance ID (uses active instance if not provided)')

    args = parser.parse_args()
    
    # Create manager with custom provisioning script if provided for provision command
    if args.command == 'provision' and hasattr(args, 'provision_script') and args.provision_script:
        manager = VastAIManager(provisioning_script_url=args.provision_script)
    else:
        manager = VastAIManager()

    # Dispatch commands
    if args.command == 'list':
        list_instances(manager, args)
    elif args.command == 'search':
        search_offers(manager, args)
    elif args.command == 'provision':
        provision_instance(manager, args)
    elif args.command == 'fix':
        fix_instance(manager, args)
    elif args.command == 'update':
        update_instance(manager, args)
    elif args.command == 'ssh':
        ssh_command(manager, args)
    elif args.command == 'endpoint':
        endpoint_info(manager, args)
    elif args.command == 'stop':
        stop_instance(manager, args)
    elif args.command == 'start':
        start_instance(manager, args)
    elif args.command == 'restart':
        restart_instance(manager, args)
    elif args.command == 'destroy':
        destroy_instance(manager, args)
    elif args.command == 'health':
        health_check(manager, args)
    elif args.command == 'test-markers':
        test_provisioning_markers(manager, args)
    else:
        parser.print_help()


if __name__ == '__main__':  # pragma: no cover
    main()
