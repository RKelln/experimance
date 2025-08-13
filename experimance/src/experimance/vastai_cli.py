"""Vast.ai management CLI (packaged entry point).

This relocates the previous scripts/vastai_cli.py into the installed experimance package.
Use via:

  uv run vastai list
  uv run vastai search --min-gpu-ram 24 --max-price 0.7
  uv run vastai provision

Legacy invocation (python scripts/vastai_cli.py ...) still works but is deprecated.

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
    test           Test image generation performance and report timing (auto-detects active instance)

Note: Commands that take an instance_id will automatically use the first running 
Experimance instance if no ID is provided.
"""

from __future__ import annotations

# NOTE: We inline the original implementation to avoid runtime dependency on the scripts/ directory
# which is not included when the package is installed. Some minor cleanup (duplicate imports) performed.

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Optional

import requests

from image_server.generators.vastai.vastai_manager import VastAIManager  # type: ignore
from image_server.generators.vastai.server.data_types import era_to_loras  # type: ignore

# --- (Original functions copied with minimal edits) ---


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_active_instance_id(manager: VastAIManager) -> Optional[int]:
    try:
        instances = manager.find_experimance_instances()
        if instances:
            instance_id = instances[0]["id"]
            print(f"Using active Experimance instance: {instance_id}")
            return instance_id
        print("❌ No running Experimance instances found")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error finding active instance: {e}")
        return None


def list_instances(manager: VastAIManager, args: argparse.Namespace):  # noqa: D401
    raw = getattr(args, "raw", False)
    instances = manager.show_instances(raw=raw)
    if isinstance(instances, str):
        print(instances)
    else:
        print(json.dumps(instances, indent=2))


def search_offers(manager: VastAIManager, args: argparse.Namespace):  # noqa: D401
    offers = manager.search_offers(
        min_gpu_ram=args.min_gpu_ram,
        max_price=args.max_price,
        dlperf=args.dlperf,
    )
    if not offers:
        print("No offers found matching criteria")
        return
    cheapest_price = min(o.get("dph_total", float("inf")) for o in offers)
    price_threshold = cheapest_price * 1.25
    rows = []
    for offer in offers:
        gpu_ram = offer.get("gpu_ram", 0) / 1024
        score = manager._calculate_offer_score(offer, cheapest_price, price_threshold)
        rows.append([
            offer.get("id", ""),
            f"{score:.1f}",
            offer.get("gpu_name", ""),
            f"{offer.get('dph_total', 0):.3f}",
            f"{offer.get('dlperf', 0):.1f}",
            f"{offer.get('dlperf_per_dphtotal', 0):.1f}",
            f"{offer.get('reliability2', 0):.3f}",
            offer.get("geolocation", ""),
            offer.get("verified", False),
            f"{offer.get('inet_down', 0):.0f}",
            f"{gpu_ram:.1f}",
        ])
    headers = [
        "ID",
        "Score",
        "GPU",
        "Price($/hr)",
        "DLPerf",
        "DLPerf/$",
        "Reliability",
        "Location",
        "Verified",
        "Down(Mbps)",
        "RAM(GB)",
    ]
    col_widths: list[int] = []
    for idx, header in enumerate(headers):
        max_len = len(header)
        for row in rows:
            cell_len = len(str(row[idx]))
            if cell_len > max_len:
                max_len = cell_len
        col_widths.append(max_len)
    header_row = " | ".join(header.ljust(col_widths[i]) for i, header in enumerate(headers))
    sep_row = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_row)
    print(sep_row)
    for row in rows:
        line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        print(line)


def provision_instance(manager: VastAIManager, args: argparse.Namespace):  # noqa: D401
    if hasattr(args, "provision_script") and args.provision_script:
        manager = VastAIManager(provisioning_script_url=args.provision_script)
    instance_id = getattr(args, "instance_id", None)
    print(f"Instance ID provided: {instance_id}")
    if instance_id is None:
        existing_instances = manager.find_experimance_instances()
    else:
        result = manager.create_instance(instance_id)
        instance_id = result.get("new_contract")
        if not instance_id:
            print(f"Failed to create instance: {result}")
            return None
        existing_instances = manager.find_experimance_instances()
    if existing_instances:
        instance_id = existing_instances[0]["id"]
        print(f"Found existing Experimance instance {instance_id}, provisioning it...")
        if manager.wait_for_instance_ready(instance_id):
            if manager.wait_for_ssh_ready(instance_id, timeout=180):
                success = manager.provision_instance_via_scp(
                    instance_id, verbose=getattr(args, "show_output", False)
                )
                if success:
                    endpoint = manager.get_model_server_endpoint(instance_id)
                    if endpoint:
                        print("✅ Instance provisioned successfully!")
                        print(f"Instance ready at {endpoint.url} (ID: {endpoint.instance_id})")
                        print(manager.get_ssh_command(endpoint.instance_id))
                    else:
                        print("✅ Instance provisioned (endpoint not yet available)")
                else:
                    print(f"❌ Failed to provision existing instance {instance_id}")
            else:
                print(f"❌ SSH not ready for existing instance {instance_id}")
        else:
            print(f"❌ Existing instance {instance_id} is not ready")
    else:
        print("No existing Experimance instances found, creating new one...")
        disable_scp = hasattr(args, "provision_script") and args.provision_script
        if disable_scp:
            print("🚫 Disabling SCP provisioning due to custom provisioning script")
        endpoint = manager.find_or_create_instance(
            create_if_none=True,
            wait_for_ready=not args.no_wait,
            provision_existing=False,
            disable_scp_provisioning=disable_scp,
            min_gpu_ram=args.min_gpu_ram,
            max_price=args.max_price,
            dlperf=args.dlperf,
        )
        if endpoint:
            print("✅ New instance created successfully!")
            print(f"Instance ready at {endpoint.url} (ID: {endpoint.instance_id})")
            print(manager.get_ssh_command(endpoint.instance_id))
        else:
            print("❌ Failed to create new instance")
            return 1


def update_instance(manager: VastAIManager, args: argparse.Namespace):  # noqa: D401
    instance_id = getattr(args, "instance_id", None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    print(f"Updating server code on instance {instance_id}...")
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    success = manager.update_server_code(instance_id, timeout=args.timeout, verbose=args.show_output)
    if success:
        print(f"✅ Instance {instance_id} updated successfully!")
        endpoint = manager.get_model_server_endpoint(instance_id)
        if endpoint:
            print(f"🌐 Model server at: {endpoint.url}")
    else:
        print(f"❌ Failed to update instance {instance_id}")


def fix_instance(manager: VastAIManager, args: argparse.Namespace):  # noqa: D401
    instance_id = getattr(args, "instance_id", None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    print(f"Fixing instance {instance_id} using SCP provisioning...")
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    success = manager.provision_instance_via_scp(
        instance_id, timeout=args.timeout, verbose=args.show_output
    )
    if success:
        print(f"✅ Instance {instance_id} fixed successfully!")
    else:
        print(f"❌ Failed to fix instance {instance_id}")


def ssh_command(manager: VastAIManager, args: argparse.Namespace):  # noqa: D401
    instance_id = getattr(args, "instance_id", None)
    debug = getattr(args, "debug", False)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    if debug:
        methods = manager.get_ssh_methods(instance_id)
        print(f"SSH methods for instance {instance_id}: {methods}")
    else:
        cmd = manager.get_ssh_command(instance_id)
        if cmd:
            _cleanup_ssh_host_keys(cmd)
            print(cmd)
        else:
            print("SSH command not available")


def _cleanup_ssh_host_keys(ssh_cmd: str):  # noqa: D401
    try:
        parts = ssh_cmd.split()
        port = None
        host = None
        for i, part in enumerate(parts):
            if part == "-p" and i + 1 < len(parts):
                port = parts[i + 1]
            elif "@" in part:
                host = part.split("@")[1]
        if host and port:
            host_entry = f"[{host}]:{port}"
            known_hosts_path = os.path.expanduser("~/.ssh/known_hosts")
            if os.path.exists(known_hosts_path):
                cleanup_cmd = ["ssh-keygen", "-f", known_hosts_path, "-R", host_entry]
                subprocess.run(cleanup_cmd, capture_output=True, text=True)
    except Exception:  # noqa: BLE001
        pass


def endpoint_info(manager: VastAIManager, args: argparse.Namespace):  # noqa: D401
    instance_id = getattr(args, "instance_id", None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    endpoint = manager.get_model_server_endpoint(instance_id)
    if endpoint:
        print(json.dumps(vars(endpoint), indent=2))
    else:
        print("Endpoint not available")


def stop_instance(manager: VastAIManager, args: argparse.Namespace):  # noqa: D401
    instance_id = getattr(args, "instance_id", None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    result = manager.stop_instance(instance_id)
    print(json.dumps(result, indent=2))


def start_instance(manager: VastAIManager, args: argparse.Namespace):  # noqa: D401
    instance_id = getattr(args, "instance_id", None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    result = manager.start_instance(instance_id)
    print(json.dumps(result, indent=2))


def restart_instance(manager: VastAIManager, args: argparse.Namespace):  # noqa: D401
    instance_id = getattr(args, "instance_id", None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    result = manager.restart_instance(instance_id)
    print(json.dumps(result, indent=2))


def destroy_instance(manager: VastAIManager, args: argparse.Namespace):  # noqa: D401
    instance_id = getattr(args, "instance_id", None)
    if instance_id is None:
        all_instances = manager.show_instances(raw=True)
        if not all_instances:
            print("No instances found to destroy")
            return
        for instance in all_instances:
            inst_id = instance.get("id")
            if inst_id is not None:
                try:
                    manager.destroy_instance(inst_id)
                    print(f"✅ Destroyed instance {inst_id}")
                except Exception as e:  # noqa: BLE001
                    print(f"❌ Failed to destroy instance {inst_id}: {e}")
        return
    result = manager.destroy_instance(instance_id)
    print(json.dumps(result, indent=2))


def health_check(manager: VastAIManager, args: argparse.Namespace):  # noqa: D401
    instance_id = getattr(args, "instance_id", None)
    if instance_id is None:
        instance_id = get_active_instance_id(manager)
        if instance_id is None:
            return
    print(f"Checking health of instance {instance_id}... (Ctrl+C to stop)\n")
    try:
        while True:
            try:
                instance_data = manager.show_instance(instance_id, raw=True)
                if isinstance(instance_data, dict):
                    actual_status = instance_data.get("actual_status", "unknown")
                    intended_status = instance_data.get("intended_status", "unknown")
                    print(f"Instance status: actual={actual_status} intended={intended_status}")
                    is_broken, error_desc = manager._is_instance_unrecoverably_broken(instance_data)
                    if is_broken:
                        print(f"🚨 Instance unrecoverably broken: {error_desc}")
                        return
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ Status error: {e}")
            endpoint = manager.get_model_server_endpoint(instance_id)
            if endpoint:
                try:
                    url = f"{endpoint.url}/healthcheck"
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        try:
                            resp_json = resp.json()
                            print(json.dumps(resp_json, indent=2))
                            if resp_json.get("status") == "ready":
                                print("✅ Ready")
                                return
                        except json.JSONDecodeError:
                            print(resp.text)
                    else:
                        print(f"HTTP {resp.status_code}: {resp.text}")
                except requests.RequestException as e:  # noqa: BLE001
                    print(f"Health request failed: {e}")
            else:
                print("Endpoint not yet available")
            print()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped")


def main() -> int:  # noqa: D401
    parser = argparse.ArgumentParser(description="Manage Vast.ai instances for Experimance image_server")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_list = subparsers.add_parser("list", help="List all instances")
    p_list.add_argument("--raw", action="store_true", help="Show raw JSON output")
    p_search = subparsers.add_parser("search", help="Search for GPU offers")
    p_search.add_argument("--min-gpu-ram", type=int, default=16)
    p_search.add_argument("--max-price", type=float, default=0.5)
    p_search.add_argument("--dlperf", type=float, default=32.0)
    p_prov = subparsers.add_parser("provision", help="Find or create and provision an instance")
    p_prov.add_argument("instance_id", type=int, nargs="?")
    p_prov.add_argument("--no-wait", action="store_true", dest="no_wait")
    p_prov.add_argument("--show-output", action="store_true")
    p_prov.add_argument("--provision-script", type=str)
    p_prov.add_argument("--min-gpu-ram", type=int, default=16)
    p_prov.add_argument("--max-price", type=float, default=0.5)
    p_prov.add_argument("--dlperf", type=float, default=32.0)
    p_fix = subparsers.add_parser("fix", help="Fix an instance using SCP provisioning")
    p_fix.add_argument("instance_id", type=int, nargs="?")
    p_fix.add_argument("--timeout", type=int, default=300)
    p_fix.add_argument("--debug", action="store_true")
    p_fix.add_argument("--show-output", action="store_true")
    p_update = subparsers.add_parser("update", help="Update server code and restart service")
    p_update.add_argument("instance_id", type=int, nargs="?")
    p_update.add_argument("--timeout", type=int, default=120)
    p_update.add_argument("--debug", action="store_true")
    p_update.add_argument("--show-output", action="store_true")
    p_ssh = subparsers.add_parser("ssh", help="Get SSH command for an instance")
    p_ssh.add_argument("instance_id", type=int, nargs="?")
    p_ssh.add_argument("--debug", action="store_true")
    p_ep = subparsers.add_parser("endpoint", help="Get model server endpoint")
    p_ep.add_argument("instance_id", type=int, nargs="?")
    for cmd in ("stop", "start", "restart", "destroy", "health"):
        sp = subparsers.add_parser(cmd, help=f"{cmd.capitalize()} an instance")
        sp.add_argument("instance_id", type=int, nargs="?")
    args = parser.parse_args()
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    if args.command == "provision" and hasattr(args, "provision_script") and args.provision_script:
        manager = VastAIManager(provisioning_script_url=args.provision_script)
    else:
        manager = VastAIManager()
    if args.command == "list":
        list_instances(manager, args)
    elif args.command == "search":
        search_offers(manager, args)
    elif args.command == "provision":
        provision_instance(manager, args)
    elif args.command == "fix":
        fix_instance(manager, args)
    elif args.command == "update":
        update_instance(manager, args)
    elif args.command == "ssh":
        ssh_command(manager, args)
    elif args.command == "endpoint":
        endpoint_info(manager, args)
    elif args.command == "stop":
        stop_instance(manager, args)
    elif args.command == "start":
        start_instance(manager, args)
    elif args.command == "restart":
        restart_instance(manager, args)
    elif args.command == "destroy":
        destroy_instance(manager, args)
    elif args.command == "health":
        health_check(manager, args)
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
