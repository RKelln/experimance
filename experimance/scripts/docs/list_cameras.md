# list_cameras.py

Discovers Reolink IP cameras on the local network using several detection strategies.

See `scripts/list_cameras.py`.  
Related: [`test_reolink_camera.md`](test_reolink_camera.md)

## Quick Start

```bash
# Comprehensive discovery (default — most thorough)
uv run python scripts/list_cameras.py

# Test a known IP directly
uv run python scripts/list_cameras.py --known-ip 192.168.2.229
```

## Discovery Modes

| Flag | Method | Speed | Notes |
|---|---|---|---|
| (default) | Comprehensive | Slow | Combines multiple strategies; most reliable |
| `--fast` | Port scan (HTTPS/443) | Fast | Finds any HTTPS device, not just Reolink |
| `--signature` | API signature probe | Medium | Credential-free; identifies Reolink by API shape |
| `--arp` | ARP table | Fastest | Only sees devices recently on the network |
| `--nmap` | nmap scan | Medium | Requires `nmap` installed |
| `--mdns` | mDNS/Bonjour | Fast | Cameras must advertise via mDNS |

```bash
uv run python scripts/list_cameras.py --fast
uv run python scripts/list_cameras.py --signature
uv run python scripts/list_cameras.py --arp
uv run python scripts/list_cameras.py --nmap
uv run python scripts/list_cameras.py --mdns
```

## Additional Options

| Flag | Description |
|---|---|
| `--known-ip IP` | Test a specific IP before broader scanning |
| `--subnet CIDR` | Restrict scan to a specific subnet (e.g. `192.168.2.0/24`) |
| `--debug` | Enable verbose logging |

## Troubleshooting

- Camera not found? Try `--known-ip <camera-ip>` to test directly
- Use `--fast` to see all HTTPS devices on the network (may include routers, NAS, etc.)
- Ping the camera first if using `--arp` (it only sees recently active devices)
- `--nmap` requires `nmap`: `sudo apt install nmap` / `brew install nmap`

## Requirements

- Network access on the same LAN as the cameras
- `services/agent/src/agent/vision/reolink_discovery.py` (loaded from project path)
- `requests`, `urllib3` for HTTP probing
- Optional: `nmap` for nmap-based discovery
