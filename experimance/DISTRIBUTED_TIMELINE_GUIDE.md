# Distributed Timeline CLI Guide

The timeline CLI now supports distributed deployments, allowing you to view logs from multiple machines seamlessly.

## Quick Start

**Local mode (default):**
```bash
uv run timeline list                    # Uses local logs
uv run timeline show 0                  # Show local session
```

**Distributed mode:**
```bash
uv run timeline --deployment list       # Auto-discovers from deployment.toml
uv run timeline --deployment stream     # Streams from multiple machines
uv run timeline --deployment show 0     # Show session from distributed logs
```

## How It Works

1. **Discovers deployment.toml** - Automatically finds your project's deployment configuration
2. **Maps services to machines** - Identifies where transcripts and prompts are located:
   - Transcripts: machines running `agent` service
   - Prompts: machines running `core` service
3. **Syncs via SSH** - Uses rsync over SSH to fetch logs to local cache
4. **Caches locally** - Stores in `~/.experimance/log_cache/` to avoid re-downloading

## SSH Integration

The system respects your SSH config. For example:

**SSH config:**
```ssh
Host ia360
  User experimance
  HostName ia360

Host iamini
  User fireproject
  HostName fireprojects-mac-mini
```

**deployment.toml:**
```toml
[machines.ubuntu]
hostname = "ia360"              # Matches SSH config
user = "experimance"
services = ["core", "image_server", "display", "health"]

[machines.macos]
hostname = "iamini"             # Matches SSH config  
user = "fireproject"
services = ["agent", "health"]
```

## Advanced Configuration

For complex network setups (e.g., different hostnames for different contexts), use `ssh_hostname` override:

```toml
[machines.ubuntu]
hostname = "ia360.local"                        # Logical hostname
ssh_hostname = "ia360"                          # SSH config hostname
# ssh_hostname = "ia360.tailscale.ts.net"      # Or tailscale hostname
user = "experimance"
services = ["core", "image_server", "display", "health"]
```

## Troubleshooting

**SSH connection failures:**
- System gracefully falls back to cached data
- Check SSH key access to remote machines
- Verify hostnames in deployment.toml match SSH config

**Missing log directories:**
- The system reports missing directories clearly:
  ```
  Remote directory /var/log/experimance/prompts does not exist on ia360
  ```
- This is expected if services aren't running yet

**Cache management:**
- Cache location: `~/.experimance/log_cache/`
- Clear cache: `rm -rf ~/.experimance/log_cache/`
- Cache is organized by machine ID and log type

## Examples

**View deployment configuration:**
```bash
uv run timeline --deployment-file projects/fire/deployment.toml list
```

**Monitor fire project across machines:**
```bash
# Set fire as current project first
uv run set-project fire

# Stream from distributed deployment  
uv run timeline --deployment stream
```

**Debug with explicit paths:**
```bash
uv run timeline --transcripts-path ~/.experimance/log_cache/macos/transcripts --prompts-path ~/.experimance/log_cache/ubuntu/prompts list
```

This distributed timeline viewer makes it easy to monitor your multi-machine installations from any development machine with SSH access to the deployment!