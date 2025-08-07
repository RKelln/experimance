#!/bin/bash
# Remote Access Monitor and Recovery Script
# Monitors SSH, Tailscale, and system health with auto-recovery

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_FILE="/var/log/experimance/remote-access.log"
STATE_FILE="/var/cache/experimance/remote-access-state.json"
HEALTH_FILE="/var/cache/experimance/health/remote_access_health.json"
CHECK_INTERVAL=60  # seconds
SSH_TIMEOUT=10     # seconds for SSH connection tests
TAILSCALE_TIMEOUT=15  # seconds for Tailscale operations

# Ensure log directory exists
sudo mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$STATE_FILE")" "$(dirname "$HEALTH_FILE")"

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | sudo tee -a "$LOG_FILE" > /dev/null
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1" | sudo tee -a "$LOG_FILE" > /dev/null
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | sudo tee -a "$LOG_FILE" > /dev/null
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1" | sudo tee -a "$LOG_FILE" > /dev/null
}

# Function to check SSH service health
check_ssh_service() {
    log "Checking SSH service status..."
    
    local ssh_status="unknown"
    local ssh_port="unknown"
    local ssh_listeners=0
    local ssh_active_connections=0
    
    # Check systemd service status
    if systemctl is-active ssh.service &>/dev/null; then
        ssh_status="active"
    else
        ssh_status="inactive"
        error "SSH service is not active"
        return 1
    fi
    
    # Check if SSH is listening on port 22
    if ss -tlnp | grep -q ":22 "; then
        ssh_port="listening"
        ssh_listeners=$(ss -tlnp | grep ":22 " | wc -l)
    else
        ssh_port="not_listening"
        error "SSH not listening on port 22"
        return 1
    fi
    
    # Count active SSH connections
    ssh_active_connections=$(ss -tn | grep ":22 " | grep ESTAB | wc -l)
    
    log "SSH service: $ssh_status, Port 22: $ssh_port, Listeners: $ssh_listeners, Active connections: $ssh_active_connections"
    return 0
}

# Function to check Tailscale connectivity
check_tailscale_connectivity() {
    log "Checking Tailscale connectivity..."
    
    local tailscale_status="unknown"
    local tailscale_ip=""
    local connectivity_ok=false
    local derp_connection=false
    
    # Check if tailscaled is running
    if systemctl is-active tailscaled.service &>/dev/null; then
        tailscale_status="active"
    else
        tailscale_status="inactive"
        error "Tailscale daemon is not active"
        return 1
    fi
    
    # Get Tailscale IP address
    if tailscale_ip=$(timeout $TAILSCALE_TIMEOUT tailscale ip -4 2>/dev/null); then
        log "Tailscale IP: $tailscale_ip"
    else
        error "Failed to get Tailscale IP address"
        return 1
    fi
    
    # Check overall connectivity
    if timeout $TAILSCALE_TIMEOUT tailscale status | grep -q "active\|idle"; then
        connectivity_ok=true
        log "Tailscale connectivity appears healthy"
    else
        connectivity_ok=false
        warn "Tailscale connectivity may be impaired"
    fi
    
    # Check DERP connection health
    local derp_health
    if derp_health=$(timeout $TAILSCALE_TIMEOUT tailscale status --json 2>/dev/null | jq -r '.Health[]? | select(.Component? == "derp")?' 2>/dev/null); then
        if [ -n "$derp_health" ] && echo "$derp_health" | grep -q '"Level":"ok"'; then
            derp_connection=true
            log "DERP connection is healthy"
        elif [ -n "$derp_health" ]; then
            derp_connection=false
            warn "DERP connection may be unhealthy: $derp_health"
        else
            # No DERP health info available, but connectivity is working
            derp_connection=true
            log "DERP connection status unknown but connectivity working"
        fi
    else
        # If we can't check DERP but overall connectivity is ok, don't fail
        if [ "$connectivity_ok" = true ]; then
            derp_connection=true
            log "DERP connection status unknown but connectivity working"
        else
            warn "Could not check DERP connection health"
        fi
    fi
    
    return 0
}

# Function to check network connectivity
check_network_connectivity() {
    log "Checking network connectivity..."
    
    local internet_ok=false
    local dns_ok=false
    local gateway_ok=false
    
    # Check internet connectivity - use || true to prevent exit on failure
    if ping -c 1 -W 5 8.8.8.8 &>/dev/null || true; then
        internet_ok=true
        log "Internet connectivity: OK"
    else
        internet_ok=false
        error "Internet connectivity: FAILED"
    fi
    
    # Check DNS resolution - use || true to prevent exit on failure
    if nslookup google.com &>/dev/null || true; then
        dns_ok=true
        log "DNS resolution: OK"
    else
        dns_ok=false
        warn "DNS resolution: FAILED"
    fi
    
    # Check default gateway - use || true to prevent exit on failure
    local gateway=$(ip route | grep default | awk '{print $3}' | head -1 || true)
    if [ -n "$gateway" ] && (ping -c 1 -W 3 "$gateway" &>/dev/null || true); then
        gateway_ok=true
        log "Gateway connectivity to $gateway: OK"
    else
        gateway_ok=false
        warn "Gateway connectivity: FAILED"
    fi
    
    # Overall network health
    if [ "$internet_ok" = true ] && [ "$dns_ok" = true ] && [ "$gateway_ok" = true ]; then
        return 0
    else
        return 1
    fi
}

# Function to check system resources
check_system_resources() {
    log "Checking system resources..."
    
    local memory_usage_percent memory_available disk_usage_percent load_avg cpu_temp
    
    # Memory usage
    memory_usage_percent=$(free | grep Mem | awk '{printf "%.1f", ($3/$2) * 100.0}')
    memory_available=$(free -h | grep Mem | awk '{print $7}')
    
    # Disk usage
    disk_usage_percent=$(df / | awk 'NR==2 {print $(NF-1)}' | sed 's/%//')
    
    # Load average
    load_avg=$(uptime | awk -F'load average:' '{print $2}')
    
    # CPU temperature (from hardware sensors)
    cpu_temp="N/A"
    
    # Try to get CPU temperature from various sources
    # 1. Primary: use sensors command if available (most portable)
    if command -v sensors &>/dev/null; then
        # Try to get CPU temperature from sensors output
        # Look for Tctl (AMD), Core temp (Intel), or Package (Intel)
        # Handle cases where there might be a prefix before the sensor name (e.g., "3:Package id 0:")
        local temp_line=$(sensors 2>/dev/null | grep -E "(Tctl:|Core [0-9]+:|Package id [0-9]+:)" | head -1)
        if [ -n "$temp_line" ]; then
            # Extract temperature - get the first temperature value (not high/crit values)
            # Look for the first occurrence of +XX.X°C pattern and extract just the number
            cpu_temp=$(echo "$temp_line" | grep -oE '\+[0-9]+\.[0-9]+°C' | head -1 | sed 's/+//;s/°C.*//' 2>/dev/null || echo "N/A")
        fi
        
        # If that didn't work, try looking for any temperature sensor that might be CPU
        if [ "$cpu_temp" = "N/A" ]; then
            cpu_temp=$(sensors 2>/dev/null | grep -E "temp[0-9]+:" | grep -v "crit\|high\|low" | head -1 | awk '{print $2}' | sed 's/+//;s/°C.*//' 2>/dev/null || echo "N/A")
        fi
    fi
    
    # 2. Fallback: direct hwmon access for k10temp (AMD) or coretemp (Intel)
    if [ "$cpu_temp" = "N/A" ]; then
        for hwmon_dir in /sys/class/hwmon/hwmon*; do
            if [ -f "$hwmon_dir/name" ]; then
                local hwmon_name=$(cat "$hwmon_dir/name" 2>/dev/null)
                if [[ "$hwmon_name" =~ ^(k10temp|coretemp)$ ]] && [ -f "$hwmon_dir/temp1_input" ]; then
                    local temp_millidegrees=$(cat "$hwmon_dir/temp1_input" 2>/dev/null)
                    if [ -n "$temp_millidegrees" ] && [ "$temp_millidegrees" -gt 0 ]; then
                        if command -v bc &>/dev/null; then
                            cpu_temp=$(echo "$temp_millidegrees / 1000" | bc -l 2>/dev/null | xargs printf "%.1f" 2>/dev/null || echo "N/A")
                        else
                            cpu_temp="N/A"
                        fi
                        break
                    fi
                fi
            fi
        done
    fi
    
    # 3. Final fallback: ACPI thermal zone
    if [ "$cpu_temp" = "N/A" ] && [ -f "/sys/class/thermal/thermal_zone0/temp" ]; then
        local temp_millidegrees=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
        if [ -n "$temp_millidegrees" ] && [ "$temp_millidegrees" -gt 0 ]; then
            if command -v bc &>/dev/null; then
                cpu_temp=$(echo "$temp_millidegrees / 1000" | bc -l 2>/dev/null | xargs printf "%.1f" 2>/dev/null || echo "N/A")
            else
                cpu_temp="N/A"
            fi
        fi
    fi
    
    log "System resources: Memory: ${memory_usage_percent}% used (${memory_available} available), Disk: ${disk_usage_percent}% used, Load:${load_avg}, CPU temp: ${cpu_temp}°C"
    
    # Check for critical resource usage
    if command -v bc &>/dev/null && (( $(echo "$memory_usage_percent > 90" | bc -l 2>/dev/null || echo 0) )); then
        error "Critical memory usage: ${memory_usage_percent}%"
        return 1
    fi
    
    if (( disk_usage_percent > 90 )); then
        error "Critical disk usage: ${disk_usage_percent}%"
        return 1
    fi
    
    return 0
}

# Function to attempt SSH service recovery
recover_ssh_service() {
    log "Attempting SSH service recovery..."
    
    # Restart SSH service
    if sudo systemctl restart ssh.service; then
        success "SSH service restarted successfully"
        sleep 3
        
        # Verify it's working
        if check_ssh_service; then
            success "SSH service recovery successful"
            return 0
        else
            error "SSH service still not healthy after restart"
            return 1
        fi
    else
        error "Failed to restart SSH service"
        return 1
    fi
}

# Function to attempt network interface recovery
recover_network_connectivity() {
    log "Starting network connectivity recovery..."
    
    # Check if we're currently connected via Tailscale
    local via_tailscale=false
    local tailscale_connection_details=""
    if [ -n "${SSH_CLIENT:-}" ] && echo "${SSH_CLIENT}" | grep -q "100\."; then
        via_tailscale=true
        tailscale_connection_details="SSH_CLIENT: ${SSH_CLIENT}"
        warn "Current SSH connection appears to be via Tailscale ($tailscale_connection_details)"
    elif [ -n "${SSH_CONNECTION:-}" ] && echo "${SSH_CONNECTION}" | grep -q "100\."; then
        via_tailscale=true  
        tailscale_connection_details="SSH_CONNECTION: ${SSH_CONNECTION}"
        warn "Current SSH connection appears to be via Tailscale ($tailscale_connection_details)"
    fi
    
    # Get current Tailscale status
    local tailscale_healthy=false
    if systemctl is-active --quiet tailscaled && command -v tailscale >/dev/null 2>&1; then
        if tailscale status >/dev/null 2>&1; then
            tailscale_healthy=true
        fi
    fi
    
    # Check how long network has been failing
    local outage_duration=$(get_network_outage_duration)
    
    # Decision matrix based on connection method, Tailscale health, and outage duration
    # Conservative thresholds:
    # - < 5 minutes: Very conservative, only try safe operations
    # - 5-10 minutes: Careful, try NetworkManager restart
    # - 10-15 minutes: Moderate, try interface restart but preserve Tailscale if connected via it
    # - > 15 minutes: Aggressive recovery, network is clearly broken
    local recovery_mode="conservative"
    
    if [ "$outage_duration" -gt 900 ]; then  # > 15 minutes
        recovery_mode="aggressive"
    elif [ "$outage_duration" -gt 600 ]; then  # > 10 minutes
        recovery_mode="moderate"
    elif [ "$outage_duration" -gt 300 ]; then   # > 5 minutes
        recovery_mode="careful"
    else
        recovery_mode="conservative"
    fi
    
    log "Network outage duration: ${outage_duration}s, using '$recovery_mode' recovery mode"
    
    # SAFETY DECISION LOGIC - Use the recovery modes
    if [ "$via_tailscale" = true ] && [ "$tailscale_healthy" = true ]; then
        warn "SAFETY: Connected via Tailscale and Tailscale is healthy"
        warn "SAFETY: Remote access is working - using mode: $recovery_mode"
        
        case "$recovery_mode" in
            "conservative")
                warn "Conservative mode: Only DNS and gentle NetworkManager operations"
                
                # Flush DNS cache gently
                if command -v systemd-resolve >/dev/null 2>&1; then
                    log "Flushing DNS cache..."
                    sudo systemd-resolve --flush-caches 2>/dev/null || true
                fi
                
                # Very gentle NetworkManager nudge only
                if systemctl is-active --quiet NetworkManager; then
                    log "Sending gentle signal to NetworkManager..."
                    sudo pkill -HUP NetworkManager 2>/dev/null || true
                    sleep 5
                    
                    if check_network_connectivity; then
                        success "Network connectivity restored via conservative methods"
                        return 0
                    fi
                fi
                
                warn "Conservative recovery insufficient, network may need manual intervention"
                return 1
                ;;
                
            "careful")
                warn "Careful mode: DNS refresh and NetworkManager restart"
                
                # DNS flush
                if command -v systemd-resolve >/dev/null 2>&1; then
                    log "Flushing DNS cache..."
                    sudo systemd-resolve --flush-caches 2>/dev/null || true
                fi
                
                # Restart NetworkManager (should not affect Tailscale)
                if systemctl is-active --quiet NetworkManager; then
                    log "Restarting NetworkManager (safe with Tailscale)..."
                    if sudo systemctl restart NetworkManager; then
                        success "NetworkManager restarted"
                        sleep 10  # Give it time to reinitialize
                        
                        if check_network_connectivity; then
                            success "Network connectivity restored via NetworkManager restart"
                            return 0
                        fi
                    fi
                fi
                
                warn "Careful recovery methods exhausted"
                return 1
                ;;
                
            "moderate")
                warn "Moderate mode: NetworkManager + interface restart (RISK: May disrupt Tailscale briefly)"
                
                # First try NetworkManager restart
                if systemctl is-active --quiet NetworkManager; then
                    log "Restarting NetworkManager..."
                    if sudo systemctl restart NetworkManager; then
                        sleep 10
                        if check_network_connectivity; then
                            success "Network connectivity restored via NetworkManager restart"
                            return 0
                        fi
                    fi
                fi
                
                # Get primary interface
                local primary_interface=$(ip route | grep default | awk '{print $5}' | head -1)
                if [ -z "$primary_interface" ]; then
                    warn "Cannot determine primary network interface"
                    return 1
                fi
                
                warn "Attempting interface restart on $primary_interface (may briefly interrupt Tailscale)"
                
                # Interface restart with monitoring
                if sudo ip link set "$primary_interface" down && sleep 3 && sudo ip link set "$primary_interface" up; then
                    success "Interface restart completed, checking connectivity..."
                    sleep 15  # Give time for network to stabilize
                    
                    if check_network_connectivity; then
                        success "Network connectivity restored via interface restart"
                        return 0
                    fi
                fi
                
                warn "Moderate recovery methods failed"
                return 1
                ;;
                
            "aggressive")
                warn "Aggressive mode: Full network recovery (RISK: May disrupt remote access)"
                warn "Network has been failing for over 15 minutes - attempting full recovery"
                
                # Try all recovery methods in sequence
                local primary_interface=$(ip route | grep default | awk '{print $5}' | head -1)
                
                # NetworkManager restart
                if systemctl is-active --quiet NetworkManager && sudo systemctl restart NetworkManager; then
                    sleep 10
                    if check_network_connectivity; then
                        success "Network restored via NetworkManager restart"
                        return 0
                    fi
                fi
                
                # Interface reset
                if [ -n "$primary_interface" ]; then
                    log "Attempting interface reset..."
                    if sudo ip link set "$primary_interface" down && sleep 5 && sudo ip link set "$primary_interface" up; then
                        sleep 20
                        if check_network_connectivity; then
                            success "Network restored via interface reset"
                            return 0
                        fi
                    fi
                    
                    # DHCP renewal
                    log "Attempting DHCP renewal..."
                    if sudo dhclient -r "$primary_interface" 2>/dev/null && sudo dhclient "$primary_interface"; then
                        sleep 15
                        if check_network_connectivity; then
                            success "Network restored via DHCP renewal"
                            return 0
                        fi
                    fi
                fi
                
                error "All aggressive recovery attempts failed"
                return 1
                ;;
        esac
    else
        # Not connected via Tailscale OR Tailscale is not healthy
        # Can be more aggressive since we're not risking losing the only access method
        warn "Not connected via Tailscale or Tailscale unhealthy - can attempt full recovery"
        
        # Get the primary network interface
        local primary_interface=$(ip route | grep default | awk '{print $5}' | head -1)
        
        if [ -z "$primary_interface" ]; then
            error "Could not determine primary network interface"
            return 1
        fi
        
        log "Primary network interface detected: $primary_interface"
        
        # Try to restart NetworkManager first
        log "Attempting to restart NetworkManager..."
        if sudo systemctl restart NetworkManager.service; then
            success "NetworkManager restarted successfully"
            sleep 10
            
            if check_network_connectivity; then
                success "Network connectivity recovery successful via NetworkManager restart"
                return 0
            fi
        fi
        
        # Try interface down/up
        log "Attempting network interface reset: $primary_interface"
        if sudo ip link set "$primary_interface" down && sleep 2 && sudo ip link set "$primary_interface" up; then
            success "Network interface reset completed"
            sleep 15
            
            if check_network_connectivity; then
                success "Network connectivity recovery successful via interface reset"
                return 0
            fi
        fi
        
        # Try DHCP renewal
        log "Attempting DHCP renewal on $primary_interface..."
        if sudo dhclient -r "$primary_interface" && sudo dhclient "$primary_interface"; then
            success "DHCP renewal completed"
            sleep 10
            
            if check_network_connectivity; then
                success "Network connectivity recovery successful via DHCP renewal"
                return 0
            fi
        fi
        
        error "All network recovery attempts failed"
        return 1
    fi
}

# Function to attempt Tailscale recovery
recover_tailscale() {
    log "Attempting Tailscale recovery..."
    
    # Try to bring Tailscale up
    if sudo timeout $TAILSCALE_TIMEOUT tailscale up; then
        success "Tailscale brought up successfully"
        sleep 5
        
        # Verify connectivity
        if check_tailscale_connectivity; then
            success "Tailscale recovery successful"
            return 0
        else
            error "Tailscale still not healthy after recovery attempt"
            return 1
        fi
    else
        error "Failed to bring Tailscale up"
        
        # Try restarting the daemon
        log "Attempting to restart Tailscale daemon..."
        if sudo systemctl restart tailscaled.service; then
            success "Tailscale daemon restarted"
            sleep 10
            
            # Try to bring it up again
            if sudo timeout $TAILSCALE_TIMEOUT tailscale up; then
                success "Tailscale recovery successful after daemon restart"
                return 0
            fi
        fi
        
        return 1
    fi
}

# Function to check if we can test remote connectivity
test_remote_connectivity() {
    log "Testing remote connectivity capabilities..."
    
    # We can't easily test if remote SSH works from the local machine
    # But we can check if the service is ready for remote connections
    
    local ssh_config_ok=true
    local sshd_config="/etc/ssh/sshd_config"
    
    # Check SSH configuration
    if [ -f "$sshd_config" ]; then
        # Check if key authentication is enabled
        if grep -q "^PubkeyAuthentication yes" "$sshd_config" || ! grep -q "^PubkeyAuthentication no" "$sshd_config"; then
            log "SSH public key authentication: enabled"
        else
            warn "SSH public key authentication may be disabled"
            ssh_config_ok=false
        fi
        
        # Check if password authentication is properly disabled
        if grep -q "^PasswordAuthentication no" "$sshd_config"; then
            log "SSH password authentication: properly disabled"
        else
            warn "SSH password authentication may be enabled (security risk)"
        fi
        
        # Check if root login is disabled
        if grep -q "^PermitRootLogin no" "$sshd_config"; then
            log "SSH root login: properly disabled"
        else
            warn "SSH root login may be enabled (security risk)"
        fi
    else
        error "SSH configuration file not found: $sshd_config"
        ssh_config_ok=false
    fi
    
    return $([ "$ssh_config_ok" = true ] && echo 0 || echo 1)
}

# Function to calculate network outage duration in seconds
get_network_outage_duration() {
    local last_success=""
    
    # Get last network success time from state file
    if [ -f "$STATE_FILE" ]; then
        last_success=$(jq -r '.last_network_success // ""' "$STATE_FILE" 2>/dev/null || echo "")
    fi
    
    # If we have no record of last success, assume it's been a long time
    if [ -z "$last_success" ] || [ "$last_success" = "null" ]; then
        echo "999999"  # Very large number to indicate unknown duration
        return
    fi
    
    # Calculate seconds since last success
    local current_epoch=$(date +%s)
    local last_success_epoch=$(date -d "$last_success" +%s 2>/dev/null || echo "0")
    
    echo $((current_epoch - last_success_epoch))
}

# Function to save health state
save_health_state() {
    local overall_status="$1"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%6NZ")
    
    # Use default values if variables are not set
    local ssh_health="${ssh_healthy:-false}"
    local tailscale_health="${tailscale_healthy:-false}"
    local network_health="${network_healthy:-false}"
    local system_health="${system_healthy:-false}"
    local last_recovery="${last_recovery_attempt:-}"
    local consecutive="${consecutive_failures:-0}"

    cat > "$STATE_FILE" << EOF
{
  "timestamp": "$timestamp",
  "overall_status": "$overall_status",
  "ssh_healthy": $ssh_health,
  "tailscale_healthy": $tailscale_healthy,
  "network_healthy": $network_health,
  "system_healthy": $system_health,
  "last_recovery_attempt": "$last_recovery",
  "consecutive_failures": $consecutive,
  "last_network_success": "$(if [ "$network_health" = true ]; then echo "$timestamp"; else [ -f "$STATE_FILE" ] && jq -r '.last_network_success // ""' "$STATE_FILE" 2>/dev/null || echo ""; fi)"
}
EOF

    # Also save in health monitoring format
    cat > "$HEALTH_FILE" << EOF
{
  "timestamp": "$timestamp",
  "service_name": "remote_access",
  "status": "$overall_status",
  "checks": [
    {
      "name": "ssh_service",
      "status": "$([ "$ssh_health" = true ] && echo "healthy" || echo "unhealthy")",
      "timestamp": "$timestamp"
    },
    {
      "name": "tailscale_connectivity", 
      "status": "$([ "$tailscale_health" = true ] && echo "healthy" || echo "unhealthy")",
      "timestamp": "$timestamp"
    },
    {
      "name": "network_connectivity",
      "status": "$([ "$network_health" = true ] && echo "healthy" || echo "unhealthy")",
      "timestamp": "$timestamp"
    },
    {
      "name": "system_resources",
      "status": "$([ "$system_health" = true ] && echo "healthy" || echo "unhealthy")",
      "timestamp": "$timestamp"
    }
  ]
}
EOF
}

# Function to perform a complete health check
perform_health_check() {
    log "=== Starting Remote Access Health Check ==="
    
    # Use global variables so they can be accessed by save_health_state
    ssh_healthy=false
    tailscale_healthy=false
    network_healthy=false
    system_healthy=false
    local overall_healthy=false
    
    # Run all checks
    if check_ssh_service; then
        ssh_healthy=true
    fi
    
    if check_tailscale_connectivity; then
        tailscale_healthy=true
    fi
    
    if check_network_connectivity; then
        network_healthy=true
    fi
    
    if check_system_resources; then
        system_healthy=true
    fi
    
    # Test remote connectivity readiness
    test_remote_connectivity
    
    # Determine overall health
    if [ "$ssh_healthy" = true ] && [ "$tailscale_healthy" = true ] && [ "$network_healthy" = true ] && [ "$system_healthy" = true ]; then
        overall_healthy=true
        success "=== All remote access checks PASSED ==="
    else
        error "=== Some remote access checks FAILED ==="
    fi
    
    # Export for use in recovery functions
    export ssh_healthy tailscale_healthy network_healthy system_healthy
    
    return $([ "$overall_healthy" = true ] && echo 0 || echo 1)
}

# Function to perform recovery actions
perform_recovery() {
    log "=== Starting Recovery Actions ==="
    
    local recovery_success=false
    local recovery_attempted=false
    
    # Attempt SSH recovery if needed
    if [ "$ssh_healthy" != true ]; then
        recovery_attempted=true
        if recover_ssh_service; then
            ssh_healthy=true
        fi
    fi
    
    # Attempt Tailscale recovery if needed
    if [ "$tailscale_healthy" != true ]; then
        recovery_attempted=true
        if recover_tailscale; then
            tailscale_healthy=true
        fi
    fi
    
    # Network issues usually require manual intervention or reboot
    if [ "$network_healthy" != true ]; then
        recovery_attempted=true
        
        # CRITICAL SAFETY: Check if we're currently connected via Tailscale
        local current_ssh_via_tailscale=false
        if [ -n "${SSH_CLIENT:-}" ] && echo "${SSH_CLIENT}" | grep -q "100\."; then
            current_ssh_via_tailscale=true
            warn "SAFETY: Current SSH connection appears to be via Tailscale (${SSH_CLIENT})"
        elif [ -n "${SSH_CONNECTION:-}" ] && echo "${SSH_CONNECTION}" | grep -q "100\."; then
            current_ssh_via_tailscale=true  
            warn "SAFETY: Current SSH connection appears to be via Tailscale (${SSH_CONNECTION})"
        fi
        
        if [ "$current_ssh_via_tailscale" = true ] && [ "$tailscale_healthy" = true ]; then
            warn "SAFETY: Current session is via Tailscale and Tailscale is healthy"
            warn "SAFETY: Network recovery could break this connection - being very conservative"
        fi
        
        log "Attempting network connectivity recovery..."
        if recover_network_connectivity; then
            network_healthy=true
        else
            warn "Network connectivity recovery failed - may require manual intervention"
        fi
    fi
    
    # System resource issues may require service restarts or cleanup
    if [ "$system_healthy" != true ]; then
        warn "System resource issues detected - consider restarting services or cleaning up"
    fi
    
    if [ "$recovery_attempted" = true ]; then
        export last_recovery_attempt=$(date -u +"%Y-%m-%dT%H:%M:%S.%6NZ")
        
        # Re-check after recovery
        if perform_health_check; then
            recovery_success=true
            success "=== Recovery SUCCESSFUL ==="
        else
            error "=== Recovery FAILED - manual intervention may be required ==="
        fi
    fi
    
    return $([ "$recovery_success" = true ] && echo 0 || echo 1)
}

# Function to run continuous monitoring
run_monitor() {
    log "Starting remote access monitoring (interval: ${CHECK_INTERVAL}s)"
    log "Press Ctrl+C to stop"
    
    local consecutive_failures=0
    local last_recovery_attempt=""
    
    # Load previous state if available
    if [ -f "$STATE_FILE" ]; then
        consecutive_failures=$(jq -r '.consecutive_failures // 0' "$STATE_FILE" 2>/dev/null || echo 0)
        last_recovery_attempt=$(jq -r '.last_recovery_attempt // ""' "$STATE_FILE" 2>/dev/null || echo "")
    fi
    
    export consecutive_failures last_recovery_attempt
    
    while true; do
        if perform_health_check || false; then
            consecutive_failures=0
            save_health_state "healthy"
        else
            consecutive_failures=$((consecutive_failures + 1))
            warn "Health check failed (consecutive failures: $consecutive_failures)"
            
            # Attempt recovery after 2 consecutive failures
            if [ $consecutive_failures -ge 2 ]; then
                if perform_recovery || false; then
                    # Recovery succeeded, reset counter
                    consecutive_failures=0
                    log "Recovery successful, resetting failure counter"
                else
                    # Recovery failed, limit retry attempts to prevent spam
                    if [ $consecutive_failures -ge 5 ]; then
                        warn "Multiple recovery attempts failed, waiting longer before next attempt"
                        consecutive_failures=3  # Reset to 3 so next attempt is in 2 cycles
                    fi
                fi
            fi
            
            save_health_state "unhealthy"
        fi
        
        sleep $CHECK_INTERVAL
    done
}

# Function to install as systemd service
install_monitor_service() {
    log "Installing remote access monitor as systemd service..."
    
    local service_file="/etc/systemd/system/remote-access-monitor.service"
    
    sudo tee "$service_file" > /dev/null << EOF
[Unit]
Description=Experimance Remote Access Monitor
After=network.target ssh.service tailscaled.service
Wants=network.target ssh.service tailscaled.service

[Service]
Type=simple
User=root
ExecStart=$SCRIPT_DIR/remote_access_monitor.sh monitor
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable remote-access-monitor.service
    
    success "Remote access monitor service installed"
    log "Start with: sudo systemctl start remote-access-monitor.service"
    log "Monitor logs with: sudo journalctl -u remote-access-monitor.service -f"
}

# Function to show status
show_status() {
    log "=== Remote Access Status ==="
    
    perform_health_check
    
    if [ -f "$STATE_FILE" ]; then
        echo ""
        log "=== Previous State ==="
        jq . "$STATE_FILE" 2>/dev/null || cat "$STATE_FILE"
    fi
    
    echo ""
    log "=== Recent Logs ==="
    if [ -f "$LOG_FILE" ]; then
        tail -20 "$LOG_FILE"
    else
        log "No log file found"
    fi
}

# Main function
main() {
    case "${1:-check}" in
        "check")
            perform_health_check
            ;;
        "recover")
            perform_health_check || true  # Don't exit on health check failure
            if ! perform_recovery; then
                exit 1
            fi
            ;;
        "monitor")
            run_monitor
            ;;
        "install")
            install_monitor_service
            ;;
        "status")
            show_status
            ;;
        "help"|"--help"|"-h")
            echo "Remote Access Monitor and Recovery Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  check    - Run one-time health check (default)"
            echo "  recover  - Run health check and attempt recovery"
            echo "  monitor  - Run continuous monitoring"
            echo "  install  - Install as systemd service"
            echo "  status   - Show current status and recent logs"
            echo "  help     - Show this help message"
            echo ""
            echo "Log file: $LOG_FILE"
            echo "State file: $STATE_FILE"
            echo "Health file: $HEALTH_FILE"
            ;;
        *)
            error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
