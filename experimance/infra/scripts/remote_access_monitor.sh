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
    
    # CPU temperature (if available)
    if command -v sensors &>/dev/null; then
        cpu_temp=$(sensors 2>/dev/null | grep -E "Core|CPU" | awk '{print $3}' | head -1 | sed 's/+//;s/°C.*//' || echo "N/A")
    else
        cpu_temp="N/A"
    fi
    
    log "System resources: Memory: ${memory_usage_percent}% used (${memory_available} available), Disk: ${disk_usage_percent}% used, Load:${load_avg}, CPU temp: ${cpu_temp}°C"
    
    # Check for critical resource usage
    if (( $(echo "$memory_usage_percent > 90" | bc -l 2>/dev/null || echo 0) )); then
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
    log "Attempting network connectivity recovery..."
    
    # Get the primary network interface (excluding loopback and Tailscale)
    local primary_interface=$(ip route | grep default | awk '{print $5}' | head -1)
    
    if [ -z "$primary_interface" ]; then
        error "Could not determine primary network interface"
        return 1
    fi
    
    log "Primary network interface detected: $primary_interface"
    
    # Try to restart NetworkManager first (gentler approach)
    log "Attempting to restart NetworkManager..."
    if sudo systemctl restart NetworkManager.service; then
        success "NetworkManager restarted successfully"
        sleep 10  # Give it time to reconnect
        
        # Check if this fixed the issue
        if check_network_connectivity; then
            success "Network connectivity recovery successful via NetworkManager restart"
            return 0
        else
            log "NetworkManager restart didn't resolve connectivity issues, trying interface reset..."
        fi
    else
        warn "Failed to restart NetworkManager, trying interface reset..."
    fi
    
    # Try interface down/up as fallback
    log "Attempting network interface reset: $primary_interface"
    
    # Bring interface down and up
    if sudo ip link set "$primary_interface" down && sleep 2 && sudo ip link set "$primary_interface" up; then
        success "Network interface reset completed"
        sleep 15  # Give it time to get DHCP lease
        
        # Check if this fixed the issue
        if check_network_connectivity; then
            success "Network connectivity recovery successful via interface reset"
            return 0
        else
            log "Interface reset didn't resolve connectivity, trying DHCP renewal..."
        fi
    else
        error "Failed to reset network interface"
    fi
    
    # Try DHCP renewal as last resort
    log "Attempting DHCP renewal on $primary_interface..."
    if sudo dhclient -r "$primary_interface" && sudo dhclient "$primary_interface"; then
        success "DHCP renewal completed"
        sleep 10
        
        # Final check
        if check_network_connectivity; then
            success "Network connectivity recovery successful via DHCP renewal"
            return 0
        fi
    else
        warn "DHCP renewal failed"
    fi
    
    error "All network recovery attempts failed"
    return 1
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
  "tailscale_healthy": $tailscale_health,
  "network_healthy": $network_health,
  "system_healthy": $system_health,
  "last_recovery_attempt": "$last_recovery",
  "consecutive_failures": $consecutive
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
                perform_recovery || true  # Don't exit on recovery failure
                consecutive_failures=0  # Reset after recovery attempt
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
