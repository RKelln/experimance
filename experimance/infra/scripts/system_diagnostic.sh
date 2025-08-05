#!/bin/bash
# System Diagnostic Script for Gallery Installation
# Identifies common issues that can cause SSH lockouts

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
DIAGNOSTIC_LOG="/var/log/experimance/system-diagnostic.log"

# Ensure log directory exists
sudo mkdir -p "$(dirname "$DIAGNOSTIC_LOG")"

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | sudo tee -a "$DIAGNOSTIC_LOG" > /dev/null
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1" | sudo tee -a "$DIAGNOSTIC_LOG" > /dev/null
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | sudo tee -a "$DIAGNOSTIC_LOG" > /dev/null
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] OK:${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] OK: $1" | sudo tee -a "$DIAGNOSTIC_LOG" > /dev/null
}

# Function to check system load and processes that could cause hangs
check_system_load() {
    log "=== System Load Analysis ==="
    
    # Get load averages
    local load_avg=$(uptime | awk -F'load average:' '{print $2}')
    log "Load averages:$load_avg"
    
    # Get CPU count for context
    local cpu_count=$(nproc)
    local load_1min=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
    
    log "CPU cores: $cpu_count"
    
    # Check if load is excessive
    if (( $(echo "$load_1min > $cpu_count * 2" | bc -l) )); then
        error "High system load detected: $load_1min (threshold: $((cpu_count * 2)))"
        
        # Show top CPU consumers
        log "Top CPU-consuming processes:"
        ps -eo pid,ppid,cmd,%cpu --sort=-%cpu | head -10
        
        return 1
    else
        success "System load is normal: $load_1min"
    fi
    
    return 0
}

# Function to check memory usage and potential OOM conditions
check_memory_usage() {
    log "=== Memory Usage Analysis ==="
    
    # Get memory info
    local total_mem=$(free -m | awk 'NR==2{print $2}')
    local used_mem=$(free -m | awk 'NR==2{print $3}')
    local available_mem=$(free -m | awk 'NR==2{print $7}')
    local memory_usage_percent=$(echo "scale=1; $used_mem * 100 / $total_mem" | bc)
    
    log "Memory: ${used_mem}MB used / ${total_mem}MB total (${memory_usage_percent}%)"
    log "Available memory: ${available_mem}MB"
    
    # Check for memory pressure
    if (( $(echo "$memory_usage_percent > 90" | bc -l) )); then
        error "Critical memory usage: ${memory_usage_percent}%"
        
        # Show memory hogs
        log "Top memory-consuming processes:"
        ps -eo pid,ppid,cmd,%mem --sort=-%mem | head -10
        
        # Check for OOM killer activity
        log "Checking for OOM killer activity in last 24 hours:"
        if dmesg -T | grep -i "killed process" | tail -5; then
            error "OOM killer has been active recently"
        else
            log "No recent OOM killer activity found"
        fi
        
        return 1
    else
        success "Memory usage is acceptable: ${memory_usage_percent}%"
    fi
    
    return 0
}

# Function to check disk usage and I/O
check_disk_health() {
    log "=== Disk Health Analysis ==="
    
    # Check disk usage
    log "Disk usage:"
    df -h | grep -E '^/dev/'
    
    # Check for critical disk usage
    local critical_partitions=()
    while IFS= read -r line; do
        local usage=$(echo "$line" | awk '{print $(NF-1)}' | sed 's/%//')
        local mount=$(echo "$line" | awk '{print $NF}')
        
        if [ "$usage" -gt 90 ]; then
            critical_partitions+=("$mount ($usage%)")
        fi
    done < <(df | grep -E '^/dev/' | grep -v '/snap/')
    
    if [ ${#critical_partitions[@]} -gt 0 ]; then
        error "Critical disk usage detected: ${critical_partitions[*]}"
        return 1
    else
        success "Disk usage is acceptable"
    fi
    
    # Check I/O wait
    local iowait=$(iostat -c 1 2 | tail -1 | awk '{print $4}' 2>/dev/null || echo "0")
    log "I/O wait: ${iowait}%"
    
    if (( $(echo "$iowait > 20" | bc -l) )); then
        warn "High I/O wait detected: ${iowait}%"
        
        # Show I/O activity
        if command -v iotop &>/dev/null; then
            log "Top I/O processes (snapshot):"
            timeout 3 iotop -b -n 1 | head -15
        fi
    fi
    
    return 0
}

# Function to check network connectivity issues
check_network_issues() {
    log "=== Network Connectivity Analysis ==="
    
    # Check network interfaces
    log "Network interfaces:"
    ip addr show | grep -E '^[0-9]+:|inet '
    
    # Check routing
    log "Default routes:"
    ip route show default
    
    # Check if network is saturated
    log "Network statistics:"
    cat /proc/net/dev | grep -E '(eth|wlan|enp|wlp)' | while read line; do
        local interface=$(echo "$line" | awk -F: '{print $1}' | tr -d ' ')
        local rx_bytes=$(echo "$line" | awk '{print $2}')
        local tx_bytes=$(echo "$line" | awk '{print $10}')
        
        log "Interface $interface: RX $(( rx_bytes / 1024 / 1024 ))MB, TX $(( tx_bytes / 1024 / 1024 ))MB"
    done
    
    # Test basic connectivity
    local connectivity_ok=true
    
    if ! ping -c 1 -W 5 8.8.8.8 &>/dev/null; then
        error "Cannot reach 8.8.8.8 (Google DNS)"
        connectivity_ok=false
    fi
    
    if ! nslookup google.com &>/dev/null; then
        error "DNS resolution failed"
        connectivity_ok=false
    fi
    
    if [ "$connectivity_ok" = true ]; then
        success "Basic network connectivity is working"
    else
        error "Network connectivity issues detected"
        return 1
    fi
    
    return 0
}

# Function to check SSH-specific issues
check_ssh_issues() {
    log "=== SSH Service Analysis ==="
    
    # Check SSH service status
    if systemctl is-active ssh.service &>/dev/null; then
        success "SSH service is active"
    else
        error "SSH service is not active"
        systemctl status ssh.service --no-pager -l
        return 1
    fi
    
    # Check if SSH is listening
    if ss -tlnp | grep -q ":22 "; then
        success "SSH is listening on port 22"
    else
        error "SSH is not listening on port 22"
        return 1
    fi
    
    # Check SSH configuration for common issues
    local sshd_config="/etc/ssh/sshd_config"
    if [ -f "$sshd_config" ]; then
        log "SSH configuration analysis:"
        
        # Check max sessions
        local max_sessions=$(grep "^MaxSessions" "$sshd_config" | awk '{print $2}' || echo "10")
        log "MaxSessions: $max_sessions"
        
        # Check max startups
        local max_startups=$(grep "^MaxStartups" "$sshd_config" | awk '{print $2}' || echo "10:30:100")
        log "MaxStartups: $max_startups"
        
        # Check authentication methods
        if grep -q "^PubkeyAuthentication yes" "$sshd_config"; then
            success "Public key authentication is enabled"
        else
            warn "Public key authentication may not be explicitly enabled"
        fi
        
        if grep -q "^PasswordAuthentication no" "$sshd_config"; then
            success "Password authentication is disabled"
        else
            warn "Password authentication may be enabled"
        fi
    fi
    
    # Check current SSH connections
    local ssh_connections=$(ss -tn | grep ":22 " | grep ESTAB | wc -l)
    log "Current SSH connections: $ssh_connections"
    
    if [ "$ssh_connections" -gt 0 ]; then
        log "Active SSH connections:"
        ss -tn | grep ":22 " | grep ESTAB
    fi
    
    return 0
}

# Function to check Tailscale-specific issues
check_tailscale_issues() {
    log "=== Tailscale Analysis ==="
    
    # Check Tailscale daemon
    if systemctl is-active tailscaled.service &>/dev/null; then
        success "Tailscale daemon is active"
    else
        error "Tailscale daemon is not active"
        systemctl status tailscaled.service --no-pager -l
        return 1
    fi
    
    # Check Tailscale status
    log "Tailscale status:"
    tailscale status || warn "Failed to get Tailscale status"
    
    # Check Tailscale IP
    local tailscale_ip
    if tailscale_ip=$(tailscale ip -4 2>/dev/null); then
        success "Tailscale IP: $tailscale_ip"
    else
        error "Failed to get Tailscale IP"
    fi
    
    # Check DERP connectivity
    log "Checking DERP connectivity..."
    if tailscale netcheck; then
        success "DERP connectivity check passed"
    else
        warn "DERP connectivity issues detected"
    fi
    
    return 0
}

# Function to check for processes that commonly cause hangs
check_problematic_processes() {
    log "=== Problematic Process Analysis ==="
    
    # Check for zombie processes
    local zombie_count=$(ps aux | awk '$8 ~ /Z/ { print $2 }' | wc -l)
    if [ "$zombie_count" -gt 0 ]; then
        warn "Zombie processes detected: $zombie_count"
        ps aux | awk '$8 ~ /Z/ { print $0 }'
    else
        success "No zombie processes found"
    fi
    
    # Check for processes in uninterruptible sleep (D state)
    local blocked_count=$(ps aux | awk '$8 ~ /D/ { print $2 }' | wc -l)
    if [ "$blocked_count" -gt 0 ]; then
        warn "Processes in uninterruptible sleep (D state): $blocked_count"
        ps aux | awk '$8 ~ /D/ { print $0 }'
        log "These processes may indicate I/O issues or kernel problems"
    else
        success "No processes stuck in uninterruptible sleep"
    fi
    
    # Check for high CPU processes
    log "Top CPU consumers:"
    ps -eo pid,ppid,cmd,%cpu,%mem --sort=-%cpu | head -10
    
    # Check for runaway Experimance processes
    local experimance_procs=$(ps aux | grep -E "(experimance|uv run)" | grep -v grep | wc -l)
    log "Experimance-related processes: $experimance_procs"
    
    if [ "$experimance_procs" -gt 20 ]; then
        warn "High number of Experimance processes detected"
        ps aux | grep -E "(experimance|uv run)" | grep -v grep
    fi
    
    return 0
}

# Function to check system logs for warning signs
check_system_logs() {
    log "=== System Log Analysis ==="
    
    # Check for kernel panics or oops
    log "Checking for kernel issues in last 24 hours:"
    if journalctl --since "24 hours ago" | grep -i "panic\|oops\|segfault" | head -5; then
        error "Kernel issues detected in logs"
    else
        success "No kernel panics or oops found"
    fi
    
    # Check for systemd service failures
    log "Recent systemd service failures:"
    systemctl --failed --no-pager || success "No failed services"
    
    # Check for SSH-related errors
    log "SSH-related errors in last 24 hours:"
    if journalctl -u ssh.service --since "24 hours ago" | grep -i "error\|failed\|refused"; then
        warn "SSH errors detected in logs"
    else
        success "No SSH errors in recent logs"
    fi
    
    # Check for Tailscale errors
    log "Tailscale errors in last 24 hours:"
    if journalctl -u tailscaled.service --since "24 hours ago" | grep -i "error\|failed" | head -5; then
        warn "Tailscale errors detected in logs"
    else
        success "No Tailscale errors in recent logs"
    fi
    
    # Check for out-of-memory conditions
    log "OOM killer activity in last 24 hours:"
    if dmesg -T | grep -i "killed process" | head -5; then
        error "OOM killer has been active"
    else
        success "No OOM killer activity"
    fi
    
    return 0
}

# Function to generate recommendations
generate_recommendations() {
    log "=== Recommendations ==="
    
    # Based on the checks, provide specific recommendations
    log "To prevent SSH lockouts, consider these recommendations:"
    
    echo "1. Monitor system resources regularly:"
    echo "   - Set up automated monitoring for CPU, memory, and disk usage"
    echo "   - Configure alerts when thresholds are exceeded"
    echo ""
    
    echo "2. Implement automatic recovery mechanisms:"
    echo "   - Install the remote access monitor service"
    echo "   - Set up health checks that restart services when needed"
    echo ""
    
    echo "3. Configure emergency access:"
    echo "   - Set up a secondary network connection (mobile hotspot)"
    echo "   - Configure wake-on-LAN if supported"
    echo "   - Ensure physical access is available when needed"
    echo ""
    
    echo "4. Optimize SSH configuration:"
    echo "   - Set reasonable MaxSessions and MaxStartups limits"
    echo "   - Enable compression to reduce bandwidth usage"
    echo "   - Consider using SSH multiplexing for multiple connections"
    echo ""
    
    echo "5. Monitor Experimance services:"
    echo "   - Set up the health monitoring service"
    echo "   - Configure automatic service restarts on failure"
    echo "   - Monitor resource usage of individual services"
    echo ""
    
    echo "6. Regular maintenance:"
    echo "   - Schedule regular reboots during off-hours"
    echo "   - Clean up log files and temporary data"
    echo "   - Update system packages regularly"
    echo ""
    
    log "Use the remote access monitor script to implement automated monitoring:"
    log "  sudo $SCRIPT_DIR/remote_access_monitor.sh install"
    log "  sudo systemctl start remote-access-monitor.service"
}

# Main diagnostic function
run_full_diagnostic() {
    log "=== Starting Full System Diagnostic ==="
    log "Diagnostic will identify potential causes of SSH lockouts"
    
    local issues_found=0
    
    # Run all checks
    if ! check_system_load; then
        issues_found=$((issues_found + 1))
    fi
    
    if ! check_memory_usage; then
        issues_found=$((issues_found + 1))
    fi
    
    if ! check_disk_health; then
        issues_found=$((issues_found + 1))
    fi
    
    if ! check_network_issues; then
        issues_found=$((issues_found + 1))
    fi
    
    if ! check_ssh_issues; then
        issues_found=$((issues_found + 1))
    fi
    
    if ! check_tailscale_issues; then
        issues_found=$((issues_found + 1))
    fi
    
    check_problematic_processes
    check_system_logs
    
    # Summary
    log "=== Diagnostic Summary ==="
    if [ "$issues_found" -eq 0 ]; then
        success "No critical issues detected"
        log "System appears healthy for remote access"
    else
        error "Found $issues_found critical issue(s) that could cause SSH lockouts"
        log "Review the issues above and take corrective action"
    fi
    
    generate_recommendations
    
    log "=== Diagnostic Complete ==="
    log "Full log saved to: $DIAGNOSTIC_LOG"
    
    return $issues_found
}

# Quick check function
run_quick_check() {
    log "=== Quick Health Check ==="
    
    local issues_found=0
    
    # Check only critical items
    if ! systemctl is-active ssh.service &>/dev/null; then
        error "SSH service is not running"
        issues_found=$((issues_found + 1))
    fi
    
    if ! systemctl is-active tailscaled.service &>/dev/null; then
        error "Tailscale daemon is not running"
        issues_found=$((issues_found + 1))
    fi
    
    if ! ping -c 1 -W 5 8.8.8.8 &>/dev/null; then
        error "No internet connectivity"
        issues_found=$((issues_found + 1))
    fi
    
    # Check system load
    local load_1min=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
    local cpu_count=$(nproc)
    if (( $(echo "$load_1min > $cpu_count * 3" | bc -l) )); then
        error "System load is very high: $load_1min"
        issues_found=$((issues_found + 1))
    fi
    
    # Check memory
    local memory_usage_percent=$(free | grep Mem | awk '{printf "%.0f", ($3/$2) * 100.0}')
    if [ "$memory_usage_percent" -gt 95 ]; then
        error "Memory usage is critical: ${memory_usage_percent}%"
        issues_found=$((issues_found + 1))
    fi
    
    if [ "$issues_found" -eq 0 ]; then
        success "Quick check passed - system appears healthy"
    else
        error "Quick check found $issues_found critical issue(s)"
        log "Run full diagnostic for detailed analysis: $0 full"
    fi
    
    return $issues_found
}

# Main function
main() {
    case "${1:-quick}" in
        "full")
            run_full_diagnostic
            ;;
        "quick")
            run_quick_check
            ;;
        "help"|"--help"|"-h")
            echo "System Diagnostic Script for Gallery Installation"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  quick - Run quick health check (default)"
            echo "  full  - Run comprehensive diagnostic"
            echo "  help  - Show this help message"
            echo ""
            echo "Log file: $DIAGNOSTIC_LOG"
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
