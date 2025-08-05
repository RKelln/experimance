#!/bin/bash
# Preventive Maintenance Script for Gallery Installation
# Prevents common issues that lead to SSH lockouts

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
MAINTENANCE_LOG="/var/log/experimance/maintenance.log"
MAINTENANCE_STATE="/var/cache/experimance/maintenance-state.json"

# Maintenance thresholds
MAX_MEMORY_PERCENT=85
MAX_DISK_PERCENT=85
MAX_LOAD_MULTIPLIER=2
MAX_LOG_SIZE_MB=100
MAX_CACHE_SIZE_MB=500
MAX_TEMP_AGE_DAYS=7

# Ensure directories exist
sudo mkdir -p "$(dirname "$MAINTENANCE_LOG")" "$(dirname "$MAINTENANCE_STATE")"

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | sudo tee -a "$MAINTENANCE_LOG" > /dev/null
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1" | sudo tee -a "$MAINTENANCE_LOG" > /dev/null
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | sudo tee -a "$MAINTENANCE_LOG" > /dev/null
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1" | sudo tee -a "$MAINTENANCE_LOG" > /dev/null
}

# Function to clean up log files
cleanup_logs() {
    log "=== Log Cleanup ==="
    
    local space_freed=0
    local logs_cleaned=0
    
    # Clean up journal logs older than 30 days
    if sudo journalctl --vacuum-time=30d &>/dev/null; then
        success "Cleaned systemd journal logs older than 30 days"
        logs_cleaned=$((logs_cleaned + 1))
    fi
    
    # Clean up large Experimance log files
    find /var/log/experimance -name "*.log" -size +${MAX_LOG_SIZE_MB}M 2>/dev/null | while read -r logfile; do
        if [ -f "$logfile" ]; then
            local size_before=$(stat -c%s "$logfile")
            
            # Keep last 1000 lines
            tail -1000 "$logfile" > "${logfile}.tmp" && mv "${logfile}.tmp" "$logfile"
            
            local size_after=$(stat -c%s "$logfile")
            local freed=$((size_before - size_after))
            
            log "Truncated large log file: $logfile (freed $(( freed / 1024 / 1024 ))MB)"
            space_freed=$((space_freed + freed))
            logs_cleaned=$((logs_cleaned + 1))
        fi
    done
    
    # Clean up user log files
    if [ -d "$REPO_DIR/logs" ]; then
        find "$REPO_DIR/logs" -name "*.log" -size +${MAX_LOG_SIZE_MB}M 2>/dev/null | while read -r logfile; do
            if [ -f "$logfile" ]; then
                local size_before=$(stat -c%s "$logfile")
                
                # Keep last 1000 lines
                tail -1000 "$logfile" > "${logfile}.tmp" && mv "${logfile}.tmp" "$logfile"
                
                local size_after=$(stat -c%s "$logfile")
                local freed=$((size_before - size_after))
                
                log "Truncated user log file: $logfile (freed $(( freed / 1024 / 1024 ))MB)"
                space_freed=$((space_freed + freed))
                logs_cleaned=$((logs_cleaned + 1))
            fi
        done
    fi
    
    if [ $logs_cleaned -gt 0 ]; then
        success "Log cleanup complete: $logs_cleaned files cleaned, $(( space_freed / 1024 / 1024 ))MB freed"
    else
        log "No log cleanup needed"
    fi
}

# Function to clean up cache and temporary files
cleanup_cache() {
    log "=== Cache Cleanup ==="
    
    local space_freed=0
    local files_cleaned=0
    
    # Clean up system cache
    if [ -d "/tmp" ]; then
        find /tmp -type f -atime +$MAX_TEMP_AGE_DAYS -delete 2>/dev/null || true
        files_cleaned=$((files_cleaned + $(find /tmp -type f -atime +$MAX_TEMP_AGE_DAYS 2>/dev/null | wc -l)))
    fi
    
    # Clean up user cache directories
    if [ -d "$REPO_DIR/cache" ]; then
        local cache_size=$(du -sm "$REPO_DIR/cache" 2>/dev/null | awk '{print $1}' || echo 0)
        
        if [ "$cache_size" -gt $MAX_CACHE_SIZE_MB ]; then
            log "Cache directory is large (${cache_size}MB), cleaning old files..."
            
            # Remove files older than 7 days
            find "$REPO_DIR/cache" -type f -mtime +7 -delete 2>/dev/null || true
            
            local new_cache_size=$(du -sm "$REPO_DIR/cache" 2>/dev/null | awk '{print $1}' || echo 0)
            local freed=$((cache_size - new_cache_size))
            
            success "Cache cleanup freed ${freed}MB"
            space_freed=$((space_freed + freed * 1024 * 1024))
        fi
    fi
    
    # Clean up Python cache
    if [ -d "$REPO_DIR/.venv" ]; then
        find "$REPO_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        find "$REPO_DIR" -name "*.pyc" -delete 2>/dev/null || true
        log "Cleaned Python cache files"
    fi
    
    # Clean up old transcripts (keep last 50)
    if [ -d "$REPO_DIR/transcripts" ]; then
        local transcript_count=$(ls -1 "$REPO_DIR/transcripts"/*.jsonl 2>/dev/null | wc -l || echo 0)
        
        if [ "$transcript_count" -gt 50 ]; then
            log "Cleaning old transcripts (keeping last 50 of $transcript_count)"
            ls -1t "$REPO_DIR/transcripts"/*.jsonl | tail -n +51 | xargs rm -f
            success "Cleaned $((transcript_count - 50)) old transcript files"
        fi
    fi
    
    success "Cache cleanup complete"
}

# Function to monitor and manage memory usage
manage_memory() {
    log "=== Memory Management ==="
    
    local memory_usage_percent=$(free | grep Mem | awk '{printf "%.0f", ($3/$2) * 100.0}')
    local memory_available=$(free -m | awk 'NR==2{print $7}')
    
    log "Memory usage: ${memory_usage_percent}% (${memory_available}MB available)"
    
    if [ "$memory_usage_percent" -gt $MAX_MEMORY_PERCENT ]; then
        warn "High memory usage detected: ${memory_usage_percent}%"
        
        # Show top memory consumers
        log "Top memory-consuming processes:"
        ps -eo pid,ppid,cmd,%mem --sort=-%mem | head -10
        
        # Check for memory leaks in Experimance services
        local experimance_mem=$(ps -eo pid,cmd,%mem | grep -E "(experimance|uv run)" | grep -v grep | awk '{sum += $3} END {print sum}' || echo 0)
        log "Experimance services using: ${experimance_mem}% of total memory"
        
        if (( $(echo "$experimance_mem > 50" | bc -l) )); then
            warn "Experimance services using high memory (${experimance_mem}%)"
            log "Consider restarting services to free memory"
            
            # Optionally restart services if memory is critical
            if [ "$memory_usage_percent" -gt 95 ]; then
                error "Critical memory usage - considering service restart"
                # This would restart services - commented out for safety
                # sudo systemctl restart experimance@experimance.target
            fi
        fi
        
        # Drop caches if memory is very low
        if [ "$memory_available" -lt 200 ]; then
            log "Very low memory available, dropping caches..."
            sudo sync
            echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
            success "Dropped system caches"
        fi
    else
        success "Memory usage is acceptable: ${memory_usage_percent}%"
    fi
}

# Function to monitor system load
manage_system_load() {
    log "=== System Load Management ==="
    
    local load_1min=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
    local cpu_count=$(nproc)
    local load_threshold=$(echo "$cpu_count * $MAX_LOAD_MULTIPLIER" | bc)
    
    log "System load: $load_1min (threshold: $load_threshold, CPUs: $cpu_count)"
    
    if (( $(echo "$load_1min > $load_threshold" | bc -l) )); then
        warn "High system load detected: $load_1min"
        
        # Show top CPU consumers
        log "Top CPU-consuming processes:"
        ps -eo pid,ppid,cmd,%cpu --sort=-%cpu | head -10
        
        # Check for runaway processes
        local high_cpu_procs=$(ps -eo pid,cmd,%cpu --sort=-%cpu | awk '$3 > 50 {print $0}' | wc -l)
        if [ "$high_cpu_procs" -gt 0 ]; then
            warn "Found $high_cpu_procs processes using >50% CPU"
            ps -eo pid,cmd,%cpu --sort=-%cpu | awk '$3 > 50 {print $0}'
        fi
        
        return 1
    else
        success "System load is acceptable: $load_1min"
    fi
    
    return 0
}

# Function to check and manage disk space
manage_disk_space() {
    log "=== Disk Space Management ==="
    
    local critical_found=false
    
    # Check each mounted filesystem
    while IFS= read -r line; do
        local filesystem=$(echo "$line" | awk '{print $1}')
        local usage_percent=$(echo "$line" | awk '{print $(NF-1)}' | sed 's/%//')
        local mount_point=$(echo "$line" | awk '{print $NF}')
        
        log "Filesystem $filesystem at $mount_point: ${usage_percent}% used"
        
        if [ "$usage_percent" -gt $MAX_DISK_PERCENT ]; then
            warn "High disk usage on $mount_point: ${usage_percent}%"
            critical_found=true
            
            # Show largest directories
            log "Largest directories in $mount_point:"
            du -sh "$mount_point"/* 2>/dev/null | sort -hr | head -5 || true
        fi
    done < <(df | grep -E '^/dev/' | grep -v '/snap/')
    
    if [ "$critical_found" = false ]; then
        success "Disk usage is acceptable on all filesystems"
    else
        warn "Critical disk usage detected - consider cleanup"
        return 1
    fi
    
    return 0
}

# Function to check and restart failed services
manage_services() {
    log "=== Service Management ==="
    
    local services_restarted=0
    
    # Check critical services
    local critical_services=("ssh" "tailscaled")
    
    for service in "${critical_services[@]}"; do
        if ! systemctl is-active "$service.service" &>/dev/null; then
            warn "Critical service $service is not active"
            
            if sudo systemctl restart "$service.service"; then
                success "Restarted $service service"
                services_restarted=$((services_restarted + 1))
            else
                error "Failed to restart $service service"
            fi
        else
            log "Service $service is running normally"
        fi
    done
    
    # Check Experimance services
    local experimance_services=(
        "core@experimance"
        "display@experimance"
        "agent@experimance"
        "audio@experimance"
        "image_server@experimance"
        "health@experimance"
    )
    
    local failed_experimance=0
    for service in "${experimance_services[@]}"; do
        if systemctl is-failed "$service.service" &>/dev/null; then
            warn "Failed Experimance service: $service"
            failed_experimance=$((failed_experimance + 1))
        fi
    done
    
    if [ $failed_experimance -gt 0 ]; then
        warn "$failed_experimance Experimance services have failed"
        log "Consider restarting Experimance services if issues persist"
    else
        success "All Experimance services are healthy"
    fi
    
    if [ $services_restarted -gt 0 ]; then
        success "Restarted $services_restarted failed services"
    fi
}

# Function to optimize network settings
optimize_network() {
    log "=== Network Optimization ==="
    
    # Check network connectivity
    if ! ping -c 1 -W 5 8.8.8.8 &>/dev/null; then
        error "No internet connectivity - network may need attention"
        return 1
    fi
    
    # Check Tailscale connectivity
    if ! timeout 10 tailscale status &>/dev/null; then
        warn "Tailscale status check failed"
        
        # Try to bring Tailscale up
        if sudo timeout 15 tailscale up &>/dev/null; then
            success "Brought Tailscale back up"
        else
            warn "Failed to bring Tailscale up - may need manual intervention"
        fi
    else
        success "Tailscale connectivity is good"
    fi
    
    # Check for high network usage that could affect SSH
    local rx_dropped=$(cat /proc/net/dev | grep -E '(eth|wlan|enp|wlp)' | awk '{sum += $5} END {print sum+0}')
    local tx_dropped=$(cat /proc/net/dev | grep -E '(eth|wlan|enp|wlp)' | awk '{sum += $13} END {print sum+0}')
    
    if [ "$rx_dropped" -gt 1000 ] || [ "$tx_dropped" -gt 1000 ]; then
        warn "Network packet drops detected (RX: $rx_dropped, TX: $tx_dropped)"
    else
        log "Network packet drops are minimal (RX: $rx_dropped, TX: $tx_dropped)"
    fi
    
    return 0
}

# Function to save maintenance state
save_maintenance_state() {
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%6NZ")
    local status="$1"
    
    cat > "$MAINTENANCE_STATE" << EOF
{
  "timestamp": "$timestamp",
  "status": "$status",
  "last_maintenance": "$timestamp",
  "actions_taken": {
    "log_cleanup": $log_cleanup_done,
    "cache_cleanup": $cache_cleanup_done,
    "memory_management": $memory_management_done,
    "service_management": $service_management_done,
    "network_optimization": $network_optimization_done
  }
}
EOF
}

# Main maintenance function
run_maintenance() {
    log "=== Starting Preventive Maintenance ==="
    
    local issues_found=0
    local log_cleanup_done=false
    local cache_cleanup_done=false
    local memory_management_done=false
    local service_management_done=false
    local network_optimization_done=false
    
    # Run cleanup operations
    cleanup_logs
    log_cleanup_done=true
    
    cleanup_cache
    cache_cleanup_done=true
    
    # Check system health and manage issues
    manage_memory
    memory_management_done=true
    
    if ! manage_system_load; then
        issues_found=$((issues_found + 1))
    fi
    
    if ! manage_disk_space; then
        issues_found=$((issues_found + 1))
    fi
    
    manage_services
    service_management_done=true
    
    if ! optimize_network; then
        issues_found=$((issues_found + 1))
    fi
    network_optimization_done=true
    
    # Export variables for save_maintenance_state
    export log_cleanup_done cache_cleanup_done memory_management_done
    export service_management_done network_optimization_done
    
    # Summary
    log "=== Maintenance Summary ==="
    if [ $issues_found -eq 0 ]; then
        success "Maintenance completed successfully - no issues found"
        save_maintenance_state "success"
    else
        warn "Maintenance completed with $issues_found issue(s) requiring attention"
        save_maintenance_state "issues_found"
    fi
    
    log "Maintenance log: $MAINTENANCE_LOG"
    
    return $issues_found
}

# Function to install maintenance as a cron job
install_cron() {
    log "Installing preventive maintenance as cron job..."
    
    # Create a script that runs maintenance with logging
    local cron_script="/usr/local/bin/experimance-maintenance"
    
    sudo tee "$cron_script" > /dev/null << EOF
#!/bin/bash
# Experimance Preventive Maintenance Cron Job
cd "$REPO_DIR"
"$SCRIPT_DIR/preventive_maintenance.sh" run
EOF
    
    sudo chmod +x "$cron_script"
    
    # Add to root's crontab (runs every 6 hours)
    local cron_entry="0 */6 * * * $cron_script"
    
    if ! sudo crontab -l 2>/dev/null | grep -q "experimance-maintenance"; then
        (sudo crontab -l 2>/dev/null; echo "$cron_entry") | sudo crontab -
        success "Preventive maintenance cron job installed (every 6 hours)"
    else
        log "Preventive maintenance cron job already exists"
    fi
    
    log "View cron jobs with: sudo crontab -l"
    log "View maintenance logs with: tail -f $MAINTENANCE_LOG"
}

# Function to uninstall cron job
uninstall_cron() {
    log "Removing preventive maintenance cron job..."
    
    if sudo crontab -l 2>/dev/null | grep -q "experimance-maintenance"; then
        sudo crontab -l 2>/dev/null | grep -v "experimance-maintenance" | sudo crontab -
        success "Preventive maintenance cron job removed"
    else
        log "No preventive maintenance cron job found"
    fi
    
    # Remove the script
    if [ -f "/usr/local/bin/experimance-maintenance" ]; then
        sudo rm -f "/usr/local/bin/experimance-maintenance"
        log "Removed maintenance script"
    fi
}

# Main function
main() {
    case "${1:-run}" in
        "run")
            run_maintenance
            ;;
        "install-cron")
            install_cron
            ;;
        "uninstall-cron")
            uninstall_cron
            ;;
        "status")
            if [ -f "$MAINTENANCE_STATE" ]; then
                log "Last maintenance status:"
                cat "$MAINTENANCE_STATE" | jq . 2>/dev/null || cat "$MAINTENANCE_STATE"
            else
                log "No maintenance state file found"
            fi
            
            if [ -f "$MAINTENANCE_LOG" ]; then
                log "Recent maintenance log entries:"
                tail -20 "$MAINTENANCE_LOG"
            fi
            ;;
        "help"|"--help"|"-h")
            echo "Preventive Maintenance Script for Gallery Installation"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  run           - Run maintenance tasks (default)"
            echo "  install-cron  - Install as cron job (every 6 hours)"
            echo "  uninstall-cron- Remove cron job"
            echo "  status        - Show last maintenance status"
            echo "  help          - Show this help message"
            echo ""
            echo "Log file: $MAINTENANCE_LOG"
            echo "State file: $MAINTENANCE_STATE"
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
