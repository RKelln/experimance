#!/bin/bash

# Experimance Quick Status Script
# Shows a simple overview of all services

set -euo pipefail

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT="${1:-experimance}"
SERVICES=(
    "experimance-core@${PROJECT}"
    "experimance-display@${PROJECT}"
    "image-server@${PROJECT}"
    "experimance-agent@${PROJECT}"
    "experimance-audio@${PROJECT}"
)

echo -e "${BLUE}=== Experimance Status (${PROJECT}) ===${NC}"
echo -e "$(date '+%Y-%m-%d %H:%M:%S')\n"

# Check services
echo -e "${BLUE}Services:${NC}"
for service in "${SERVICES[@]}"; do
    if systemctl is-active "$service" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $service"
    else
        echo -e "  ${RED}✗${NC} $service ($(systemctl is-active "$service" 2>/dev/null || echo "unknown"))"
    fi
done

# Check target
echo -e "\n${BLUE}Target:${NC}"
target="experimance@${PROJECT}.target"
if systemctl is-active "$target" &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} $target"
else
    echo -e "  ${RED}✗${NC} $target ($(systemctl is-active "$target" 2>/dev/null || echo "unknown"))"
fi

# System resources
echo -e "\n${BLUE}System Resources:${NC}"

# Memory
if command -v free &>/dev/null; then
    mem_info=$(free -h | grep "Mem:")
    mem_used=$(echo "$mem_info" | awk '{print $3}')
    mem_total=$(echo "$mem_info" | awk '{print $2}')
    echo -e "  Memory: ${mem_used}/${mem_total}"
fi

# Disk
if command -v df &>/dev/null; then
    disk_info=$(df -h / | tail -1)
    disk_used=$(echo "$disk_info" | awk '{print $3}')
    disk_total=$(echo "$disk_info" | awk '{print $2}')
    disk_percent=$(echo "$disk_info" | awk '{print $5}')
    echo -e "  Disk: ${disk_used}/${disk_total} (${disk_percent})"
fi

# Load average
if [[ -f /proc/loadavg ]]; then
    load_avg=$(cut -d' ' -f1 /proc/loadavg)
    echo -e "  Load: ${load_avg}"
fi

# GPU info (if nvidia-smi is available)
if command -v nvidia-smi &>/dev/null; then
    echo -e "\n${BLUE}GPU:${NC}"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r name mem_used mem_total gpu_util; do
        echo -e "  $name: ${mem_used}MB/${mem_total}MB, ${gpu_util}% util"
    done
fi

# Recent errors
echo -e "\n${BLUE}Recent Errors (last 1 hour):${NC}"
error_count=0
for service in "${SERVICES[@]}"; do
    errors=$(journalctl -u "$service" --since "1 hour ago" --grep "ERROR|CRITICAL|Failed" --no-pager -q 2>/dev/null | wc -l)
    if [[ $errors -gt 0 ]]; then
        echo -e "  ${RED}✗${NC} $service: $errors errors"
        ((error_count++))
    fi
done

if [[ $error_count -eq 0 ]]; then
    echo -e "  ${GREEN}✓${NC} No errors found"
fi

# Port check
echo -e "\n${BLUE}ZMQ Ports:${NC}"
for port in 5555 5556 5557 5558; do
    if netstat -ln 2>/dev/null | grep -q ":$port "; then
        echo -e "  ${GREEN}✓${NC} Port $port: listening"
    else
        echo -e "  ${RED}✗${NC} Port $port: not listening"
    fi
done

echo ""
