#!/bin/bash
# Safe SSH Security Hardening Script
# Tests SSH configuration safely without locking out remote users

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKUP_DIR="/var/backups/experimance/ssh"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

# Function to create backup
create_backup() {
    log "Creating backup of SSH configuration..."
    
    sudo mkdir -p "$BACKUP_DIR"
    sudo cp /etc/ssh/sshd_config "$BACKUP_DIR/sshd_config_${TIMESTAMP}.backup"
    
    success "Backup created: $BACKUP_DIR/sshd_config_${TIMESTAMP}.backup"
}

# Function to check if we're connected via SSH
check_ssh_connection() {
    if [ -n "${SSH_CLIENT:-}" ] || [ -n "${SSH_TTY:-}" ]; then
        return 0  # We are connected via SSH
    else
        return 1  # Not connected via SSH
    fi
}

# Function to verify SSH keys are set up
verify_ssh_keys() {
    log "Verifying SSH key setup for experimance user..."
    
    if [ ! -f "/home/experimance/.ssh/authorized_keys" ]; then
        error "No authorized_keys file found for experimance user!"
        error "You must set up SSH keys before disabling password authentication"
        return 1
    fi
    
    local key_count=$(wc -l < /home/experimance/.ssh/authorized_keys)
    log "Found $key_count authorized key(s) for experimance user"
    
    if [ "$key_count" -eq 0 ]; then
        error "authorized_keys file is empty!"
        return 1
    fi
    
    # Check permissions
    local perms=$(stat -c "%a" /home/experimance/.ssh/authorized_keys)
    if [ "$perms" != "600" ]; then
        warn "authorized_keys permissions are $perms, should be 600"
        sudo chmod 600 /home/experimance/.ssh/authorized_keys
        success "Fixed authorized_keys permissions"
    fi
    
    success "SSH keys are properly configured"
    return 0
}

# Function to test SSH configuration
test_ssh_config() {
    log "Testing SSH configuration syntax..."
    
    if sudo sshd -t; then
        success "SSH configuration syntax is valid"
        return 0
    else
        error "SSH configuration has syntax errors!"
        return 1
    fi
}

# Function to create a safe test configuration
create_test_config() {
    log "Creating test SSH configuration..."
    
    # Create a test config that's more secure but safe
    sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.test
    
    # Apply security settings to test config
    sudo sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config.test
    sudo sed -i 's/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config.test
    sudo sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config.test
    sudo sed -i 's/^#\?ChallengeResponseAuthentication.*/ChallengeResponseAuthentication no/' /etc/ssh/sshd_config.test
    
    # Add additional security settings if not present
    if ! grep -q "^MaxAuthTries" /etc/ssh/sshd_config.test; then
        echo "MaxAuthTries 3" | sudo tee -a /etc/ssh/sshd_config.test > /dev/null
    fi
    
    if ! grep -q "^LoginGraceTime" /etc/ssh/sshd_config.test; then
        echo "LoginGraceTime 30" | sudo tee -a /etc/ssh/sshd_config.test > /dev/null
    fi
    
    if ! grep -q "^MaxSessions" /etc/ssh/sshd_config.test; then
        echo "MaxSessions 3" | sudo tee -a /etc/ssh/sshd_config.test > /dev/null
    fi
    
    success "Test configuration created"
}

# Function to test the configuration on alternate port
test_on_alternate_port() {
    log "Testing SSH on alternate port 2222..."
    
    # Start SSH on port 2222 with test config
    sudo /usr/sbin/sshd -f /etc/ssh/sshd_config.test -p 2222 -D &
    local test_pid=$!
    
    sleep 2
    
    # Test if it's listening
    if ss -tlnp | grep -q ":2222 "; then
        success "Test SSH daemon is running on port 2222"
        
        # Try to connect (this will only work if you have keys set up)
        log "You can now test the connection in another terminal:"
        log "  ssh -p 2222 experimance@localhost"
        log "  ssh -p 2222 experimance@$(hostname -I | awk '{print $1}')"
        
        read -p "Press Enter when you've successfully tested the connection on port 2222..."
        
        # Stop test daemon
        sudo kill $test_pid 2>/dev/null || true
        
        return 0
    else
        error "Test SSH daemon failed to start"
        sudo kill $test_pid 2>/dev/null || true
        return 1
    fi
}

# Function to apply the secure configuration safely
apply_secure_config() {
    log "Applying secure SSH configuration..."
    
    # Copy the tested config over the main config
    sudo cp /etc/ssh/sshd_config.test /etc/ssh/sshd_config
    
    # Test the main config one more time
    if ! test_ssh_config; then
        error "Configuration test failed! Restoring backup..."
        sudo cp "$BACKUP_DIR/sshd_config_${TIMESTAMP}.backup" /etc/ssh/sshd_config
        return 1
    fi
    
    # Reload SSH configuration (safer than restart)
    log "Reloading SSH configuration (not restarting)..."
    if sudo systemctl reload ssh; then
        success "SSH configuration reloaded successfully"
        
        # Verify the service is still running
        if systemctl is-active ssh &>/dev/null; then
            success "SSH service is still active"
            return 0
        else
            error "SSH service is not active after reload!"
            return 1
        fi
    else
        error "Failed to reload SSH configuration!"
        return 1
    fi
}

# Function to show current SSH configuration
show_current_config() {
    log "Current SSH configuration:"
    echo "----------------------------------------"
    
    # Show relevant security settings
    grep -E "^#?PasswordAuthentication|^#?PubkeyAuthentication|^#?PermitRootLogin|^#?ChallengeResponseAuthentication" /etc/ssh/sshd_config | while read line; do
        if [[ $line == \#* ]]; then
            echo -e "${YELLOW}$line${NC} (commented - using default)"
        else
            echo -e "${GREEN}$line${NC}"
        fi
    done
    
    echo "----------------------------------------"
}

# Function to show what will be changed
show_proposed_changes() {
    log "Proposed security changes:"
    echo "----------------------------------------"
    echo -e "${GREEN}PasswordAuthentication no${NC} (disable password login)"
    echo -e "${GREEN}PubkeyAuthentication yes${NC} (enable key-based login)"
    echo -e "${GREEN}PermitRootLogin no${NC} (disable root login)"
    echo -e "${GREEN}ChallengeResponseAuthentication no${NC} (disable challenge-response)"
    echo -e "${GREEN}MaxAuthTries 3${NC} (limit authentication attempts)"
    echo -e "${GREEN}LoginGraceTime 30${NC} (30 second login timeout)"
    echo -e "${GREEN}MaxSessions 3${NC} (limit concurrent sessions)"
    echo "----------------------------------------"
}

# Main function
main() {
    case "${1:-status}" in
        "status")
            show_current_config
            ;;
        "secure")
            log "=== SSH Security Hardening (Safe Mode) ==="
            
            if check_ssh_connection; then
                log "Detected SSH connection - using safe procedure"
            else
                log "Local access detected - proceeding normally"
            fi
            
            show_current_config
            show_proposed_changes
            
            if ! verify_ssh_keys; then
                error "SSH keys are not properly configured. Set up keys first!"
                exit 1
            fi
            
            read -p "Continue with SSH hardening? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log "Aborted by user"
                exit 0
            fi
            
            create_backup
            create_test_config
            
            if ! test_ssh_config; then
                exit 1
            fi
            
            if check_ssh_connection; then
                log "Testing configuration on alternate port first..."
                if ! test_on_alternate_port; then
                    error "Test on alternate port failed!"
                    exit 1
                fi
            fi
            
            if apply_secure_config; then
                success "SSH has been successfully hardened!"
                log "Password authentication is now disabled"
                log "Only key-based authentication is allowed"
                
                # Clean up test config
                sudo rm -f /etc/ssh/sshd_config.test
            else
                error "Failed to apply secure configuration"
                exit 1
            fi
            ;;
        "restore")
            if [ -z "${2:-}" ]; then
                log "Available backups:"
                ls -la "$BACKUP_DIR"/sshd_config_*.backup 2>/dev/null || log "No backups found"
                echo
                log "Usage: $0 restore <backup_file>"
                exit 1
            fi
            
            local backup_file="$2"
            if [ ! -f "$backup_file" ]; then
                backup_file="$BACKUP_DIR/$backup_file"
            fi
            
            if [ ! -f "$backup_file" ]; then
                error "Backup file not found: $backup_file"
                exit 1
            fi
            
            log "Restoring SSH configuration from: $backup_file"
            sudo cp "$backup_file" /etc/ssh/sshd_config
            
            if test_ssh_config; then
                sudo systemctl reload ssh
                success "SSH configuration restored successfully"
            else
                error "Restored configuration has syntax errors!"
                exit 1
            fi
            ;;
        "test-keys")
            verify_ssh_keys
            ;;
        "help"|"--help"|"-h")
            echo "Safe SSH Security Hardening Script"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  status     - Show current SSH configuration (default)"
            echo "  secure     - Safely apply security hardening"
            echo "  restore    - Restore from backup"
            echo "  test-keys  - Verify SSH keys are set up"
            echo "  help       - Show this help message"
            echo ""
            echo "This script safely hardens SSH by:"
            echo "  - Disabling password authentication"
            echo "  - Enabling only key-based authentication"
            echo "  - Disabling root login"
            echo "  - Adding security limits"
            echo ""
            echo "Safety features:"
            echo "  - Creates backup before changes"
            echo "  - Tests configuration syntax"
            echo "  - Tests on alternate port first (if remote)"
            echo "  - Uses reload instead of restart"
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
