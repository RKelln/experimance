# Infrastructure Implementation TODO

## Phase 1: Core Setup (1 hour)

### Prerequisites
- [ ] Create `experimance` user on target system
- [ ] Install `uv` package manager
- [ ] Clone repository to `/home/experimance/experimance`
- [ ] Install Python 3.11+ and dependencies

### Service Installation
- [ ] Copy systemd service files to `/etc/systemd/system/`
- [ ] Create required directories (`/var/cache/experimance`, `/var/log/experimance`)
- [ ] Set proper permissions for experimance user
- [ ] Test service installation: `sudo ./infra/scripts/deploy.sh experimance install`

### Initial Testing
- [ ] Start services: `sudo ./infra/scripts/deploy.sh experimance start`
- [ ] Check status: `./infra/scripts/status.sh experimance`
- [ ] Test restart: `sudo ./infra/scripts/deploy.sh experimance restart`

## Phase 2: Monitoring Setup (45 minutes)

### Email Configuration
- [ ] Set up Gmail app password (or alternative SMTP)
- [ ] Configure environment variables:
  ```bash
  export ALERT_EMAIL="your-email@gmail.com"
  export ALERT_EMAIL_TO="your-email@gmail.com"
  export ALERT_EMAIL_PASSWORD="your-app-password"
  ```
- [ ] Test email alerts: `./infra/scripts/healthcheck.py`

### Web Dashboard
- [ ] **IMPORTANT**: Change default password in `infra/monitoring/dashboard.py`
- [ ] Test web dashboard: `./infra/monitoring/dashboard.py`
- [ ] Verify mobile access at `http://installation-ip:8080`

### Automated Monitoring
- [ ] Add cron jobs: `sudo crontab -e` (use `infra/monitoring/crontab` as template)
- [ ] Test health check automation
- [ ] Verify log rotation is working

## Phase 3: SSH & Remote Access (15 minutes)

### SSH Setup
- [ ] Enable SSH: `sudo systemctl enable ssh && sudo systemctl start ssh`
- [ ] Add your public key to `~/.ssh/authorized_keys`
- [ ] Test SSH access from your machine
- [ ] Test SSH key-based authentication

### Security
- [ ] Disable password authentication in `/etc/ssh/sshd_config` (optional)
- [ ] Configure firewall rules if needed
- [ ] Test remote access to web dashboard

## Phase 4: Testing & Validation (30 minutes)

### Service Testing
- [ ] Test individual service failures (kill processes, check auto-restart)
- [ ] Test resource exhaustion scenarios
- [ ] Test ZMQ communication between services
- [ ] Verify image generation pipeline works

### Update Testing
- [ ] Test update process: `sudo ./infra/scripts/update.sh experimance`
- [ ] Test rollback functionality
- [ ] Verify services restart after updates

### Monitoring Testing
- [ ] Trigger alerts (stop services, simulate failures)
- [ ] Verify email notifications work
- [ ] Test web dashboard restart functionality
- [ ] Check log aggregation and rotation

## Phase 5: Documentation & Handoff (15 minutes)

### Documentation
- [ ] Print emergency reference card: `infra/docs/emergency-reference.md`
- [ ] Document installation-specific details (IP addresses, credentials)
- [ ] Create contact information sheet
- [ ] Document any custom configuration

### Final Checklist
- [ ] All services running and healthy
- [ ] Monitoring alerts working
- [ ] SSH access confirmed
- [ ] Web dashboard accessible
- [ ] Emergency procedures tested
- [ ] Documentation complete

## Deployment Commands Summary

```bash
# Phase 1: Core Setup
sudo ./infra/scripts/deploy.sh experimance install
sudo ./infra/scripts/deploy.sh experimance start

# Phase 2: Monitoring
export ALERT_EMAIL="your-email@gmail.com"
export ALERT_EMAIL_TO="your-email@gmail.com"
export ALERT_EMAIL_PASSWORD="your-app-password"
./infra/scripts/healthcheck.py
./infra/monitoring/dashboard.py &

# Phase 3: SSH (from your machine)
ssh-copy-id experimance@installation-ip
ssh experimance@installation-ip

# Phase 4: Testing
sudo ./infra/scripts/update.sh experimance
./infra/scripts/status.sh experimance
```

## Common Issues & Solutions

### Services Won't Start
- Check logs: `sudo journalctl -u experimance-core@experimance -n 50`
- Check permissions: `ls -la /var/cache/experimance`
- Reset services: `sudo systemctl reset-failed`

### Email Alerts Not Working
- Test SMTP settings: `python3 -c "import smtplib; print('SMTP OK')"`
- Check Gmail app password setup
- Verify environment variables are set

### Web Dashboard Not Accessible
- Check if process is running: `ps aux | grep dashboard`
- Check firewall: `sudo ufw status`
- Test local access: `curl http://localhost:8080`

### SSH Access Issues
- Check SSH service: `sudo systemctl status ssh`
- Verify public key: `cat ~/.ssh/authorized_keys`
- Check SSH config: `sudo nano /etc/ssh/sshd_config`

## Nice-to-Have Additions

### After Basic Setup Works
- [ ] Set up automatic backups to cloud storage
- [ ] Configure additional notification channels (SMS, Slack)
- [ ] Add performance monitoring dashboard
- [ ] Set up log analysis and search
- [ ] Create additional project configurations

### For Long-term Maintenance
- [ ] Set up automated security updates
- [ ] Create maintenance schedules
- [ ] Add more detailed performance metrics
- [ ] Set up remote debugging tools
- [ ] Create disaster recovery procedures

## Success Criteria

✅ **All services start automatically on boot**
✅ **Failed services restart automatically**
✅ **Email alerts work for critical failures**
✅ **Web dashboard shows accurate status**
✅ **SSH access works from remote locations**
✅ **Updates can be applied remotely**
✅ **Emergency procedures are documented**

**Total Time Estimate: 2.5 hours**
