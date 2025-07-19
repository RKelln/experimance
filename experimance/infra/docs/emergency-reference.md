# Experimance Emergency Quick Reference

## 🚨 Emergency Contacts & Info

- **Installation IP**: ________________
- **SSH User**: `experimance`
- **Web Dashboard**: `http://IP:8080` (admin/experimance2024)
- **Git Repo**: https://github.com/RKelln/experimance

## 🔧 Quick Commands

### Check Status
```bash
# Quick overview
/home/experimance/experimance/infra/scripts/status.sh

# Detailed service status
sudo systemctl status experimance-core@experimance
```

### Restart Everything
```bash
# Restart all services
sudo /home/experimance/experimance/infra/scripts/deploy.sh experimance restart

# Or individual service
sudo systemctl restart experimance-core@experimance
```

### View Logs
```bash
# Core service logs
sudo journalctl -u experimance-core@experimance -f

# All recent errors
sudo journalctl --since "1 hour ago" --grep "ERROR|CRITICAL"
```

### Emergency Stop
```bash
# Stop everything
sudo /home/experimance/experimance/infra/scripts/deploy.sh experimance stop

# Kill hanging processes
sudo pkill -f experimance
```

## 🆘 Common Issues

### "Services Won't Start"
1. Check logs: `sudo journalctl -u experimance-core@experimance -n 50`
2. Check permissions: `ls -la /var/cache/experimance`
3. Reset: `sudo systemctl reset-failed && sudo systemctl start experimance-core@experimance`

### "No Image Generation"
1. Check image server: `sudo systemctl status image-server@experimance`
2. Check GPU: `nvidia-smi` (if local generation)
3. Check API keys in `/home/experimance/experimance/projects/experimance/.env`

### "Display Issues"
1. Check X11: `echo $DISPLAY` (should be `:0`)
2. Check permissions: `xauth list`
3. Restart display: `sudo systemctl restart experimance-display@experimance`

### "High Resource Usage"
1. Check: `htop` or `nvidia-smi`
2. Reduce image quality in config
3. Restart services to clear memory

### "Can't SSH In"
1. Try physical access
2. Check network: `ping IP`
3. Check SSH service: `sudo systemctl status ssh`

## 🔄 Recovery Procedures

### Soft Reset
```bash
sudo /home/experimance/experimance/infra/scripts/deploy.sh experimance restart
```

### Hard Reset
```bash
sudo /home/experimance/experimance/infra/scripts/deploy.sh experimance stop
cd /home/experimance/experimance
git reset --hard origin/main
sudo /home/experimance/experimance/infra/scripts/deploy.sh experimance start
```

### Update & Restart
```bash
sudo /home/experimance/experimance/infra/scripts/update.sh experimance
```

### Rollback
```bash
cd /home/experimance/experimance
git log --oneline -5  # Find previous commit
git reset --hard <commit-hash>
sudo /home/experimance/experimance/infra/scripts/deploy.sh experimance restart
```

## 📱 Remote Monitoring

### Email Alerts
- Check `/var/log/experimance/healthcheck.log` for alert history
- Email issues: Check SMTP settings in healthcheck.py

### Web Dashboard
- Access: `http://IP:8080`
- Username: `admin`
- Password: `experimance2024`

### Health Check
```bash
# Manual health check
/home/experimance/experimance/infra/scripts/healthcheck.py

# Check cron jobs
sudo crontab -l | grep experimance
```

## 🔍 Debugging

### Check ZMQ Communication
```bash
# Test ZMQ ports
netstat -ln | grep ':555'

# Test connection
python3 -c "
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect('tcp://localhost:5555')
print('ZMQ OK')
"
```

### Check Process Tree
```bash
# Find all experimance processes
ps aux | grep experimance

# Check systemd tree
systemctl list-units | grep experimance
```

### Check File Permissions
```bash
# Key directories
ls -la /var/cache/experimance
ls -la /var/log/experimance
ls -la /home/experimance/experimance

# Fix permissions if needed
sudo chown -R experimance:experimance /var/cache/experimance
sudo chown -R experimance:experimance /var/log/experimance
```

## 📞 When to Call for Help

Call immediately if:
- ❌ Installation is completely unresponsive
- ❌ Smoke or burning smell
- ❌ Audience complaining about broken experience
- ❌ Multiple service failures that don't resolve

Call within 2 hours if:
- ⚠️ Single service repeatedly failing
- ⚠️ Performance severely degraded
- ⚠️ Unusual resource usage
- ⚠️ Cannot update or restart services

Monitor and document if:
- ℹ️ Occasional service restarts
- ℹ️ Minor performance issues
- ℹ️ Log warnings

---

**Print this page and keep it near the installation!**
