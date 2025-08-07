#!/usr/bin/env python3
"""
Simple web dashboard for Experimance monitoring
Provides a mobile-friendly interface for checking system status
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import aiohttp
from aiohttp import web
from aiohttp.web import Request, Response
import aiohttp_basicauth

# Configuration
CONFIG = {
    "host": "0.0.0.0",
    "port": 8080,
    "auth_user": "admin",
    "auth_password": "experimance2024",  # Change this!
    "refresh_interval": 30,
    "script_dir": Path(__file__).parent.parent / "scripts"
}

async def get_system_status():
    """Get current system status."""
    try:
        # Run status script
        result = subprocess.run(
            [str(CONFIG["script_dir"] / "status.sh")],
            capture_output=True,
            text=True
        )
        
        # Parse the output (simplified)
        lines = result.stdout.split('\n')
        status = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "system": {},
            "errors": []
        }
        
        # Parse services status
        in_services = False
        for line in lines:
            if "Services:" in line:
                in_services = True
                continue
            elif in_services and line.strip().startswith("✓"):
                service_name = line.split()[1].split("@")[0]
                status["services"][service_name] = "running"
            elif in_services and line.strip().startswith("✗"):
                service_name = line.split()[1].split("@")[0]
                status["services"][service_name] = "stopped"
            elif "System Resources:" in line:
                in_services = False
        
        return status
        
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "services": {},
            "system": {},
            "errors": [str(e)]
        }

async def index(request: Request) -> Response:
    """Main dashboard page."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experimance Dashboard</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background: #f5f5f5;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .header { 
                text-align: center; 
                margin-bottom: 30px;
                color: #333;
            }
            .status-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px;
            }
            .status-card { 
                background: #f8f9fa; 
                padding: 20px; 
                border-radius: 8px;
                border-left: 4px solid #007bff;
            }
            .status-card h3 { 
                margin-top: 0; 
                color: #495057;
            }
            .service-list { 
                list-style: none; 
                padding: 0; 
            }
            .service-list li { 
                padding: 8px 0; 
                border-bottom: 1px solid #dee2e6;
            }
            .service-list li:last-child { 
                border-bottom: none; 
            }
            .status-running { 
                color: #28a745; 
                font-weight: bold;
            }
            .status-stopped { 
                color: #dc3545; 
                font-weight: bold;
            }
            .refresh-btn { 
                background: #007bff; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                border-radius: 4px; 
                cursor: pointer;
                font-size: 16px;
                margin-right: 10px;
            }
            .refresh-btn:hover { 
                background: #0056b3; 
            }
            .action-btn { 
                background: #6c757d; 
                color: white; 
                border: none; 
                padding: 8px 16px; 
                border-radius: 4px; 
                cursor: pointer;
                margin-right: 5px;
                font-size: 14px;
            }
            .action-btn:hover { 
                background: #545b62; 
            }
            .action-btn.restart { 
                background: #ffc107; 
                color: #212529;
            }
            .action-btn.restart:hover { 
                background: #e0a800; 
            }
            .timestamp { 
                color: #6c757d; 
                font-size: 14px;
                text-align: center;
                margin-top: 20px;
            }
            .error { 
                background: #f8d7da; 
                color: #721c24; 
                padding: 10px; 
                border-radius: 4px; 
                margin-bottom: 20px;
            }
            .image-section { 
                margin-top: 30px; 
                text-align: center;
            }
            .latest-image { 
                max-width: 100%; 
                max-height: 400px; 
                border-radius: 8px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                margin: 10px 0;
            }
            .image-info { 
                color: #6c757d; 
                font-size: 14px; 
                margin-top: 10px;
            }
            @media (max-width: 768px) {
                .container { 
                    margin: 10px; 
                    padding: 15px; 
                }
                .status-grid { 
                    grid-template-columns: 1fr; 
                }
                .latest-image { 
                    max-height: 300px; 
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Experimance Dashboard</h1>
                <button class="refresh-btn" onclick="location.reload()">Refresh</button>
                <button class="action-btn restart" onclick="restartAll()">Restart All</button>
            </div>
            
            <div id="status-content">
                <div class="status-grid">
                    <div class="status-card">
                        <h3>Loading...</h3>
                        <p>Fetching system status...</p>
                    </div>
                </div>
            </div>
            
            <div class="image-section">
                <h3>Latest Generated Image</h3>
                <div id="image-content">
                    <p>Loading image...</p>
                </div>
            </div>
            
            <div class="timestamp" id="timestamp"></div>
        </div>
        
        <script>
            async function loadStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    
                    document.getElementById('timestamp').textContent = 
                        'Last updated: ' + new Date(data.timestamp).toLocaleString();
                    
                    let html = '<div class="status-grid">';
                    
                    // Services card
                    html += '<div class="status-card">';
                    html += '<h3>Services</h3>';
                    html += '<ul class="service-list">';
                    
                    for (const [service, status] of Object.entries(data.services)) {
                        const statusClass = status === 'running' ? 'status-running' : 'status-stopped';
                        const statusText = status === 'running' ? '✓ Running' : '✗ Stopped';
                        html += `<li>${service}: <span class="${statusClass}">${statusText}</span></li>`;
                    }
                    
                    html += '</ul></div>';
                    
                    // System card
                    html += '<div class="status-card">';
                    html += '<h3>System</h3>';
                    html += '<ul class="service-list">';
                    
                    for (const [key, value] of Object.entries(data.system)) {
                        html += `<li>${key}: ${value}</li>`;
                    }
                    
                    html += '</ul></div>';
                    
                    // Errors card
                    if (data.errors && data.errors.length > 0) {
                        html += '<div class="status-card">';
                        html += '<h3>Errors</h3>';
                        html += '<ul class="service-list">';
                        
                        for (const error of data.errors) {
                            html += `<li class="status-stopped">${error}</li>`;
                        }
                        
                        html += '</ul></div>';
                    }
                    
                    html += '</div>';
                    
                    document.getElementById('status-content').innerHTML = html;
                    
                } catch (error) {
                    document.getElementById('status-content').innerHTML = 
                        '<div class="error">Error loading status: ' + error.message + '</div>';
                }
            }
            
            async function restartAll() {
                if (confirm('Are you sure you want to restart all services?')) {
                    try {
                        const response = await fetch('/api/restart', { method: 'POST' });
                        const result = await response.json();
                        alert(result.message);
                        setTimeout(loadStatus, 2000);
                    } catch (error) {
                        alert('Error restarting services: ' + error.message);
                    }
                }
            }
            
            async function loadLatestImage() {
                try {
                    // Get image metadata first
                    const infoResponse = await fetch('/api/image-info');
                    const imageInfo = await infoResponse.json();
                    
                    if (imageInfo.exists) {
                        // Create image element with cache-busting timestamp
                        const imgUrl = `/api/latest-image?t=${imageInfo.timestamp}`;
                        const imgElement = `
                            <img src="${imgUrl}" class="latest-image" alt="Latest generated image">
                            <div class="image-info">
                                File: ${imageInfo.filename}<br>
                                Generated: ${new Date(imageInfo.modified).toLocaleString()}<br>
                                Size: ${(imageInfo.size / 1024).toFixed(1)} KB
                            </div>
                        `;
                        document.getElementById('image-content').innerHTML = imgElement;
                    } else {
                        document.getElementById('image-content').innerHTML = 
                            '<p style="color: #6c757d;">No image generated yet</p>';
                    }
                    
                } catch (error) {
                    document.getElementById('image-content').innerHTML = 
                        '<p style="color: #dc3545;">Error loading image: ' + error.message + '</p>';
                }
            }
            
            // Load status on page load
            loadStatus();
            loadLatestImage();
            
            // Auto-refresh every 30 seconds
            setInterval(() => {
                loadStatus();
                loadLatestImage();
            }, 30000);
        </script>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')

async def api_status(request: Request) -> Response:
    """API endpoint for system status."""
    status = await get_system_status()
    return web.json_response(status)

async def api_restart(request: Request) -> Response:
    """API endpoint to restart services."""
    try:
        # Get current project
        project = "experimance"  # Default
        try:
            with open("/etc/experimance/current_project", "r") as f:
                project = f.read().strip()
        except FileNotFoundError:
            pass
        
        # Restart services
        result = subprocess.run(
            [str(CONFIG["script_dir"] / "deploy.sh"), project, "restart"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return web.json_response({"message": "Services restarted successfully"})
        else:
            return web.json_response(
                {"error": "Failed to restart services", "details": result.stderr},
                status=500
            )
            
    except Exception as e:
        return web.json_response(
            {"error": "Internal server error", "details": str(e)},
            status=500
        )

async def api_latest_image(request: Request) -> Response:
    """API endpoint to serve the latest generated image."""
    try:
        latest_image = Path("/home/experimance/experimance/media/images/generated/latest.jpg")
        if latest_image.exists():
            return web.FileResponse(latest_image)
        else:
            return web.Response(text="No latest image found", status=404)
    except Exception as e:
        return web.Response(text=f"Error: {str(e)}", status=500)

async def api_image_info(request: Request) -> Response:
    """API endpoint to get latest image metadata."""
    try:
        latest_image = Path("/home/experimance/experimance/media/images/generated/latest.jpg")
        if latest_image.exists():
            stat = latest_image.stat()
            # Get the actual file it points to (since it's a symlink)
            actual_file = latest_image.resolve()
            return web.json_response({
                "exists": True,
                "timestamp": stat.st_mtime,
                "size": stat.st_size,
                "filename": actual_file.name,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        else:
            return web.json_response({"exists": False})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def init_app():
    """Initialize the web application."""
    app = web.Application()
    
    # Add basic auth
    auth = aiohttp_basicauth.BasicAuthMiddleware(
        CONFIG["auth_user"], CONFIG["auth_password"]
    )
    app.middlewares.append(auth)
    
    # Routes
    app.router.add_get('/', index)
    app.router.add_get('/api/status', api_status)
    app.router.add_post('/api/restart', api_restart)
    app.router.add_get('/api/latest-image', api_latest_image)
    app.router.add_get('/api/image-info', api_image_info)
    
    return app

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(f"Usage: {sys.argv[0]} [--help]")
        print("Starts the Experimance monitoring dashboard")
        print(f"Access at: http://localhost:{CONFIG['port']}")
        print(f"Username: {CONFIG['auth_user']}")
        print(f"Password: {CONFIG['auth_password']}")
        return
    
    app = asyncio.run(init_app())
    
    print(f"Starting Experimance dashboard on http://{CONFIG['host']}:{CONFIG['port']}")
    print(f"Username: {CONFIG['auth_user']}")
    print(f"Password: {CONFIG['auth_password']}")
    
    web.run_app(app, host=CONFIG["host"], port=CONFIG["port"])

if __name__ == "__main__":
    main()
