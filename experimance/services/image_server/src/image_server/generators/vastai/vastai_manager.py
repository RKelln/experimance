"""
Vast.ai Instance Manager for Experimance Image Generation

This module provides programmatic management of vast.ai instances for the 
experimance image generation service. It handles instance creation, monitoring,
and provides the external endpoint information for API access.

Usage:
    from vastai_manager import VastAIManager
    
    manager = VastAIManager()
    instance = manager.find_or_create_instance()
    endpoint = manager.get_model_server_endpoint(instance['id'])
    print(f"Model server available at: {endpoint}")
"""

import json
import os
import time
import subprocess
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from experimance_common.constants import PROJECT_ROOT
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class InstanceEndpoint:
    """Information about an accessible instance endpoint."""
    public_ip: str
    external_port: int
    internal_port: int
    url: str
    instance_id: int
    status: str


class VastAIManager:
    """Manages vast.ai instances for experimance image generation."""
    
    def __init__(self, api_key: Optional[str] = None, provisioning_script_url: Optional[str] = None):
        """
        Initialize the VastAI manager.
        
        Args:
            api_key: Optional API key. If not provided, will use vastai CLI auth.
            provisioning_script_url: Optional URL for the provisioning script. If not provided,
                                   will use the default experimance provisioning script.
        """
        self.api_key = api_key or os.getenv("VASTAI_API_KEY")
        self.experimance_template_id = "d1328a4225c2f54cb1908912be208bf4"  # PyTorch template
        self.experimance_template_name = "PyTorch (Vast) web accessible"  # Custom template name
        
        # Allow custom provisioning script URL or use default
        # NOTE: PROVISIONING_SCRIPT env var is documented but appears broken in VastAI PyTorch template
        # See: https://cloud.vast.ai/template/readme/b28b2d2625172e089509d1c0331258b8
        # "Runs any custom setup script defined in the PROVISIONING_SCRIPT environment variable"
        # Using SCP provisioning as workaround until this is fixed
        default_provisioning_url = "https://gist.githubusercontent.com/RKelln/21ad3ecb4be1c1d0d55a8f1524ff9b14/raw"
        self.provisioning_script_url = provisioning_script_url or os.getenv("VASTAI_PROVISIONING_SCRIPT", default_provisioning_url)
        
        self.required_env_vars = {
            "GITHUB_ACCESS_TOKEN": os.getenv("GITHUB_ACCESS_TOKEN", ""),
            "PROVISIONING_SCRIPT": self.provisioning_script_url
        }
        self.disk_size = 20 # Gigabytes
        
    def _run_vastai_command(self, cmd: List[str], raw: bool = True) -> Any:
        """Run a vastai CLI command and return parsed JSON result."""
        try:
            command_args = []
            if raw:
                command_args.append("--raw")
            if self.api_key:
                command_args.append("--api-key")
                command_args.append(self.api_key)
            #command_args = ["--raw", "--api-key", self.api_key] if self.api_key else ["--raw"]
            command = (
                ["uv", "tool", "run", "vastai"] +
                cmd +
                command_args
            )
            # Filter out any accidental None values
            command = [str(x) for x in command if x is not None]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            if raw:
                return json.loads(result.stdout)
            else:
                return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"VastAI command failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse VastAI response: {e}")
            logger.error(f"$ {' '.join(command)}: {result.stdout}")
            raise

    def show_instances(self, raw: bool = True) -> List[Dict[str, Any]]:
        """List all instances for the current user."""
        return self._run_vastai_command(["show", "instances"], raw=raw)

    def show_instance(self, instance_id: int, raw: bool = True) -> Dict[str, Any]:
        """Get detailed information about a specific instance."""
        return self._run_vastai_command(["show", "instance", str(instance_id)], raw=raw)

    def find_experimance_instances(self) -> List[Dict[str, Any]]:
        """Find all running experimance instances."""
        instances = self.show_instances(raw=True)  # Force raw=True to get JSON
        experimance_instances = []
        
        for instance in instances:
            # Check if it's our custom template and has our environment variables
            template_id = instance.get("template_hash_id", "")
            template_name = instance.get("template_name", "")
            
            # Match either by template ID or name (for backwards compatibility)
            is_experimance_template = (
                template_id == self.experimance_template_id or
                self.experimance_template_name in template_name
            )

            logger.debug(f"Checking instance {instance['id']} - Template ID: {template_id}, Name: {template_name}")
            
            if (is_experimance_template and
                instance.get("actual_status") == "running"):
                experimance_instances.append(instance)
        
        return experimance_instances
    
    def get_ssh_command(self, instance_id: int, prefer_direct: bool = True) -> Optional[str]:
        """
        Get the SSH command to connect to an instance.
        
        Provides both direct SSH (via public IP and mapped port) and proxy SSH
        (via VastAI's SSH gateway). Direct SSH is preferred as it's more reliable.
        
        Args:
            instance_id: The vast.ai instance ID
            prefer_direct: If True, prefer direct SSH over proxy SSH
            
        Returns:
            SSH command string, or None if not available
        """
        try:
            instance = self.show_instance(instance_id)
            
            # Try direct SSH first (via public IP and port mapping)
            if prefer_direct:
                public_ip = instance.get("public_ipaddr")
                ports = instance.get("ports", {})
                ssh_port_mapping = ports.get("22/tcp")
                
                if public_ip and ssh_port_mapping:
                    try:
                        external_ssh_port = int(ssh_port_mapping[0]["HostPort"])
                        direct_ssh = f"ssh -p {external_ssh_port} root@{public_ip}"
                        logger.debug(f"Direct SSH available: {direct_ssh}")
                        return direct_ssh
                    except (IndexError, ValueError, KeyError) as e:
                        logger.debug(f"Failed to parse direct SSH port mapping: {e}")
                
                logger.debug(f"Direct SSH not available for instance {instance_id}, trying proxy SSH")
            
            # Fall back to proxy SSH (via VastAI's SSH gateway)
            ssh_host = instance.get("ssh_host")
            ssh_port = instance.get("ssh_port")
            
            if ssh_host and ssh_port:
                proxy_ssh = f"ssh -p {ssh_port} root@{ssh_host}"
                logger.debug(f"Proxy SSH available: {proxy_ssh}")
                return proxy_ssh
            
            # If we get here, neither method worked
            logger.warning(f"No SSH access method found for instance {instance_id}")
            logger.debug(f"Instance data: public_ip={instance.get('public_ipaddr')}, "
                        f"ssh_host={instance.get('ssh_host')}, ssh_port={instance.get('ssh_port')}, "
                        f"ports={instance.get('ports', {})}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get SSH command for instance {instance_id}: {e}")
            return None

    def get_ssh_methods(self, instance_id: int) -> Dict[str, Optional[str]]:
        """
        Get all available SSH connection methods for debugging.
        
        Args:
            instance_id: The vast.ai instance ID
            
        Returns:
            Dictionary with 'direct' and 'proxy' SSH commands
        """
        try:
            instance = self.show_instance(instance_id)
            methods: Dict[str, Optional[str]] = {"direct": None, "proxy": None}
            
            # Direct SSH (via public IP and port mapping)
            public_ip = instance.get("public_ipaddr")
            ports = instance.get("ports", {})
            ssh_port_mapping = ports.get("22/tcp")
            
            if public_ip and ssh_port_mapping:
                try:
                    external_ssh_port = int(ssh_port_mapping[0]["HostPort"])
                    methods["direct"] = f"ssh -p {external_ssh_port} root@{public_ip}"
                except (IndexError, ValueError, KeyError):
                    pass
            
            # Proxy SSH (via VastAI's SSH gateway)
            ssh_host = instance.get("ssh_host")
            ssh_port = instance.get("ssh_port")
            
            if ssh_host and ssh_port:
                methods["proxy"] = f"ssh -p {ssh_port} root@{ssh_host}"
            
            return methods
            
        except Exception as e:
            logger.error(f"Failed to get SSH methods for instance {instance_id}: {e}")
            return {"direct": None, "proxy": None}

    def get_model_server_endpoint(self, instance_id: int) -> Optional[InstanceEndpoint]:
        """
        Get the external endpoint for the model server (port 8000).
        
        Args:
            instance_id: The vast.ai instance ID
            
        Returns:
            InstanceEndpoint with connection information, or None if not available
        """
        try:
            instance = self.show_instance(instance_id)
            
            # Extract port mapping for internal port 8000
            ports = instance.get("ports", {})
            port_8000_mapping = ports.get("8000/tcp")
            
            if not port_8000_mapping:
                logger.warning(f"No port 8000 mapping found for instance {instance_id}")
                return None
            
            # Get the external port mapping
            external_port = int(port_8000_mapping[0]["HostPort"])
            public_ip = instance.get("public_ipaddr")
            
            if not public_ip:
                logger.warning(f"No public IP found for instance {instance_id}")
                return None
            
            url = f"http://{public_ip}:{external_port}"
            
            return InstanceEndpoint(
                public_ip=public_ip,
                external_port=external_port,
                internal_port=8000,
                url=url,
                instance_id=instance_id,
                status=instance.get("actual_status", "unknown")
            )
            
        except Exception as e:
            logger.error(f"Failed to get endpoint for instance {instance_id}: {e}")
            return None
    
    def wait_for_instance_ready(self, instance_id: int, timeout: int = 600) -> bool:
        """
        Wait for an instance to be running and have port mappings available.
        
        Args:
            instance_id: The instance ID to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if instance is ready, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                instance = self.show_instance(instance_id, raw=True)
                status = instance.get("actual_status", "unknown")
                
                # Check if instance is running
                if status == "running":
                    # Check if port mappings are available
                    ports = instance.get("ports", {})
                    if "8000/tcp" in ports and "1111/tcp" in ports:
                        logger.info(f"Instance {instance_id} is ready and has port mappings")
                        return True
                    else:
                        logger.info(f"Instance {instance_id} is running but ports not yet mapped")
                else:
                    logger.info(f"Instance {instance_id} status: {status}")
                
                time.sleep(15)  # Check more frequently
                
            except Exception as e:
                logger.warning(f"Error checking instance {instance_id}: {e}")
                time.sleep(15)
        
        logger.error(f"Timeout waiting for instance {instance_id} to be ready")
        return False
    
    def wait_for_ssh_ready(self, instance_id: int, timeout: int = 300) -> bool:
        """
        Wait for SSH to be accessible on an instance.
        
        Args:
            instance_id: The instance ID to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if SSH is ready, False if timeout
        """
        logger.info(f"Waiting for SSH to be ready on instance {instance_id}...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                ssh_cmd = self.get_ssh_command(instance_id)
                if not ssh_cmd:
                    logger.debug(f"SSH command not available for instance {instance_id}")
                    time.sleep(10)
                    continue
                
                # Parse SSH connection details
                parts = ssh_cmd.split()
                port = parts[2]  # After -p
                host_and_user = parts[3]  # root@HOST
                host = host_and_user.split('@')[1]
                user = host_and_user.split('@')[0]
                
                # Test SSH connectivity with a simple command
                test_cmd = [
                    "ssh", "-p", port,
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "LogLevel=ERROR",
                    "-o", "ConnectTimeout=10",
                    "-o", "BatchMode=yes",
                    f"{user}@{host}",
                    "echo 'SSH ready'"
                ]
                
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=15)
                if result.returncode == 0:
                    logger.info(f"SSH is ready on instance {instance_id}")
                    return True
                else:
                    logger.debug(f"SSH not ready yet: {result.stderr.strip()}")
                
            except Exception as e:
                logger.debug(f"SSH test failed: {e}")
            
            time.sleep(10)  # Wait 10 seconds between attempts
        
        logger.error(f"Timeout waiting for SSH to be ready on instance {instance_id}")
        return False
    
    def provision_instance_via_scp(self, instance_id: int, timeout: int = 300, verbose: bool = False) -> bool:
        """
        Provision an instance by SCPing the provisioning script and running it.
        
        This is currently required because the PROVISIONING_SCRIPT environment variable
        feature is broken in VastAI's PyTorch template (as of July 2025).
        
        Bug details:
        - Environment variable is correctly passed to container
        - Template documentation claims it should execute the script: 
          https://cloud.vast.ai/template/readme/b28b2d2625172e089509d1c0331258b8
        - But /opt/instance-tools/bin/entrypoint.sh does not actually run the script
        
        Args:
            instance_id: The instance ID to provision
            timeout: Maximum time to wait for provisioning to complete
            verbose: If True, stream script output to console in real-time
            
        Returns:
            True if provisioning succeeded, False otherwise
        """
        logger.info(f"Starting SCP provisioning for instance {instance_id}")
        
        # Wait for SSH to be ready first
        if not self.wait_for_ssh_ready(instance_id, timeout=180):  # 3 minutes for SSH
            logger.error(f"SSH not ready for instance {instance_id}, cannot provision")
            return False
        
        # Get SSH connection info
        ssh_command = self.get_ssh_command(instance_id)
        if not ssh_command:
            logger.error(f"Cannot get SSH command for instance {instance_id}")
            return False
        
        # Extract SSH connection details
        try:
            # Parse ssh command: "ssh -p PORT root@HOST"
            parts = ssh_command.split()
            port = parts[2]  # After -p
            host_and_user = parts[3]  # root@HOST
            host = host_and_user.split('@')[1]
            user = host_and_user.split('@')[0]
        except (IndexError, ValueError) as e:
            logger.error(f"Failed to parse SSH command '{ssh_command}': {e}")
            return False
        
        # Get the path to the provisioning script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        provisioning_script_path = os.path.join(script_dir, "server", "vast_provisioning.sh")
        
        if not os.path.exists(provisioning_script_path):
            logger.error(f"Provisioning script not found at {provisioning_script_path}")
            return False
        
        logger.info(f"Found provisioning script at {provisioning_script_path}")
        
        remote_path = "/workspace/vast_provisioning.sh"

        try:
            # SCP the script to the instance
            logger.info(f"Copying provisioning script to instance {instance_id}...")
            scp_cmd = [
                "scp", "-P", port,
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", "ConnectTimeout=30",
                "-o", "BatchMode=yes",
                provisioning_script_path,
                f"{user}@{host}:{remote_path}"
            ]
            
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.error(f"SCP failed: {result.stderr}")
                return False
            
            logger.info("Successfully copied provisioning script")
            
            # Make the script executable and run it
            logger.info("Running provisioning script on instance...")
            
            ssh_cmd = [
                "ssh", "-p", port,
                "-o", "StrictHostKeyChecking=no", 
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-o", "ConnectTimeout=30",
                "-o", "BatchMode=yes",
                f"{user}@{host}",
                f"chmod +x {remote_path} && {remote_path}"
            ]
            
            logger.info(f"Executing: {' '.join(ssh_cmd[:7])} ... [script execution]")
            
            if verbose:
                # Stream output in real-time when verbose mode is enabled
                print(f"\n🔧 Running provisioning script on instance {instance_id}...")
                print("=" * 60)
                
                process = subprocess.Popen(
                    ssh_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                output_lines = []
                try:
                    if process.stdout:
                        while True:
                            output = process.stdout.readline()
                            if output == '' and process.poll() is not None:
                                break
                            if output:
                                print(output.strip())
                                output_lines.append(output)
                    
                    process.wait(timeout=timeout)
                    print("=" * 60)
                    
                    if process.returncode == 0:
                        logger.info("Provisioning script completed successfully")
                        print("✅ Provisioning script completed successfully!")
                        return True
                    else:
                        logger.error(f"Provisioning script failed with exit code {process.returncode}")
                        print(f"❌ Provisioning script failed with exit code {process.returncode}")
                        
                        # Even if the script failed, check if the service is actually running
                        print("⚠️  Checking if service is running despite script failure...")
                        if self._check_service_health(instance_id):
                            logger.warning("Service appears to be healthy despite script failure, considering provisioning successful")
                            print("✅ Service is healthy despite script failure, provisioning considered successful!")
                            return True
                        else:
                            print("❌ Service is not healthy, provisioning failed")
                            return False
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.error(f"Provisioning script timed out after {timeout} seconds")
                    print(f"❌ Provisioning script timed out after {timeout} seconds")
                    return False
            else:
                # Original behavior - capture output and only show on error/debug
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
                
                if result.returncode == 0:
                    logger.info("Provisioning script completed successfully")
                    logger.debug(f"Script output: {result.stdout}")
                    return True
                else:
                    logger.error(f"Provisioning script failed with exit code {result.returncode}")
                    if result.stderr:
                        logger.error(f"Script stderr: {result.stderr}")
                    if result.stdout:
                        logger.error(f"Script stdout: {result.stdout}")
                    logger.debug(f"Full SSH command: {' '.join(ssh_cmd)}")
                    
                    # Even if the script failed, check if the service is actually running
                    logger.warning("Checking if service is running despite script failure...")
                    if self._check_service_health(instance_id):
                        logger.warning("Service appears to be healthy despite script failure, considering provisioning successful")
                        return True
                    else:
                        return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Provisioning script timed out after {timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"Failed to provision instance via SCP: {e}")
            return False
    
    def _check_service_health(self, instance_id: int, timeout: int = 30) -> bool:
        """
        Check if the Experimance image server is healthy on the given instance.
        
        Args:
            instance_id: The Vast.ai instance ID
            timeout: Timeout in seconds for health check
            
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            endpoint = self.get_model_server_endpoint(instance_id)
            if not endpoint:
                logger.debug(f"Could not get endpoint for instance {instance_id}")
                return False
            
            import requests
            response = requests.get(f"{endpoint.url}/healthcheck", timeout=timeout)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') == 'healthy':
                    logger.debug(f"Health check passed for instance {instance_id}")
                    return True
                else:
                    logger.debug(f"Health check failed: service not healthy")
                    return False
            else:
                logger.debug(f"Health check failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.debug(f"Health check failed for instance {instance_id}: {e}")
            return False

    def _wait_for_service_healthy(self, instance_id: int, timeout: int = 180) -> bool:
        """
        Wait for the Experimance image server to become healthy.
        
        Args:
            instance_id: The Vast.ai instance ID
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if service becomes healthy, False if timeout
        """
        logger.info(f"Waiting for service to become healthy on instance {instance_id} (timeout: {timeout}s)...")
        start_time = time.time()
        endpoint_url = None
        
        while time.time() - start_time < timeout:
            # Try to get endpoint each iteration - it might not be available initially
            endpoint = self.get_model_server_endpoint(instance_id)
            if endpoint:
                if endpoint_url != endpoint.url:
                    endpoint_url = endpoint.url
                    logger.info(f"Endpoint now available, checking health at: {endpoint_url}/healthcheck")
                
                # Now try health check
                if self._check_service_health(instance_id, timeout=10):
                    logger.info(f"Service is healthy on instance {instance_id}")
                    return True
            else:
                # Endpoint not available yet - this is normal early in instance lifecycle
                elapsed = int(time.time() - start_time)
                if elapsed % 30 == 0:  # Log every 30 seconds
                    logger.info(f"Endpoint not available yet, waiting for port mappings... ({elapsed}s elapsed)")
                else:
                    logger.debug(f"Endpoint not available yet, waiting for port mappings... ({elapsed}s elapsed)")
            
            elapsed = int(time.time() - start_time)
            if elapsed % 30 == 0 and endpoint:  # Log every 30 seconds when we have endpoint
                logger.info(f"Service not healthy yet, waiting... ({elapsed}s elapsed)")
            elif endpoint:
                logger.debug(f"Service not healthy yet, waiting... ({elapsed}s elapsed)")
                
            time.sleep(5)  # Check every 5 seconds
        
        elapsed = int(time.time() - start_time)
        if not endpoint:
            logger.warning(f"Endpoint never became available within {timeout} seconds on instance {instance_id} (waited {elapsed}s)")
        else:
            logger.warning(f"Service did not become healthy within {timeout} seconds on instance {instance_id} (waited {elapsed}s)")
        return False

    def update_server_code(self, instance_id: int, timeout: int = 120, verbose: bool = False) -> bool:
        """
        Update server code on an existing instance by uploading Python files and restarting service.
        
        Args:
            instance_id: The instance ID to update
            timeout: Timeout for operations in seconds
            verbose: Show command output in real-time
            
        Returns:
            True if update succeeded, False otherwise
        """
        logger.info(f"Updating server code on instance {instance_id}")
        
        # Get SSH connection info
        ssh_cmd = self.get_ssh_command(instance_id)
        if not ssh_cmd:
            logger.error(f"Could not get SSH command for instance {instance_id}")
            return False
        
        # Parse SSH command string to get connection details and convert to list
        ssh_cmd_parts = ssh_cmd.split()
        host, port, user = None, None, None
        for i, part in enumerate(ssh_cmd_parts):
            if part == "-p" and i + 1 < len(ssh_cmd_parts):
                port = ssh_cmd_parts[i + 1]
            elif "@" in part:
                user, host = part.split("@", 1)
        
        if not all([host, port, user]):
            logger.error(f"Could not parse SSH connection details from: {' '.join(ssh_cmd_parts)}")
            return False
        
        # Get the server directory path
        import os
        from pathlib import Path
        current_file = Path(__file__).resolve()
        server_dir = current_file.parent / "server"
        
        if not server_dir.exists():
            logger.error(f"Server directory not found: {server_dir}")
            return False
        
        # need to remove the path before the root experimance directory and add to the remote path
        relative_server_dir = server_dir.relative_to(PROJECT_ROOT)
        logger.debug(f"Relative server directory: {relative_server_dir}")
        remote_server_dir = f"/workspace/experimance/experimance/{relative_server_dir}"

        logger.info(f"Found server directory at {server_dir}")

        try:
            # SCP all Python files from the server directory
            logger.info(f"Uploading server code to instance {instance_id}...")
            python_files = list(server_dir.glob("*.py"))
            
            if not python_files:
                logger.error(f"No Python files found in {server_dir}")
                return False
            
            logger.info(f"Found {len(python_files)} Python files to upload")
            
            for py_file in python_files:
                if verbose:
                    print(f"📁 Uploading {py_file.name}...")
                
                scp_cmd = [
                    "scp", "-P", port,
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "LogLevel=ERROR",
                    "-o", "ConnectTimeout=30",
                    "-o", "BatchMode=yes",
                    str(py_file),
                    f"{user}@{host}:{remote_server_dir}"
                ]
                
                result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    logger.error(f"Failed to upload {py_file.name}: {result.stderr}")
                    return False
                
                if verbose:
                    print(f"✅ {py_file.name} uploaded successfully")
            
            logger.info("All server files uploaded successfully")
            
            # Restart the service using supervisorctl
            restart_cmd = ssh_cmd_parts + ["supervisorctl", "restart", "experimance-image-server"]
            
            if verbose:
                print(f"\n🔄 Restarting image server service...")
                print("=" * 40)
            
            logger.info("Restarting experimance-image-server service...")
            logger.debug(f"Executing: {' '.join(restart_cmd[:7])} ... [service restart]")
            
            if verbose:
                process = subprocess.Popen(
                    restart_cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                try:
                    if process.stdout:
                        while True:
                            output = process.stdout.readline()
                            if output == '' and process.poll() is not None:
                                break
                            if output:
                                print(output.strip())
                    
                    process.wait(timeout=30)
                    print("=" * 40)
                    
                    if process.returncode == 0:
                        logger.info("Service restarted successfully")
                        return True
                    else:
                        logger.error(f"Service restart failed with exit code {process.returncode}")
                        return False
                        
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.error("Service restart timed out")
                    return False
            else:
                # Non-verbose mode
                result = subprocess.run(restart_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    logger.info("Service restarted successfully")
                    logger.debug(f"Restart output: {result.stdout}")
                    return True
                else:
                    logger.error(f"Service restart failed with exit code {result.returncode}")
                    if result.stderr:
                        logger.error(f"Restart stderr: {result.stderr}")
                    if result.stdout:
                        logger.error(f"Restart stdout: {result.stdout}")
                    return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Update operation timed out after {timeout} seconds")
            return False
        except Exception as e:
            logger.error(f"Failed to update server code: {e}")
            return False
    
    def provision_existing_instance(self, instance_id: int) -> bool:
        """
        Manually provision an existing instance using SCP.
        Useful for instances that were created without proper provisioning.
        
        Args:
            instance_id: The instance ID to provision
            
        Returns:
            True if provisioning succeeded, False otherwise
        """
        logger.info(f"Manually provisioning existing instance {instance_id}")
        
        # Check if instance is running
        try:
            instance = self.show_instance(instance_id)
            if instance.get("actual_status") != "running":
                logger.error(f"Instance {instance_id} is not running (status: {instance.get('actual_status')})")
                return False
        except Exception as e:
            logger.error(f"Failed to check instance status: {e}")
            return False
        
        # Wait for SSH to be available
        if not self.wait_for_instance_ready(instance_id, timeout=300):
            logger.error(f"Instance {instance_id} is not ready for SSH connection")
            return False
        
        # Run SCP provisioning
        return self.provision_instance_via_scp(instance_id)
    
    def search_offers(self, 
                     min_gpu_ram: int = 16,  # 16GB minimum for optimized SDXL
                     max_price: float = 0.5,     # $0.50/hour max
                     dlperf: float = 32.0,       # 3090 +
                     verified_only: bool = True) -> List[Dict[str, Any]]:
        """
        Search for suitable GPU offers.
        
        Args:
            min_gpu_ram: Minimum GPU RAM in GB
            max_price: Maximum price per hour
            dlperf: Minimum DLPerf score
            verified_only: Only show verified hosts
            
        Returns:
            List of available offers sorted by smart selection criteria
        """
        # Build query string
        # Note: VastAI API is inconsistent - search query expects GB but returns MB
        # Search uses GB units, but results come back in MB
        # Set good minimums for other factors in the search query
        query_parts = [
            f"reliability>0.95",        # Minimum reliability
            f"gpu_ram>={min_gpu_ram}",  # Search API expects GB
            f"dph_total<={max_price}",  # Maximum price
            f"dlperf>={dlperf}",        # Minimum DLPerf
            f"num_gpus=1",              # Single GPU instances
            f"geolocation in [CA,US]",  # Geographic preference
            f"inet_down>500",           # Minimum 500 Mbps download
            f"inet_up>100",             # Minimum 100 Mbps upload
            f"cuda_vers>=12.2",         # Modern CUDA version
            f"pcie_bw>=12"              # Minimum PCIe bandwidth
        ]
        
        if verified_only:
            query_parts.append("verified=true")

        query_string = " ".join(query_parts)
        
        cmd = [
            "search", "offers",
            query_string,
            "--order", "dph_total",  # Sort by price initially
            "--limit", "15"          # Get more results for smart selection
        ]
        
        offers = self._run_vastai_command(cmd)
        
        # Apply smart selection algorithm
        return self._smart_select_offers(offers)
    
    def _smart_select_offers(self, offers: List[Dict[str, Any]], 
                           price_tolerance: float = 0.5) -> List[Dict[str, Any]]:
        """
        Apply smart selection algorithm to rank offers by value, not just price.
        
        Args:
            offers: List of offers from VastAI API
            price_tolerance: Price tolerance factor (0.5 = 50% above cheapest)
            
        Returns:
            List of offers sorted by smart selection score
        """
        if not offers:
            return offers
            
        # Find the cheapest offer price
        cheapest_price = min(offer.get('dph_total', float('inf')) for offer in offers)
        price_threshold = cheapest_price * (1 + price_tolerance)
        
        # Score each offer
        scored_offers = []
        for offer in offers:
            score = self._calculate_offer_score(offer, cheapest_price, price_threshold)
            scored_offers.append((offer, score))
        
        # Sort by score (highest first) and return just the offers
        scored_offers.sort(key=lambda x: x[1], reverse=True)
        return [offer for offer, score in scored_offers]
    
    def _calculate_offer_score(self, offer: Dict[str, Any], 
                              cheapest_price: float, 
                              price_threshold: float) -> float:
        """
        Calculate a smart selection score for an offer.
        
        Args:
            offer: Individual offer from VastAI API
            cheapest_price: Price of the cheapest available offer
            price_threshold: Maximum price to consider in this tier
            
        Returns:
            Score (higher is better)
        """
        price = offer.get('dph_total', float('inf'))
        
        # If price is too high, give very low score
        if price > price_threshold:
            return 0.0
        
        # Base metrics
        dlperf_per_dollar = offer.get('dlperf_per_dphtotal', 0)  # Primary value metric
        reliability = offer.get('reliability2', 0)
        
        # Additional performance metrics for slight bonuses
        cpu_cores_effective = offer.get('cpu_cores_effective', 0)
        cpu_ghz = offer.get('cpu_ghz', 0)
        gpu_mem_bw = offer.get('gpu_mem_bw', 0)  # GPU memory bandwidth
        pcie_bw = offer.get('pcie_bw', 0)  # PCIe bandwidth
        cpu_ram = offer.get('cpu_ram', 0)  # CPU RAM in MB
        
        # Calculate score components
        # DLPerf per dollar is the primary factor (higher is better)
        perf_per_dollar_score = dlperf_per_dollar * 1.0  # Main weight
        
        # Reliability bonus (slight bonus for more reliable hosts)
        reliability_bonus = (reliability - 0.95) * 20.0 if reliability > 0.95 else 0.0
        
        # CPU performance bonus (slight bonus for better CPU)
        cpu_score = min(cpu_cores_effective * cpu_ghz / 100, 5.0)  # Capped at 5 points
        
        # Memory bandwidth bonus (slight bonus for faster GPU memory)
        gpu_mem_bonus = min(gpu_mem_bw / 1000, 3.0)  # Capped at 3 points
        
        # PCIe bandwidth bonus (slight bonus for faster PCIe)
        pcie_bonus = min(pcie_bw / 16, 2.0)  # Capped at 2 points
        
        # CPU RAM bonus (slight bonus for more system RAM)
        cpu_ram_gb = cpu_ram / 1024  # Convert MB to GB
        ram_bonus = min(cpu_ram_gb / 32, 2.0)  # Capped at 2 points
        
        # Price preference - prefer cheaper within the tolerance
        # Normalize price difference to 0-1 range within tolerance
        max_price_diff = price_threshold - cheapest_price
        if max_price_diff > 0:
            price_factor = 1.0 - ((price - cheapest_price) / max_price_diff)
            price_bonus = price_factor * 10.0  # Small bonus for being cheaper
        else:
            price_bonus = 0.0
        
        # Total score
        total_score = (
            perf_per_dollar_score +  # Primary: performance per dollar
            reliability_bonus +      # Secondary: reliability above minimum
            price_bonus +            # Tertiary: price preference within tolerance
            cpu_score +              # Small bonus: CPU performance
            gpu_mem_bonus +          # Small bonus: GPU memory bandwidth
            pcie_bonus +             # Small bonus: PCIe bandwidth
            ram_bonus                # Small bonus: System RAM
        )
        
        return max(0.0, total_score)  # Ensure non-negative score
    
    def create_instance(self, offer_id: int) -> Dict[str, Any]:
        """
        Create a new instance from an offer using the custom experimance template.
        
        Args:
            offer_id: The offer ID to rent
            
        Returns:
            Instance creation result
        """
        # Build environment string with all env vars and port mappings
        env_parts = []
        
        # Add environment variables
        for key, value in self.required_env_vars.items():
            env_parts.append(f"-e {key}={value}")
        
        # Add port mappings
        # FIXME: these are already prt of the template, so not needed here
        #env_parts.append("-p 1111:1111/tcp")
        #env_parts.append("-p 8000:8000/tcp")
        
        env_string = " ".join(env_parts)
        
        # Build the command
        cmd = [
            "create", "instance", str(offer_id),
            "--template_hash", self.experimance_template_id,
            "--disk", str(self.disk_size),
            "--env", env_string,
            "--ssh"
        ]
        
        return self._run_vastai_command(cmd)
    
    def stop_instance(self, instance_id: int) -> Dict[str, Any]:
        """
        Stop an instance (keeps it allocated but stops billing).
        
        Args:
            instance_id: The instance ID to stop
            
        Returns:
            Stop result
        """
        cmd = ["stop", "instance", str(instance_id)]
        return self._run_vastai_command(cmd)
    
    def start_instance(self, instance_id: int) -> Dict[str, Any]:
        """
        Start a stopped instance.
        
        Args:
            instance_id: The instance ID to start
            
        Returns:
            Start result
        """
        cmd = ["start", "instance", str(instance_id)]
        return self._run_vastai_command(cmd)
    
    def restart_instance(self, instance_id: int) -> Dict[str, Any]:
        """
        Restart an instance (stop then start).
        
        Args:
            instance_id: The instance ID to restart
            
        Returns:
            Restart result
        """
        logger.info(f"Restarting instance {instance_id} (stop then start)")
        
        # Stop the instance first
        logger.info(f"Stopping instance {instance_id}...")
        stop_result = self.stop_instance(instance_id)
        
        # Wait a moment for the stop to take effect
        import time
        time.sleep(5)
        
        # Start the instance
        logger.info(f"Starting instance {instance_id}...")
        start_result = self.start_instance(instance_id)
        
        # Return the start result as it's the final operation
        return start_result
    
    def destroy_instance(self, instance_id: int) -> Dict[str, Any]:
        """
        Destroy an instance (terminates and stops billing).
        
        Args:
            instance_id: The instance ID to destroy
            
        Returns:
            Destroy result
        """
        cmd = ["destroy", "instance", str(instance_id)]
        return self._run_vastai_command(cmd)
    
    def find_or_create_instance(self, 
                               create_if_none: bool = True,
                               wait_for_ready: bool = True,
                               provision_existing: bool = False,
                               disable_scp_provisioning: bool = False,
                               min_gpu_ram: int = 16,
                               max_price: float = 0.5,
                               dlperf: float = 32.0) -> Optional[InstanceEndpoint]:
        """
        Find an existing experimance instance or create a new one.
        
        Args:
            create_if_none: Create a new instance if none found
            wait_for_ready: Wait for the instance to be fully ready
            provision_existing: Whether to run SCP provisioning on existing instances
            disable_scp_provisioning: If True, skip SCP provisioning fallback (useful for testing PROVISIONING_SCRIPT env var)
            min_gpu_ram: Minimum GPU RAM in GB for new instances
            max_price: Maximum price per hour for new instances
            dlperf: Minimum DLPerf score for new instances
            
        Returns:
            InstanceEndpoint for the ready instance, or None
        """
        # First, check for existing instances
        existing_instances = self.find_experimance_instances()
        
        if existing_instances:
            instance = existing_instances[0]  # Use the first one
            instance_id = instance["id"]
            logger.info(f"Found existing instance {instance_id}")
            
            if wait_for_ready:
                if self.wait_for_instance_ready(instance_id):
                    # Optionally provision existing instances
                    if provision_existing:
                        logger.info(f"Provisioning existing instance {instance_id}")
                        
                        # Wait for SSH to be ready before provisioning
                        if self.wait_for_ssh_ready(instance_id, timeout=180):  # 3 minutes for SSH
                            if self.provision_instance_via_scp(instance_id):
                                logger.info("SCP provisioning of existing instance completed successfully")
                            else:
                                logger.warning("SCP provisioning of existing instance failed, but continuing anyway")
                        else:
                            logger.warning(f"SSH not ready for existing instance {instance_id}, skipping provisioning")
                    elif disable_scp_provisioning:
                        logger.info(f"SCP provisioning disabled for testing - relying on PROVISIONING_SCRIPT environment variable")
                    
                    return self.get_model_server_endpoint(instance_id)
                else:
                    logger.error(f"Existing instance {instance_id} is not responding")
                    return None
            else:
                return self.get_model_server_endpoint(instance_id)
        
        if not create_if_none:
            logger.info("No existing instances found and create_if_none=False")
            return None
        
        # Create a new instance
        logger.info("No existing instances found, creating new one...")
        
        # Search for offers with specified criteria
        offers = self.search_offers(
            min_gpu_ram=min_gpu_ram,
            max_price=max_price,
            dlperf=dlperf
        )
        if not offers:
            logger.error("No suitable offers found")
            return None
        
        # Use the cheapest offer
        best_offer = offers[0]
        offer_id = best_offer["id"]
        price = best_offer["dph_total"]
        gpu_name = best_offer["gpu_name"]
        
        logger.info(f"Creating instance with offer {offer_id}: {gpu_name} at ${price:.3f}/hour")
        
        # Create the instance
        result = self.create_instance(offer_id)
        instance_id = result.get("new_contract")
        
        if not instance_id:
            logger.error(f"Failed to create instance: {result}")
            return None
        
        logger.info(f"Created instance {instance_id}, waiting for it to be ready...")
        
        if wait_for_ready:
            if self.wait_for_instance_ready(instance_id):
                if disable_scp_provisioning:
                    logger.info(f"Instance {instance_id} is ready. SCP provisioning disabled - relying on PROVISIONING_SCRIPT environment variable")
                    # Wait for the provisioning script to complete and service to become healthy
                    logger.info("Waiting for PROVISIONING_SCRIPT to complete and service to become healthy...")
                    if self._wait_for_service_healthy(instance_id, timeout=180):  # 3 minutes for provisioning
                        logger.info("Service is healthy - PROVISIONING_SCRIPT completed successfully")
                    else:
                        logger.warning("Service did not become healthy within timeout - PROVISIONING_SCRIPT may have failed")
                else:
                    # Give PROVISIONING_SCRIPT a chance to work first, then fall back to SCP if needed
                    logger.info(f"Instance {instance_id} is ready. Waiting for PROVISIONING_SCRIPT to complete...")

                    # Wait for the service to become healthy first (PROVISIONING_SCRIPT should handle this)
                    if self._wait_for_service_healthy(instance_id, timeout=180):  # 3 minutes for PROVISIONING_SCRIPT
                        logger.info("Service became healthy - PROVISIONING_SCRIPT worked successfully, skipping SCP fallback")
                    else:
                        logger.warning("Service not healthy after 180s, falling back to SCP provisioning...")
                        
                        # Now wait for SSH to be ready before attempting SCP
                        if self.wait_for_ssh_ready(instance_id, timeout=60):  # 1 minute for SSH
                            logger.info("SSH is ready, running provisioning script via SCP...")
                            
                            if self.provision_instance_via_scp(instance_id):
                                logger.info("SCP provisioning completed successfully")
                            else:
                                logger.warning("SCP provisioning failed, but continuing anyway")
                        else:
                            logger.error("SSH not ready, cannot run SCP fallback provisioning")

                return self.get_model_server_endpoint(instance_id)
            else:
                logger.error(f"New instance {instance_id} failed to become ready")
                return None
        else:
            return self.get_model_server_endpoint(instance_id)


def main():
    """Example usage of the VastAI manager."""
    import argparse
    
    logging.basicConfig(level=logging.DEBUG)
    
    parser = argparse.ArgumentParser(description="VastAI Manager for Experimance")
    parser.add_argument("--provision-existing", action="store_true", 
                       help="Force provisioning of existing instances")
    parser.add_argument("--provision-instance", type=int, metavar="INSTANCE_ID",
                       help="Manually provision a specific instance by ID")
    parser.add_argument("--create", action="store_true",
                       help="Create new instance if none found")
    parser.add_argument("--provision-script", type=str, metavar="URL",
                       help="Custom provisioning script URL to use instead of default")
    
    args = parser.parse_args()
    
    manager = VastAIManager(provisioning_script_url=args.provision_script)

    if args.provision_instance:
        print(f"Manually provisioning instance {args.provision_instance}...")
        if manager.provision_existing_instance(args.provision_instance):
            print("✅ Provisioning completed successfully")
        else:
            print("❌ Provisioning failed")
        return

    # Find or create an instance
    print("Finding or creating experimance instance...")
    endpoint = manager.find_or_create_instance(
        create_if_none=args.create, 
        provision_existing=args.provision_existing
    )
    
    if endpoint:
        print(f"✅ Model server ready at: {endpoint.url}")
        print(f"   Instance ID: {endpoint.instance_id}")
        print(f"   Public IP: {endpoint.public_ip}")
        print(f"   External Port: {endpoint.external_port}")
        print(f"   Status: {endpoint.status}")
        
        # Test the health endpoint
        try:
            import requests
            response = requests.get(f"{endpoint.url}/healthcheck", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health check passed")
                print(f"   Models loaded: {health_data.get('models_loaded', [])}")
                print(f"   Uptime: {health_data.get('uptime', 0):.1f}s")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to test health endpoint: {e}")
    else:
        print("❌ Failed to get a ready instance")


if __name__ == "__main__":
    main()
