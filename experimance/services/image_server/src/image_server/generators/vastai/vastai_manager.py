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
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the VastAI manager.
        
        Args:
            api_key: Optional API key. If not provided, will use vastai CLI auth.
        """
        self.api_key = api_key or os.getenv("VASTAI_API_KEY")
        self.experimance_template_id = "e77f3f3425011b283b73c093feb2d600"
        self.experimance_template_name = "PyTorch (Vast) web accessible"  # Custom template name
        self.required_env_vars = {
            "GITHUB_ACCESS_TOKEN": os.getenv("GITHUB_ACCESS_TOKEN", ""),
            "PROVISIONING_SCRIPT": "https://gist.githubusercontent.com/RKelln/21ad3ecb4be1c1d0d55a8f1524ff9b14/raw/vast_experimance_provisioning.sh"
        }
        self.disk_size = 20 # Gigabytes
        
    def _run_vastai_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a vastai CLI command and return parsed JSON result."""
        try:
            command_args = ["--raw", "--api-key", self.api_key] if self.api_key else ["--raw"]
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
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"VastAI command failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse VastAI response: {e}")
            logger.error(f"$ {' '.join(command)}: {result.stdout}")
            raise
    
    def show_instances(self) -> List[Dict[str, Any]]:
        """List all instances for the current user."""
        return self._run_vastai_command(["show", "instances"])
    
    def show_instance(self, instance_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific instance."""
        return self._run_vastai_command(["show", "instance", str(instance_id)])
    
    def find_experimance_instances(self) -> List[Dict[str, Any]]:
        """Find all running experimance instances."""
        instances = self.show_instances()
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
        Wait for an instance to be running and the model server to be accessible.
        
        Args:
            instance_id: The instance ID to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if instance is ready, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                endpoint = self.get_model_server_endpoint(instance_id)
                
                if endpoint and endpoint.status == "running":
                    # Test if the model server is responding
                    try:
                        response = requests.get(
                            f"{endpoint.url}/healthcheck",
                            timeout=10
                        )
                        if response.status_code == 200:
                            logger.info(f"Instance {instance_id} is ready at {endpoint.url}")
                            return True
                    except requests.RequestException:
                        # Server not ready yet, continue waiting
                        pass
                
                logger.info(f"Waiting for instance {instance_id} to be ready...")
                time.sleep(30)
                
            except Exception as e:
                logger.warning(f"Error checking instance {instance_id}: {e}")
                time.sleep(30)
        
        logger.error(f"Timeout waiting for instance {instance_id} to be ready")
        return False
    
    def search_offers(self, 
                     min_gpu_ram: int = 20,  # 20GB minimum for SDXL
                     max_price: float = 0.5,     # $0.50/hour max
                     dlperf: float = 32.0,       # 3090 +
                     verified_only: bool = True) -> List[Dict[str, Any]]:
        """
        Search for suitable GPU offers.
        
        Args:
            min_gpu_ram: Minimum GPU RAM in MB
            max_price: Maximum price per hour
            gpu_name: Preferred GPU name pattern
            verified_only: Only show verified hosts
            
        Returns:
            List of available offers
        """
        cmd = [
            "search", "offers",
            "'"
            f"reliability>0.95",
            f"gpu_ram>={min_gpu_ram}",
            f"dph_total<={max_price}",
            f"dlperf>={dlperf}",
            f"num_gpus=1",
            f"geolocation in [CA,US]",
            f"inet_down>250",
            f"cuda_vers>=12.2"
        ]
        
        if verified_only:
            cmd.append("verified = true")

        cmd.append("'")
        
        cmd.extend([
            "--order", "dph_total",  # Sort by price
            "--limit", "10"          # Top 10 results
        ])
        
        return self._run_vastai_command(cmd)
    
    def create_instance(self, offer_id: int) -> Dict[str, Any]:
        """
        Create a new instance from an offer using the custom experimance template.
        
        Args:
            offer_id: The offer ID to rent
            
        Returns:
            Instance creation result
        """
        # Prepare environment variables
        env_args = []
        for key, value in self.required_env_vars.items():
            env_args.extend(["--env", f"{key}={value}"])
        
        # Add port mappings
        port_args = [
            "--port", "1111:1111/tcp",   # Instance Portal
            "--port", "8000:8000/tcp"    # Model Server
        ]
        
        cmd = [
            "create", "instance", str(offer_id),
            "--template-id", self.experimance_template_id,  # Use our custom template
            "--disk", self.disk_size  # 40GB disk
        ] + env_args + port_args
        
        return self._run_vastai_command(cmd)
    
    def find_or_create_instance(self, 
                               create_if_none: bool = True,
                               wait_for_ready: bool = True) -> Optional[InstanceEndpoint]:
        """
        Find an existing experimance instance or create a new one.
        
        Args:
            create_if_none: Create a new instance if none found
            wait_for_ready: Wait for the instance to be fully ready
            
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
        
        # Search for offers
        offers = self.search_offers()
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
                return self.get_model_server_endpoint(instance_id)
            else:
                logger.error(f"New instance {instance_id} failed to become ready")
                return None
        else:
            return self.get_model_server_endpoint(instance_id)


def main():
    """Example usage of the VastAI manager."""
    logging.basicConfig(level=logging.DEBUG)
    
    manager = VastAIManager()

    # Find or create an instance
    print("Finding or creating experimance instance...")
    endpoint = manager.find_or_create_instance(create_if_none=False)
    
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
