"""
OSC bridge for communication between the Experimance Audio Service and SuperCollider.

This module provides functionality to send OSC messages to SuperCollider,
enabling control of the audio engine from the Experimance system.
"""

import logging
from pythonosc import udp_client
from typing import Any, Dict, List, Optional, Union

from experimance_common.constants import DEFAULT_PORTS

logger = logging.getLogger(__name__)



class OscBridge:
    """Bridge for sending OSC messages to SuperCollider."""
    
    host: str
    port: int
    client: Optional[udp_client.SimpleUDPClient] = None

    def __init__(self, host: str = "localhost", port: int = DEFAULT_PORTS["audio_osc_send_port"]):
        """Initialize the OSC bridge.
        
        Args:
            host: SuperCollider host address
            port: SuperCollider OSC listening port
        """
        self.host = host
        self.port = port
        self.client = None
        self._connect()
    
    def _connect(self) -> bool:
        """Initialize the OSC client connection.
        
        Returns:
            bool: True if connection was successful, False otherwise
        """
        try:
            self.client = udp_client.SimpleUDPClient(self.host, self.port)
            logger.info(f"OSC client initialized at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Error initializing OSC client: {e}")
            return False
    
    def send_spacetime(self, biome: str, era: str) -> bool:
        """Send spacetime context to SuperCollider.
        
        Args:
            biome: Current biome name
            era: Current era name
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/spacetime", [biome, era])
            logger.debug(f"Sent spacetime: biome={biome}, era={era}")
            return True
        except Exception as e:
            logger.error(f"Error sending spacetime OSC message: {e}")
            return False
    
    def include_tag(self, tag: str) -> bool:
        """Include a sound tag in the active set.
        
        Args:
            tag: Tag to include
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/include", [tag])
            logger.debug(f"Sent include tag: {tag}")
            return True
        except Exception as e:
            logger.error(f"Error sending include OSC message: {e}")
            return False
    
    def exclude_tag(self, tag: str) -> bool:
        """Exclude a sound tag from the active set.
        
        Args:
            tag: Tag to exclude
            
        Returns:
            bool: True if message was sent successfully
        """
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/exclude", [tag])
            logger.debug(f"Sent exclude tag: {tag}")
            return True
        except Exception as e:
            logger.error(f"Error sending exclude OSC message: {e}")
            return False
    
    def listening(self, start: bool) -> bool:
        """Trigger listening UI sound effect.
        
        Args:
            start: True to start, False to stop
            
        Returns:
            bool: True if message was sent successfully
        """
        status = "start" if start else "stop"
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/listening", [status])
            logger.debug(f"Sent listening: {status}")
            return True
        except Exception as e:
            logger.error(f"Error sending listening OSC message: {e}")
            return False
    
    def speaking(self, start: bool) -> bool:
        """Trigger speaking UI sound effect.
        
        Args:
            start: True to start, False to stop
            
        Returns:
            bool: True if message was sent successfully
        """
        status = "start" if start else "stop"
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/speaking", [status])
            logger.debug(f"Sent speaking: {status}")
            return True
        except Exception as e:
            logger.error(f"Error sending speaking OSC message: {e}")
            return False
    
    def transition(self, start: bool) -> bool:
        """Trigger transition sound effect.
        
        Args:
            start: True to start, False to stop
            
        Returns:
            bool: True if message was sent successfully
        """
        status = "start" if start else "stop"
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/transition", [status])
            logger.debug(f"Sent transition: {status}")
            return True
        except Exception as e:
            logger.error(f"Error sending transition OSC message: {e}")
            return False
    
    def reload_configs(self) -> bool:
        """Trigger configuration reload in SuperCollider.
        
        Returns:
            bool: True if message was sent successfully
        """
        try:
            assert self.client is not None, "OSC client is not initialized"
            self.client.send_message("/reload", [])
            logger.info("Sent reload command to SuperCollider")
            return True
        except Exception as e:
            logger.error(f"Error sending reload OSC message: {e}")
            return False
