"""
Schema definitions for Experimance inter-service messages.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


class Era(str, Enum):
    """Era definitions for the Experimance installation.
    
    These values must match the 'eras' array in data/experimance_config.json
    """
    WILDERNESS = "wilderness"
    PRE_INDUSTRIAL = "pre_industrial"
    EARLY_INDUSTRIAL = "early_industrial"
    LATE_INDUSTRIAL = "late_industrial"
    MODERN = "modern"
    CURRENT = "current"
    FUTURE = "future"
    DYSTOPIA = "dystopia"
    RUINS = "ruins"


class Biome(str, Enum):
    """Biome definitions for the Experimance installation.
    
    These values must match the 'biomes' array in data/experimance_config.json
    """
    RAINFOREST = "rainforest"
    TEMPERATE_FOREST = "temperate_forest"
    BOREAL_FOREST = "boreal_forest"
    DECIDUOUS_FOREST = "deciduous_forest"
    DESERT = "desert"
    MOUNTAIN = "mountain"
    TROPICAL_ISLAND = "tropical_island"
    RIVER = "river"
    TUNDRA = "tundra"
    STEPPE = "steppe"
    COASTAL = "coastal"
    SWAMP = "swamp"
    PLAINS = "plains"
    ARCTIC = "arctic"
    JUNGLE = "jungle"


class TransitionStyle(str, Enum):
    """Transition style definitions."""
    DISSOLVE = "dissolve"
    MORPH = "morph"
    WIPE = "wipe"
    SIMPLE = "simple"


class DisplayContentType(str, Enum):
    """Content types for display media."""
    IMAGE = "image"
    IMAGE_SEQUENCE = "image_sequence"
    VIDEO = "video"


class DisplayTransitionType(str, Enum):
    """Transition types for display media."""
    NONE = "none"                    # No transition, direct display
    FADE = "fade"                    # Simple fade transition
    DISSOLVE = "dissolve"            # Cross-dissolve
    SLIDE = "slide"                  # Slide transition
    MORPH = "morph"                  # Morphing transition
    IMAGE_SEQUENCE = "image_sequence"  # Play image sequence
    VIDEO = "video"                  # Play video transition


class MessageSchema(BaseModel):
    """Base schema for all messages.
    
    Provides helpful utility methods for accessing attributes.
    """
    
    def get(self, key: str, missing_value: Optional[Any] = None) -> Any:
        """
        Return the value for a given attribute, or None if not found.
        Follows doct.get() semantics.
        
        Returns:
            Value of the attribute or None if not found
        """
        return getattr(self, key, missing_value)

    # allow for "key in MessageBase"
    def __contains__(self, key: str) -> bool:
        """
        Check if the message contains a specific key.
        
        Args:
            key: The key to check for in the message
            
        Returns:
            True if the key exists, False otherwise
        """
        return hasattr(self, key)

    def __getitem__(self, key: str) -> Any:
        """
        Get an attribute value using dictionary-style access.
        
        Args:
            key: The attribute name to access
            
        Returns:
            The value of the attribute
            
        Raises:
            KeyError: If the attribute doesn't exist
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set an attribute value using dictionary-style access.
        
        Args:
            key: The attribute name to set
            value: The value to set
        """
        setattr(self, key, value)

class MessageBase(MessageSchema):
    """Base class for all message types."""
    type: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Union[Dict[str, Any], 'MessageBase']:
        """
        Create the appropriate message object based on the 'type' field in the data.
        
        Args:
            data: Dictionary data from JSON deserialization with a 'type' field
            
        Returns:
            Either the original dict (if type unknown) or the appropriate MessageBase subclass instance
        """
        # Get all MessageBase subclasses
        subclasses = cls._get_all_subclasses()
        
        # Create a mapping from type strings to classes
        type_to_class = {}
        for subclass in subclasses:
            # Get the default value for the 'type' field
            if hasattr(subclass, '__fields__') and 'type' in subclass.__fields__:
                field_info = subclass.__fields__['type']
                if hasattr(field_info, 'default') and field_info.default:
                    type_to_class[field_info.default] = subclass
            # Fallback: try to create an instance and get the type
            elif hasattr(subclass, 'model_fields') and 'type' in subclass.model_fields:
                field_info = subclass.model_fields['type']
                if hasattr(field_info, 'default') and field_info.default:
                    type_to_class[field_info.default] = subclass
        
        message_type = data.get("type")
        if message_type and message_type in type_to_class:
            schema_class = type_to_class[message_type]
            try:
                return schema_class(**data)
            except Exception:
                # If conversion fails, return original dict
                return data
        else:
            # Unknown message type, return as dict
            return data

    @classmethod
    def to_message_type(cls, data: Union[Dict[str, Any], 'MessageBase'], target_class=None) -> 'MessageBase':
        """
        Convert MessageDataType to a specific message type.
        
        Args:
            data: Message data (dict or MessageBase instance)
            target_class: Target MessageBase subclass to convert to. If None, uses the calling class.
            
        Returns:
            Instance of target_class if conversion successful, None otherwise.
            When called on a subclass (e.g. RenderRequest.to_message_type(data)), 
            returns an instance of that subclass type.
            
        Raises:
            ValidationError: If the data doesn't match the target schema
            
        Example:
            # Convert to RenderRequest using the class method
            try:
                render_request = RenderRequest.to_message_type(message_data)  # type: RenderRequest | None
            except ValidationError as e:
                logger.error(f"Invalid RenderRequest data: {e}")
            
            # Or specify target class explicitly (old way still works)
            render_request = MessageBase.to_message_type(message_data, RenderRequest)
        """
        # If no target_class specified, use the calling class
        if target_class is None:
            target_class = cls
        
        # If data is already the target type, return it directly
        if type(data).__name__ == target_class.__name__ and isinstance(data, MessageBase):
            return data
        # Convert dict to target class  
        elif isinstance(data, dict):
            return target_class(**data)
        # Convert MessageBase to dict then target class
        elif isinstance(data, MessageBase):
            data_dict = data.model_dump()
            return target_class(**data_dict)
        else:
            raise TypeError(f"Cannot convert {type(data).__name__} to {target_class.__name__}")
    
    @classmethod 
    def _get_all_subclasses(cls):
        """Recursively get all subclasses of MessageBase."""
        subclasses = set(cls.__subclasses__())
        for subclass in list(subclasses):
            subclasses.update(subclass._get_all_subclasses())
        return subclasses


class EraChanged(MessageBase):
    """Event published when the era changes."""
    type: str = "EraChanged"
    era: Era
    biome: Biome


# used as a mix-in for schemas that have generated images
class ImageSource(MessageSchema):
    """Base class for image sources."""
    image_data: Optional[str] = None  # Base64 encoded PNG
    uri: Optional[str] = None
    _temp_file: Optional[str] = None  # Denotes if temporary file path

class RenderRequest(MessageBase):
    """Request to generate a new image."""
    type: str = "RenderRequest"
    
    request_id: str  # Unique identifier for tracking the request
    era: Era
    biome: Biome
    prompt: str
    negative_prompt: Optional[str] = None
    style: Optional[str] = None  # Optional style hint
    seed: Optional[int] = None
    reference_image: Optional[ImageSource] = None  # Optional reference image to guide generation
    depth_map: Optional[ImageSource] = None  # Optional depth map URI for depth-aware generation

class IdleStatus(MessageBase):
    """Event published when the idle status changes."""
    type: str = "Idle"
    status: bool  # True = now idle, False = exiting idle


class ImageReady(MessageBase):
    """Event published when a new image is ready."""
    type: str = "ImageReady"
    request_id: str
    uri: str  # URI to the image (file://, http://, etc.)


class TransitionReady(MessageBase):
    """Event published when a transition video/sequence is ready."""
    type: str = "TransitionReady"
    transition_id: str
    uri: str  # URI to video or image sequence manifest
    is_video: bool  # True if URI points to video file, False if image sequence
    loop: bool = False  # Always False for transitions
    final_frame_uri: str  # URI to the final frame to display after transition


class LoopReady(MessageBase):
    """Event published when a loop animation is ready."""
    type: str = "LoopReady"
    loop_id: str
    uri: str  # URI to video or image sequence manifest
    is_video: bool  # True if URI points to video file, False if image sequence
    duration_s: Optional[float] = None  # Duration in seconds if known


class AgentControlEventPayload(BaseModel):
    """Base class for agent control event payloads."""
    pass


class SuggestBiomePayload(AgentControlEventPayload):
    """Payload for suggesting a biome."""
    biome_suggestion: Biome


class AudiencePresentPayload(AgentControlEventPayload):
    """Payload for audience presence detection."""
    status: bool


class SpeechDetectedPayload(AgentControlEventPayload):
    """Payload for speech detection."""
    is_speaking: bool


class AgentControlEvent(MessageBase):
    """Event published by the agent to control other services."""
    type: str = "AgentControlEvent"
    sub_type: str  # "SuggestBiome", "AudiencePresent", "SpeechDetected"
    payload: Dict  # Structure varies based on sub_type


class TransitionRequest(MessageBase):
    """Request to generate a transition between two images."""
    type: str = "TransitionRequest"
    request_id: str
    from_image_uri: str  # URI to the source image
    to_image_uri: str  # URI to the destination image
    style: TransitionStyle = TransitionStyle.DISSOLVE
    duration_frames: int  # Number of frames for the transition


class LoopRequest(MessageBase):
    """Request to generate a loop animation from a still image."""
    type: str = "LoopRequest"
    request_id: str
    still_image_uri: str  # URI to the still image to animate
    style: str  # Hint for the animation style


class ContentType(str, Enum):
    """Types of content that can be displayed."""
    IMAGE = "image"                    # Single static image
    IMAGE_SEQUENCE = "image_sequence"  # Sequence of images (for transitions)
    VIDEO = "video"                    # Video file

    def __str__(self):
        """Return the string representation of the content type."""
        return self.value
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, str):
            return self.value == value
        return super().__eq__(value)

class DisplayMedia(MessageBase, ImageSource):
    """Message for sending media content to display service."""
    type: str = "DisplayMedia"
    
    # Content specification
    content_type: ContentType
    request_id: Optional[str] = None      # Unique identifier tracking the request through pipeline
    
    # For IMAGE content_type (now part of ImageSource)
    # image_data: Optional[Any] = None      # Image data (numpy array, PIL, etc.)
    # uri: Optional[str] = None             # File URI for image
    
    # For IMAGE_SEQUENCE content_type  
    sequence_path: Optional[str] = None   # Path to directory with numbered images
    
    # For VIDEO content_type
    video_path: Optional[str] = None      # Path to video file
    
    # Display properties (override defaults in display service)
    duration: Optional[float] = None      # Duration in seconds (for sequences/videos)
    loop: bool = False                    # Whether to loop the content
    fade_in: Optional[float] = None       # Fade in duration in seconds
    fade_out: Optional[float] = None      # Fade out duration in seconds
    
    # Context information
    era: Optional[Era] = None
    biome: Optional[Biome] = None
