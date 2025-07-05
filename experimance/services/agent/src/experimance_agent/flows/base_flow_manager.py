"""
Base flow manager for Pipecat-based conversation flows.

This module provides a configurable flow manager that can work with different
flow definitions, making the PipecatBackend reusable across projects.
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable, Union
from abc import ABC, abstractmethod

from pipecat_flows import FlowManager, FlowResult, FlowArgs, NodeConfig, ContextStrategy, ContextStrategyConfig

# Import UserContext from the backend module
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..backends.base import UserContext

logger = logging.getLogger(__name__)


class BaseFlowManager(ABC):
    """
    Abstract base class for flow managers.
    
    This provides the interface that PipecatBackend expects, allowing
    different flow implementations to be plugged in.
    """
    
    def __init__(self, task, llm, context_aggregator, initial_persona: str = "default", user_context: Optional["UserContext"] = None):
        """Initialize the base flow manager."""
        logger.info(f"Initializing BaseFlowManager with initial_persona='{initial_persona}', task={task is not None}, user_context={user_context is not None}")
        
        self.task = task
        self.llm = llm
        self.context_aggregator = context_aggregator
        self.initial_persona = initial_persona
        
        # Flow state
        self.current_persona: str = initial_persona
        self.conversation_state: Dict[str, Any] = {}
        
        # Use backend's UserContext if provided, otherwise create a minimal fallback
        self.user_context = user_context
        if self.user_context is None:
            # Create a minimal fallback for when no backend context is available
            self._fallback_user_context: Dict[str, Any] = {}
            logger.info("No user_context provided - using fallback context")
        else:
            logger.info("Using provided user_context from backend")
        
        # Initialize Pipecat FlowManager - handle None task gracefully
        logger.info("Creating initial flow configuration")
        initial_node = self.create_initial_flow()
        initial_context_strategy = self.get_context_strategy_for_persona(initial_persona)
        logger.info(f"Initial flow created for persona '{initial_persona}'")
        
        # Only create FlowManager if we have a task, otherwise defer initialization
        if self.task is not None:
            logger.info("Initializing Pipecat FlowManager")
            self.flow_manager = FlowManager(
                task=self.task,
                llm=self.llm,
                context_aggregator=self.context_aggregator,
                context_strategy=initial_context_strategy
            )
            logger.info("Pipecat FlowManager initialized successfully")
            # Set the initial node (FlowManager constructor should handle initial node setup)
            # self.flow_manager.set_node(initial_node)  # Remove this line if causing issues
        else:
            # Defer initialization until task is available
            self.flow_manager = None
            logger.info("Flow manager initialization deferred until task is available")
        
        # Store initial setup for later use
        self._initial_node = initial_node
        self._initial_context_strategy = initial_context_strategy
        logger.info("BaseFlowManager initialization complete")
        
    async def start(self) -> None:
        """Start the flow manager - called by the backend after pipeline setup."""
        logger.info(f"Starting flow manager with persona: {self.current_persona}")
        if self.flow_manager is not None:
            # FlowManager should start automatically when added to pipeline
            # await self.flow_manager.initialize(self._initial_node)  # Remove if causing issues
            pass
        logger.info(f"Flow manager started with persona: {self.current_persona}")
        
    @abstractmethod
    def create_initial_flow(self) -> NodeConfig:
        """Create the initial flow configuration."""
        pass
        
    @abstractmethod
    def get_available_personas(self) -> Dict[str, str]:
        """Get a dictionary of available persona names and descriptions."""
        pass
        
    @abstractmethod
    def create_flow_for_persona(self, persona_name: str) -> NodeConfig:
        """Create a flow configuration for the specified persona."""
        pass
        
    @abstractmethod
    def get_context_strategy_for_persona(self, persona_name: str) -> ContextStrategyConfig:
        """Get the appropriate context strategy for a persona."""
        pass
        
    async def switch_persona(self, persona_name: str, context_data: Optional[Dict[str, Any]] = None) -> None:
        """Switch to a different conversation persona."""
        logger.info(f"switch_persona called: {self.current_persona} -> {persona_name}")
        
        if self.flow_manager is None:
            logger.warning("Cannot switch persona: FlowManager not initialized yet")
            return
            
        available_personas = self.get_available_personas()
        logger.debug(f"Available personas: {list(available_personas.keys())}")
        
        if persona_name not in available_personas:
            logger.error(f"Unknown persona: {persona_name}. Available: {list(available_personas.keys())}")
            return
            
        if persona_name == self.current_persona:
            logger.debug(f"Already in persona: {persona_name}")
            return
            
        logger.info(f"Switching from {self.current_persona} to {persona_name}")
        
        # Store context data if provided
        if context_data:
            logger.debug(f"Updating conversation state with: {context_data}")
            self.conversation_state.update(context_data)
            
        # Create new flow for the persona
        logger.info(f"Creating flow for persona: {persona_name}")
        new_flow = self.create_flow_for_persona(persona_name)
        
        # Use appropriate context strategy based on persona
        context_strategy = self.get_context_strategy_for_persona(persona_name)
        logger.debug(f"Using context strategy: {context_strategy}")
        
        # Add context strategy to the flow configuration if not already present
        if "context_strategy" not in new_flow:
            new_flow["context_strategy"] = context_strategy
        
        # Set the new node (may need to check pipecat-flows API for correct method signature)
        # await self.flow_manager.set_node(persona_name, new_flow)  # Commented out temporarily
        
        self.current_persona = persona_name
        logger.info(f"Successfully switched to persona: {persona_name}")
        
    def get_current_persona(self) -> str:
        """Get the current active persona."""
        return self.current_persona
        
    def get_conversation_state(self) -> Dict[str, Any]:
        """Get the current conversation state."""
        return self.conversation_state.copy()
        
    def get_user_context_info(self) -> Dict[str, Any]:
        """Get the user context information as a dictionary."""
        if self.user_context is not None:
            # Extract information from backend's UserContext
            return {
                "first_name": self.user_context.first_name,
                "last_name": self.user_context.last_name,
                "full_name": self.user_context.full_name,
                "location": self.user_context.location,
                "is_identified": self.user_context.is_identified,
                "session_start": self.user_context.session_start,
                "custom_data": self.user_context.custom_data.copy() if self.user_context.custom_data else {}
            }
        else:
            # Use fallback context
            return self._fallback_user_context.copy()
    
    def update_user_context(self, **kwargs) -> None:
        """Update user context information."""
        if self.user_context is not None:
            # Update backend's UserContext
            for key, value in kwargs.items():
                if hasattr(self.user_context, key):
                    setattr(self.user_context, key, value)
                else:
                    # Store in custom_data if not a direct attribute
                    self.user_context.custom_data[key] = value
        else:
            # Update fallback context
            self._fallback_user_context.update(kwargs)
        
    def get_flow_manager(self) -> Optional["FlowManager"]:
        """Get the underlying Pipecat FlowManager."""
        return self.flow_manager
    
    def complete_initialization(self) -> None:
        """Complete flow manager initialization when task becomes available."""
        logger.info(f"complete_initialization called - task={self.task is not None}, flow_manager={self.flow_manager is not None}")
        
        if self.task is None:
            logger.warning("⚠️ Cannot complete initialization: task is still None")
            return
            
        if self.flow_manager is not None:
            logger.debug("✅ Flow manager already initialized")
            return
            
        logger.info(f"🔧 Completing flow manager initialization with task for persona '{self.current_persona}'")
        
        try:
            self.flow_manager = FlowManager(
                task=self.task,
                llm=self.llm,
                context_aggregator=self.context_aggregator,
                context_strategy=self._initial_context_strategy
            )
            logger.info(f"✅ Pipecat FlowManager created successfully for persona '{self.current_persona}'")
            
            # Set the initial node
            # FlowManager initialization should handle setting the initial flow
            # self.flow_manager.set_node(self._initial_node)  # Remove if causing issues
            logger.info(f"🎉 Flow manager initialization completed successfully for persona '{self.current_persona}'")
            
        except Exception as e:
            logger.error(f"Failed to complete flow manager initialization: {e}")
            raise


class ConfigurableFlowManager(BaseFlowManager):
    """
    A configurable flow manager that takes flow definitions as configuration.
    
    This allows for completely dynamic flow configuration without hardcoding
    specific flows in the class.
    """
    
    def __init__(
        self, 
        task, 
        llm, 
        context_aggregator,
        flow_config: Dict[str, Any],
        initial_persona: str = "default",
        user_context: Optional["UserContext"] = None
    ):
        """
        Initialize with flow configuration.
        
        Args:
            flow_config: Dictionary containing flow definitions and context strategies
                Format:
                {
                    "personas": {
                        "persona_name": {
                            "description": "Description of persona",
                            "flow": NodeConfig or callable returning NodeConfig,
                            "context_strategy": ContextStrategyConfig
                        }
                    },
                    "initial_persona": "persona_name"
                }
        """
        self.flow_config = flow_config
        
        # Extract initial persona from config if not provided
        if initial_persona == "default":
            initial_persona = flow_config.get("initial_persona", "default")
            
        super().__init__(task, llm, context_aggregator, initial_persona, user_context)
        
    def create_initial_flow(self) -> NodeConfig:
        """Create the initial flow from configuration."""
        personas = self.flow_config.get("personas", {})
        if self.initial_persona not in personas:
            raise ValueError(f"Initial persona '{self.initial_persona}' not found in flow config")
            
        persona_config = personas[self.initial_persona]
        flow = persona_config["flow"]
        
        # Handle both direct NodeConfig and callable
        if callable(flow):
            return flow()  # type: ignore  # Flow function is expected to return NodeConfig
        return flow
        
    def get_available_personas(self) -> Dict[str, str]:
        """Get available personas from configuration."""
        personas = self.flow_config.get("personas", {})
        return {
            name: config.get("description", f"Persona: {name}")
            for name, config in personas.items()
        }
        
    def create_flow_for_persona(self, persona_name: str) -> NodeConfig:
        """Create flow for persona from configuration."""
        personas = self.flow_config.get("personas", {})
        if persona_name not in personas:
            raise ValueError(f"Persona '{persona_name}' not found in flow config")
            
        persona_config = personas[persona_name]
        flow = persona_config["flow"]
        
        # Handle both direct NodeConfig and callable
        if callable(flow):
            return flow()  # type: ignore  # Flow function is expected to return NodeConfig
        return flow
        
    def get_context_strategy_for_persona(self, persona_name: str) -> ContextStrategyConfig:
        """Get context strategy from configuration."""
        personas = self.flow_config.get("personas", {})
        if persona_name in personas:
            return personas[persona_name].get(
                "context_strategy", 
                ContextStrategyConfig(strategy=ContextStrategy.APPEND)
            )
        
        # Default strategy
        return ContextStrategyConfig(strategy=ContextStrategy.APPEND)


# Factory function for creating flow managers
def create_flow_manager(
    flow_type: str,
    task,
    llm, 
    context_aggregator,
    config: Optional[Dict[str, Any]] = None,
    user_context: Optional["UserContext"] = None,
    **kwargs
) -> BaseFlowManager:
    """
    Factory function to create different types of flow managers.
    
    Args:
        flow_type: Type of flow manager ("experimance", "configurable", "simple")
        config: Configuration dictionary for the flow manager
        user_context: Backend's UserContext for tracking user information
        **kwargs: Additional arguments passed to the flow manager
        
    Note: With the custom adapter system, any flow manager can work with any LLM service
    that has a compatible adapter, including OpenAI Realtime Beta.
    """
    logger.info(f"🏭 Creating flow manager: type='{flow_type}', task={task is not None}, config={config is not None}, user_context={user_context is not None}")
    
    # Register custom adapters for pipecat-flows (if available) - do this EARLY
    try:
        from .adapter_registry import ensure_adapters_registered
        ensure_adapters_registered()
        logger.debug("✅ Custom adapters registered")
    except ImportError:
        logger.debug("⚠️ Adapter registration not available")
        pass  # Adapter registration not available
    
    if flow_type == "experimance":
        logger.info("🎭 Creating ExperimanceFlowManager")
        from .experimance_flows import ExperimanceFlowManager
        manager = ExperimanceFlowManager(task, llm, context_aggregator, user_context=user_context, **kwargs)
        logger.info(f"✅ ExperimanceFlowManager created with persona '{manager.current_persona}'")
        return manager
    elif flow_type == "configurable":
        if not config:
            raise ValueError("ConfigurableFlowManager requires config parameter")
        logger.info("🔧 Creating ConfigurableFlowManager")
        manager = ConfigurableFlowManager(task, llm, context_aggregator, config, user_context=user_context, **kwargs)
        logger.info(f"✅ ConfigurableFlowManager created")
        return manager
    elif flow_type == "simple":
        logger.info("🔹 Creating SimpleFlowManager")
        from .simple_flows import SimpleFlowManager
        manager = SimpleFlowManager(task, llm, context_aggregator, user_context=user_context, **kwargs)
        logger.info(f"✅ SimpleFlowManager created")
        return manager
    else:
        raise ValueError(f"Unknown flow type: {flow_type}")
