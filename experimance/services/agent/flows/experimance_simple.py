#!/usr/bin/env python3
"""
Simple Experimance Flow Configuration

A basic flow configuration for testing the PipecatBackendV2.
"""

# Simple flow configuration for testing
flow_config = {
    "initial_node": "welcome",
    "nodes": {
        "welcome": {
            "role_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a voice assistant for the Experimance art installation. "
                        "You must ALWAYS use the available functions to progress the conversation. "
                        "This is a voice conversation and your responses will be converted to audio. "
                        "Keep responses conversational, brief, and engaging. "
                        "Avoid outputting special characters and emojis."
                    )
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": (
                        "Greet visitors warmly."
                    )
                }
            ]
        }
    }
}
