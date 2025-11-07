#!/usr/bin/env python3
"""
Simplified magic demo - debug version to isolate point sprite issues.
Based directly on the working point_sprite_test.py
"""

import pyglet
from pyglet.gl import *
from pyglet.graphics.shader import Shader, ShaderProgram
import ctypes
import numpy as np
import time

# Simple vertex shader - based on working test
VS_SIMPLE = r'''
#version 330 core
layout(location=0) in vec2 aPosition;
uniform float uPointSize;

void main() {
    gl_Position = vec4(aPosition, 0.0, 1.0);
    gl_PointSize = uPointSize;
}
'''

# Fragment shader using Shadertoy firefly technique with edge cleanup
FS_SIMPLE = r'''
#version 330 core
out vec4 fragColor;

uniform float uFireflySize;
uniform float uCenterBrightness; 
uniform float uGlowSize;
uniform float uGlowBrightness;
uniform float uVariation;  // for size variation per firefly

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    // Get distance from center of point sprite
    vec2 p = gl_PointCoord * 2.0 - 1.0;  // Convert to -1 to 1 range
    float distance = length(p);  // Actual distance from center
    
    // Discard pixels outside circular boundary to prevent square artifacts
    if (distance > 1.0) discard;
    
    // Add per-firefly variation based on gl_PrimitiveID would be ideal, 
    // but we can use PointCoord as a seed for some variation
    float variation = rand(gl_PointCoord + vec2(0.123, 0.456)) * uVariation + (1.0 - uVariation);
    
    // Use the Shadertoy technique: size / distance - size
    // This creates a bright center that falls off sharply
    float firefly_size = uFireflySize * variation;
    float center_intensity = max(firefly_size / distance - firefly_size, 0.0) * uCenterBrightness;
    
    // Add a controllable outer glow with smooth falloff at edges
    float outer_glow = exp(-uGlowSize * distance) * uGlowBrightness * variation;
    
    // Add edge smoothing to prevent harsh cutoff at point sprite boundary
    float edge_fade = smoothstep(0.9, 1.0, distance);
    outer_glow *= (1.0 - edge_fade);
    
    float total_intensity = center_intensity + outer_glow;
    
    // Warm firefly color with slight variation
    vec3 base_color = vec3(1.0, 0.8, 0.4);
    vec3 color = base_color * total_intensity;
    
    fragColor = vec4(color, 1.0);  // additive blending handles transparency
}
'''

class SimpleMagicDemo:
    def __init__(self):
        self.window = pyglet.window.Window(1920, 1080, caption="Simple Magic Demo Test")
        
        # Create shader program
        vert_shader = Shader(VS_SIMPLE, 'vertex')
        frag_shader = Shader(FS_SIMPLE, 'fragment')
        self.program = ShaderProgram(vert_shader, frag_shader)
        
        # Create firefly positions - random placement for testing overlaps
        num_fireflies = 100
        positions = []
        np.random.seed(42)  # consistent random placement for testing
        for i in range(num_fireflies):
            x = np.random.uniform(-0.9, 0.9)  # random x position
            y = np.random.uniform(-0.9, 0.9)  # random y position
            positions.extend([x, y])
        
        self.positions = np.array(positions, dtype=np.float32)
        
        # Create VAO and VBO
        self.vao = GLuint()
        glGenVertexArrays(1, ctypes.pointer(self.vao))
        glBindVertexArray(self.vao)
        
        self.vbo = GLuint()
        glGenBuffers(1, ctypes.pointer(self.vbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.positions.nbytes, self.positions.ctypes.data, GL_STATIC_DRAW)
        
        # Set up vertex attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, ctypes.c_void_p(0))
        
        glBindVertexArray(0)
        
        self.num_points = len(self.positions) // 2
        self.point_size = 48.0  # larger to accommodate bigger glows
        
        # Firefly parameters for real-time adjustment
        self.firefly_size = 0.12  # slightly smaller center as requested
        self.center_brightness = 1.0  # brightness of the center
        self.glow_size = 2.5  # how tight the glow is (higher = tighter)
        self.glow_brightness = 0.4  # slightly larger/brighter glow as requested
        self.variation = 0.3  # amount of size variation between fireflies
        
        # Set up event handlers
        @self.window.event
        def on_draw():
            self.draw()
            
        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.ESCAPE:
                pyglet.app.exit()
            elif symbol == pyglet.window.key.UP:
                self.point_size += 8.0
                print(f"Point size: {self.point_size}")
            elif symbol == pyglet.window.key.DOWN:
                self.point_size = max(8.0, self.point_size - 8.0)
                print(f"Point size: {self.point_size}")
            # Firefly size controls
            elif symbol == pyglet.window.key.Q:
                self.firefly_size += 0.01
                print(f"Firefly size (center): {self.firefly_size:.3f}")
            elif symbol == pyglet.window.key.A:
                self.firefly_size = max(0.01, self.firefly_size - 0.01)
                print(f"Firefly size (center): {self.firefly_size:.3f}")
            # Center brightness controls  
            elif symbol == pyglet.window.key.W:
                self.center_brightness += 0.1
                print(f"Center brightness: {self.center_brightness:.2f}")
            elif symbol == pyglet.window.key.S:
                self.center_brightness = max(0.1, self.center_brightness - 0.1)
                print(f"Center brightness: {self.center_brightness:.2f}")
            # Glow size controls (higher = tighter glow)
            elif symbol == pyglet.window.key.E:
                self.glow_size += 0.2
                print(f"Glow tightness: {self.glow_size:.1f}")
            elif symbol == pyglet.window.key.D:
                self.glow_size = max(0.5, self.glow_size - 0.2)
                print(f"Glow tightness: {self.glow_size:.1f}")
            # Glow brightness controls
            elif symbol == pyglet.window.key.R:
                self.glow_brightness += 0.05
                print(f"Glow brightness: {self.glow_brightness:.3f}")
            elif symbol == pyglet.window.key.F:
                self.glow_brightness = max(0.0, self.glow_brightness - 0.05)
                print(f"Glow brightness: {self.glow_brightness:.3f}")
            # Variation controls
            elif symbol == pyglet.window.key.T:
                self.variation = min(1.0, self.variation + 0.05)
                print(f"Size variation: {self.variation:.3f}")
            elif symbol == pyglet.window.key.G:
                self.variation = max(0.0, self.variation - 0.05)
                print(f"Size variation: {self.variation:.3f}")
            elif symbol == pyglet.window.key.P:
                print("=== Current Settings ===")
                print(f"Point size: {self.point_size}")
                print(f"Firefly size (center): {self.firefly_size:.3f}")
                print(f"Center brightness: {self.center_brightness:.2f}")
                print(f"Glow tightness: {self.glow_size:.1f}")
                print(f"Glow brightness: {self.glow_brightness:.3f}")
                print(f"Size variation: {self.variation:.3f}")
    
    def draw(self):
        # Clear with dark background like magic demo
        glClearColor(0.0, 0.0, 0.0, 1.0)
        self.window.clear()
        
        # Enable blending for glow effect - back to additive but controlled
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)  # additive blending for glows
        
        # Enable programmable point size
        glEnable(GL_PROGRAM_POINT_SIZE)
        
        self.program.use()
        self.program['uPointSize'] = self.point_size
        self.program['uFireflySize'] = self.firefly_size
        self.program['uCenterBrightness'] = self.center_brightness
        self.program['uGlowSize'] = self.glow_size
        self.program['uGlowBrightness'] = self.glow_brightness
        self.program['uVariation'] = self.variation
        
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, self.num_points)
        glBindVertexArray(0)
        
        glDisable(GL_PROGRAM_POINT_SIZE)
        glDisable(GL_BLEND)

def main():
    demo = SimpleMagicDemo()
    print("Simple Magic Demo Test - Interactive Firefly Tuning")
    print("================================================")
    print("Random firefly placement for testing overlaps and edge behavior.")
    print("")
    print("Controls:")
    print("UP/DOWN arrows - Change point size")
    print("Q/A - Firefly size (center brightness area)")
    print("W/S - Center brightness intensity")
    print("E/D - Glow tightness (higher = tighter glow)")
    print("R/F - Glow brightness")
    print("T/G - Size variation between fireflies")
    print("P - Print current settings")
    print("ESC - Exit")
    print("")
    print("Now with random placement - test overlaps and high glow tightness...")
    pyglet.app.run()

if __name__ == "__main__":
    main()
