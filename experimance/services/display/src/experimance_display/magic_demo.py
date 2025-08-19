#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fog + Fairy Lights (screen-space) demo for pyglet with OSC controls.
- No real depth or segmentation inputs; this is a *mocked* playground to tune the look and measure perf.
- Handles venue-specific horizontal stretch via uXStretch so fog patterns / particle glows look correct after stretch.
- Three passes on a single framebuffer: Base (background), Fog (screen-space raymarch), Lights (additive point sprites).
- OSC controls: see routes near the bottom.
Tested targets:
- Linux/Windows + RTX (OpenGL 3.3 core+)
- macOS (OpenGL 4.1 core, pyglet auto-sel        # Mock depth + masks
        depth, trees, sky, ground = make_mock_depth_and_masks(W, H)
        self.tex_depth = create_texture_2d(W, H, gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT, depth.ctypes.data_as(ctypes.c_void_p))
        self.tex_trees = create_texture_2d(W, H, gl.GL_R8,   gl.GL_RED, gl.GL_UNSIGNED_BYTE, (np.clip(trees,0,1)*255).astype(np.uint8).ctypes.data_as(ctypes.c_void_p))
        self.tex_sky   = create_texture_2d(W, H, gl.GL_R8,   gl.GL_RED, gl.GL_UNSIGNED_BYTE, (np.clip(sky,0,1)*255).astype(np.uint8).ctypes.data_as(ctypes.c_void_p))
        self.tex_ground= create_texture_2d(W, H, gl.GL_R8,   gl.GL_RED, gl.GL_UNSIGNED_BYTE, (np.clip(ground,0,1)*255).astype(np.uint8).ctypes.data_as(ctypes.c_void_p))

        # Blue-noise-ish tile (random) – good enough for testing
        blu = make_random_tile(128, 128, seed=7)
        self.tex_blue = create_texture_2d(128, 128, gl.GL_R8, gl.GL_RED, gl.GL_UNSIGNED_BYTE, blu.ctypes.data_as(ctypes.c_void_p), wrap=gl.GL_REPEAT)t). Keep shaders 330 core friendly.

Dependencies:
    pip install pyglet PyOpenGL python-osc numpy

Run:
    python fog_fireflies_osc_pyglet.py

Send OSC (examples, using python-osc's SimpleUDPClient in another shell/script):
    /view/x_stretch 6.0
    /fog/density 0.045
    /fog/falloff 1.8
    /fog/noise_scale 0.9
    /fog/wind 0.12
    /lights/count 500
    /lights/size 4.0
    /lights/speed 0.06
    /lights/twinkle 0.5
    /mood/set "deep_forest"

Keyboard:
    [M] toggle mirror, [1]/[2]/[3] moods, [ESC] quit
"""

import os, sys, math, threading, time, random
import numpy as np
import ctypes

import pyglet
from pyglet import gl
from pyglet.window import key
from pyglet.graphics import shader

# OSC server
from pythonosc import dispatcher, osc_server

# ----------------------------- GL helpers ------------------------------------

def check_gl_error(where=""):
    err = gl.glGetError()
    if err != gl.GL_NO_ERROR:
        print(f"[GL ERROR] {where}: 0x{err:04X}", file=sys.stderr)

# Use pyglet's built-in shader system instead of raw OpenGL
def create_shader_program(vs_src, fs_src):
    """Create a shader program using pyglet's shader system"""
    vertex_shader = shader.Shader(vs_src, 'vertex')
    fragment_shader = shader.Shader(fs_src, 'fragment')
    program = shader.ShaderProgram(vertex_shader, fragment_shader)
    return program

# ----------------------------- Geometry --------------------------------------

def make_fullscreen_quad():
    # 2D clip-space quad (x,y) and uv
    verts = np.array([
        -1.0, -1.0,  0.0, 0.0,
         1.0, -1.0,  1.0, 0.0,
         1.0,  1.0,  1.0, 1.0,
        -1.0,  1.0,  0.0, 1.0,
    ], dtype=np.float32)
    idx = np.array([0,1,2,  0,2,3], dtype=np.uint32)

    vao = gl.GLuint()
    gl.glGenVertexArrays(1, ctypes.pointer(vao))
    gl.glBindVertexArray(vao)

    vbo = gl.GLuint()
    gl.glGenBuffers(1, ctypes.pointer(vbo))
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts.ctypes.data, gl.GL_STATIC_DRAW)

    ebo = gl.GLuint()
    gl.glGenBuffers(1, ctypes.pointer(ebo))
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx.ctypes.data, gl.GL_STATIC_DRAW)

    stride = 4 * 4  # 4 floats per vertex
    # pos (vec2)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
    # uv (vec2)
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8))

    gl.glBindVertexArray(0)
    return vao, vbo, ebo, idx.size

# ----------------------------- Textures --------------------------------------

def create_texture_2d(width, height, internal_fmt, fmt, typ, data=None, filter=gl.GL_LINEAR, wrap=gl.GL_CLAMP_TO_EDGE):
    tex = gl.GLuint()
    gl.glGenTextures(1, ctypes.pointer(tex))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, filter)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, filter)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, wrap)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, wrap)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal_fmt, width, height, 0, fmt, typ, data)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tex

def upload_texture_2d(tex, width, height, fmt, typ, data):
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, width, height, fmt, typ, data)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

def make_random_tile(w=128, h=128, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.random((h, w), dtype=np.float32)
    return (a * 255).astype(np.uint8)

def make_mock_depth_and_masks(width, height):
    """
    Produce simple *plausible* depth + masks.
    Depth: closer near bottom (ground), farther near top (sky).
    Trees: noisy vertical structures.
    Sky: top band.
    Ground: bottom band.
    """
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]

    # Depth in meters-ish: near=2m at bottom, far=40m at top + hill
    hill = 0.15 * np.sin(2*np.pi*(x*0.25) + 1.3) * np.exp(-y*1.5)
    depth = 2.0 + (40.0 - 2.0) * np.clip(y + hill, 0.0, 1.0)

    # Trees: create organic forest silhouette with varied individual trees
    rng = np.random.default_rng(42)
    trees = np.zeros((height, width), dtype=np.float32)
    
    # Generate individual tree trunks as points, then expand them
    n_trees = 25  # number of individual trees
    for i in range(n_trees):
        # Random tree position (x) and size
        tree_x = rng.random()
        tree_width = 0.015 + rng.random() * 0.025  # varying widths
        tree_height = 0.4 + rng.random() * 0.4    # varying heights
        tree_base_y = 0.1 + rng.random() * 0.2    # varying ground levels
        
        # Create distance field for this tree
        x_dist = np.abs(x - tree_x)
        tree_mask = np.exp(-x_dist / tree_width)  # soft falloff
        
        # Height mask - trees grow from ground up with varied heights  
        y_mask = np.clip((tree_base_y + tree_height - y) / tree_height, 0.0, 1.0)
        
        # Add some vertical texture/irregularity
        trunk_noise = np.sin(y * 30.0 + i * 2.1) * 0.3 + 0.7
        tree_mask = tree_mask * trunk_noise  # use explicit multiplication
        
        # Combine with height mask and add to forest
        individual_tree = tree_mask * y_mask
        trees = np.maximum(trees, individual_tree * 0.9)
    
    # Add some understory/bush layer
    undergrowth_noise = rng.random((height, width), dtype=np.float32)
    undergrowth = (undergrowth_noise > 0.85).astype(np.float32)
    undergrowth *= np.clip((0.25 - y) * 4.0, 0.0, 1.0)  # only in lower areas
    trees = np.maximum(trees, undergrowth * 0.6)
    
    # Final cleanup: ensure trees don't extend into sky area and smooth edges
    trees *= np.clip((0.85 - y) * 2.0, 0.0, 1.0)
    # Light blur for more natural edges
    trees = (np.roll(trees, 1, axis=0) + trees + np.roll(trees, -1, axis=0)) / 3.0
    trees = (np.roll(trees, 1, axis=1) + trees + np.roll(trees, -1, axis=1)) / 3.0

    # Sky: top band with soft edge
    sky = np.clip((y - 0.7) * 3.0, 0.0, 1.0).astype(np.float32)

    # Ground: bottom band with soft falloff
    ground = np.clip((0.2 - y) * 5.0, 0.0, 1.0).astype(np.float32)

    return depth, trees, sky, ground

# ----------------------------- Shaders ---------------------------------------

VS_QUAD = r'''
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main() {
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
'''

FS_BASE = r'''
#version 330 core
in vec2 vUV;
out vec4 fragColor;

// Simple forest-y gradient background (replace with your image later)
uniform float uTime;
uniform float uXStretch;
uniform bool  uMirror;

vec2 contentUV(vec2 uv){
    if(uMirror){
        float t = fract(uv.x * 2.0);
        uv.x = (t < 1.0) ? t : (2.0 - t);
    }
    return uv;
}

void main(){
    vec2 uv = contentUV(vUV);
    // deep forest gradient with slight horizontal variation
    float g = mix(0.05, 0.12, uv.y);
    float band = 0.02*sin(uv.x*40.0*uXStretch + uTime*0.2);
    vec3 base = vec3(0.04, 0.09, 0.07) + vec3(0.02,0.03,0.03)*(uv.y + band);
    fragColor = vec4(base, 1.0);
}
'''

FS_FOG = r'''
#version 330 core
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uDepth;   // R32F linear meters
uniform sampler2D uTrees;   // R8   0..1
uniform sampler2D uSky;     // R8
uniform sampler2D uGround;  // R8
uniform sampler2D uBlue;    // R8  128x128 noise tile

uniform float uTime;
uniform float uFogBase;
uniform float uFogFalloff;
uniform vec3  uFogColorLow;
uniform vec3  uFogColorHigh;
uniform float uNoiseScale;
uniform float uWind;
uniform float uXStretch;
uniform bool  uMirror;
uniform vec2  uResolution;

// Hash-based value noise (3D), tri-linear interpolation (cheap fBm seed)
float hash3D(vec3 p){
    // from IQ-style hashing
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float noise3D(vec3 p){
    vec3 i = floor(p);
    vec3 f = fract(p);
    float n000 = hash3D(i + vec3(0,0,0));
    float n100 = hash3D(i + vec3(1,0,0));
    float n010 = hash3D(i + vec3(0,1,0));
    float n110 = hash3D(i + vec3(1,1,0));
    float n001 = hash3D(i + vec3(0,0,1));
    float n101 = hash3D(i + vec3(1,0,1));
    float n011 = hash3D(i + vec3(0,1,1));
    float n111 = hash3D(i + vec3(1,1,1));
    vec3 u = f*f*(3.0-2.0*f);
    float n00 = mix(n000, n100, u.x);
    float n01 = mix(n001, n101, u.x);
    float n10 = mix(n010, n110, u.x);
    float n11 = mix(n011, n111, u.x);
    float n0 = mix(n00, n10, u.y);
    float n1 = mix(n01, n11, u.y);
    return mix(n0, n1, u.z);
}

vec2 contentUV(vec2 uv){
    if(uMirror){
        float t = fract(uv.x * 2.0);
        uv.x = (t < 1.0) ? t : (2.0 - t);
    }
    return uv;
}

vec2 stretchAware(vec2 d){
    return vec2(d.x / uXStretch, d.y);
}

void main(){
    vec2 uv = contentUV(vUV);

    float surfT = texture(uDepth, uv).r;              // linear meters
    if(surfT <= 0.0){
        fragColor = vec4(0.0);
        return;
    }

    // Masks
    float mTrees  = texture(uTrees,  uv).r;
    float mSky    = texture(uSky,    uv).r;
    float mGround = texture(uGround, uv).r;

    // integrate along a simple "view ray" (screen-space approx)
    const int N = 8;
    float t  = 0.0;
    float dt = surfT / float(N);
    // blue-noise dither to reduce banding
    ivec2 pix = ivec2(gl_FragCoord.xy) & 127;
    float bn = texelFetch(uBlue, pix, 0).r / 255.0;
    t += dt * bn;

    vec3 accum = vec3(0.0);
    float trans = 1.0;

    for(int i=0; i<N; ++i){
        float ti = t + 0.5*dt;

        // Fake world pos from uv + "depth along ray": we just need a y to drive height fog
        vec2 uvl = vec2(uv.x * uXStretch, uv.y);
        float worldY = mix(-1.0, 5.0, uv.y) - ti*0.02; // descend slightly with depth

        float dens = uFogBase * exp(-uFogFalloff * worldY);
        // bias by masks
        float bias = 1.0;
        bias *= mix(1.0, 1.6, mGround);
        bias *= mix(1.0, 0.6, mSky);
        bias *= mix(1.0, 0.8, mTrees);
        dens *= bias;

        // animated 3D noise in logical (stretch-compensated) domain
        vec3 q = vec3(uvl * uNoiseScale, (uTime * uWind + ti*0.02) * uNoiseScale);
        float n = noise3D(q);
        
        // Add secondary layer for more complex movement
        vec3 q2 = vec3(uvl * uNoiseScale * 0.5, (uTime * uWind * 1.3 + ti*0.015) * uNoiseScale);
        float n2 = noise3D(q2);
        float combined_noise = mix(n, n2, 0.4);
        
        dens *= mix(0.4, 1.6, combined_noise);  // stronger contrast for more visible patterns

        float a = 1.0 - exp(-dens * dt);
        vec3  c = mix(uFogColorLow, uFogColorHigh, clamp(worldY*0.15, 0.0, 1.0));
        accum += trans * a * c;
        trans *= (1.0 - a);
        t += dt;
    }

    float alpha = clamp(1.0 - trans, 0.0, 1.0);
    fragColor = vec4(accum, alpha);
}
'''

VS_PARTICLES = r'''
#version 330 core
layout(location=0) in vec2 aUV;      // base position in [0,1]
layout(location=1) in float aDepth;  // pseudo-depth in meters
layout(location=2) in vec2 aSeed;    // random seed
layout(location=3) in float aHue;    // 0..1

out vec2  vUV;
out float vDepth;
out float vHue;
out vec2  vSeed;
out vec2  vVelocity; // pass velocity direction to fragment shader

uniform float uTime;
uniform float uSpeed;
uniform float uXStretch;
uniform bool  uMirror;

uniform vec2  uResolution;
uniform float uPointSize; // base size in px

// 2D pseudo-random noise for motion (deterministic)
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p){
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i + vec2(0,0));
    float b = hash(i + vec2(1,0));
    float c = hash(i + vec2(0,1));
    float d = hash(i + vec2(1,1));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

vec2 contentUV(vec2 uv){
    if(uMirror){
        float t = fract(uv.x * 2.0);
        uv.x = (t < 1.0) ? t : (2.0 - t);
    }
    return uv;
}

void main(){
    // base uv in logical domain for motion
    vec2 uv = contentUV(aUV);
    float t = uTime * uSpeed;

    // flow field drift (curl-ish via two perpendicular gradients)
    float n1 = noise(uv * 1.7 + aSeed + vec2(0.0, t*0.2));
    float n2 = noise(uv * 1.9 + aSeed.yx + vec2(t*0.2, 0.0));
    vec2  dir = normalize(vec2(n1 - 0.5, n2 - 0.5) + 1e-3);

    vec2 drift = dir * 0.08 * sin(t*0.7 + aSeed.x*6.2831);
    vec2 pos = uv + drift;
    // wrap
    pos = fract(pos);

    // Calculate velocity for oriented particles - pure random orientation
    float orientationAngle = aSeed.x * 6.2831; // completely random 0-2π based on seed
    float c = cos(orientationAngle);
    float s = sin(orientationAngle);
    vVelocity = vec2(c, s); // pure random direction

    // Pass to fragment for occlusion and color
    vUV = pos;
    vDepth = aDepth;
    vHue = aHue;
    vSeed = aSeed;

    // Convert to clip space
    vec2 clip = pos * 2.0 - 1.0;
    gl_Position = vec4(clip, 0.0, 1.0);

    // Variable point size based on speed for streak effect
    float speed = length(vVelocity);
    gl_PointSize = uPointSize * (1.0 + speed * 2.0);
}
'''

FS_PARTICLES = r'''
#version 330 core
in vec2  vUV;
in float vDepth;
in float vHue;
in vec2  vSeed;
in vec2  vVelocity; // velocity direction from vertex shader
out vec4 fragColor;

uniform sampler2D uDepth;   // scene depth for occlusion
uniform sampler2D uTrees;   // soften edges in foliage
uniform float uXStretch;
uniform float uTwinkle;     // 0..1
uniform float uAudioEnergy; // 0..1

// HSV -> RGB (simple)
vec3 hsv2rgb(vec3 c){
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main(){
    // Create oriented firefly streak instead of circle
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    
    // Simple approach: directly use random orientation from vertex shader
    // vVelocity already contains random cos/sin direction
    vec2 orientation = normalize(vVelocity);
    
    // Project the point onto the orientation axis
    float along = dot(p, orientation);        // distance along the streak direction
    float across = dot(p, vec2(-orientation.y, orientation.x)); // distance perpendicular
    
    // Account for stretch in the across direction only
    across *= uXStretch;
    
    // Create an elongated ellipse - shorter and thicker than before
    float streak_length = 1.4 + 0.3 * sin(vSeed.x * 12.0);  // 1.4-1.7x length (was 2.0-2.5x)
    float streak_width = 0.6 + 0.15 * sin(vSeed.y * 15.0);  // 0.6-0.75x width (was 0.4-0.5x)
    
    float ellipse = (along * along) / (streak_length * streak_length) + 
                    (across * across) / (streak_width * streak_width);
    
    if(ellipse > 1.0) discard;
    float r2 = ellipse;
    
    if(r2 > 1.0) discard;

    // Depth occlusion
    float dScene = texture(uDepth, vUV).r;
    float occl = smoothstep(-0.02, 0.10, dScene - vDepth); // behind -> 0 .. in front -> 1
    
    // Much lighter tree occlusion - don't completely hide fireflies in trees
    float tree_mask = texture(uTrees, vUV).r;
    occl *= mix(1.0, 0.9, tree_mask);  // only slightly dim in trees instead of 0.75

    // Multiple radial falloffs for strong bloom effect with pulse
    float core = exp(-6.0 * r2);        // bright center
    float mid_bloom = exp(-3.0 * r2);   // medium bloom
    float wide_bloom = exp(-1.0 * r2);  // wide bloom
    
    // Enhanced pulse effect running along firefly body
    float pulse_speed = 3.0 + 2.0 * sin(vSeed.y * 17.0);  // vary pulse speed per firefly
    float pulse_phase = vSeed.x * 6.2831;  // unique phase per firefly
    float pulse_time = uTwinkle * 8.0 + pulse_phase;
    
    // Create traveling pulse effect along the streak
    float pulse_pos = fract(pulse_time * pulse_speed);  // 0-1 position along body
    float body_pos = (along + streak_length) / (2.0 * streak_length);  // normalize body position 0-1
    float pulse_distance = abs(body_pos - pulse_pos);
    float pulse_intensity = exp(-15.0 * pulse_distance);  // sharp pulse
    
    // Not all fireflies get strong pulses - some are more steady
    float pulse_strength = 0.3 + 0.7 * sin(vSeed.x * 23.0);  // 30-100% pulse strength
    float pulse_effect = mix(1.0, 1.0 + pulse_intensity * 2.0, pulse_strength);
    
    float fall = (core + mid_bloom * 0.6 + wide_bloom * 0.3) * pulse_effect;

    // Twinkle & audio
    float tw = 0.5 + 0.5 * sin(vSeed.x*17.0 + vSeed.y*23.0 + uAudioEnergy*2.4 + uTwinkle*10.0);
    float base_intensity = mix(0.7, 1.6, tw);  // base twinkle range
    
    // Add depth-based brightness variation for more realistic depth
    float depth_factor = (vDepth - 5.0) / 30.0;  // normalize depth from 5-35m to 0-1
    float depth_dimming = mix(1.0, 0.15, depth_factor);  // far fireflies much dimmer (15% brightness)
    
    // Add extra random brightness variation per firefly
    float brightness_variation = 0.3 + 0.7 * sin(vSeed.y * 23.0);  // 30-100% base brightness
    
    float I = base_intensity * depth_dimming * brightness_variation;

    // Add subtle color variation - slight hue shifts and saturation changes
    float hue_variation = vHue + 0.05 * sin(vSeed.y * 19.0);  // ±5% hue shift
    float sat_variation = 0.75 + 0.15 * sin(vSeed.x * 21.0);  // 75-90% saturation variation
    
    vec3 col = hsv2rgb(vec3(hue_variation, sat_variation, 1.0));
    vec3 glow = col * fall * I * occl;

    // Much stronger additive bloom - especially visible in fog
    fragColor = vec4(glow * 1.5, 1.0);  // 1.5x multiplier for stronger bloom
}
'''

# ----------------------------- App State -------------------------------------

class State:
    def __init__(self):
        self.width  = 1920
        self.height = 1080
        self.x_stretch = 6.0
        self.mirror = True  # re-enable mirroring with better particle distribution

        # Fog params - much more visible
        self.fog_base = 0.25  # increased significantly for much better visibility
        self.fog_falloff = 1.5  # reduced falloff for more fog at height
        self.fog_color_low = (0.12, 0.20, 0.15)  # much higher contrast
        self.fog_color_high = (0.25, 0.35, 0.25)  # much higher contrast
        self.noise_scale = 1.2  # increased for more visible patterns
        self.wind = 0.25  # increased wind speed for more movement

        # Lights - much smaller point size
        self.lights_count = 640  # increased from 480 for more fireflies
        self.point_size = 4.0  # reduced from 12.0 - much smaller fireflies
        self.speed = 0.025  # much slower for nicer movement
        self.twinkle = 0.5
        self.audio_energy = 0.0

        # Time
        self.t0 = time.time()
        self.time = 0.0

        self.moods = {
            "deep_forest": dict(
                fog_base=0.25, fog_falloff=1.5,
                fog_color_low=(0.12,0.20,0.15),
                fog_color_high=(0.25,0.35,0.25),
                noise_scale=1.2, wind=0.25,  # increased for more visible movement
                lights_count=600, point_size=4.0, speed=0.025, twinkle=0.5  # more fireflies
            ),
            "moonlit": dict(
                fog_base=0.22, fog_falloff=1.4,
                fog_color_low=(0.08,0.12,0.18),
                fog_color_high=(0.18,0.25,0.35),
                noise_scale=1.0, wind=0.20,  # more visible movement
                lights_count=550, point_size=3.5, speed=0.022, twinkle=0.6  # more fireflies
            ),
            "dawn": dict(
                fog_base=0.28, fog_falloff=1.3,
                fog_color_low=(0.20,0.18,0.12),
                fog_color_high=(0.40,0.32,0.20),
                noise_scale=1.1, wind=0.22,  # more visible movement
                lights_count=720, point_size=4.5, speed=0.028, twinkle=0.4  # more fireflies
            ),
        }

    def set_mood(self, name):
        if name in self.moods:
            m = self.moods[name]
            self.fog_base = m["fog_base"]
            self.fog_falloff = m["fog_falloff"]
            self.fog_color_low = m["fog_color_low"]
            self.fog_color_high = m["fog_color_high"]
            self.noise_scale = m["noise_scale"]
            self.wind = m["wind"]
            self.lights_count = m["lights_count"]
            self.point_size = m["point_size"]
            self.speed = m["speed"]
            self.twinkle = m["twinkle"]
            print(f"[mood] {name}")
        else:
            print(f"[mood] unknown: {name}")

# ----------------------------- Particle data ---------------------------------

def init_particles(n):
    # Use uniform random distribution instead of Hammersley for better coverage
    rng = np.random.default_rng(42)  # Fixed seed for consistency
    
    aUV = np.zeros((n,2), dtype=np.float32)
    aDepth = np.zeros((n,1), dtype=np.float32)
    aSeed = np.zeros((n,2), dtype=np.float32)
    aHue  = np.zeros((n,1), dtype=np.float32)

    for i in range(n):
        # Uniform random distribution across full screen
        aUV[i,0] = rng.random()
        aUV[i,1] = rng.random()
        aDepth[i,0] = 5.0 + 30.0 * rng.random()  # 5..35 m
        aSeed[i,0] = rng.random()
        aSeed[i,1] = rng.random()
        aHue[i,0]  = 0.30 + 0.10 * rng.random()  # greens

    return aUV, aDepth, aSeed, aHue

# ----------------------------- Main App --------------------------------------

class App:
    def __init__(self, W=1920, H=1080):
        self.state = State()
        self.state.width, self.state.height = W, H

        config = pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=4,
                                  major_version=3, minor_version=3, depth_size=0, stencil_size=0)
        try:
            self.win = pyglet.window.Window(width=W, height=H, config=config, resizable=True, caption="Fog + Fairy Lights + OSC")
        except Exception as e:
            # fallback without multisampling
            config = pyglet.gl.Config(double_buffer=True, major_version=3, minor_version=3)
            self.win = pyglet.window.Window(width=W, height=H, config=config, resizable=True, caption="Fog + Fairy Lights + OSC")

        self.keys = key.KeyStateHandler()
        self.win.push_handlers(self.keys)

        # Compile programs
        self.prog_base = create_shader_program(VS_QUAD, FS_BASE)
        self.prog_fog  = create_shader_program(VS_QUAD, FS_FOG)
        self.prog_pts  = create_shader_program(VS_PARTICLES, FS_PARTICLES)

        # Quad
        self.vao_quad, self.vbo_quad, self.ebo_quad, self.quad_count = make_fullscreen_quad()

        # Mock depth + masks
        depth, trees, sky, ground = make_mock_depth_and_masks(W, H)
        self.tex_depth = create_texture_2d(W, H, gl.GL_R32F, gl.GL_RED, gl.GL_FLOAT, depth.ctypes.data_as(ctypes.c_void_p))
        self.tex_trees = create_texture_2d(W, H, gl.GL_R8,   gl.GL_RED, gl.GL_UNSIGNED_BYTE, (np.clip(trees,0,1)*255).astype(np.uint8).ctypes.data_as(ctypes.c_void_p))
        self.tex_sky   = create_texture_2d(W, H, gl.GL_R8,   gl.GL_RED, gl.GL_UNSIGNED_BYTE, (np.clip(sky,0,1)*255).astype(np.uint8).ctypes.data_as(ctypes.c_void_p))
        self.tex_ground= create_texture_2d(W, H, gl.GL_R8,   gl.GL_RED, gl.GL_UNSIGNED_BYTE, (np.clip(ground,0,1)*255).astype(np.uint8).ctypes.data_as(ctypes.c_void_p))

        # Blue-noise-ish tile (random) – good enough for testing
        blu = make_random_tile(128, 128, seed=7)
        self.tex_blue = create_texture_2d(128, 128, gl.GL_R8, gl.GL_RED, gl.GL_UNSIGNED_BYTE, blu.ctypes.data_as(ctypes.c_void_p), wrap=gl.GL_REPEAT)

        # Particles
        self.max_particles = 2000
        aUV, aDepth, aSeed, aHue = init_particles(self.max_particles)
        self.n_particles = self.state.lights_count

        # VBOs
        self.vao_pts = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.pointer(self.vao_pts))
        gl.glBindVertexArray(self.vao_pts)

        def make_vbo(initial, loc, comps, dtype=gl.GL_FLOAT):
            buf = gl.GLuint()
            gl.glGenBuffers(1, ctypes.pointer(buf))
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, initial.nbytes, initial.ctypes.data, gl.GL_STATIC_DRAW)
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, comps, dtype, gl.GL_FALSE, 0, ctypes.c_void_p(0))
            return buf

        self.vbo_uv   = make_vbo(aUV,   0, 2)
        self.vbo_dep  = make_vbo(aDepth,1, 1)
        self.vbo_seed = make_vbo(aSeed, 2, 2)
        self.vbo_hue  = make_vbo(aHue,  3, 1)

        gl.glBindVertexArray(0)

        # GL state
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        # Set additive blending for brighter particle effects
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE)

        # OSC server
        self.osc_state = self.state  # alias
        self.start_osc_server()

        # Event handlers
        @self.win.event
        def on_draw():
            self.draw()

        @self.win.event
        def on_key_press(symbol, modifiers):
            if symbol == key.ESCAPE:
                pyglet.app.exit()
            elif symbol == key.M:
                self.state.mirror = not self.state.mirror
                print("[view] mirror =", self.state.mirror)
            elif symbol in (key._1, key._2, key._3):
                mood = {key._1:"deep_forest", key._2:"moonlit", key._3:"dawn"}[symbol]
                self.state.set_mood(mood)

        pyglet.clock.schedule_interval(self.update_time, 1/120.0)

    # ---------------- OSC ----------------

    def start_osc_server(self, host="0.0.0.0", port=9000):
        disp = dispatcher.Dispatcher()

        def set_float(attr):
            def _f(addr, val):
                setattr(self.osc_state, attr, float(val))
                print(f"[osc] {attr} = {getattr(self.osc_state, attr)}")
            return _f

        def set_int(attr):
            def _f(addr, val):
                setattr(self.osc_state, attr, int(val))
                print(f"[osc] {attr} = {getattr(self.osc_state, attr)}")
            return _f

        disp.map("/view/x_stretch", set_float("x_stretch"))
        disp.map("/view/mirror",    lambda addr, v: setattr(self.osc_state, "mirror", bool(int(v))))
        disp.map("/fog/density",    set_float("fog_base"))
        disp.map("/fog/falloff",    set_float("fog_falloff"))
        disp.map("/fog/noise_scale",set_float("noise_scale"))
        disp.map("/fog/wind",       set_float("wind"))
        disp.map("/lights/count",   set_int("lights_count"))
        disp.map("/lights/size",    set_float("point_size"))
        disp.map("/lights/speed",   set_float("speed"))
        disp.map("/lights/twinkle", set_float("twinkle"))
        disp.map("/audio/energy",   set_float("audio_energy"))
        disp.map("/mood/set",       lambda addr, s: self.osc_state.set_mood(str(s)))

        self.osc_server = osc_server.ThreadingOSCUDPServer((host, port), disp)
        self.osc_thread = threading.Thread(target=self.osc_server.serve_forever, daemon=True)
        self.osc_thread.start()
        print(f"[osc] Listening on udp://{host}:{port}")

    # ---------------- Frame update/draw ----------------

    def update_time(self, dt):
        self.state.time = time.time() - self.state.t0
        # Clamp/validate counts
        self.n_particles = int(max(0, min(self.state.lights_count, self.max_particles)))

    def draw(self):
        W, H = self.win.get_size()

        gl.glViewport(0, 0, W, H)
        gl.glClearColor(0.02, 0.04, 0.05, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Pass A: base background
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ZERO)  # overwrite
        self.prog_base.use()
        self.prog_base['uTime'] = self.state.time
        self.prog_base['uXStretch'] = float(self.state.x_stretch)
        self.prog_base['uMirror'] = int(self.state.mirror)
        gl.glBindVertexArray(self.vao_quad)
        gl.glDrawElements(gl.GL_TRIANGLES, self.quad_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))

        # Pass B: fog (alpha blend)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.prog_fog.use()
        def bind_tex(unit, tex, name, target=gl.GL_TEXTURE_2D):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(target, tex)
            self.prog_fog[name] = unit

        bind_tex(0, self.tex_depth, "uDepth")
        bind_tex(1, self.tex_trees, "uTrees")
        bind_tex(2, self.tex_sky,   "uSky")
        bind_tex(3, self.tex_ground,"uGround")
        bind_tex(4, self.tex_blue,  "uBlue")

        self.prog_fog['uTime'] = self.state.time
        self.prog_fog['uFogBase'] = self.state.fog_base
        self.prog_fog['uFogFalloff'] = self.state.fog_falloff
        self.prog_fog['uFogColorLow'] = self.state.fog_color_low
        self.prog_fog['uFogColorHigh'] = self.state.fog_color_high
        self.prog_fog['uNoiseScale'] = self.state.noise_scale
        self.prog_fog['uWind'] = self.state.wind
        self.prog_fog['uXStretch'] = float(self.state.x_stretch)
        self.prog_fog['uMirror'] = int(self.state.mirror)
        # self.prog_fog['uResolution'] = (float(W), float(H))  # Unused, commented out

        gl.glBindVertexArray(self.vao_quad)
        gl.glDrawElements(gl.GL_TRIANGLES, self.quad_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))

        # Pass C: particles (additive)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
        self.prog_pts.use()
        # bind shared textures for occlusion
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_depth)
        self.prog_pts['uDepth'] = 0
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_trees)
        self.prog_pts['uTrees'] = 1

        self.prog_pts['uTime'] = self.state.time
        self.prog_pts['uSpeed'] = self.state.speed
        self.prog_pts['uXStretch'] = float(self.state.x_stretch)
        self.prog_pts['uMirror'] = int(self.state.mirror)
        # self.prog_pts['uResolution'] = (float(W), float(H))  # Unused, commented out
        self.prog_pts['uPointSize'] = float(self.state.point_size)
        self.prog_pts['uTwinkle'] = float(self.state.twinkle)
        self.prog_pts['uAudioEnergy'] = float(self.state.audio_energy)

        gl.glBindVertexArray(self.vao_pts)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.n_particles)
        gl.glBindVertexArray(0)

        check_gl_error("draw")

def main():
    W = int(os.environ.get("WIN_W", "1920"))
    H = int(os.environ.get("WIN_H", "1080"))
    app = App(W, H)
    pyglet.app.run()

if __name__ == "__main__":
    main()