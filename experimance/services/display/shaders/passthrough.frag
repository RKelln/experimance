#version 150 core

in vec2 v_tex;
out vec4 frag;

uniform sampler2D scene_texture;
uniform float time;

void main() {
    vec2 uv = v_tex;
    vec4 color = texture(scene_texture, uv);
    
    // Simple passthrough with slight time-based color shift to test animation
    color.rgb += sin(time) * 0.1;
    
    frag = color;
}
