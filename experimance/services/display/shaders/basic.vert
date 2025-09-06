#version 150 core

in vec2 position;
in vec2 tex_coords;
out vec2 v_tex;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_tex = tex_coords;
}
