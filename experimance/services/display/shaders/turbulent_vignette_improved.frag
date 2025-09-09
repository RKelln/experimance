#version 330 core

in vec2 tex_coords;
out vec4 FragColor;

uniform float time;
uniform vec2 resolution;
uniform float vignette_strength = 0.7;
uniform float turbulence_amount = 0.25;
uniform float gradient_zone = 0.25;     // Configurable gradient transition zone
uniform float speed = 1.0;              // Global speed multiplier for all animations
uniform float horizontal_compression = 1.0;  // Compression factor for horizontal coordinates (6.0 for 6x projector stretch)

// Improved noise function
float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Smooth noise with better interpolation
float smooth_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    
    float a = noise(i);
    float b = noise(i + vec2(1.0, 0.0));
    float c = noise(i + vec2(0.0, 1.0));
    float d = noise(i + vec2(1.0, 1.0));
    
    // Smoother interpolation
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Multi-octave fractal noise for turbulence
float fractal_noise(vec2 p) {
    float n = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    for (int i = 0; i < 5; i++) {
        n += smooth_noise(p * freq) * amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    return n;
}

// Enhanced noise function with more octaves for fine detail
float detailed_noise(vec2 p) {
    float n = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    for (int i = 0; i < 8; i++) {  // Even more octaves for ultra-fine detail
        n += smooth_noise(p * freq) * amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    return n;
}

// Ultra-detailed noise for micro-textures
float micro_noise(vec2 p) {
    float n = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    for (int i = 0; i < 6; i++) {
        n += smooth_noise(p * freq) * amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    return n;
}

// Swirl function to create rotating, twisted patterns with tighter control
vec2 swirl(vec2 p, float strength) {
    float len = length(p);
    float angle = atan(p.y, p.x) + len * strength;
    return vec2(cos(angle), sin(angle)) * len;
}

// Tight swirl function for smaller, more concentrated spirals
vec2 tight_swirl(vec2 p, float strength, float frequency) {
    vec2 center = fract(p * frequency) - 0.5;
    float len = length(center);
    float angle = atan(center.y, center.x) + len * strength * 8.0;  // Much tighter spirals
    return p + (vec2(cos(angle), sin(angle)) * len - center) * 0.1;
}

// Ridged noise for more dramatic cloud features
float ridged_noise(vec2 p) {
    return 1.0 - abs(smooth_noise(p) * 2.0 - 1.0);
}

// Domain warping for organic distortion
vec2 domain_warp(vec2 p, float time) {
    vec2 q = vec2(fractal_noise(p + vec2(0.0, 0.0)),
                  fractal_noise(p + vec2(5.2, 1.3) + time * speed * 0.05));
    
    vec2 r = vec2(fractal_noise(p + 4.0 * q + vec2(1.7, 9.2) + time * speed * 0.1),
                  fractal_noise(p + 4.0 * q + vec2(8.3, 2.8) + time * speed * 0.08));
    
    return p + r * 0.5;
}

void main() {
    vec2 uv = tex_coords;
    
    // Apply horizontal compression for aspect ratio correction (6x projector stretch)
    // This compresses the horizontal dimension for distance calculations
    vec2 compressed_uv = uv;
    compressed_uv.x = (compressed_uv.x - 0.5) * horizontal_compression + 0.5;
    
    // This is a masking shader - start with full transparency
    // The alpha channel will define the mask
    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);  // Completely transparent base
    
    // Create single smooth transition with configurable zones
    // float full_mask_zone = 0.08;      // Small fully masked area at very edge (now configurable)
    // float gradient_zone = 0.25;       // Large gradient zone for smooth transition (now configurable)
    
    // Calculate distances from edges (0 = at edge, 1 = at center)
    // Use compressed UV for distance calculations to account for horizontal stretch
    float dist_from_top = compressed_uv.y;
    float dist_from_bottom = 1.0 - compressed_uv.y;
    
    // For horizontal stretched displays, we also need to consider side distances for proper effect distribution
    float dist_from_left = compressed_uv.x;
    float dist_from_right = 1.0 - compressed_uv.x;
    
    // Create smooth, continuous edge masks
    float top_base_mask = 0.0;
    float bottom_base_mask = 0.0;
    
    // Single smooth transition for top with more transparency in larger area
    if (dist_from_top < gradient_zone) {
        // Create a smooth curve that's steep near the edge, gentle toward center
        float normalized_dist = dist_from_top / gradient_zone;
        // Custom curve: steep at edge (0), gentle toward center (1)
        float curve = smoothstep(0.0, 1.0, normalized_dist);
        curve = curve * curve;  // Square for steeper falloff at edge
        
        // Make the larger mask area more transparent by scaling down the mask strength
        float mask_strength = 1.0 - normalized_dist * 0.7;  // Reduce strength as we move away from edge
        top_base_mask = (1.0 - curve) * mask_strength;
    }
    
    // Single smooth transition for bottom with more transparency in larger area
    if (dist_from_bottom < gradient_zone) {
        // Create a smooth curve that's steep near the edge, gentle toward center
        float normalized_dist = dist_from_bottom / gradient_zone;
        // Custom curve: steep at edge (0), gentle toward center (1)
        float curve = smoothstep(0.0, 1.0, normalized_dist);
        curve = curve * curve;  // Square for steeper falloff at edge
        
        // Make the larger mask area more transparent by scaling down the mask strength
        float mask_strength = 1.0 - normalized_dist * 0.7;  // Reduce strength as we move away from edge
        bottom_base_mask = (1.0 - curve) * mask_strength;
    }
    
    // Generate flowing, steam-like turbulence with ultra-fine texture and tight swirls
    // Top flow moves UPWARD (negative Y), bottom flow moves downward - much slower, more balanced movement
    // Use compressed UV to account for horizontal stretching in the flow calculations
    vec2 top_flow_coord = compressed_uv * vec2(8.0, 6.0) + vec2(time * speed * 0.015, -time * speed * 0.06);  // Fixed: negative Y for upward flow
    vec2 bottom_flow_coord = compressed_uv * vec2(8.0, 6.0) + vec2(time * speed * -0.01, -time * speed * 0.045);
    
    // Apply domain warping for organic distortion
    vec2 warped_top = domain_warp(top_flow_coord, time * speed * 0.06);
    vec2 warped_bottom = domain_warp(bottom_flow_coord, time * speed * 0.06);
    
    // Apply multiple layers of swirls - regular and tight
    vec2 swirled_top = swirl(warped_top, 0.15 + sin(time * speed * 0.03) * 0.05);
    vec2 tight_swirled_top = tight_swirl(swirled_top, 0.4, 3.0 + sin(time * speed * 0.025) * 0.5);
    
    vec2 swirled_bottom = swirl(warped_bottom, 0.1 + cos(time * speed * 0.035) * 0.05);
    vec2 tight_swirled_bottom = tight_swirl(swirled_bottom, 0.35, 2.8 + cos(time * speed * 0.028) * 0.4);
    
    // Generate ultra-detailed noise patterns with multiple techniques
    float top_steam_base = detailed_noise(tight_swirled_top);
    float top_steam_micro = micro_noise(tight_swirled_top * 1.5);
    float top_steam_ridged = ridged_noise(tight_swirled_top * 2.0);
    float top_steam = mix(mix(top_steam_base, top_steam_micro, 0.4), top_steam_ridged, 0.3);
    
    float bottom_steam_base = detailed_noise(tight_swirled_bottom);
    float bottom_steam_micro = micro_noise(tight_swirled_bottom * 1.3);
    float bottom_steam_ridged = ridged_noise(tight_swirled_bottom * 2.2);
    float bottom_steam = mix(mix(bottom_steam_base, bottom_steam_micro, 0.4), bottom_steam_ridged, 0.3);
    
    // Add multiple layers of cross-flow turbulence with enhanced texture
    vec2 cross_turb_coord1 = domain_warp(uv * vec2(12.0, 8.0) + vec2(time * speed * 0.025, time * speed * 0.012), time * speed * 0.045);
    vec2 tight_cross1 = tight_swirl(cross_turb_coord1, 0.3, 4.0);
    float cross_turbulence1 = mix(detailed_noise(tight_cross1), micro_noise(tight_cross1 * 1.2), 0.5);
    
    vec2 cross_turb_coord2 = swirl(uv * vec2(16.0, 12.0) + vec2(time * speed * -0.012, time * speed * 0.025), 0.1);
    vec2 tight_cross2 = tight_swirl(cross_turb_coord2, 0.25, 3.5);
    float cross_turbulence2 = fractal_noise(tight_cross2);
    
    vec2 cross_turb_coord3 = domain_warp(uv * vec2(20.0, 16.0) + vec2(time * speed * 0.018, time * speed * -0.006), time * speed * 0.06);
    vec2 tight_cross3 = tight_swirl(cross_turb_coord3, 0.2, 5.0);
    float cross_turbulence3 = ridged_noise(tight_cross3);
    
    // Add ultra-fine detail layers for maximum texture
    vec2 detail_coord = uv * vec2(24.0, 18.0) + vec2(sin(time * speed * 0.09) * 0.02, cos(time * speed * 0.075) * 0.02);
    vec2 tight_detail = tight_swirl(detail_coord, 0.15, 6.0);
    float fine_detail = micro_noise(tight_detail) * 0.3;
    
    vec2 ultra_detail_coord = uv * vec2(32.0, 24.0) + vec2(sin(time * speed * 0.12) * 0.01, cos(time * speed * 0.105) * 0.01);
    vec2 ultra_tight_detail = tight_swirl(ultra_detail_coord, 0.1, 8.0);
    float ultra_fine_detail = micro_noise(ultra_tight_detail) * 0.2;
    
    // Combine multiple turbulence layers for ultra-complex, textured cloud patterns
    float combined_cross_turb = (cross_turbulence1 * 0.35 + cross_turbulence2 * 0.25 + cross_turbulence3 * 0.2 + fine_detail * 0.15 + ultra_fine_detail * 0.05);
    
    // Create enhanced steam patterns with multiple texture layers
    float top_steam_pattern = mix(top_steam, combined_cross_turb, 0.6);
    float bottom_steam_pattern = mix(bottom_steam, combined_cross_turb, 0.6);
    
    // Add smoother large-scale turbulence near the unmasked area to break up transition lines
    // Use gentler transition zones to avoid visible boundaries
    float edge_influence = max(top_base_mask, bottom_base_mask);
    float center_distance = min(dist_from_top, dist_from_bottom);
    float smooth_transition_zone = smoothstep(0.1, 0.4, center_distance) * (1.0 - smoothstep(0.4, 0.7, center_distance));
    
    // Generate larger-scale turbulence for transition areas
    vec2 large_turb_coord = uv * vec2(4.0, 3.0) + vec2(time * speed * 0.006, time * speed * 0.009);
    vec2 warped_large = domain_warp(large_turb_coord, time * speed * 0.03);
    float large_turbulence = fractal_noise(warped_large) * 0.6;  // Reduced intensity
    
    // Apply smoother transition distortion to avoid visible lines
    float transition_distortion = smooth_transition_zone * large_turbulence * edge_influence * 0.2;  // Much gentler application
    
    // Apply much stronger steam distortion to the edge masks
    float top_steam_influence = top_base_mask * turbulence_amount * 1.2;  // Much stronger
    float bottom_steam_influence = bottom_base_mask * turbulence_amount * 1.2;
    
    // Create flowing steam edges with more dramatic variation and transition line breaking
    float top_mask = top_base_mask + (top_steam_pattern - 0.5) * top_steam_influence + transition_distortion;
    float bottom_mask = bottom_base_mask + (bottom_steam_pattern - 0.5) * bottom_steam_influence + transition_distortion;
    
    // Add additional wispy details by modulating the mask strength itself
    float wispy_modulation = (combined_cross_turb - 0.5) * 0.4;
    top_mask = top_mask * (1.0 + wispy_modulation * top_base_mask);
    bottom_mask = bottom_mask * (1.0 + wispy_modulation * bottom_base_mask);
    
    // Ensure masks stay within bounds
    top_mask = clamp(top_mask, 0.0, 1.0);
    bottom_mask = clamp(bottom_mask, 0.0, 1.0);
    
    // Combine the masks
    float combined_mask = max(top_mask, bottom_mask);
    
    // The mask defines transparency: 0 = fully transparent (masked), 1 = fully opaque (visible)
    float visibility = 1.0 - combined_mask * vignette_strength;
    visibility = clamp(visibility, 0.0, 1.0);
    
    // For a vignette effect, we want the EDGES to mask the background, not the center
    // So we need to INVERT the visibility for the alpha channel
    // High visibility (center) → Low alpha (transparent, let background through)
    // Low visibility (edges) → High alpha (opaque, show our mask effect)
    color.a = 1.0 - visibility;
    
    // Optional: Add subtle steam/cloud coloring in the masked areas
    // This creates a slight tint effect rather than pure transparency
    // vec3 steam_color = vec3(0.95, 0.95, 1.0) * 0.02; // Very faint bluish steam
    // float steam_visibility = combined_mask * 0.1; // Much more subtle
    // color.rgb = steam_color * steam_visibility;
    
    FragColor = color;
}
