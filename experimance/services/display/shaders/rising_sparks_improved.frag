#version 330 core

in vec2 tex_coords;
out vec4 FragColor;

uniform float time;
uniform vec2 resolution;
uniform float spark_intensity = 0.5;
uniform float horizontal_compression = 1.0;  // Compression factor for horizontal coordinates (6.0 for 6x projector stretch)

// Hash function for pseudo-random numbers
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Generate a random 2D vector
vec2 hash2(vec2 p) {
    return fract(sin(vec2(dot(p, vec2(12.9898, 78.233)), 
                          dot(p, vec2(139.234, 98.187)))) * 43758.5453);
}

// Convert RGB to luminance
float luminance(vec3 color) {
    return dot(color, vec3(0.299, 0.587, 0.114));
}

// Generate procedural bright spots to replace scene_texture sampling
vec3 generate_procedural_bright_spots(vec2 pos) {
    vec3 brightest_color = vec3(0.0);
    float max_brightness = 0.0;
    
    // Create a grid of potential bright spots
    for (float x = -2.0; x <= 2.0; x += 1.0) {
        for (float y = -2.0; y <= 2.0; y += 1.0) {
            vec2 sample_pos = pos + vec2(x, y) * 0.005;
            
            // Check bounds
            if (sample_pos.x >= 0.0 && sample_pos.x <= 1.0 && 
                sample_pos.y >= 0.0 && sample_pos.y <= 1.0) {
                
                // Generate procedural bright spots using noise
                float spot_seed = hash(floor(sample_pos * 50.0));
                
                // Only create bright spots occasionally
                if (spot_seed > 0.85) {
                    // Create a bright spot with some color variation
                    float brightness = 0.3 + spot_seed * 0.7;
                    
                    // Color variation for more interesting sparks
                    vec3 spot_color;
                    if (spot_seed > 0.95) {
                        // Rare white hot spots
                        spot_color = vec3(1.0, 0.95, 0.8) * brightness;
                    } else if (spot_seed > 0.9) {
                        // Orange/yellow spots
                        spot_color = vec3(1.0, 0.6, 0.2) * brightness;
                    } else {
                        // Reddish spots
                        spot_color = vec3(0.8, 0.3, 0.1) * brightness;
                    }
                    
                    float spot_brightness = luminance(spot_color);
                    if (spot_brightness > max_brightness) {
                        max_brightness = spot_brightness;
                        brightest_color = spot_color;
                    }
                }
            }
        }
    }
    
    return brightest_color;
}

// Single pixel spark with cross pattern, glow, and twinkling
float spark_pattern(vec2 uv, vec2 spark_pos, float intensity, float time, float spark_seed) {
    vec2 diff = uv - spark_pos;
    
    // Apply horizontal compression to the difference vector
    // This compresses the spark's horizontal dimension while keeping it in the same location
    diff.x *= horizontal_compression;
    
    float dist = length(diff);
    
    // Twinkling effect - multiple frequencies for organic variation
    float twinkle1 = sin(time * 8.0 + spark_seed * 15.0) * 0.5 + 0.5;
    float twinkle2 = sin(time * 12.0 + spark_seed * 20.0) * 0.3 + 0.7;
    float twinkle3 = sin(time * 5.0 + spark_seed * 10.0) * 0.2 + 0.8;
    float twinkle = twinkle1 * twinkle2 * twinkle3; // Combined twinkling
    
    // Apply twinkling to intensity - clamped to prevent going too dim
    float twinkling_intensity = intensity * (0.5 + 1.0 * twinkle); // Never goes below 50%
    
    // Single pixel core (very tight) with twinkling
    float core = exp(-dist * 2000.0) * twinkling_intensity;
    
    // Cross pattern (+ shape) with subtle twinkling
    float cross_twinkle = 0.7 + 0.3 * twinkle; // Gentler twinkling for cross
    float cross_h = exp(-abs(diff.y) * 800.0) * exp(-abs(diff.x) * 200.0) * twinkling_intensity * 0.6 * cross_twinkle;
    float cross_v = exp(-abs(diff.x) * 800.0) * exp(-abs(diff.y) * 200.0) * twinkling_intensity * 0.6 * cross_twinkle;
    
    // X pattern (diagonal) with twinkling
    float diag_twinkle = 0.8 + 0.2 * twinkle; // Even gentler for diagonals
    float diag1 = exp(-abs(diff.x - diff.y) * 400.0) * exp(-dist * 300.0) * twinkling_intensity * 0.4 * diag_twinkle;
    float diag2 = exp(-abs(diff.x + diff.y) * 400.0) * exp(-dist * 300.0) * twinkling_intensity * 0.4 * diag_twinkle;
    
    // Very faint glow (steady, no twinkling for subtle base)
    float glow = exp(-dist * 100.0) * intensity * 0.1;
    
    return core + cross_h + cross_v + diag1 + diag2 + glow;
}

void main() {
    vec2 uv = tex_coords;
    
    // This is an additive effect shader - start with full transparency
    // We'll only add color where sparks appear
    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);  // Completely transparent base
    
    // Spark accumulation
    vec3 spark_contribution = vec3(0.0);
    
    // Variable number of sparks over time - between 20 and 45
    float time_variation = sin(time * 0.3) * sin(time * 0.7) * sin(time * 0.13); // Complex variation
    int base_sparks = 32;
    int spark_variation = int(15.0 * time_variation); // +/- 15 sparks
    int num_sparks = base_sparks + spark_variation;
    
    for (int i = 0; i < 60; i++) { // Max loop size for GPU compatibility
        if (i >= num_sparks) break; // Dynamic break based on current spark count
        
        // Random spark seed
        float spark_id = float(i);
        vec2 random_seed = vec2(spark_id * 0.1, spark_id * 0.2);
        
        // Random spawn position (ONLY in lower half of screen)
        vec2 random_pos = hash2(random_seed * 100.0);
        vec2 spawn_pos = vec2(random_pos.x, random_pos.y * 0.5); // Y: 0.0 to 0.5 (lower half in texture coords)
        
        // Generate procedural bright spots instead of sampling texture
        vec3 source_color = generate_procedural_bright_spots(spawn_pos);
        float source_brightness = luminance(source_color);
        
        // Only create spark if there's some brightness in the procedural area
        if (source_brightness > 0.15) {  // Lower threshold for more sparks
            // Unique timing and properties for this spark
            float spark_seed = hash(random_seed);
            
            // Individual spark lifetime (2-6 seconds) 
            float spark_cycle_time = 2.0 + spark_seed * 4.0; // Much more variable lifetime
            float cycle_offset = spark_seed * spark_cycle_time;
            float current_time = mod(time * 0.6 + cycle_offset, spark_cycle_time);
            float life_progress = current_time / spark_cycle_time;
            
            // Individual spark velocity (how fast it rises)
            float spark_velocity = 0.8 + spark_seed * 1.2; // 0.8x to 2.0x speed variation
            
            // Individual visibility duration (some sparks last longer than others)
            float visibility_duration = 0.6 + spark_seed * 0.3; // 60% to 90% of lifetime
            
            // Spark visible for its individual duration
            if (life_progress < visibility_duration) {
                // Calculate base upward movement with individual velocity
                float base_rise_distance = 0.6; // Increased base travel distance
                float individual_rise_distance = base_rise_distance * spark_velocity;
                float current_y = spawn_pos.y + life_progress * individual_rise_distance;
                
                // Add wind gusts - create occasional strong horizontal forces
                float gust_frequency = 0.2; // How often gusts occur
                float gust_time = time * gust_frequency;
                float gust_strength = sin(gust_time + spark_seed * 5.0) * sin(gust_time * 2.3 + spark_seed * 3.0);
                gust_strength = smoothstep(0.4, 1.0, abs(gust_strength)) * sign(gust_strength); // Only strong gusts
                
                // Swirl and swoop movement
                float swirl_time = time * 0.8 + spark_seed * 10.0;
                float swirl_radius = 0.02 * life_progress * spark_velocity; // Faster sparks swirl more
                float swirl_angle = swirl_time * (2.0 + spark_seed * 3.0); // Variable swirl speed
                
                // Swooping motion - periodic arcs (affected by individual velocity)
                float swoop_frequency = (1.5 + spark_seed * 1.0) * spark_velocity; // Faster sparks swoop faster
                float swoop_phase = sin(time * swoop_frequency + spark_seed * 6.28);
                float swoop_x = swoop_phase * 0.015 * life_progress; // Horizontal swooping
                float swoop_y = abs(swoop_phase) * 0.008 * life_progress; // Slight vertical swooping
                
                // Thermal updraft effect - faster sparks get more upward boost
                float thermal_boost = spark_velocity * 0.1 * sin(time * 0.3 + spark_seed * 3.0) * life_progress;
                
                // Combine all movements
                float total_drift_x = 
                    gust_strength * 0.08 * life_progress +  // Wind gusts
                    sin(swirl_angle) * swirl_radius +        // Swirl motion
                    swoop_x;                                 // Swooping
                    
                float total_drift_y = 
                    cos(swirl_angle) * swirl_radius * 0.5 +  // Swirl motion (reduced Y)
                    swoop_y +                                // Swooping
                    thermal_boost;                           // Thermal updrafts
                
                // Small flutter for organic feel (scaled by velocity)
                float flutter_scale = 0.5 + spark_velocity * 0.5; // Faster sparks flutter more
                float flutter_x = sin(time * 4.0 + spark_seed * 8.0) * 0.003 * flutter_scale;
                float flutter_y = cos(time * 5.0 + spark_seed * 7.0) * 0.002 * flutter_scale;
                
                // Final spark position
                vec2 spark_pos = vec2(
                    spawn_pos.x + total_drift_x + flutter_x,
                    current_y + total_drift_y + flutter_y
                );
                
                // Only render if spark is still on screen
                if (spark_pos.x >= 0.0 && spark_pos.x <= 1.0 && 
                    spark_pos.y >= 0.0 && spark_pos.y <= 1.0) {
                    
                    // Age-based fading
                    float age_fade = 1.0 - smoothstep(0.0, visibility_duration, life_progress);
                    age_fade = pow(age_fade, 0.8); // Slightly slower fade
                    
                    // Additional fade based on distance traveled (faster sparks fade differently)
                    float travel_fade = 1.0 - (life_progress * spark_velocity) * 0.2;
                    travel_fade = max(travel_fade, 0.3); // Don't fade completely
                    
                    // Velocity affects brightness (faster sparks can be brighter)
                    float velocity_brightness = 0.8 + spark_velocity * 0.4;
                    
                    // Calculate spark pattern intensity
                    float spark_pattern_intensity = age_fade * travel_fade * velocity_brightness * source_brightness * spark_intensity * 1.5;
                    
                    // Get spark contribution with twinkling
                    float pattern_strength = spark_pattern(uv, spark_pos, spark_pattern_intensity, time, spark_seed);
                    
                    if (pattern_strength > 0.001) {
                        // Enhanced spark color - preserve more of the original color
                        vec3 enhanced_color = source_color;
                        
                        // Add fire/spark enhancement but keep original character
                        vec3 fire_tint = vec3(1.2, 0.8, 0.3); // Warm tint
                        enhanced_color = enhanced_color * fire_tint; // Multiply to enhance while preserving
                        
                        // Brighten based on original brightness
                        enhanced_color *= 1.5 + source_brightness;
                        
                        // Add spark contribution
                        spark_contribution += enhanced_color * pattern_strength;
                    }
                }
            }
        }
    }
    
    // Apply spark effects with alpha blending instead of additive
    // For alpha blending, we want: sparks * alpha + background * (1-alpha)
    // So we'll use a more subdued approach that layers on top of the background
    
    // Scale down the sparks for alpha blending (they were designed for additive)
    vec3 final_sparks = spark_contribution * 0.5; // Reduce intensity
    
    // For alpha blending, we want to "add light" to what's underneath
    // So we'll make sparks appear as bright additions
    color.rgb = final_sparks;
    
    // Use spark brightness as alpha - where there are sparks, they'll blend with background
    float spark_brightness = max(max(final_sparks.r, final_sparks.g), final_sparks.b);
    color.a = spark_brightness; // Direct alpha from brightness
    
    FragColor = color;
}
