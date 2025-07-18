#!/usr/bin/env python3
"""
Test script for the VastAI Manager

This script demonstrates how to use the VastAI manager to find/create instances
and interact with the experimance model server.
"""

import time
import json
import os
import requests
from vastai_manager import VastAIManager, InstanceEndpoint

from experimance_common.constants import GENERATED_IMAGES_DIR

TEST_PROMPTS = {
    "current+jungle": {
        "positive": "satellite image, jungle, river, dense urban metropolis, 2020s, river ferry, gated community, illegal gold mining, big box store, music festival, rivers, water, hills, valleys, detailed photorealistic, New Guinea Jungle, busy roads and highways, vibrant color, entangled, chaotic, dense, crowded, traffic, parking lots",
        "negative": "blurry, sketch, cartoon, illustration, oversaturated, fish-eye lens, vein, tentacle, vine"
    },
    "wilderness+deciduous_forest": {
        "positive": "colorful satellite image, coastal, pre-industrial landscape, wilderness, rivers, water, hills, valleys, vibrant hyper detailed photorealistic maximum detail, Norwegian Fjords",
        "negative": "blurry, sketch, cartoon, illustration, oversaturated, fish-eye lens, vein, tentacle, modern, plastic, path, road, roads, buildings, humans"
    },
    "current+boreal_forest_1": {
        "positive": "colorful satellite image, boreal forest, dense urban metropolis, 2020s, in the style of experimance, highways, suburban housing, power plant, company store for loggers, music festival, rivers, water, hills, valleys, vibrant hyper detailed photorealistic, Canadian Boreal Forest, busy roads and highways, in the style of (Burtynsky:1.1) and (Richter:1.1), vibrant (colorful), entangled, chaotic, dense, crowded, traffic, parking lots",
        "negative": "blurry, sketch, cartoon, illustration, oversaturated, fish-eye lens, vein, tentacle"
    },
    "current+boreal_forest_2": {
        "positive": "colorful satellite image, boreal forest, dense urban metropolis, 2020s, in the style of experimance, logging road, luxury condo tower, factories, pollution, skyscraper, music festival, rivers, water, hills, valleys, vibrant hyper detailed photorealistic, Alaskan Boreal Forest, busy roads and highways, in the style of (Burtynsky:1.1) and (Richter:1.1), vibrant (colorful), entangled, chaotic, dense, crowded, traffic, parking lots",
        "negative": "blurry, sketch, cartoon, illustration, oversaturated, fish-eye lens, vein, tentacle"
    },
    "current+river": {
        "positive": "colorful satellite image, river, dense urban metropolis, 2020s, in the style of experimance, light rail, suburban housing, factories, pollution, pop-up food market, urban park, rivers, water, hills, valleys, vibrant hyper detailed photorealistic, Mississippi River, busy roads and highways, in the style of (Burtynsky:1.1) and (Richter:1.1), vibrant (colorful), entangled, chaotic, dense, crowded, traffic, parking lots",
        "negative": "blurry, sketch, cartoon, illustration, oversaturated, fish-eye lens, vein, tentacle"
    }
}


def test_image_generation(endpoint: InstanceEndpoint):
    """Test image generation with the model server."""
    print(f"\n🧪 Testing image generation at {endpoint.url}")
    
    # Generate a mock depth map to send
    import base64
    import io
    import numpy as np
    from PIL import Image
    
    # Create a simple depth map (radial gradient)
    width, height = 1024, 1024
    center_x, center_y = width // 2, height // 2
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Normalize to 0-255 range and invert (closer = brighter)
    depth_array = (255 * (1 - distance / max_distance)).astype(np.uint8)
    depth_image = Image.fromarray(depth_array, mode='L')
    
    # Convert to base64
    buffer = io.BytesIO()
    depth_image.save(buffer, format='PNG')
    depth_map_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    # Use one of the TEST_PROMPTS with experimance era
    test_prompt_key = "current+jungle"
    test_prompt_data = TEST_PROMPTS[test_prompt_key]
    
    model = "lightning"  # Use the hyper model
    era = "experimance"  # Use the experimance LoRA
    steps = 6  # Use recommended 6 steps
    cfg = 2.0
    seed = 123456

    # Test payload with real experimance prompts and depth map
    test_payload = {
        "prompt": test_prompt_data["positive"],
        "negative_prompt": test_prompt_data["negative"],
        "depth_map_b64": depth_map_b64,
        "mock_depth": False,  # Using real depth map
        "model": model,
        "era": era, 
        "steps": steps,
        "cfg": cfg,
        "width": 1024,
        "height": 1024,
        "controlnet_strength": 0.8,
        "lora_strength": 1.0,
        "seed": seed,  # Fixed seed for reproducibility
        "scheduler": "auto",  # Use optimized default scheduler
        #"use_karras_sigmas": False  # Force Karras sigmas for better quality
    }
    
    print(f"   🎨 Using prompt: {test_prompt_key}")
    print(f"   🧬 Using era: experimance LoRA")
    print(f"   🗺️  Using generated depth map")
    
    try:
        print("   Sending generation request...")
        start_time = time.time()
        
        response = requests.post(
            f"{endpoint.url}/generate",
            headers={"Content-Type": "application/json"},
            json=test_payload,
            timeout=60
        )
        
        generation_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            cfg_used = result.get("metadata", {}).get("cfg", "unknown")
            assert cfg_used == cfg
            seed_used = result.get("seed_used", "unknown")
            assert seed_used == seed

            print(f"   ✅ Generation successful!")
            print(f"   ⏱️  Total time: {generation_time:.1f}s")
            print(f"   🎨 Model generation time: {result.get('generation_time', 0):.1f}s")
            print(f"   🎲 Seed used: {result.get('seed_used', 'unknown')}")
            print(f"   📏 Steps: {result.get('metadata', {}).get('steps', 'unknown')}")
            print(f"   🔧 CFG: {result.get('metadata', {}).get('cfg', 'unknown')}")
            print(f"   🧬 Era used: {result.get('era_used', 'none')}")
            print(f"   💪 LoRA strength: {result.get('metadata', {}).get('lora_strength', 'none')}")
            print(f"   🎯 ControlNet strength: {result.get('metadata', {}).get('controlnet_strength', 'unknown')}")
            
            # Check if we got image data
            if 'image_b64' in result:
                print(f"   📸 Image generated (base64 length: {len(result['image_b64'])})")
                
                # Save the image to see it
                import base64
                from datetime import datetime
                
                image_data = base64.b64decode(result['image_b64'])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = GENERATED_IMAGES_DIR / f"test_vastai_experimance_{timestamp}_{model}_cfg{cfg}_steps{steps}_{seed}.png"
                
                with open(filename, 'wb') as f:
                    f.write(image_data)
                
                print(f"   💾 Image saved as: {filename}")
                print(f"   📂 Full path: {os.path.abspath(filename)}")
            
            return True
        else:
            print(f"   ❌ Generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return False


def test_model_endpoints(endpoint: InstanceEndpoint):
    """Test various model server endpoints."""
    print(f"\n🔍 Testing model server endpoints at {endpoint.url}")
    
    endpoints_to_test = [
        ("/healthcheck", "Health check"),
        ("/models", "Available models"),
    ]
    
    for path, description in endpoints_to_test:
        try:
            response = requests.get(f"{endpoint.url}{path}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ {description}: OK")
                
                if path == "/healthcheck":
                    print(f"      Models loaded: {data.get('models_loaded', [])}")
                    print(f"      Memory usage: {data.get('memory_usage', {}).get('ram_mb', 0):.0f} MB RAM")
                    print(f"      Uptime: {data.get('uptime', 0):.1f}s")
                    
                elif path == "/models":
                    print(f"      Available models: {data.get('available_models', [])}")
                    print(f"      Available eras: {data.get('available_eras', [])}")
            else:
                print(f"   ❌ {description}: {response.status_code}")
                
        except requests.RequestException as e:
            print(f"   ❌ {description}: {e}")


def main():
    """Main test function."""
    print("🚀 VastAI Manager Test Script")
    print("=" * 50)
    
    # Initialize manager
    manager = VastAIManager()

    # test search offers
    # offers = manager.search_offers()
    # print(offers)
    # exit()
    
    # List existing instances
    print("\n📋 Checking for existing experimance instances...")
    existing_instances = manager.find_experimance_instances()
    
    if existing_instances:
        print(f"   Found {len(existing_instances)} existing instance(s):")
        for instance in existing_instances:
            print(f"   - Instance {instance['id']}: {instance.get('actual_status', 'unknown')}")
    else:
        print("   No existing experimance instances found")
    
    # Find or create instance
    print("\n🔧 Finding or creating ready instance...")
    endpoint = manager.find_or_create_instance(
        create_if_none=True,
        wait_for_ready=True
    )
    
    if not endpoint:
        print("❌ Failed to get a ready instance")
        return
    
    print(f"✅ Instance ready!")
    print(f"   Instance ID: {endpoint.instance_id}")
    print(f"   Public IP: {endpoint.public_ip}")
    print(f"   External Port: {endpoint.external_port}")
    print(f"   URL: {endpoint.url}")
    print(f"   Status: {endpoint.status}")
    
    # Test endpoints
    test_model_endpoints(endpoint)
    
    # Test image generation
    test_image_generation(endpoint)
    
    print("\n🎉 Testing complete!")
    print(f"\nYour model server is ready at: {endpoint.url}")
    print("You can now use this endpoint for your experimance installation.")


if __name__ == "__main__":
    main()
