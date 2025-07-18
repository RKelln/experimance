#!/usr/bin/env python3
"""
Test script for the VastAI Manager

This script demonstrates how to use the VastAI manager to find/create instances
and interact with the experimance model server.
"""

import time
import json
import os
import argparse
import requests
from typing import Optional
from .vastai_manager import VastAIManager, InstanceEndpoint

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


def test_offers_search(manager: VastAIManager, min_gpu_ram: int = 20, max_price: float = 0.5, dlperf: float = 32.0):
    """Test searching for offers and show what would be selected."""
    print(f"\n🔍 Searching for offers...")
    print(f"   Parameters: min_gpu_ram={min_gpu_ram}GB, max_price=${max_price:.2f}/hr, dlperf>={dlperf}")
    
    try:
        offers = manager.search_offers(
            min_gpu_ram=min_gpu_ram,
            max_price=max_price,
            dlperf=dlperf
        )
        
        if not offers:
            print("   ❌ No offers found matching criteria")
            return None
        
        print(f"   ✅ Found {len(offers)} offer(s)")
        print(f"\n   📊 Top offers (ranked by smart selection):")
        
        # Calculate scores for display
        cheapest_price = min(offer.get('dph_total', float('inf')) for offer in offers)
        price_threshold = cheapest_price * 1.25  # 25% tolerance
        
        for i, offer in enumerate(offers[:5]):  # Show top 5
            gpu_name = offer.get('gpu_name', 'unknown')
            price = offer.get('dph_total', 0)
            reliability = offer.get('reliability2', 0)
            gpu_ram = offer.get('gpu_ram', 0) / 1024  # Convert MB to GB for display
            dlperf_val = offer.get('dlperf', 0)
            dlperf_per_dollar = offer.get('dlperf_per_dphtotal', 0)
            location = offer.get('geolocation', 'unknown')
            inet_down = offer.get('inet_down', 0)
            verified = offer.get('verified', False)
            
            # Calculate the score for this offer
            score = manager._calculate_offer_score(offer, cheapest_price, price_threshold)
            
            selected_marker = "🏆 " if i == 0 else f"   {i+1}. "
            print(f"{selected_marker}Offer {offer['id']}: {gpu_name} ({gpu_ram:.1f}GB)")
            print(f"       Price: ${price:.3f}/hr | DLPerf: {dlperf_val:.1f} | DLPerf/$: {dlperf_per_dollar:.1f}")
            print(f"       Reliability: {reliability:.3f} | Location: {location} | Verified: {verified}")
            print(f"       Download: {inet_down:.0f}Mbps | Smart Score: {score:.1f}")
            print()
        
        best_offer = offers[0]
        print(f"   🎯 Smart selection: Offer {best_offer['id']} - {best_offer['gpu_name']} at ${best_offer['dph_total']:.3f}/hr")
        print(f"      (DLPerf/$ ratio: {best_offer.get('dlperf_per_dphtotal', 0):.1f})")
        
        return best_offer
        
    except Exception as e:
        print(f"   ❌ Failed to search offers: {e}")
        return None


def test_provision_instance(manager: VastAIManager, offer_id: Optional[int] = None, wait_for_ready: bool = True):
    """Test provisioning a new instance."""
    print(f"\n🚀 Provisioning new instance...")
    
    if offer_id is None:
        print("   🔍 Searching for best offer...")
        offers = manager.search_offers()
        if not offers:
            print("   ❌ No offers found")
            return None
        
        best_offer = offers[0]
        offer_id = best_offer['id']
        print(f"   🎯 Selected offer {offer_id}: {best_offer['gpu_name']} at ${best_offer['dph_total']:.3f}/hr")
    
    if not offer_id:
        print("   ❌ No offer ID provided or found")
        return None

    try:
        print(f"   📝 Creating instance from offer {offer_id}...")
        result = manager.create_instance(offer_id)
        instance_id = result.get("new_contract")
        
        if not instance_id:
            print(f"   ❌ Failed to create instance: {result}")
            return None
        
        print(f"   ✅ Instance {instance_id} created successfully!")
        
        # Get SSH command for the new instance
        ssh_command = manager.get_ssh_command(instance_id)
        if ssh_command:
            print(f"   🔗 SSH: {ssh_command}")
        
        if wait_for_ready:
            print(f"   ⏳ Waiting for instance to be ready...")
            if manager.wait_for_instance_ready(instance_id):
                endpoint = manager.get_model_server_endpoint(instance_id)
                if endpoint:
                    print(f"   🎉 Instance ready at {endpoint.url}")
                    if ssh_command:
                        print(f"   🔗 SSH: {ssh_command}")
                    return endpoint
                else:
                    print(f"   ❌ Failed to get endpoint for instance {instance_id}")
                    return None
            else:
                print(f"   ❌ Instance failed to become ready")
                return None
        else:
            print(f"   ℹ️  Instance is being provisioned (not waiting for ready)")
            return manager.get_model_server_endpoint(instance_id)
            
    except Exception as e:
        print(f"   ❌ Failed to provision instance: {e}")
        return None


def test_list_instances(manager: VastAIManager):
    """Test listing existing instances."""
    print(f"\n📋 Listing existing instances...")
    
    try:
        # List all instances
        all_instances = manager.show_instances()
        print(f"   Total instances: {len(all_instances)}")
        
        if all_instances:
            print(f"\n   📊 All instances:")
            for instance in all_instances:
                status = instance.get('actual_status', 'unknown')
                template_name = instance.get('template_name', 'unknown')
                price = instance.get('dph_total', 0)
                gpu_name = instance.get('gpu_name', 'unknown')
                
                print(f"   - Instance {instance['id']}: {status}")
                print(f"     Template: {template_name}")
                print(f"     GPU: {gpu_name} | Price: ${price:.3f}/hr")
                
                # Add SSH command if available
                ssh_command = manager.get_ssh_command(instance['id'])
                if ssh_command:
                    print(f"     SSH: {ssh_command}")
                
                print()
        
        # List experimance instances specifically
        experimance_instances = manager.find_experimance_instances()
        print(f"   Experimance instances: {len(experimance_instances)}")
        
        if experimance_instances:
            print(f"\n   🎨 Experimance instances:")
            for instance in experimance_instances:
                print(f"   - Instance {instance['id']}: {instance.get('actual_status', 'unknown')}")
                endpoint = manager.get_model_server_endpoint(instance['id'])
                if endpoint:
                    print(f"     URL: {endpoint.url}")
                
                # Add SSH command
                ssh_command = manager.get_ssh_command(instance['id'])
                if ssh_command:
                    print(f"     SSH: {ssh_command}")
                
                print()
        
        return experimance_instances
        
    except Exception as e:
        print(f"   ❌ Failed to list instances: {e}")
        return []


def main():
    """Main test function with argparse support."""
    parser = argparse.ArgumentParser(
        description="Test VastAI Manager functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test searching for offers (shows what would be selected)
  python -m test_vastai_manager --offers

  # Test provisioning a new instance
  python -m test_vastai_manager --provision

  # Test image generation with existing instance
  python -m test_vastai_manager --generate

  # List all instances
  python -m test_vastai_manager --list

  # Full test suite (default behavior)
  python -m test_vastai_manager --full

  # Search offers with custom parameters
  python -m test_vastai_manager --offers --max-price 0.3 --min-gpu-ram 24
        """
    )
    
    # Action selection (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument('--offers', action='store_true', 
                             help='Test searching for offers and show selection')
    action_group.add_argument('--provision', action='store_true',
                             help='Test provisioning a new instance')
    action_group.add_argument('--generate', action='store_true',
                             help='Test image generation with existing instance')
    action_group.add_argument('--list', action='store_true',
                             help='List all instances')
    action_group.add_argument('--full', action='store_true',
                             help='Run full test suite (default)')
    
    # Offer search parameters
    parser.add_argument('--min-gpu-ram', type=int, default=20,
                       help='Minimum GPU RAM in GB (default: 20)')
    parser.add_argument('--max-price', type=float, default=0.5,
                       help='Maximum price per hour (default: 0.5)')
    parser.add_argument('--dlperf', type=float, default=32.0,
                       help='Minimum DLPerf score (default: 32.0)')
    
    # Provisioning parameters
    parser.add_argument('--offer-id', type=int,
                       help='Specific offer ID to provision (if not specified, best will be selected)')
    parser.add_argument('--no-wait', action='store_true',
                       help='Don\'t wait for instance to be ready when provisioning')
    
    # Generation parameters
    parser.add_argument('--instance-id', type=int,
                       help='Specific instance ID to use for generation (if not specified, will find existing)')
    
    args = parser.parse_args()
    
    # If no action specified, default to full test
    if not any([args.offers, args.provision, args.generate, args.list]):
        args.full = True
    
    print("🚀 VastAI Manager Test Script")
    print("=" * 50)
    
    # Initialize manager
    manager = VastAIManager()
    
    if args.offers:
        test_offers_search(manager, args.min_gpu_ram, args.max_price, args.dlperf)
    
    elif args.provision:
        endpoint = test_provision_instance(manager, args.offer_id, not args.no_wait)
        if endpoint:
            print(f"\n✅ Provisioning successful!")
            print(f"   URL: {endpoint.url}")
            print(f"   Instance ID: {endpoint.instance_id}")
    
    elif args.generate:
        # Find existing instance or use specified one
        if args.instance_id:
            endpoint = manager.get_model_server_endpoint(args.instance_id)
            if not endpoint:
                print(f"❌ Instance {args.instance_id} not found or not accessible")
                return
        else:
            existing_instances = manager.find_experimance_instances()
            if not existing_instances:
                print("❌ No existing experimance instances found")
                print("   Use --provision to create a new instance first")
                return
            
            instance_id = existing_instances[0]['id']
            endpoint = manager.get_model_server_endpoint(instance_id)
            if not endpoint:
                print(f"❌ Instance {instance_id} not accessible")
                return
        
        print(f"🎨 Testing image generation with instance {endpoint.instance_id}")
        test_model_endpoints(endpoint)
        test_image_generation(endpoint)
    
    elif args.list:
        test_list_instances(manager)
    
    elif args.full:
        # Original full test behavior
        test_list_instances(manager)
        
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
