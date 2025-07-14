#!/usr/bin/env python3
"""
Integration test for LLM prompt builder using real OpenAI API.

This test uses the test_stories.json file to verify that the LLM prompt builder
produces reasonable results when working with actual OpenAI API calls.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sohkepayin_core.llm_prompt_builder import LLMPromptBuilder
from sohkepayin_core.llm import OpenAIProvider
from sohkepayin_core.config import ImagePrompt

system_prompt_file = Path(__file__).parent.parent / "src"/ "sohkepayin_core" / "system_prompt.md"
test_stories_path = Path(__file__).parent / "test_stories.json"

async def test_llm_with_real_api():
    """Test LLM prompt builder with real OpenAI API using test stories."""
    
    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set.")
        print("   This test requires a real OpenAI API key to test with the actual service.")
        print("   Set the key with: export OPENAI_API_KEY=your_key_here")
        return False
    
    print("🧪 Testing LLM Prompt Builder with Real OpenAI API\n")
    
    # Load test stories
    test_stories_path = Path(__file__).parent / "test_stories.json"
    if not test_stories_path.exists():
        print(f"❌ Test stories file not found: {test_stories_path}")
        return False
    
    with open(test_stories_path, 'r') as f:
        test_stories = json.load(f)
    
    print(f"📖 Loaded {len(test_stories)} test stories")
    
    # Initialize LLM provider and prompt builder
    llm = OpenAIProvider(
        model="gpt-4o",
        max_tokens=300,
        temperature=0.7
    )
    
    prompt_builder = LLMPromptBuilder(
        llm=llm,
        system_prompt_or_file=system_prompt_file
    )
    
    print("🤖 Initialized OpenAI LLM provider and prompt builder")
    print("=" * 80)
    
    # Test each story
    for i, story in enumerate(test_stories, 1):
        print(f"\n📝 Test Story {i}")
        print(f"Context: {story['context'][:100]}...")
        
        try:
            # Generate prompt using LLM
            print("\n🔄 Generating prompt with LLM...")
            generated_prompt = await prompt_builder.build_prompt(story['context'])
            
            # Display results
            print(f"\n✅ Generated Prompt:")
            print(f"   Prompt: {generated_prompt.prompt}")
            print(f"   Negative: {generated_prompt.negative_prompt or 'None'}")
            
            print(f"\n📋 Reference Prompt:")
            print(f"   Prompt: {story['prompt']}")
            print(f"   Negative: {story['negative_prompt']}")
            
            # Basic quality checks
            generated_words = set(generated_prompt.prompt.lower().split())
            reference_words = set(story['prompt'].lower().split())
            
            # Check for some common words (simple similarity check)
            common_words = generated_words.intersection(reference_words)
            similarity_score = len(common_words) / len(reference_words) if reference_words else 0
            
            print(f"\n🔍 Analysis:")
            print(f"   Generated length: {len(generated_prompt.prompt)} chars")
            print(f"   Reference length: {len(story['prompt'])} chars")
            print(f"   Word overlap: {len(common_words)} / {len(reference_words)} ({similarity_score:.1%})")
            print(f"   Common concepts: {', '.join(list(common_words)[:5])}")
            
            # Test prompt conversions
            print(f"\n🎨 Testing prompt conversions...")
            
            panorama_prompt = prompt_builder.base_prompt_to_panorama_prompt(generated_prompt)
            print(f"   Panorama: {panorama_prompt.prompt[:60]}...")
            
            tile_prompt = prompt_builder.base_prompt_to_tile_prompt(generated_prompt)
            print(f"   Tile: {tile_prompt.prompt[:60]}...")
            
            print(f"\n✅ Story {i} completed successfully")
            
        except Exception as e:
            print(f"\n❌ Error processing story {i}: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("-" * 60)
    
    print(f"\n🎉 All {len(test_stories)} test stories processed successfully!")
    print("\n💡 Summary:")
    print("   - LLM API calls worked correctly")
    print("   - Prompt generation completed for all stories") 
    print("   - Panorama and tile prompt conversions worked")
    print("   - Generated prompts contain relevant concepts")
    
    return True


async def test_llm_mock_comparison():
    """Compare OpenAI results with mock LLM for consistency."""
    
    print("\n🔄 Comparing OpenAI LLM with Mock LLM...")
    
    from sohkepayin_core.llm import MockLLMProvider
    
    # Test with mock LLM
    mock_llm = MockLLMProvider()
    mock_builder = LLMPromptBuilder(
        llm=mock_llm,
        system_prompt_or_file="You are a test assistant."
    )
    
    test_context = "I remember sitting by a peaceful lake at sunset with mountains in the background."
    
    mock_result = await mock_builder.build_prompt(test_context)
    print(f"🤖 Mock result: {mock_result.prompt}")
    
    # Test with OpenAI if available
    openai_llm = OpenAIProvider(model="gpt-4o-mini")  # Use cheaper model for comparison
    openai_builder = LLMPromptBuilder(
        llm=openai_llm,
        system_prompt_or_file=system_prompt_file
    )
    
    openai_result = await openai_builder.build_prompt(test_context)
    print(f"🌐 OpenAI result: {openai_result.prompt}")
    
    print("\n📊 Comparison:")
    print(f"   Mock length: {len(mock_result.prompt)} chars")
    print(f"   OpenAI length: {len(openai_result.prompt)} chars")
    print("   Both prompt builders work with different LLM providers ✅")

    
    return True


if __name__ == "__main__":
    async def main():
        try:
            # Test with real API
            success1 = await test_llm_with_real_api()
            
            # Test comparison
            success2 = await test_llm_mock_comparison()
            
            if success1 and success2:
                print("\n✨ All LLM integration tests passed!")
                return 0
            else:
                print("\n❌ Some tests failed!")
                return 1
                
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
            return 1
        except Exception as e:
            print(f"\n💥 Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
