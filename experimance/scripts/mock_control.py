#!/usr/bin/env python3
"""
Simple control script for the mock audience detector.

Usage:
    python mock_control.py present             # Signal presence
    python mock_control.py absent              # Signal absence  
    python mock_control.py count 3             # Set person count to 3
    python mock_control.py status              # Show current control directory status
    python mock_control.py -i                  # Interactive mode with keyboard controls
    python mock_control.py --interactive       # Interactive mode with keyboard controls
    python mock_control.py --test-scenarios    # Run automated test scenarios
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add the agent source to the path so we can import the helper
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "agent" / "src"))

from agent.vision.mock_detector import FileControlHelper


def get_timestamp():
    """Get current timestamp in a readable format."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds


def log_step(message: str):
    """Log a test step with timestamp."""
    timestamp = get_timestamp()
    print(f"[{timestamp}] {message}")


def wait_with_countdown(seconds: int, message: str = ""):
    """Wait with a visible countdown and timestamps."""
    if message:
        log_step(f"⏱️  {message}")
    
    start_time = get_timestamp()
    for i in range(seconds, 0, -1):
        print(f"  [{get_timestamp()}] ⏱️  {i}s remaining...", end='\r')
        time.sleep(1)
    end_time = get_timestamp()
    print(f"  [{end_time}] ⏱️  Done! (started at {start_time})           ")


def test_scenario_idle_timeout():
    """Test scenario: Verify idle timeout behavior with presence detection."""
    print("🧪 Test Scenario: Idle Timeout with Presence Detection")
    print("=" * 55)
    print()
    print("This test verifies that:")
    print("1. Agent sends re-engagement message when someone is present but idle")
    print("2. Agent ends conversation when no one is present during timeout")
    print("3. No infinite re-engagement loops")
    print()
    
    helper = FileControlHelper()
    
    try:
        # Start with no audience
        log_step("🎬 Step 1: Start with no audience")
        helper.set_absent()
        wait_with_countdown(3, "Setting initial state...")
        
        # Signal presence to start conversation
        log_step("🎬 Step 2: Person arrives - should trigger greeting")
        helper.set_present()
        wait_with_countdown(8, "Waiting for agent to detect presence and start conversation...")
        
        # Wait for idle timeout (should get re-engagement message)
        log_step("🎬 Step 3: Stay silent and wait for idle timeout re-engagement")
        log_step("   📋 Expected: Agent should send 'I'm still here...' message after 30s of silence")
        wait_with_countdown(40, "Waiting for idle timeout (30s + buffer)...")
        
        # Wait a bit more to see the re-engagement message
        log_step("🎬 Step 4: Continue waiting to observe re-engagement message")
        wait_with_countdown(10, "Time for re-engagement message to be spoken...")
        
        # Person leaves after re-engagement 
        log_step("🎬 Step 5: Person leaves - agent should end conversation gracefully")
        helper.set_absent()
        wait_with_countdown(10, "Person left, agent should detect absence and end conversation...")
        
        # Wait to ensure no more re-engagement messages
        log_step("🎬 Step 6: Verify no more re-engagement messages")
        log_step("   📋 Expected: NO additional re-engagement messages should appear")
        wait_with_countdown(15, "Monitoring for any unexpected re-engagement messages...")
        
        log_step("✅ Test completed! Check agent logs for expected behavior.")
        print()
        
    except KeyboardInterrupt:
        log_step("🛑 Test interrupted by user")
        helper.set_absent()


def test_scenario_rapid_changes():
    """Test scenario: Rapid presence changes."""
    print("🧪 Test Scenario: Rapid Presence Changes")
    print("=" * 40)
    print()
    print("This tests handling of rapid presence detection changes")
    print()
    
    helper = FileControlHelper()
    
    try:
        changes = [
            (False, "Start: No one present"),
            (True, "Person arrives"),
            (False, "Person leaves quickly"),
            (True, "Person returns"),
            (False, "Person leaves again"),
            (True, "Person arrives final time"),
        ]
        
        for i, (present, description) in enumerate(changes, 1):
            log_step(f"🎬 Step {i}: {description}")
            if present:
                helper.set_present()
            else:
                helper.set_absent()
            wait_with_countdown(3, "Waiting for detection...")
            
        # Final cleanup
        log_step("🎬 Final: Cleanup - set to absent")
        helper.set_absent()
        
        log_step("✅ Rapid changes test completed!")
        print()
        
    except KeyboardInterrupt:
        log_step("🛑 Test interrupted by user")
        helper.set_absent()


def test_scenario_multiple_people():
    """Test scenario: Multiple people detection."""
    print("🧪 Test Scenario: Multiple People Detection")
    print("=" * 42)
    print()
    print("This tests handling of varying person counts")
    print()
    
    helper = FileControlHelper()
    
    try:
        scenarios = [
            (0, "Start: Empty room"),
            (1, "One person arrives"),
            (3, "More people join (3 total)"),
            (5, "Busy room (5 people)"),
            (2, "Some people leave (2 remaining)"),
            (1, "Down to 1 person"),
            (0, "Last person leaves"),
        ]
        
        for i, (count, description) in enumerate(scenarios, 1):
            log_step(f"🎬 Step {i}: {description}")
            if count == 0:
                helper.set_absent()
            else:
                helper.set_count(count)
            wait_with_countdown(4, f"Setting count to {count}...")
            
        log_step("✅ Multiple people test completed!")
        print()
        
    except KeyboardInterrupt:
        log_step("🛑 Test interrupted by user")
        helper.set_absent()


def test_scenario_conversation_cycle():
    """Test scenario: Complete conversation cycle."""
    print("🧪 Test Scenario: Complete Conversation Cycle")
    print("=" * 45)
    print()
    print("This tests a realistic conversation flow:")
    print("1. Person arrives -> greeting")
    print("2. Conversation active")  
    print("3. Person leaves -> goodbye")
    print("4. Cooldown period")
    print("5. New person arrives")
    print()
    
    helper = FileControlHelper()
    
    try:
        # Person arrives
        log_step("🎬 Step 1: Person arrives")
        helper.set_present()
        wait_with_countdown(8, "Waiting for greeting...")
        
        # Simulate conversation active period
        log_step("🎬 Step 2: Conversation active period")
        wait_with_countdown(10, "Simulating active conversation...")
        
        # Person leaves
        log_step("🎬 Step 3: Person leaves")
        helper.set_absent()
        wait_with_countdown(5, "Agent should detect departure and say goodbye...")
        
        # Cooldown period
        log_step("🎬 Step 4: Cooldown period")
        log_step("   📋 Expected: No new conversations should start during cooldown")
        wait_with_countdown(8, "Cooldown period (no new conversations should start)...")
        
        # New person arrives
        log_step("🎬 Step 5: New person arrives after cooldown")
        helper.set_present()
        wait_with_countdown(8, "Should start new conversation...")
        
        # Cleanup
        helper.set_absent()
        
        log_step("✅ Conversation cycle test completed!")
        print()
        
    except KeyboardInterrupt:
        log_step("🛑 Test interrupted by user")
        helper.set_absent()


def test_scenario_re_engagement_only():
    """Test scenario: Focus specifically on re-engagement behavior."""
    print("🧪 Test Scenario: Re-engagement Message Test")
    print("=" * 45)
    print()
    print("This test specifically focuses on the re-engagement feature:")
    print("1. Person arrives and gets greeting")
    print("2. Wait for idle timeout to trigger re-engagement")
    print("3. Verify re-engagement message is sent")
    print("4. Then person leaves")
    print()
    
    helper = FileControlHelper()
    
    try:
        # Start clean
        log_step("🎬 Step 1: Start with no audience")
        helper.set_absent()
        wait_with_countdown(3, "Setting initial state...")
        
        # Person arrives
        log_step("🎬 Step 2: Person arrives")
        helper.set_present()
        wait_with_countdown(10, "Waiting for greeting to complete...")
        
        # Wait specifically for re-engagement (give extra time)
        log_step("🎬 Step 3: Wait for re-engagement (staying present and silent)")
        log_step("   📋 Expected: 'I'm still here if you'd like to continue our conversation.'")
        log_step("   📋 This should happen ~30 seconds after greeting ends")
        wait_with_countdown(45, "Waiting for idle timeout and re-engagement message...")
        
        # Give time for the re-engagement message to be delivered
        log_step("🎬 Step 4: Allow time for re-engagement message delivery")
        wait_with_countdown(10, "Time for message to be spoken...")
        
        # Wait to see if another re-engagement happens (it shouldn't immediately)
        log_step("🎬 Step 5: Monitor for second re-engagement (should take another 30s)")
        wait_with_countdown(25, "Checking if second re-engagement cycle starts...")
        
        # Finally, person leaves
        log_step("🎬 Step 6: Person leaves")
        helper.set_absent()
        wait_with_countdown(5, "Person departed...")
        
        log_step("✅ Re-engagement test completed!")
        print()
        
    except KeyboardInterrupt:
        log_step("🛑 Test interrupted by user")
        helper.set_absent()


def run_test_scenarios():
    """Run all automated test scenarios."""
    start_timestamp = get_timestamp()
    print("🚀 Running Automated Test Scenarios")
    print("=" * 40)
    print(f"📅 Test session started at: {start_timestamp}")
    print()
    print("These scenarios test various presence detection edge cases.")
    print("Monitor the agent logs during testing to verify expected behavior.")
    print()
    print("💡 Tip: Use timestamps to correlate test steps with agent logs!")
    print("Press Ctrl+C to skip to next test or exit.")
    print()
    
    scenarios = [
        ("Idle Timeout Test", test_scenario_idle_timeout),
        ("Re-engagement Only Test", test_scenario_re_engagement_only),
        ("Rapid Changes Test", test_scenario_rapid_changes), 
        ("Multiple People Test", test_scenario_multiple_people),
        ("Conversation Cycle Test", test_scenario_conversation_cycle),
    ]
    
    for name, test_func in scenarios:
        try:
            log_step(f"▶️  Starting: {name}")
            test_func()
            
            # Ask if user wants to continue
            try:
                response = input("Continue to next test? (y/N): ").strip().lower()
                if response not in ['y', 'yes']:
                    log_step("Tests stopped by user.")
                    break
            except KeyboardInterrupt:
                log_step("Tests stopped by user.")
                break
                
        except KeyboardInterrupt:
            log_step(f"Skipping {name}...")
            continue
    
    # Final cleanup
    helper = FileControlHelper()
    helper.set_absent()
    end_timestamp = get_timestamp()
    log_step(f"🏁 All tests completed! Final state: absent")
    print(f"📅 Test session ended at: {end_timestamp}")


def show_status(control_dir: str = "/tmp/mock_detector"):
    """Show current status of control directory."""
    control_path = Path(control_dir)
    
    print(f"Mock detector control directory: {control_path}")
    print(f"Directory exists: {control_path.exists()}")
    
    if control_path.exists():
        files = list(control_path.iterdir())
        if files:
            print("Control files present:")
            for file in files:
                print(f"  - {file.name}")
        else:
            print("No control files present")
    
    print("\nUsage:")
    print("  python mock_control.py present     # Signal presence")
    print("  python mock_control.py absent      # Signal absence")
    print("  python mock_control.py count N     # Set person count to N")
    print("  python mock_control.py status      # Show this status")
    print("  python mock_control.py -i          # Interactive mode")


def interactive_mode():
    """Interactive keyboard control mode."""
    print("🎮 Interactive Mock Detector Control")
    print("====================================")
    print()
    print("Commands:")
    print("  p       - Signal presence (1 person)")
    print("  a       - Signal absence (0 people)")
    print("  c<N>    - Set person count to N (e.g., 'c3' for 3 people)")
    print("  s       - Show status")
    print("  h       - Show this help")
    print("  t       - Run test scenarios")
    print("  q       - Quit interactive mode")
    print()
    print("Press Enter after each command...")
    print()
    
    helper = FileControlHelper()
    
    try:
        while True:
            try:
                # Show prompt and get input
                command = input("mock> ").strip().lower()
                
                if not command:
                    continue
                    
                if command == 'q' or command == 'quit':
                    print("Goodbye!")
                    break
                    
                elif command == 'p' or command == 'present':
                    helper.set_present()
                    log_step("✅ Signaled: Person present")
                    
                elif command == 'a' or command == 'absent':
                    helper.set_absent()
                    log_step("✅ Signaled: No one present")
                    
                elif command.startswith('c'):
                    # Extract number from command like 'c3'
                    try:
                        count_str = command[1:]
                        if count_str:
                            count = int(count_str)
                            if count < 0:
                                print("❌ Error: Count cannot be negative")
                                continue
                            helper.set_count(count)
                            log_step(f"✅ Signaled: {count} people present")
                        else:
                            print("❌ Error: Please specify a number (e.g., 'c3')")
                    except ValueError:
                        print(f"❌ Error: '{count_str}' is not a valid number")
                        
                elif command == 's' or command == 'status':
                    print()
                    show_status()
                    print()
                    
                elif command == 't' or command == 'test':
                    print()
                    run_test_scenarios()
                    print()
                    
                elif command == 'h' or command == 'help':
                    print()
                    print("Commands:")
                    print("  p       - Signal presence (1 person)")
                    print("  a       - Signal absence (0 people)") 
                    print("  c<N>    - Set person count to N (e.g., 'c3')")
                    print("  s       - Show status")
                    print("  t       - Run test scenarios")
                    print("  h       - Show this help")
                    print("  q       - Quit")
                    print()
                    
                else:
                    print(f"❌ Unknown command: '{command}'. Type 'h' for help.")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Type 'q' to quit gracefully.")
                
            except EOFError:
                print("\nGoodbye!")
                break
                
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


def main():
    # Check for test scenarios flag
    if len(sys.argv) >= 2 and sys.argv[1] in ['--test-scenarios', '--test', '-t']:
        run_test_scenarios()
        return
        
    # Check for interactive mode flag
    if len(sys.argv) >= 2 and sys.argv[1] in ['-i', '--interactive']:
        interactive_mode()
        return
        
    if len(sys.argv) < 2:
        show_status()
        return
        
    command = sys.argv[1].lower()
    helper = FileControlHelper()
    
    if command == "present":
        helper.set_present()
        print("✅ Signaled: Person present")
        
    elif command == "absent":
        helper.set_absent()
        print("✅ Signaled: No one present")
        
    elif command == "count":
        if len(sys.argv) < 3:
            print("Error: count command requires a number")
            print("Usage: python mock_control.py count <number>")
            sys.exit(1)
            
        try:
            count = int(sys.argv[2])
            helper.set_count(count)
            print(f"✅ Signaled: {count} people present")
        except ValueError:
            print(f"Error: '{sys.argv[2]}' is not a valid number")
            sys.exit(1)
            
    elif command == "status":
        show_status()
        
    else:
        print(f"Unknown command: {command}")
        show_status()
        sys.exit(1)


if __name__ == "__main__":
    main()
