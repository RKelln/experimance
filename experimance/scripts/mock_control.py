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
"""

import sys
import time
from pathlib import Path

# Add the agent source to the path so we can import the helper
sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "agent" / "src"))

from agent.vision.mock_detector import FileControlHelper


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
                    print("✅ Signaled: Person present")
                    
                elif command == 'a' or command == 'absent':
                    helper.set_absent()
                    print("✅ Signaled: No one present")
                    
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
                            print(f"✅ Signaled: {count} people present")
                        else:
                            print("❌ Error: Please specify a number (e.g., 'c3')")
                    except ValueError:
                        print(f"❌ Error: '{count_str}' is not a valid number")
                        
                elif command == 's' or command == 'status':
                    print()
                    show_status()
                    print()
                    
                elif command == 'h' or command == 'help':
                    print()
                    print("Commands:")
                    print("  p       - Signal presence (1 person)")
                    print("  a       - Signal absence (0 people)") 
                    print("  c<N>    - Set person count to N (e.g., 'c3')")
                    print("  s       - Show status")
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
        
    elif command == "absent":
        helper.set_absent()
        
    elif command == "count":
        if len(sys.argv) < 3:
            print("Error: count command requires a number")
            print("Usage: python mock_control.py count <number>")
            sys.exit(1)
            
        try:
            count = int(sys.argv[2])
            helper.set_count(count)
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
