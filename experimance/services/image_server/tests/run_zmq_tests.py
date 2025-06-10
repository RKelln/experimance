#!/usr/bin/env python3
"""
Image Server ZMQ Communication Test Runner.

This script orchestrates running all the ZMQ communication tests for the Image Server Service
to verify that the messaging system is working correctly and detect any configuration issues.

$ uv run -m services.image_server.tests.run_zmq_tests
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_runner")


def run_shell_command(command: List[str], timeout: int = 30) -> Tuple[int, str, str]:
    """Run a shell command and return the exit code, stdout, and stderr.
    
    Args:
        command: Command and arguments as a list
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    logger.debug(f"Running command: {' '.join(command)}")
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.error("Command timed out")
        return -1, "", "Command timed out"
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return -1, "", str(e)


def print_test_banner(test_name: str, width: int = 80):
    """Print a banner for a test.
    
    Args:
        test_name: Name of the test
        width: Width of the banner
    """
    print("\n" + "=" * width)
    print(f"TEST: {test_name}".center(width))
    print("=" * width + "\n")


def verify_image_server_running() -> bool:
    """Check if the Image Server Service is running.
    
    Returns:
        True if running, False otherwise
    """
    # Check for running processes
    retcode, stdout, stderr = run_shell_command(["ps", "aux"])
    if retcode != 0:
        logger.error("Failed to check for running processes")
        return False
    
    if "image_server" in stdout and "python" in stdout:
        logger.info("✅ Image Server process appears to be running")
        return True
    else:
        logger.warning("❌ Image Server process does not appear to be running")
        return False


def validate_zmq_configuration() -> bool:
    """Run the ZMQ address validation script.
    
    Returns:
        True if validation passes, False otherwise
    """
    print_test_banner("ZMQ Address Validation")
    
    script_path = Path(__file__).parent / "validate_zmq_addresses.py"
    if not script_path.exists():
        logger.error(f"ZMQ address validation script not found at {script_path}")
        return False
    
    retcode, stdout, stderr = run_shell_command([sys.executable, str(script_path)])
    print(stdout)
    if stderr:
        print(stderr)
    
    if retcode == 0:
        logger.info("✅ ZMQ address validation succeeded")
        return True
    else:
        logger.error("❌ ZMQ address validation failed")
        return False


def run_zmq_message_test() -> bool:
    """Run the basic ZMQ messaging test.
    
    Returns:
        True if test passes, False otherwise
    """
    print_test_banner("Basic ZMQ Messaging Test")
    
    script_path = Path(__file__).parent / "test_zmq_messaging.py"
    if not script_path.exists():
        logger.error(f"ZMQ messaging test script not found at {script_path}")
        return False
    
    retcode, stdout, stderr = run_shell_command([sys.executable, str(script_path)])
    print(stdout)
    if stderr:
        print(stderr)
    
    if retcode == 0:
        logger.info("✅ Basic ZMQ messaging test succeeded")
        return True
    else:
        logger.error("❌ Basic ZMQ messaging test failed")
        return False


def run_zmq_render_request_test() -> bool:
    """Run the comprehensive ZMQ render request test.
    
    Returns:
        True if test passes, False otherwise
    """
    print_test_banner("Comprehensive ZMQ Render Request Test")
    
    script_path = Path(__file__).parent / "test_zmq_render_request.py"
    if not script_path.exists():
        logger.error(f"ZMQ render request test script not found at {script_path}")
        return False
    
    retcode, stdout, stderr = run_shell_command([sys.executable, str(script_path)])
    print(stdout)
    if stderr:
        print(stderr)
    
    if retcode == 0:
        logger.info("✅ Comprehensive ZMQ render request test succeeded")
        return True
    else:
        logger.error("❌ Comprehensive ZMQ render request test failed")
        return False


def run_cli_functionality_test() -> bool:
    """Test the CLI functionality directly.
    
    Returns:
        True if test passes, False otherwise
    """
    print_test_banner("CLI Functionality Test")
    
    cli_path = Path(__file__).parent.parent / "src" / "image_server" / "cli.py"
    if not cli_path.exists():
        logger.error(f"CLI script not found at {cli_path}")
        return False
    
    # Test listing prompts
    print("Testing CLI prompt listing...")
    retcode, stdout, stderr = run_shell_command([
        sys.executable, str(cli_path), "--list-prompts"
    ])
    
    if retcode != 0:
        logger.error("❌ CLI prompt listing failed")
        print(stderr)
        return False
    
    print("Sample prompts:")
    print(stdout)
    
    # Test sending a simple request with no wait
    print("\nTesting CLI simple request (no wait)...")
    retcode, stdout, stderr = run_shell_command([
        sys.executable, str(cli_path),
        "--prompt", "A test image with blue sky",
        "--no-wait"
    ])
    
    if retcode != 0:
        logger.error("❌ CLI simple request failed")
        print(stderr)
        return False
    
    print(stdout)
    logger.info("✅ CLI functionality test succeeded")
    return True


def run_all_tests():
    """Run all ZMQ communication tests."""
    print_test_banner("IMAGE SERVER ZMQ COMMUNICATION TEST SUITE", 100)
    
    # Check if image server is running
    server_running = verify_image_server_running()
    if not server_running:
        print("\n⚠️  WARNING: The Image Server service does not appear to be running.")
        print("The tests may fail if the service is not available.")
        proceed = input("Do you want to continue anyway? (y/N): ").lower() == 'y'
        if not proceed:
            print("Tests cancelled.")
            return
    
    # Run all tests
    tests = [
        ("ZMQ Configuration Validation", validate_zmq_configuration),
        ("Basic ZMQ Messaging Test", run_zmq_message_test),
        ("Comprehensive ZMQ Render Request Test", run_zmq_render_request_test),
        ("CLI Functionality Test", run_cli_functionality_test)
    ]
    
    results = {}
    
    for name, test_func in tests:
        print(f"\nRunning test: {name}")
        try:
            result = test_func()
            results[name] = result
        except Exception as e:
            logger.error(f"Error running test {name}: {e}", exc_info=True)
            results[name] = False
    
    # Print summary
    print_test_banner("TEST SUMMARY", 100)
    print(f"{'Test':<40} | {'Result':<20}")
    print(f"{'-' * 40} | {'-' * 20}")
    
    all_passed = True
    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:<40} | {status:<20}")
        if not result:
            all_passed = False
    
    print("\nOverall Result:", "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED")
    

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ZMQ communication tests for the Image Server Service"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--test", "-t",
        choices=["config", "basic", "render", "cli", "all"],
        default="all",
        help="Specific test to run (default: all)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.test == "config":
        return 0 if validate_zmq_configuration() else 1
    elif args.test == "basic":
        return 0 if run_zmq_message_test() else 1
    elif args.test == "render":
        return 0 if run_zmq_render_request_test() else 1
    elif args.test == "cli":
        return 0 if run_cli_functionality_test() else 1
    else:  # all
        run_all_tests()
        return 0


if __name__ == "__main__":
    # $ uv run -m services.image_server.tests.run_zmq_tests
    sys.exit(main())
