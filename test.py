#!/usr/bin/env python3
"""
Test Runner Wrapper for Helmet and Number Plate Detection System
Provides easy access to the test suite from the project root directory.

Usage:
    python test.py [options]
    
This is a wrapper that calls the main test runner in tests/run_tests.py
All arguments are passed through to the main test runner.

Examples:
    python test.py --all --verbose
    python test.py --unit
    python test.py --performance --quick
    python test.py --coverage
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Main entry point that delegates to tests/run_tests.py"""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    test_runner_path = project_root / "tests" / "run_tests.py"
    
    # Check if test runner exists
    if not test_runner_path.exists():
        print("❌ Error: Test runner not found at tests/run_tests.py")
        print("Please ensure the tests directory and run_tests.py exist.")
        sys.exit(1)
    
    # Print banner
    print("🧪 Helmet and Number Plate Detection System - Test Runner")
    print("=" * 60)
    print(f"Project Root: {project_root}")
    print(f"Test Runner: {test_runner_path}")
    print("=" * 60)
    
    # Prepare command to run the actual test runner
    cmd = [sys.executable, str(test_runner_path)] + sys.argv[1:]
    
    try:
        # Execute the test runner with all arguments passed through
        result = subprocess.run(cmd, cwd=project_root)
        sys.exit(result.returncode)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()