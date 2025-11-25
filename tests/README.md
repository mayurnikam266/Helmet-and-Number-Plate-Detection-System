# Tests Directory

This directory contains the comprehensive test suite for the Helmet and Number Plate Detection System, implementing the testing strategy outlined in **Section 5.0 System Testing and Validation** of the project documentation.

## 📁 Directory Structure

```
tests/
├── test_suite.py              # Main comprehensive test suite
├── integration_tests.py       # Integration testing scenarios
├── performance_tests.py       # Performance benchmarking tests
├── run_tests.py              # Test runner and orchestrator
├── demo_tests.py             # Demo script for showcasing capabilities
├── pytest.ini               # Pytest configuration
├── test_requirements.txt     # Testing dependencies
├── TESTING_README.md         # Complete testing documentation
└── README.md                 # This file
```

## 🚀 Quick Start

### From Project Root Directory

```bash
# Run all tests using the wrapper script
python test.py --all --verbose

# Run specific test categories
python test.py --unit
python test.py --performance
python test.py --integration
```

### From Tests Directory

```bash
# Navigate to tests directory
cd tests

# Install testing dependencies
pip install -r test_requirements.txt

# Run complete test suite
python run_tests.py --all --verbose

# Run quick tests (no slow/GPU tests)
python run_tests.py --quick

# Run with coverage
python run_tests.py --coverage

# Demo the test capabilities
python demo_tests.py
```

### Using pytest directly

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_suite.py -v

# Run tests with specific markers
pytest tests/ -m unit -v
pytest tests/ -m performance -v

# Run specific test class
pytest tests/test_suite.py::TestViolationTracker -v
```

## 📋 Test Categories

### Unit Tests (TC-F Series)
- **ViolationTracker**: Cooldown logic, duplicate prevention
- **OCR Module**: Text extraction, cleaning, validation  
- **Database Module**: Data persistence, file management

### Integration Tests (TC-I Series)
- **Detection Pipeline**: YOLOv8 → Violation Analysis → OCR → Tracking
- **Data Flow**: End-to-end violation processing
- **Error Propagation**: Graceful error handling

### Performance Tests (TC-P Series)
- **GPU Performance**: 15-30 FPS target
- **CPU Performance**: 3-8 FPS target
- **Memory Usage**: ≤4GB GPU memory
- **OCR Latency**: 100-200ms per plate

### System Tests (TC-S Series)
- **Input Source Handling**: Video files and camera feeds
- **Error Tolerance**: System resilience
- **Stability**: Long-running operation

### User Acceptance Tests (TC-U Series)
- **Web Interface**: Streamlit controls functionality
- **Data Accessibility**: CSV export and visualization
- **Real-time Updates**: Live violation logging

## 🎯 Performance Targets

The test suite validates these documented performance targets:

| Component | Metric | Target | Test Case |
|-----------|--------|--------|-----------|
| GPU Processing | FPS | 15-30 | TC-P-001 |
| CPU Processing | FPS | 3-8 | TC-P-002 |
| GPU Memory | Usage | ≤4GB | TC-P-003 |
| OCR Latency | Time per plate | 100-200ms | TC-P-004 |
| Detection | mAP@0.5 | 85-90% | TC-F-001 |
| OCR | Accuracy | 90-95% | TC-F-002 |

## ⚙️ Configuration

### Pytest Configuration
- **Coverage**: HTML and terminal reports
- **Markers**: unit, integration, performance, gpu, slow
- **Test Discovery**: Automatic test file detection
- **Logging**: Detailed test execution logs

### Test Markers
```bash
# Run only unit tests
pytest tests/ -m unit

# Run only performance tests (excluding slow ones)
pytest tests/ -m "performance and not slow"

# Run GPU tests only
pytest tests/ -m gpu

# Skip slow tests
pytest tests/ -m "not slow"
```

## 📊 Coverage Analysis

```bash
# Generate coverage report
python run_tests.py --coverage

# View HTML coverage report
open htmlcov/index.html  # or start htmlcov/index.html on Windows
```

## 🔧 Adding New Tests

### Test File Naming
- Use `test_*.py` or `*_test.py` pattern
- Place in appropriate category file or create new file

### Test Function Naming
```python
def test_tc_f_001_description(self):
    """
    TC-F-001: Test case description
    Requirement Source: Documentation section
    """
    # Test implementation
    pass
```

### Test Markers
```python
@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.gpu
@pytest.mark.slow
def test_example():
    # Test code
    pass
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd tests/
   python -c "import sys; print(sys.path)"
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r test_requirements.txt
   pip install -r ../requirements.txt
   ```

3. **GPU Tests Failing**
   ```bash
   # Skip GPU tests if no GPU available
   python run_tests.py --quick
   pytest tests/ -m "not gpu"
   ```

4. **Model File Missing**
   ```bash
   # Ensure best.pt exists
   ls ../models/best.pt
   ```

## 📈 Test Results

Test results are saved to `test_results.json` for CI/CD integration:

```json
{
  "timestamp": "2025-11-25 14:30:00",
  "total_duration": 45.2,
  "results": {
    "unit": {"status": "passed", "duration": 8.5},
    "integration": {"status": "passed", "duration": 12.3}
  }
}
```

## 🔗 Related Documentation

- **TESTING_README.md**: Comprehensive testing documentation
- **../PROJECT_DOCUMENTATION.md**: Full project specifications
- **../README.md**: Main project README

## 🚀 Continuous Integration

The test suite is designed for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    cd tests
    pip install -r test_requirements.txt
    python run_tests.py --all --coverage
```

For complete testing documentation, see [TESTING_README.md](TESTING_README.md).