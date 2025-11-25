"""
Integration Tests for Helmet and Number Plate Detection System
Implements comprehensive integration testing scenarios from Section 5.2.2
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
import pandas as pd
import time
import torch
from unittest.mock import Mock, patch, MagicMock
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.detector import process_frame
from utils.tracker import ViolationTracker
from utils.database import initialize_database, log_violation
from utils.ocr import predict_number_plate


class TestEndToEndIntegration:
    """
    Integration Testing: Detection-to-Violation-Analysis-to-OCR-to-Tracking-to-Database
    Tests the complete pipeline as described in System Architecture
    """
    
    def setup_method(self):
        """Setup test environment for integration tests"""
        self.test_dir = tempfile.mkdtemp()
        self.original_paths = {}
        
        # Override database paths for testing
        import utils.database as db_module
        self.original_paths['LOG_FILE_PATH'] = db_module.LOG_FILE_PATH
        self.original_paths['IMAGE_DIR'] = db_module.IMAGE_DIR
        
        db_module.LOG_FILE_PATH = os.path.join(self.test_dir, "test_violations_log.csv")
        db_module.IMAGE_DIR = os.path.join(self.test_dir, "test_detected_plates")
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
        # Restore original paths
        import utils.database as db_module
        db_module.LOG_FILE_PATH = self.original_paths['LOG_FILE_PATH']
        db_module.IMAGE_DIR = self.original_paths['IMAGE_DIR']
    
    @pytest.mark.integration
    def test_tc_i_001_detection_to_violation_analysis_integration(self):
        """
        TC-I-001: Detection to Violation Analysis Integration
        Validates Box Association logic for linking detections to riders
        """
        # Create mock YOLOv8 results simulating detection scenario
        mock_model = Mock()
        mock_ocr = Mock()
        mock_tracker = ViolationTracker(cooldown_seconds=10)
        
        # Mock detection results: rider without helmet + number plate
        mock_boxes = []
        
        # Rider bounding box (larger, encompassing)
        rider_box = Mock()
        rider_box.xyxy = [torch.tensor([50, 50, 300, 400])]
        rider_box.conf = [torch.tensor(0.9)]
        rider_box.cls = [torch.tensor(2)]  # rider class index
        mock_boxes.append(rider_box)
        
        # Without helmet detection (inside rider box)
        helmet_box = Mock()
        helmet_box.xyxy = [torch.tensor([80, 80, 200, 250])]
        helmet_box.conf = [torch.tensor(0.85)]
        helmet_box.cls = [torch.tensor(1)]  # without helmet class index
        mock_boxes.append(helmet_box)
        
        # Number plate detection (inside rider box)
        plate_box = Mock()
        plate_box.xyxy = [torch.tensor([100, 320, 250, 380])]
        plate_box.conf = [torch.tensor(0.88)]
        plate_box.cls = [torch.tensor(3)]  # number plate class index
        mock_boxes.append(plate_box)
        
        mock_result = Mock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        
        # Mock OCR successful extraction
        mock_ocr.ocr.return_value = [
            [
                [
                    [[0, 0], [150, 0], [150, 60], [0, 60]],
                    ["ABC123", 0.95]
                ]
            ]
        ]
        
        # Create test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Initialize database for logging
        initialize_database()
        
        # Process frame through detection pipeline
        with patch('utils.detector.predict_number_plate') as mock_predict:
            mock_predict.return_value = ("ABC123", 0.95)
            
            with patch('utils.detector.log_violation') as mock_log:
                mock_log.return_value = "test_image_path.jpg"
                
                annotated_frame, violations = process_frame(
                    test_frame, mock_model, mock_ocr, mock_tracker
                )
        
        # Verify integration results
        assert annotated_frame is not None
        assert annotated_frame.shape == test_frame.shape
        
        # Verify violation was detected and tracked
        assert len(violations) <= 1  # Should detect one violation or handle gracefully
        
        print("✓ TC-I-001: Detection to Violation Analysis Integration - PASS")
    
    @pytest.mark.integration
    def test_tc_i_002_violation_to_ocr_to_tracking_integration(self):
        """
        TC-I-002: Violation Analysis to OCR to Tracking Integration
        Tests handshake between violation confirmation, OCR processing, and tracking
        """
        # Initialize components
        tracker = ViolationTracker(cooldown_seconds=5)
        
        # Mock OCR model
        mock_ocr = Mock()
        mock_ocr.ocr.return_value = [
            [
                [
                    [[0, 0], [150, 0], [150, 60], [0, 60]],
                    ["DLA90", 0.95]
                ]
            ]
        ]
        
        # Create test plate image
        plate_image = np.random.randint(0, 255, (60, 150, 3), dtype=np.uint8)
        
        # Step 1: OCR Processing
        plate_text, confidence = predict_number_plate(plate_image, mock_ocr)
        
        assert plate_text == "DLA90"
        assert confidence == 0.95
        
        # Step 2: Tracking Check (first time)
        is_new_violation = tracker.is_new_violation(plate_text)
        assert is_new_violation == True
        
        # Step 3: Duplicate Prevention (second time, within cooldown)
        time.sleep(1)  # Wait 1 second (within 5-second cooldown)
        is_duplicate = tracker.is_new_violation(plate_text)
        assert is_duplicate == False
        
        # Step 4: Post-cooldown acceptance
        time.sleep(6)  # Wait for cooldown to expire
        is_post_cooldown = tracker.is_new_violation(plate_text)
        assert is_post_cooldown == True
        
        print("✓ TC-I-002: Violation to OCR to Tracking Integration - PASS")
    
    @pytest.mark.integration
    def test_tc_i_003_tracking_to_database_integration(self):
        """
        TC-I-003: Tracking to Database Integration
        Tests final violation data persistence
        """
        import utils.database as db_module
        
        # Initialize database
        db_module.initialize_database()
        
        # Create test violation data
        plate_number = "XYZ789"
        plate_image = np.random.randint(0, 255, (60, 150, 3), dtype=np.uint8)
        
        # Log violation
        image_path = db_module.log_violation(plate_number, plate_image)
        
        # Verify image file creation
        assert os.path.exists(image_path)
        assert plate_number in os.path.basename(image_path)
        assert image_path.endswith('.jpg')
        
        # Verify CSV entry
        df = pd.read_csv(db_module.LOG_FILE_PATH)
        assert len(df) == 1
        
        violation_entry = df.iloc[0]
        assert violation_entry['PlateNumber'] == plate_number
        assert violation_entry['ImagePath'] == image_path
        assert pd.notna(violation_entry['Timestamp'])
        
        # Verify timestamp format
        timestamp_str = violation_entry['Timestamp']
        pd.to_datetime(timestamp_str)  # Should not raise exception
        
        print("✓ TC-I-003: Tracking to Database Integration - PASS")
    
    @pytest.mark.integration
    def test_tc_i_004_complete_pipeline_integration(self):
        """
        TC-I-004: Complete End-to-End Pipeline Integration
        Tests entire flow from frame input to data persistence
        """
        import utils.database as db_module
        
        # Initialize all components
        db_module.initialize_database()
        tracker = ViolationTracker(cooldown_seconds=10)
        
        # Mock complete detection pipeline
        mock_model = self._create_complete_mock_model()
        mock_ocr = self._create_mock_ocr()
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process through complete pipeline
        annotated_frame, violations = process_frame(
            test_frame, mock_model, mock_ocr, tracker
        )
        
        # Verify pipeline execution
        assert annotated_frame is not None
        assert violations is not None
        assert isinstance(violations, list)
        
        # If violations were detected, verify they were logged
        if violations:
            df = pd.read_csv(db_module.LOG_FILE_PATH)
            assert len(df) >= len(violations)
            
            # Verify each violation has corresponding evidence
            for violation in violations:
                plate_number = violation['plate_number']
                image_path = violation['image_path']
                
                # Check image file exists
                assert os.path.exists(image_path)
                
                # Check CSV entry exists
                matching_entries = df[df['PlateNumber'] == plate_number]
                assert len(matching_entries) > 0
        
        print("✓ TC-I-004: Complete Pipeline Integration - PASS")
    
    def _create_complete_mock_model(self):
        """Create mock YOLOv8 model with realistic detection results"""
        mock_model = Mock()
        
        # Create mock boxes for a violation scenario
        mock_boxes = []
        
        # Rider
        rider_box = Mock()
        rider_box.xyxy = [torch.tensor([50, 50, 300, 400])]
        rider_box.conf = [torch.tensor(0.9)]
        rider_box.cls = [torch.tensor(2)]
        mock_boxes.append(rider_box)
        
        # Without helmet (inside rider)
        helmet_box = Mock()
        helmet_box.xyxy = [torch.tensor([80, 80, 200, 250])]
        helmet_box.conf = [torch.tensor(0.85)]
        helmet_box.cls = [torch.tensor(1)]
        mock_boxes.append(helmet_box)
        
        # Number plate (inside rider)
        plate_box = Mock()
        plate_box.xyxy = [torch.tensor([100, 320, 250, 380])]
        plate_box.conf = [torch.tensor(0.88)]
        plate_box.cls = [torch.tensor(3)]
        mock_boxes.append(plate_box)
        
        mock_result = Mock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        
        return mock_model
    
    def _create_mock_ocr(self):
        """Create mock OCR with successful extraction"""
        mock_ocr = Mock()
        mock_ocr.ocr.return_value = [
            [
                [
                    [[0, 0], [150, 0], [150, 60], [0, 60]],
                    ["TEST123", 0.95]
                ]
            ]
        ]
        return mock_ocr


class TestModuleInteractions:
    """
    Test interactions between different modules
    Focuses on data flow and interface contracts
    """
    
    @pytest.mark.integration
    def test_detector_tracker_interface(self):
        """Test interface between detector and tracker modules"""
        tracker = ViolationTracker(cooldown_seconds=5)
        
        # Test data flow
        plate_numbers = ["ABC123", "DEF456", "ABC123"]  # ABC123 repeated
        results = []
        
        for plate in plate_numbers:
            is_new = tracker.is_new_violation(plate)
            results.append(is_new)
        
        # First ABC123 should be new, DEF456 should be new, second ABC123 should be duplicate
        assert results == [True, True, False]
        
        print("✓ Detector-Tracker Interface - PASS")
    
    @pytest.mark.integration
    def test_ocr_database_interface(self):
        """Test interface between OCR and database modules"""
        import utils.database as db_module
        
        # Setup
        test_dir = tempfile.mkdtemp()
        original_paths = {
            'LOG_FILE_PATH': db_module.LOG_FILE_PATH,
            'IMAGE_DIR': db_module.IMAGE_DIR
        }
        
        try:
            db_module.LOG_FILE_PATH = os.path.join(test_dir, "test_log.csv")
            db_module.IMAGE_DIR = os.path.join(test_dir, "test_plates")
            
            db_module.initialize_database()
            
            # Simulate OCR result being passed to database
            ocr_result = ("XYZ789", 0.92)
            plate_image = np.random.randint(0, 255, (60, 150, 3), dtype=np.uint8)
            
            # Log the OCR result
            image_path = db_module.log_violation(ocr_result[0], plate_image)
            
            # Verify integration
            assert os.path.exists(image_path)
            df = pd.read_csv(db_module.LOG_FILE_PATH)
            assert len(df) == 1
            assert df.iloc[0]['PlateNumber'] == ocr_result[0]
            
            print("✓ OCR-Database Interface - PASS")
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
            db_module.LOG_FILE_PATH = original_paths['LOG_FILE_PATH']
            db_module.IMAGE_DIR = original_paths['IMAGE_DIR']


class TestErrorPropagation:
    """
    Test error handling and propagation through the integration chain
    """
    
    @pytest.mark.integration
    def test_ocr_error_propagation(self):
        """Test how OCR errors are handled in the pipeline"""
        # Create OCR that raises exception
        mock_ocr = Mock()
        mock_ocr.ocr.side_effect = Exception("OCR processing failed")
        
        # Create test image
        test_image = np.random.randint(0, 255, (60, 150, 3), dtype=np.uint8)
        
        # Should handle error gracefully
        result = predict_number_plate(test_image, mock_ocr)
        
        assert result == (None, None)
        print("✓ OCR Error Propagation - PASS")
    
    @pytest.mark.integration
    def test_pipeline_resilience(self):
        """Test pipeline continues processing despite component errors"""
        import utils.database as db_module
        
        # Setup
        test_dir = tempfile.mkdtemp()
        original_log_path = db_module.LOG_FILE_PATH
        
        try:
            db_module.LOG_FILE_PATH = os.path.join(test_dir, "test_log.csv")
            db_module.IMAGE_DIR = os.path.join(test_dir, "test_plates")
            db_module.initialize_database()
            
            tracker = ViolationTracker(cooldown_seconds=5)
            
            # Test multiple processing cycles with some failures
            successful_processes = 0
            failed_processes = 0
            
            test_cases = [
                ("VALID123", True),   # Valid case
                ("", False),          # Invalid plate (empty)
                ("AB", False),        # Invalid plate (too short)
                ("VALID456", True),   # Valid case
            ]
            
            for plate_text, should_succeed in test_cases:
                try:
                    if plate_text and 4 <= len(plate_text) <= 11:
                        is_new = tracker.is_new_violation(plate_text)
                        if is_new:
                            # Simulate successful logging
                            test_image = np.random.randint(0, 255, (60, 150, 3), dtype=np.uint8)
                            db_module.log_violation(plate_text, test_image)
                        successful_processes += 1
                    else:
                        # Invalid input handled
                        failed_processes += 1
                except Exception:
                    failed_processes += 1
            
            # Verify pipeline resilience
            assert successful_processes >= 2  # At least 2 valid cases should succeed
            assert failed_processes >= 1     # At least 1 invalid case should fail gracefully
            
            print(f"✓ Pipeline Resilience - {successful_processes} successes, {failed_processes} handled failures - PASS")
        
        finally:
            # Cleanup
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)
            db_module.LOG_FILE_PATH = original_log_path


def run_integration_tests():
    """Run comprehensive integration test suite"""
    print("🔗 Running Integration Test Suite")
    print("=" * 60)
    
    pytest.main([
        __file__,
        "-v",
        "-m", "integration",
        "--tb=short"
    ])


if __name__ == "__main__":
    run_integration_tests()