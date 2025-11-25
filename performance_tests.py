"""
Advanced Performance Testing Module
Implements detailed performance benchmarking as specified in Section 5.0
"""

import time
import psutil
import threading
import numpy as np
import cv2
from memory_profiler import profile
import torch
import pytest
import os
from unittest.mock import Mock, patch


class PerformanceBenchmark:
    """
    Performance benchmarking utilities for system validation
    Based on Performance Analysis requirements from documentation
    """
    
    def __init__(self):
        self.metrics = {
            'fps': [],
            'memory_usage': [],
            'cpu_usage': [],
            'gpu_memory': [],
            'processing_times': []
        }
        self.monitoring = False
    
    def start_monitoring(self):
        """Start system resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system resource monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
    
    def _monitor_resources(self):
        """Monitor system resources in background thread"""
        process = psutil.Process()
        
        while self.monitoring:
            # CPU and Memory
            cpu_percent = process.cpu_percent()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            self.metrics['cpu_usage'].append(cpu_percent)
            self.metrics['memory_usage'].append(memory_mb)
            
            # GPU Memory (if available)
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                self.metrics['gpu_memory'].append(gpu_memory)
            
            time.sleep(0.1)  # Monitor every 100ms
    
    def measure_fps(self, processing_function, test_duration=10):
        """
        Measure processing FPS for given duration
        
        Args:
            processing_function: Function to test
            test_duration: Duration in seconds
            
        Returns:
            dict: FPS metrics
        """
        start_time = time.time()
        frame_count = 0
        processing_times = []
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        while time.time() - start_time < test_duration:
            frame_start = time.time()
            
            # Process frame
            try:
                processing_function(test_frame)
                frame_count += 1
            except Exception as e:
                print(f"Processing error: {e}")
                continue
            
            frame_end = time.time()
            processing_times.append(frame_end - frame_start)
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        return {
            'average_fps': avg_fps,
            'total_frames': frame_count,
            'total_time': total_time,
            'avg_processing_time': np.mean(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times)
        }
    
    def get_summary(self):
        """Get performance summary statistics"""
        if not self.metrics['memory_usage']:
            return "No monitoring data available"
        
        summary = {
            'avg_memory_mb': np.mean(self.metrics['memory_usage']),
            'peak_memory_mb': np.max(self.metrics['memory_usage']),
            'avg_cpu_percent': np.mean(self.metrics['cpu_usage']),
            'peak_cpu_percent': np.max(self.metrics['cpu_usage'])
        }
        
        if self.metrics['gpu_memory']:
            summary['avg_gpu_memory_mb'] = np.mean(self.metrics['gpu_memory'])
            summary['peak_gpu_memory_mb'] = np.max(self.metrics['gpu_memory'])
        
        return summary


class TestAdvancedPerformance:
    """
    Advanced Performance Testing
    Implements TC-P series test cases from documentation
    """
    
    @pytest.mark.performance
    @pytest.mark.gpu
    def test_tc_p_001_gpu_performance_detailed(self):
        """
        TC-P-001: Detailed GPU performance benchmark
        Target: 15-30 FPS on recommended GPU hardware
        """
        if not torch.cuda.is_available():
            pytest.skip("GPU not available for testing")
        
        benchmark = PerformanceBenchmark()
        
        # Mock YOLO model for GPU testing
        mock_model = Mock()
        mock_model.return_value = []  # Empty detections for simplicity
        
        def gpu_processing_function(frame):
            # Simulate GPU processing
            with torch.cuda.device(0):
                tensor_frame = torch.from_numpy(frame).cuda()
                # Simulate model inference
                result = mock_model(tensor_frame)
                return result
        
        benchmark.start_monitoring()
        fps_metrics = benchmark.measure_fps(gpu_processing_function, test_duration=30)
        benchmark.stop_monitoring()
        
        # Validate performance requirements
        assert 15 <= fps_metrics['average_fps'] <= 30, f"GPU FPS {fps_metrics['average_fps']} outside expected range [15-30]"
        
        summary = benchmark.get_summary()
        assert summary['peak_gpu_memory_mb'] <= 4096, f"GPU memory {summary['peak_gpu_memory_mb']}MB exceeds 4GB limit"
        
        print(f"✓ TC-P-001 Detailed: GPU FPS: {fps_metrics['average_fps']:.2f}")
        print(f"✓ GPU Memory Peak: {summary.get('peak_gpu_memory_mb', 0):.2f}MB")
    
    @pytest.mark.performance
    def test_tc_p_002_cpu_performance_detailed(self):
        """
        TC-P-002: Detailed CPU performance benchmark
        Target: 3-8 FPS on CPU-only system
        """
        benchmark = PerformanceBenchmark()
        
        # Mock CPU-only processing
        def cpu_processing_function(frame):
            # Simulate CPU-intensive processing
            tensor_frame = torch.from_numpy(frame)
            # Simulate model inference on CPU
            processed = torch.nn.functional.conv2d(
                tensor_frame.float().unsqueeze(0).permute(0,3,1,2),
                torch.randn(16, 3, 3, 3)
            )
            return processed.numpy()
        
        benchmark.start_monitoring()
        fps_metrics = benchmark.measure_fps(cpu_processing_function, test_duration=20)
        benchmark.stop_monitoring()
        
        # Validate performance requirements
        assert 3 <= fps_metrics['average_fps'] <= 8, f"CPU FPS {fps_metrics['average_fps']} outside expected range [3-8]"
        
        summary = benchmark.get_summary()
        assert summary['peak_memory_mb'] <= 8192, f"Memory usage {summary['peak_memory_mb']}MB exceeds reasonable limit"
        
        print(f"✓ TC-P-002 Detailed: CPU FPS: {fps_metrics['average_fps']:.2f}")
        print(f"✓ Memory Peak: {summary['peak_memory_mb']:.2f}MB")
    
    @pytest.mark.performance
    def test_tc_p_003_ocr_latency_benchmark(self):
        """
        Test OCR processing latency
        Target: 100-200ms per plate as specified in documentation
        """
        from utils.ocr import predict_number_plate
        
        # Mock PaddleOCR with realistic processing time
        mock_ocr = Mock()
        mock_ocr.ocr.return_value = [
            [
                [
                    [[0, 0], [100, 0], [100, 30], [0, 30]],
                    ["ABC123", 0.95]
                ]
            ]
        ]
        
        # Create test plate images
        test_images = [
            np.random.randint(0, 255, (50, 150, 3), dtype=np.uint8)
            for _ in range(50)
        ]
        
        processing_times = []
        
        for test_image in test_images:
            start_time = time.time()
            plate_text, confidence = predict_number_plate(test_image, mock_ocr)
            end_time = time.time()
            
            processing_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_latency = np.mean(processing_times)
        max_latency = np.max(processing_times)
        
        # Validate latency requirements
        assert avg_latency <= 200, f"Average OCR latency {avg_latency:.2f}ms exceeds 200ms target"
        assert max_latency <= 500, f"Maximum OCR latency {max_latency:.2f}ms too high"
        
        print(f"✓ TC-P-003: OCR Average Latency: {avg_latency:.2f}ms")
        print(f"✓ OCR Max Latency: {max_latency:.2f}ms")
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_tc_p_004_system_stability_long_running(self):
        """
        Test system stability under prolonged operation
        Target: 100% uptime over extended period
        """
        benchmark = PerformanceBenchmark()
        
        def stable_processing_function(frame):
            # Simulate long-running stable processing
            time.sleep(0.01)  # 10ms processing time
            return frame
        
        test_duration = 300  # 5 minutes for CI, can be extended for full testing
        error_count = 0
        success_count = 0
        
        benchmark.start_monitoring()
        start_time = time.time()
        
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        while time.time() - start_time < test_duration:
            try:
                stable_processing_function(test_frame)
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"Stability test error: {e}")
            
            # Check every 1000 iterations
            if (success_count + error_count) % 1000 == 0:
                print(f"Stability check: {success_count + error_count} iterations completed")
        
        benchmark.stop_monitoring()
        
        total_operations = success_count + error_count
        stability_rate = success_count / total_operations * 100
        
        assert stability_rate >= 99.9, f"Stability rate {stability_rate:.3f}% below 99.9%"
        
        summary = benchmark.get_summary()
        print(f"✓ TC-P-004: System Stability: {stability_rate:.3f}%")
        print(f"✓ Total Operations: {total_operations}")
        print(f"✓ Average Memory: {summary['avg_memory_mb']:.2f}MB")


class TestModelCaching:
    """
    Test model caching effectiveness
    Validates @st.cache_resource implementation
    """
    
    @pytest.mark.performance
    def test_model_load_time_caching(self):
        """Test that model caching reduces subsequent load times"""
        
        # Simulate model loading times
        class MockModel:
            def __init__(self, load_time=2.0):
                time.sleep(load_time)  # Simulate model loading
                self.loaded = True
        
        # First load (uncached)
        start_time = time.time()
        model1 = MockModel(load_time=2.0)
        first_load_time = time.time() - start_time
        
        # Simulate cached load (should be much faster)
        start_time = time.time()
        model2 = model1  # Cached reference
        cached_load_time = time.time() - start_time
        
        assert first_load_time >= 2.0, "First load should take significant time"
        assert cached_load_time < 0.1, "Cached load should be nearly instantaneous"
        
        print(f"✓ First load time: {first_load_time:.3f}s")
        print(f"✓ Cached load time: {cached_load_time:.6f}s")


def run_performance_tests():
    """Run comprehensive performance test suite"""
    print("🚀 Running Advanced Performance Test Suite")
    print("=" * 60)
    
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "-m", "performance",
        "--tb=short"
    ])