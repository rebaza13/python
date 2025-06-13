"""
Unit tests for the ideal function selection system.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import modules to test
from data_handler import DataReader, TrainingFunction, IdealFunction, BaseFunction
from model_trainer import ModelTrainer
from database import DatabaseManager, TrainingData, IdealFunctions, TestDataMappings
from exceptions import DataProcessingError, ValidationError, DatabaseError
from main import IdealFunctionSelector


class TestBaseFunction(unittest.TestCase):
    """Test cases for BaseFunction class."""
    
    def setUp(self):
        """Set up test data."""
        self.x_values = np.array([1, 2, 3, 4, 5])
        self.y_values = np.array([2, 4, 6, 8, 10])
    
    def test_validation_error_different_lengths(self):
        """Test validation error for different array lengths."""
        x_wrong = np.array([1, 2, 3])
        with self.assertRaises(ValidationError):
            TrainingFunction(x_wrong, self.y_values, 1)
    
    def test_interpolation(self):
        """Test interpolation functionality."""
        train_func = TrainingFunction(self.x_values, self.y_values, 1)
        # Test exact point
        self.assertEqual(train_func.interpolate_y(3), 6)
        # Test interpolated point
        self.assertEqual(train_func.interpolate_y(2.5), 5)


class TestTrainingFunction(unittest.TestCase):
    """Test cases for TrainingFunction class."""
    
    def setUp(self):
        """Set up test data."""
        self.x_values = np.array([1, 2, 3, 4])
        self.y_values = np.array([1, 4, 9, 16])
        self.train_func = TrainingFunction(self.x_values, self.y_values, 1)
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.train_func.dataset_number, 1)
        self.assertEqual(self.train_func.function_id, "training_1")
    
    def test_deviation_calculation(self):
        """Test deviation calculation with another function."""
        # Create an identical ideal function
        ideal_func = IdealFunction(self.x_values, self.y_values, 1)
        deviation = self.train_func.calculate_deviation(ideal_func)
        self.assertEqual(deviation, 0)  # Should be zero for identical functions
        
        # Create a different ideal function
        different_y = np.array([2, 5, 10, 17])
        different_ideal = IdealFunction(self.x_values, different_y, 2)
        deviation = self.train_func.calculate_deviation(different_ideal)
        expected = sum([(1-2)**2, (4-5)**2, (9-10)**2, (16-17)**2])
        self.assertEqual(deviation, expected)


class TestIdealFunction(unittest.TestCase):
    """Test cases for IdealFunction class."""
    
    def setUp(self):
        """Set up test data."""
        self.x_values = np.array([1, 2, 3, 4])
        self.y_values = np.array([1, 4, 9, 16])
        self.ideal_func = IdealFunction(self.x_values, self.y_values, 5)
    
    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.ideal_func.function_number, 5)
        self.assertEqual(self.ideal_func.function_id, "ideal_5")
        self.assertIsNone(self.ideal_func.max_training_deviation)
    
    def test_max_training_deviation(self):
        """Test setting and getting max training deviation."""
        self.ideal_func.set_max_training_deviation(2.5)
        self.assertEqual(self.ideal_func.max_training_deviation, 2.5)
    
    def test_test_point_validation(self):
        """Test test point validation logic."""
        self.ideal_func.set_max_training_deviation(1.0)
        
        # Valid point (deviation <= sqrt(2) * max_training_deviation)
        is_valid, deviation = self.ideal_func.is_test_point_valid(2, 4.5)
        threshold = np.sqrt(2) * 1.0
        self.assertTrue(is_valid)
        self.assertEqual(deviation, 0.5)
        
        # Invalid point (deviation > sqrt(2) * max_training_deviation)
        is_valid, deviation = self.ideal_func.is_test_point_valid(2, 6.5)
        self.assertFalse(is_valid)
        self.assertEqual(deviation, 2.5)
    
    def test_validation_error_no_max_deviation(self):
        """Test validation error when max deviation not set."""
        with self.assertRaises(ValidationError):
            self.ideal_func.is_test_point_valid(2, 4)


class TestDataReader(unittest.TestCase):
    """Test cases for DataReader class."""
    
    def setUp(self):
        """Set up test data and temporary files."""
        self.data_reader = DataReader()
        
        # Create temporary CSV files for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Training data
        train_data = pd.DataFrame({'x': [1, 2, 3], 'y': [2, 4, 6]})
        self.train_file = os.path.join(self.temp_dir, 'train_test.csv')
        train_data.to_csv(self.train_file, index=False)
        
        # Ideal data
        ideal_data = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [1, 2, 3],
            'y2': [2, 4, 6],
            'y3': [3, 6, 9]
        })
        self.ideal_file = os.path.join(self.temp_dir, 'ideal_test.csv')
        ideal_data.to_csv(self.ideal_file, index=False)
        
        # Test data
        test_data = pd.DataFrame({'x': [1.5, 2.5], 'y': [3, 5]})
        self.test_file = os.path.join(self.temp_dir, 'test_test.csv')
        test_data.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_read_training_data(self):
        """Test reading training data."""
        # Create a training file with multiple y columns (y1, y2, y3, y4) as expected by new API
        train_data_multi = pd.DataFrame({
            'x': [1, 2, 3],
            'y1': [2, 4, 6],
            'y2': [3, 6, 9],
            'y3': [4, 8, 12],
            'y4': [5, 10, 15]
        })
        multi_train_file = os.path.join(self.temp_dir, 'multi_train_test.csv')
        train_data_multi.to_csv(multi_train_file, index=False)
        
        training_functions = self.data_reader.read_training_data(multi_train_file)
        
        self.assertEqual(len(training_functions), 4)
        self.assertIn(1, training_functions)
        self.assertIn(2, training_functions)
        self.assertIn(3, training_functions)
        self.assertIn(4, training_functions)
        self.assertIsInstance(training_functions[1], TrainingFunction)
        
        train_func = training_functions[1]
        np.testing.assert_array_equal(train_func.x_values, [1, 2, 3])
        np.testing.assert_array_equal(train_func.y_values, [2, 4, 6])
    
    def test_read_ideal_data(self):
        """Test reading ideal data."""
        ideal_functions = self.data_reader.read_ideal_data(self.ideal_file)
        
        self.assertEqual(len(ideal_functions), 3)  # y1, y2, y3
        self.assertIn(1, ideal_functions)
        self.assertIn(2, ideal_functions)
        self.assertIn(3, ideal_functions)
        
        ideal_func = ideal_functions[2]
        np.testing.assert_array_equal(ideal_func.x_values, [1, 2, 3])
        np.testing.assert_array_equal(ideal_func.y_values, [2, 4, 6])
    
    def test_read_test_data(self):
        """Test reading test data."""
        test_data = self.data_reader.read_test_data(self.test_file)
        
        self.assertEqual(len(test_data), 2)
        self.assertIn('x', test_data.columns)
        self.assertIn('y', test_data.columns)
    
    def test_invalid_file_error(self):
        """Test error handling for invalid files."""
        with self.assertRaises(DataProcessingError):
            self.data_reader.read_training_data(['nonexistent_file.csv'])


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer class."""
    
    def setUp(self):
        """Set up test data."""
        self.model_trainer = ModelTrainer()
        
        # Create test training functions
        x_vals = np.array([1, 2, 3, 4])
        self.training_functions = {
            1: TrainingFunction(x_vals, np.array([1, 4, 9, 16]), 1),
            2: TrainingFunction(x_vals, np.array([2, 8, 18, 32]), 2)
        }
        
        # Create test ideal functions
        self.ideal_functions = {
            1: IdealFunction(x_vals, np.array([1, 4, 9, 16]), 1),  # Perfect match for training 1
            2: IdealFunction(x_vals, np.array([2, 8, 18, 32]), 2), # Perfect match for training 2
            3: IdealFunction(x_vals, np.array([0, 1, 4, 9]), 3)   # Different function
        }
    
    def test_find_best_ideal_functions(self):
        """Test finding best ideal functions."""
        best_matches = self.model_trainer.find_best_ideal_functions(
            self.training_functions, self.ideal_functions)
        
        # Should find perfect matches
        self.assertEqual(best_matches[1], 1)
        self.assertEqual(best_matches[2], 2)
        
        # Check that deviations were recorded
        self.assertIn(1, self.model_trainer.training_deviations)
        self.assertIn(2, self.model_trainer.training_deviations)
        
        # Perfect matches should have zero deviation
        self.assertEqual(self.model_trainer.training_deviations[1]['total_deviation'], 0)
        self.assertEqual(self.model_trainer.training_deviations[2]['total_deviation'], 0)
    
    def test_assign_test_data(self):
        """Test test data assignment."""
        # First find best functions
        self.model_trainer.find_best_ideal_functions(
            self.training_functions, self.ideal_functions)
        
        # Create test data
        test_data = pd.DataFrame({
            'x': [2, 3],
            'y': [4, 9]  # Should match ideal function 1 perfectly
        })
        
        selected_ideal_functions = {1: self.ideal_functions[1], 2: self.ideal_functions[2]}
        assignments = self.model_trainer.assign_test_data(test_data, selected_ideal_functions)
        
        self.assertEqual(len(assignments), 2)
        
        # Both points should be assigned to function 1 (perfect match)
        for assignment in assignments:
            self.assertEqual(assignment['assigned_ideal_function'], 1)
            self.assertEqual(assignment['deviation'], 0)


class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager class."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_manager = DatabaseManager(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test database."""
        self.db_manager.close()
        os.unlink(self.temp_db.name)
    
    def test_database_creation(self):
        """Test database creation."""
        self.db_manager.create_database()
        self.assertIsNotNone(self.db_manager.engine)
        self.assertIsNotNone(self.db_manager.Session)
    
    def test_session_creation(self):
        """Test session creation."""
        self.db_manager.create_database()
        session = self.db_manager.get_session()
        self.assertIsNotNone(session)
        session.close()
    
    def test_session_error_without_init(self):
        """Test error when getting session without initialization."""
        with self.assertRaises(DatabaseError):
            self.db_manager.get_session()


class TestIdealFunctionSelector(unittest.TestCase):
    """Integration tests for the main application."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test CSV files
        x_vals = np.linspace(-10, 10, 20)
        
        # Create a single training file with multiple y columns (y1, y2, y3, y4)
        train_data = {'x': x_vals}
        for i in range(1, 5):
            train_data[f'y{i}'] = x_vals ** i  # Different polynomial functions
        
        train_df = pd.DataFrame(train_data)
        self.training_file = os.path.join(self.temp_dir, 'train.csv')
        train_df.to_csv(self.training_file, index=False)
        
        # Ideal file with multiple functions
        ideal_data = {'x': x_vals}
        for i in range(1, 11):  # Create 10 ideal functions
            ideal_data[f'y{i}'] = x_vals ** (i % 4 + 1) + np.random.normal(0, 0.1, len(x_vals))
        
        ideal_df = pd.DataFrame(ideal_data)
        self.ideal_file = os.path.join(self.temp_dir, 'ideal.csv')
        ideal_df.to_csv(self.ideal_file, index=False)
        
        # Test file
        test_x = np.random.uniform(-10, 10, 10)
        test_y = test_x ** 2 + np.random.normal(0, 1, len(test_x))
        test_data = pd.DataFrame({'x': test_x, 'y': test_y})
        self.test_file = os.path.join(self.temp_dir, 'test.csv')
        test_data.to_csv(self.test_file, index=False)
        
        # Database file
        self.db_file = os.path.join(self.temp_dir, 'test.db')
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('visualizer.DataVisualizer.create_comprehensive_plot')
    @patch('visualizer.DataVisualizer.create_deviation_analysis_plot')
    def test_complete_analysis_integration(self, mock_deviation_plot, mock_comprehensive_plot):
        """Test complete analysis integration."""
        # Mock the visualization methods to avoid file creation during tests
        mock_comprehensive_plot.return_value = None
        mock_deviation_plot.return_value = None
        
        selector = IdealFunctionSelector(self.db_file)
          # This should run without errors
        selector.run_complete_analysis(
            self.training_file,
            self.ideal_file,
            self.test_file
        )
        
        # Verify that data was loaded
        self.assertEqual(len(selector.training_functions), 4)
        self.assertGreater(len(selector.ideal_functions), 0)
        self.assertIsNotNone(selector.test_data)
        
        # Verify that best matches were found
        self.assertEqual(len(selector.best_matches), 4)
        
        # Verify that test assignments were made
        self.assertGreater(len(selector.test_assignments), 0)


if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestBaseFunction,
        TestTrainingFunction,
        TestIdealFunction,
        TestDataReader,
        TestModelTrainer,
        TestDatabaseManager,
        TestIdealFunctionSelector
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print(f"\nAll tests passed successfully! ✅")
    else:
        print(f"\nSome tests failed! ❌")
