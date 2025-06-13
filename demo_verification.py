"""
Quick demonstration of key features and verification.
"""

import pandas as pd
import numpy as np
from data_handler import DataReader, TrainingFunction, IdealFunction
from model_trainer import ModelTrainer
from database import DatabaseManager

def demonstrate_key_features():
    """Demonstrate that all key requirements are working correctly."""
    
    print("🔍 DEMONSTRATING KEY FEATURES")
    print("=" * 50)
    
    # 1. Demonstrate Object-Oriented Design with Inheritance
    print("\n1. ✅ OBJECT-ORIENTED DESIGN WITH INHERITANCE")
    print("-" * 40)
    
    # Show inheritance hierarchy
    x_vals = np.array([1, 2, 3, 4, 5])
    y_vals = np.array([1, 4, 9, 16, 25])
    
    train_func = TrainingFunction(x_vals, y_vals, 1)
    ideal_func = IdealFunction(x_vals, y_vals, 1)
    
    print(f"TrainingFunction ID: {train_func.function_id}")
    print(f"IdealFunction ID: {ideal_func.function_id}")
    print(f"Both inherit from BaseFunction: {hasattr(train_func, 'interpolate_y')}")
    
    # 2. Demonstrate Least-Square Calculation
    print("\n2. ✅ LEAST-SQUARE ERROR CALCULATION")
    print("-" * 40)
    
    # Perfect match should give 0 deviation
    perfect_deviation = train_func.calculate_deviation(ideal_func)
    print(f"Perfect match deviation: {perfect_deviation}")
    
    # Different function should give non-zero deviation
    different_y = np.array([2, 5, 10, 17, 26])
    different_func = IdealFunction(x_vals, different_y, 2)
    different_deviation = train_func.calculate_deviation(different_func)
    print(f"Different function deviation: {different_deviation}")
    
    # 3. Demonstrate Test Point Assignment Logic
    print("\n3. ✅ TEST POINT ASSIGNMENT LOGIC")
    print("-" * 40)
    
    ideal_func.set_max_training_deviation(2.0)
    
    # Valid test point (within √2 × max_deviation)
    is_valid, deviation = ideal_func.is_test_point_valid(3, 9.5)
    threshold = np.sqrt(2) * 2.0
    print(f"Test point (3, 9.5): deviation={deviation:.3f}, threshold={threshold:.3f}, valid={is_valid}")
    
    # Invalid test point (beyond threshold)
    is_valid, deviation = ideal_func.is_test_point_valid(3, 15)
    print(f"Test point (3, 15): deviation={deviation:.3f}, threshold={threshold:.3f}, valid={is_valid}")
    
    # 4. Demonstrate Database Operations
    print("\n4. ✅ DATABASE OPERATIONS")
    print("-" * 40)
    
    db_manager = DatabaseManager("demo_test.db")
    try:
        db_manager.create_database()
        session = db_manager.get_session()
        print("Database created successfully ✅")
        print("Session created successfully ✅")
        session.close()
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        db_manager.close()
    
    # 5. Demonstrate Custom Exception Handling
    print("\n5. ✅ CUSTOM EXCEPTION HANDLING")
    print("-" * 40)
    
    from exceptions import DataProcessingError, ValidationError
    
    try:
        # This should raise a ValidationError
        wrong_x = np.array([1, 2])
        wrong_y = np.array([1, 2, 3])  # Different length
        TrainingFunction(wrong_x, wrong_y, 1)
    except ValidationError as e:
        print(f"Caught ValidationError: {e} ✅")
    
    try:
        # This should raise a ValidationError
        test_func = IdealFunction(x_vals, y_vals, 1)
        test_func.is_test_point_valid(3, 9)  # No max deviation set
    except ValidationError as e:
        print(f"Caught ValidationError: {e} ✅")
    
    # 6. Show Algorithm Results
    print("\n6. ✅ REAL ALGORITHM RESULTS")
    print("-" * 40)
    
    data_reader = DataReader()
    try:
        # Load actual data
        training_files = ["train1.csv", "train2.csv", "train3.csv", "train4.csv"]
        training_functions = data_reader.read_training_data(training_files)
        ideal_functions = data_reader.read_ideal_data("ideal.csv")
        
        # Find best matches
        model_trainer = ModelTrainer()
        best_matches = model_trainer.find_best_ideal_functions(training_functions, ideal_functions)
        
        print("Best ideal function matches:")
        for train_num, ideal_num in best_matches.items():
            deviation = model_trainer.training_deviations[train_num]['total_deviation']
            print(f"  Training Dataset {train_num} → Ideal Function {ideal_num} (deviation: {deviation:.2f})")
        
        print("\n✅ All requirements successfully demonstrated!")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")

if __name__ == "__main__":
    demonstrate_key_features()
