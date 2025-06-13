"""
Main application that orchestrates the entire ideal function selection process.
"""

import os
from typing import List
from database import DatabaseManager, TrainingData, IdealFunctions, TestDataMappings
from data_handler import DataReader
from model_trainer import ModelTrainer
from visualizer import DataVisualizer
from exceptions import DataProcessingError, DatabaseError


class IdealFunctionSelector:
    """Main application class that coordinates all components."""
    
    def __init__(self, db_path: str = "ideal_functions.db"):
        """
        Initialize the ideal function selector.
        
        Args:
            db_path (str): Path to the database file
        """
        self.db_manager = DatabaseManager(db_path)
        self.data_reader = DataReader()
        self.model_trainer = ModelTrainer()
        self.visualizer = DataVisualizer()
          # Data storage
        self.training_functions = {}
        self.ideal_functions = {}
        self.test_data = None
        self.test_assignments = []
        self.best_matches = {}
    
    def run_complete_analysis(self, 
                            training_file: str,
                            ideal_file: str,
                            test_file: str):
        """
        Run the complete ideal function selection analysis.
        
        Args:
            training_file (str): Path to training data file
            ideal_file (str): Path to ideal functions file
            test_file (str): Path to test data file
        """
        try:
            print("Starting Ideal Function Selection Analysis...")
            print("="*50)
            
            # Step 1: Initialize database
            print("1. Initializing database...")
            self.db_manager.create_database()
            
            # Step 2: Load data
            print("2. Loading data files...")
            self.training_functions = self.data_reader.read_training_data(training_file)
            self.ideal_functions = self.data_reader.read_ideal_data(ideal_file)
            self.test_data = self.data_reader.read_test_data(test_file)
            
            # Step 3: Store data in database
            print("3. Storing data in database...")
            self._store_data_in_database()
            
            # Step 4: Find best ideal functions
            print("4. Finding best ideal functions for each training dataset...")
            self.best_matches = self.model_trainer.find_best_ideal_functions(
                self.training_functions, self.ideal_functions)
            
            # Step 5: Assign test data
            print("5. Assigning test data to ideal functions...")
            selected_ideal_functions = {train_num: self.ideal_functions[ideal_num] 
                                       for train_num, ideal_num in self.best_matches.items()}
            
            self.test_assignments = self.model_trainer.assign_test_data(
                self.test_data, selected_ideal_functions)
            
            # Step 6: Store test assignments in database
            print("6. Storing test assignments in database...")
            self._store_test_assignments()
            
            # Step 7: Create visualizations
            print("7. Creating visualizations...")
            self.visualizer.create_comprehensive_plot(
                self.training_functions, selected_ideal_functions, self.test_assignments)
            
            self.visualizer.create_deviation_analysis_plot(
                self.model_trainer.training_deviations, self.test_assignments)
            
            # Step 8: Generate and display summary
            print("8. Generating results summary...")
            training_summary = self.model_trainer.get_training_summary()
            stats = self.visualizer.create_summary_statistics(training_summary, self.test_assignments)
            self.visualizer.print_results_summary(stats)
            
            print("\nAnalysis completed successfully!")
            print("Check the generated HTML files for visualizations.")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise
        finally:
            self.db_manager.close()
    
    def _store_data_in_database(self):
        """Store all loaded data in the database."""
        session = self.db_manager.get_session()
        
        try:
            # Store training data
            for train_num, train_func in self.training_functions.items():
                for x, y in zip(train_func.x_values, train_func.y_values):
                    # Create a record with all training datasets
                    existing_record = session.query(TrainingData).filter_by(x=x).first()
                    
                    if existing_record is None:
                        record = TrainingData(x=x, y1=0, y2=0, y3=0, y4=0)
                        session.add(record)
                        session.flush()
                        existing_record = record
                    
                    # Set the appropriate y value
                    setattr(existing_record, f'y{train_num}', y)
            
            session.commit()
            
            # Store ideal functions
            if self.ideal_functions:
                first_ideal = list(self.ideal_functions.values())[0]
                for i, x in enumerate(first_ideal.x_values):
                    y_values = {}
                    for ideal_num in range(1, 51):
                        if ideal_num in self.ideal_functions:
                            y_values[f'y{ideal_num}'] = self.ideal_functions[ideal_num].y_values[i]
                        else:
                            y_values[f'y{ideal_num}'] = 0
                    
                    record = IdealFunctions(x=x, **y_values)
                    session.add(record)
            
            session.commit()
            print("Data successfully stored in database")
            
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Error storing data in database: {str(e)}")
        finally:
            session.close()
    
    def _store_test_assignments(self):
        """Store test data assignments in the database."""
        session = self.db_manager.get_session()
        
        try:
            for assignment in self.test_assignments:
                record = TestDataMappings(
                    x=assignment['x'],
                    y=assignment['y'],
                    assigned_ideal_function=assignment['assigned_ideal_function'],
                    deviation=assignment['deviation']
                )
                session.add(record)
            
            session.commit()
            print("Test assignments successfully stored in database")
            
        except Exception as e:
            session.rollback()
            raise DatabaseError(f"Error storing test assignments: {str(e)}")
        finally:
            session.close()


def main():
    """Main function to run the application."""
    # File paths
    training_file = "train.csv"
    ideal_file = "ideal.csv"
    test_file = "test.csv"
    
    # Verify files exist
    all_files = [training_file, ideal_file, test_file]
    missing_files = [f for f in all_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing files: {missing_files}")
        print("Please ensure all required CSV files are in the current directory.")
        return
      # Run the analysis
    try:
        selector = IdealFunctionSelector()
        selector.run_complete_analysis(training_file, ideal_file, test_file)
    except Exception as e:
        print(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()
