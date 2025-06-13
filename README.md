# Ideal Function Selection System

A comprehensive Python application that finds the best ideal functions for training datasets using least-square error analysis and assigns test data points to the selected functions based on deviation criteria.

## Features

- **Object-Oriented Design**: Uses inheritance with base classes and specialized implementations
- **Database Integration**: SQLite database with SQLAlchemy ORM for data persistence
- **Data Processing**: Pandas for efficient data handling and analysis
- **Interactive Visualizations**: Bokeh for creating beautiful, interactive plots
- **Comprehensive Testing**: Unit tests for all major components
- **Error Handling**: Custom exceptions and robust error handling
- **Documentation**: Thorough code documentation with docstrings

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 1.5.0
- numpy >= 1.21.0
- sqlalchemy >= 1.4.0
- bokeh >= 2.4.0
- matplotlib >= 3.5.0
- pytest >= 7.0.0

## Project Structure

```
ai_python/
├── main.py                 # Main application entry point
├── data_handler.py         # Data loading and processing classes
├── model_trainer.py        # Model training and selection logic
├── database.py            # Database models and management
├── visualizer.py          # Visualization and plotting
├── exceptions.py          # Custom exception classes
├── test_suite.py          # Comprehensive unit tests
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── train1.csv            # Training dataset 1
├── train2.csv            # Training dataset 2
├── train3.csv            # Training dataset 3
├── train4.csv            # Training dataset 4
├── ideal.csv             # 50 ideal functions
└── test.csv              # Test data points
```

## How It Works

### 1. Data Loading
The system loads:
- 4 training datasets (train1.csv - train4.csv)
- 50 ideal functions (ideal.csv)
- Test data points (test.csv)

### 2. Best Function Selection
For each training dataset, the system:
- Calculates least-square error against all 50 ideal functions
- Selects the ideal function with the minimum error
- Records the maximum deviation for later use

### 3. Test Data Assignment
For each test data point:
- Calculates deviation from each selected ideal function
- Assigns the point if deviation ≤ √2 × max_training_deviation
- Records the assignment and deviation in the database

### 4. Visualization
Creates interactive plots showing:
- Training data points
- Selected ideal functions
- Test data assignments
- Deviation analysis

## Usage

### Quick Start

Simply run the main application:

```bash
python main.py
```

This will automatically:
1. Create a SQLite database
2. Load all CSV files
3. Find the best ideal functions
4. Assign test data points
5. Generate visualizations
6. Display a results summary

### Custom Usage

```python
from main import IdealFunctionSelector

# Initialize the selector
selector = IdealFunctionSelector("custom_database.db")

# Run complete analysis
selector.run_complete_analysis(
    training_files=["train1.csv", "train2.csv", "train3.csv", "train4.csv"],
    ideal_file="ideal.csv",
    test_file="test.csv"
)
```

### Running Tests

Execute the comprehensive test suite:

```bash
python test_suite.py
```

Or using pytest:

```bash
pytest test_suite.py -v
```

## Object-Oriented Design

### Class Hierarchy

```
BaseFunction (Abstract)
├── TrainingFunction
└── IdealFunction

DataReader
ModelTrainer
DataVisualizer
DatabaseManager
IdealFunctionSelector (Main Controller)
```

### Key Classes

- **BaseFunction**: Abstract base class defining common functionality
- **TrainingFunction**: Inherits from BaseFunction, handles training data
- **IdealFunction**: Inherits from BaseFunction, handles ideal functions
- **DataReader**: Loads and validates CSV data
- **ModelTrainer**: Implements the selection algorithm
- **DataVisualizer**: Creates plots and visualizations
- **DatabaseManager**: Handles all database operations

### Custom Exceptions

- **DataProcessingError**: For data loading and processing issues
- **ValidationError**: For data validation failures
- **DatabaseError**: For database-related problems

## Database Schema

The system creates three tables:

### training_data
- id (Primary Key)
- x (Float)
- y1, y2, y3, y4 (Float) - Training dataset values

### ideal_functions
- id (Primary Key)
- x (Float)
- y1 through y50 (Float) - All ideal function values

### test_data_mappings
- id (Primary Key)
- x, y (Float) - Test point coordinates
- assigned_ideal_function (Integer) - Selected function number
- deviation (Float) - Calculated deviation

## Output Files

The application generates:

1. **ideal_functions.db** - SQLite database with all data
2. **results_visualization.html** - Main interactive plot
3. **deviation_analysis.html** - Deviation analysis plots

## Algorithm Details

### Least-Square Error Calculation

For each training dataset and ideal function pair:
```
error = Σ(y_training - y_ideal)²
```

### Test Point Assignment Criteria

A test point (x, y) is assigned to an ideal function if:
```
|y_test - y_ideal(x)| ≤ √2 × max_training_deviation
```

Where `max_training_deviation` is the largest deviation observed when the ideal function was fitted to the training data.

## Example Output

```
==============================================================
IDEAL FUNCTION SELECTION RESULTS SUMMARY
==============================================================

TRAINING RESULTS:
  Dataset 1 → Ideal Function 23
    Total Deviation: 157.2345
    Max Point Deviation: 3.4567

  Dataset 2 → Ideal Function 8
    Total Deviation: 203.1234
    Max Point Deviation: 4.2345

TEST DATA ASSIGNMENT:
  Total Test Points: 100
  Assigned Points: 87
  Unassigned Points: 13
  Assignment Rate: 87.00%

TEST DEVIATION STATISTICS:
  Mean Deviation: 1.2345
  Std Deviation: 0.8765
  Min Deviation: 0.0123
  Max Deviation: 4.5678
==============================================================
```

## Error Handling

The system includes robust error handling for:
- Missing or corrupt CSV files
- Invalid data formats
- Database connection issues
- Calculation errors
- Visualization problems

All errors are logged with descriptive messages and proper exception types.

## Best Practices Implemented

- **SOLID Principles**: Single responsibility, dependency injection
- **Documentation**: Comprehensive docstrings and comments
- **Testing**: Unit tests for all major components
- **Error Handling**: Custom exceptions with proper hierarchy
- **Type Hints**: Full type annotations for better code clarity
- **Data Validation**: Input validation at multiple levels
- **Separation of Concerns**: Each module has a specific responsibility

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is for educational purposes and demonstrates best practices in Python development, data analysis, and machine learning concepts.
