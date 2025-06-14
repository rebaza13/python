"""
Data handling classes with object-oriented design and inheritance.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from exceptions import DataProcessingError, ValidationError


class BaseFunction(ABC):
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray, function_id: str):
        self.x_values = x_values
        self.y_values = y_values
        self.function_id = function_id
        self.validate_data()

    def validate_data(self):
        if len(self.x_values) != len(self.y_values):
            raise ValidationError(f"X and Y arrays must have the same length for function {self.function_id}")

    @abstractmethod
    def calculate_deviation(self, other_function: 'BaseFunction') -> float:
        pass

    def interpolate_y(self, x: float) -> float:
        return np.interp(x, self.x_values, self.y_values)

class TrainingFunction(BaseFunction):
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray, dataset_number: int):
        super().__init__(x_values, y_values, f"training_{dataset_number}")
        self.dataset_number = dataset_number

    def calculate_deviation(self, other_function: BaseFunction) -> float:
        try:
            squared_differences = []
            for x, y in zip(self.x_values, self.y_values):
                other_y = other_function.interpolate_y(x)
                squared_differences.append((y - other_y) ** 2)
            return sum(squared_differences)
        except Exception as e:
            raise DataProcessingError(f"Error calculating deviation: {str(e)}", "training_function")

class IdealFunction(BaseFunction):
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray, function_number: int):
        super().__init__(x_values, y_values, f"ideal_{function_number}")
        self.function_number = function_number
        self.max_training_deviation = None

    def calculate_deviation(self, other_function: BaseFunction) -> float:
        return other_function.calculate_deviation(self)

    def set_max_training_deviation(self, deviation: float):
        self.max_training_deviation = deviation

    def is_test_point_valid(self, x: float, y: float) -> Tuple[bool, float]:
        if self.max_training_deviation is None:
            raise ValidationError("Max training deviation not set for ideal function")
        ideal_y = self.interpolate_y(x)
        deviation = abs(y - ideal_y)
        threshold = np.sqrt(2) * self.max_training_deviation
        return deviation <= threshold, deviation

class DataReader:
    def __init__(self):
        self.training_data = {}
        self.ideal_data = None
        self.test_data = None

    def read_training_data(self, file_path: str) -> Dict[int, TrainingFunction]:
        try:
            training_functions = {}
            df = pd.read_csv(file_path)
            if 'x' not in df.columns:
                raise DataProcessingError(f"No 'x' column found in {file_path}", "training_data")
            x_values = df['x'].values
            for i in range(1, 5):
                y_col = f'y{i}'
                if y_col not in df.columns:
                    raise DataProcessingError(f"No '{y_col}' column found in {file_path}", "training_data")
                y_values = df[y_col].values
                training_functions[i] = TrainingFunction(x_values, y_values, i)
            self.training_data = training_functions
            print(f"Successfully loaded {len(training_functions)} training datasets")
            return training_functions
        except Exception as e:
            raise DataProcessingError(f"Error reading training data: {str(e)}", "training_data")

    def read_ideal_data(self, file_path: str) -> Dict[int, IdealFunction]:
        try:
            df = pd.read_csv(file_path)
            if 'x' not in df.columns:
                raise DataProcessingError("No 'x' column found in ideal functions file", "ideal_data")
            x_values = df['x'].values
            ideal_functions = {}
            for i in range(1, 51):
                y_col = f'y{i}'
                if y_col in df.columns:
                    y_values = df[y_col].values
                    ideal_functions[i] = IdealFunction(x_values, y_values, i)
            self.ideal_data = ideal_functions
            print(f"Successfully loaded {len(ideal_functions)} ideal functions")
            return ideal_functions
        except Exception as e:
            raise DataProcessingError(f"Error reading ideal data: {str(e)}", "ideal_data")

    def read_test_data(self, file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            if 'x' not in df.columns or 'y' not in df.columns:
                raise DataProcessingError("Invalid column names in test data file", "test_data")
            self.test_data = df
            print(f"Successfully loaded {len(df)} test data points")
            return df
        except Exception as e:
            raise DataProcessingError(f"Error reading test data: {str(e)}", "test_data")
