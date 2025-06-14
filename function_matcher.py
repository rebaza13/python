"""
Model training and selection logic for finding the best ideal functions.
"""

import numpy as np
from typing import Dict, List, Tuple
from data_loader import TrainingFunction, IdealFunction
from exceptions import DataProcessingError

class ModelTrainer:
    def __init__(self):
        self.best_ideal_functions = {}
        self.training_deviations = {}

    def find_best_ideal_functions(self, training_functions: Dict[int, TrainingFunction], ideal_functions: Dict[int, IdealFunction]) -> Dict[int, int]:
        try:
            best_matches = {}
            for train_num, train_func in training_functions.items():
                print(f"Finding best ideal function for training dataset {train_num}...")
                best_ideal_num = None
                min_deviation = float('inf')
                best_deviations = []
                for ideal_num, ideal_func in ideal_functions.items():
                    deviation = train_func.calculate_deviation(ideal_func)
                    if deviation < min_deviation:
                        min_deviation = deviation
                        best_ideal_num = ideal_num
                        best_deviations = self._calculate_point_deviations(train_func, ideal_func)
                if best_ideal_num is None:
                    raise DataProcessingError(f"No ideal function found for training dataset {train_num}")
                best_matches[train_num] = best_ideal_num
                max_dev = max(best_deviations) if best_deviations else 0
                ideal_functions[best_ideal_num].set_max_training_deviation(max_dev)
                self.training_deviations[train_num] = {
                    'total_deviation': min_deviation,
                    'max_point_deviation': max_dev,
                    'point_deviations': best_deviations
                }
                print(f"  Best match: Ideal function {best_ideal_num} (deviation: {min_deviation:.4f})")
            self.best_ideal_functions = {train_num: ideal_functions[ideal_num] for train_num, ideal_num in best_matches.items()}
            return best_matches
        except Exception as e:
            raise DataProcessingError(f"Error finding best ideal functions: {str(e)}", "model_training")

    def _calculate_point_deviations(self, train_func: TrainingFunction, ideal_func: IdealFunction) -> List[float]:
        deviations = []
        for x, y in zip(train_func.x_values, train_func.y_values):
            ideal_y = ideal_func.interpolate_y(x)
            deviations.append(abs(y - ideal_y))
        return deviations

    def assign_test_data(self, test_data, selected_ideal_functions: Dict[int, IdealFunction]) -> List[Dict]:
        try:
            assignments = []
            for _, row in test_data.iterrows():
                x, y = row['x'], row['y']
                best_assignment = None
                min_deviation = float('inf')
                for train_num, ideal_func in selected_ideal_functions.items():
                    is_valid, deviation = ideal_func.is_test_point_valid(x, y)
                    if is_valid and deviation < min_deviation:
                        min_deviation = deviation
                        best_assignment = {
                            'x': x,
                            'y': y,
                            'assigned_ideal_function': ideal_func.function_number,
                            'deviation': deviation,
                            'training_dataset': train_num
                        }
                if best_assignment is None:
                    assignments.append({
                        'x': x,
                        'y': y,
                        'assigned_ideal_function': None,
                        'deviation': None,
                        'training_dataset': None
                    })
                else:
                    assignments.append(best_assignment)
            assigned_count = len([a for a in assignments if a['assigned_ideal_function'] is not None])
            print(f"Assigned {assigned_count} out of {len(assignments)} test points")
            return assignments
        except Exception as e:
            raise DataProcessingError(f"Error assigning test data: {str(e)}", "test_assignment")

    def get_training_summary(self) -> Dict:
        summary = {
            'total_training_datasets': len(self.best_ideal_functions),
            'best_matches': {},
            'deviations': self.training_deviations
        }
        for train_num, ideal_func in self.best_ideal_functions.items():
            summary['best_matches'][train_num] = {
                'ideal_function_number': ideal_func.function_number,
                'total_deviation': self.training_deviations[train_num]['total_deviation'],
                'max_point_deviation': self.training_deviations[train_num]['max_point_deviation']
            }
        return summary
