"""
Alternative visualization using matplotlib for immediate viewing.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_handler import DataReader
from model_trainer import ModelTrainer

def create_matplotlib_visualization():
    """Create matplotlib visualizations as backup to Bokeh."""
    
    
    print("Creating matplotlib visualization...")
      # Load data
    data_reader = DataReader()
    training_file = "train.csv"
    training_functions = data_reader.read_training_data(training_file)
    ideal_functions = data_reader.read_ideal_data("ideal.csv")
    test_data = data_reader.read_test_data("test.csv")
    
    # Find best matches
    model_trainer = ModelTrainer()
    best_matches = model_trainer.find_best_ideal_functions(training_functions, ideal_functions)
    
    # Get selected ideal functions
    selected_ideal_functions = {train_num: ideal_functions[ideal_num] 
                               for train_num, ideal_num in best_matches.items()}
    
    # Assign test data
    test_assignments = model_trainer.assign_test_data(test_data, selected_ideal_functions)
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'orange', 'green', 'red']
    
    # Plot training data
    for i, (train_num, train_func) in enumerate(training_functions.items()):
        plt.scatter(train_func.x_values, train_func.y_values, 
                   c=colors[i], alpha=0.7, s=50, 
                   label=f'Training Dataset {train_num}')
    
    # Plot selected ideal functions
    for i, (train_num, ideal_func) in enumerate(selected_ideal_functions.items()):
        x_range = np.linspace(min(ideal_func.x_values), max(ideal_func.x_values), 200)
        y_range = [ideal_func.interpolate_y(x) for x in x_range]
        plt.plot(x_range, y_range, 
                color=colors[i], linewidth=2, alpha=0.8,
                label=f'Ideal Function {ideal_func.function_number}')
    
    # Plot test data
    assigned_x = [a['x'] for a in test_assignments if a['assigned_ideal_function'] is not None]
    assigned_y = [a['y'] for a in test_assignments if a['assigned_ideal_function'] is not None]
    unassigned_x = [a['x'] for a in test_assignments if a['assigned_ideal_function'] is None]
    unassigned_y = [a['y'] for a in test_assignments if a['assigned_ideal_function'] is None]
    
    if assigned_x:
        plt.scatter(assigned_x, assigned_y, 
                   c='darkgreen', marker='s', s=100, alpha=0.8,
                   label=f'Assigned Test Points ({len(assigned_x)})')
    
    if unassigned_x:
        plt.scatter(unassigned_x, unassigned_y, 
                   c='darkred', marker='^', s=100, alpha=0.8,
                   label=f'Unassigned Test Points ({len(unassigned_x)})')
    
    plt.xlabel('X Values', fontsize=12)
    plt.ylabel('Y Values', fontsize=12)
    plt.title('Ideal Function Selection Results', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('matplotlib_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Matplotlib visualization saved as 'matplotlib_visualization.png'")
    
    # Print summary
    print(f"\n📊 VISUALIZATION SUMMARY:")
    print(f"Training Datasets: 4 (shown as colored circles)")
    print(f"Selected Ideal Functions: 4 (shown as colored lines)")
    print(f"Test Points Assigned: {len(assigned_x)} (green squares)")
    print(f"Test Points Unassigned: {len(unassigned_x)} (red triangles)")
    print(f"Total Test Points: {len(test_assignments)}")
    
    # Print best matches
    print(f"\n🎯 BEST MATCHES:")
    for train_num, ideal_num in best_matches.items():
        deviation = model_trainer.training_deviations[train_num]['total_deviation']
        print(f"Training Dataset {train_num} → Ideal Function {ideal_num} (deviation: {deviation:.2f})")

if __name__ == "__main__":
    create_matplotlib_visualization()
