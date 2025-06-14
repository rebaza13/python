"""
Visualization module using Bokeh for interactive plots.
"""

from bokeh.plotting import figure, save, output_file
from bokeh.layouts import column, row
from bokeh.models import HoverTool
from bokeh.palettes import Category10
import pandas as pd
import numpy as np
from typing import Dict, List
from data_loader import TrainingFunction, IdealFunction


class DataVisualizer:
    """Handles all visualization tasks using Bokeh."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.colors = Category10[10]
    
    def create_comprehensive_plot(self, 
                                training_functions: Dict[int, TrainingFunction],
                                selected_ideal_functions: Dict[int, IdealFunction],
                                test_assignments: List[Dict],
                                output_filename: str = "results_visualization.html"):
        """
        Create a comprehensive visualization showing all data and results.
        
        Args:
            training_functions (Dict[int, TrainingFunction]): Training datasets
            selected_ideal_functions (Dict[int, IdealFunction]): Selected ideal functions
            test_assignments (List[Dict]): Test data assignments
            output_filename (str): Output HTML filename
        """
        output_file(output_filename)
        
        # Create the main plot
        p = figure(width=1000, height=600, 
                  title="Ideal Function Selection Results",
                  x_axis_label="X Values",
                  y_axis_label="Y Values",
                  toolbar_location="above")
          # Plot training data
        for i, (train_num, train_func) in enumerate(training_functions.items()):
            color = self.colors[i % len(self.colors)]
            p.scatter(train_func.x_values, train_func.y_values, 
                     size=8, color=color, alpha=0.7, marker='circle',
                     legend_label=f"Training Dataset {train_num}")
        
        # Plot selected ideal functions
        for i, (train_num, ideal_func) in enumerate(selected_ideal_functions.items()):
            color = self.colors[i % len(self.colors)]
            # Create a denser x range for smooth ideal function curves
            x_range = np.linspace(min(ideal_func.x_values), max(ideal_func.x_values), 200)
            y_range = [ideal_func.interpolate_y(x) for x in x_range]
            
            p.line(x_range, y_range, 
                  line_width=2, color=color, alpha=0.8,
                  legend_label=f"Ideal Function {ideal_func.function_number}")
        
        # Plot test data points
        assigned_x = [a['x'] for a in test_assignments if a['assigned_ideal_function'] is not None]
        assigned_y = [a['y'] for a in test_assignments if a['assigned_ideal_function'] is not None]
        assigned_functions = [a['assigned_ideal_function'] for a in test_assignments if a['assigned_ideal_function'] is not None]
        assigned_deviations = [a['deviation'] for a in test_assignments if a['assigned_ideal_function'] is not None]
        
        unassigned_x = [a['x'] for a in test_assignments if a['assigned_ideal_function'] is None]
        unassigned_y = [a['y'] for a in test_assignments if a['assigned_ideal_function'] is None]
          # Assigned test points
        if assigned_x:
            assigned_source = p.scatter(assigned_x, assigned_y, 
                                       size=10, color='green', alpha=0.8, marker='square',
                                       legend_label="Assigned Test Points")
            
            # Add hover tool for assigned points
            hover_assigned = HoverTool(renderers=[assigned_source],
                                     tooltips=[
                                         ("X", "@x"),
                                         ("Y", "@y"),
                                         ("Ideal Function", "@ideal_function"),
                                         ("Deviation", "@deviation{0.000}")
                                     ])
            
            # Add custom data to the source
            assigned_source.data_source.add(assigned_functions, 'ideal_function')
            assigned_source.data_source.add(assigned_deviations, 'deviation')
            p.add_tools(hover_assigned)
          # Unassigned test points
        if unassigned_x:
            p.scatter(unassigned_x, unassigned_y, 
                     size=10, color='red', alpha=0.8, marker='triangle',
                     legend_label="Unassigned Test Points")
        
        # Customize legend
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        
        # Save the plot
        save(p)
        print(f"Visualization saved as {output_filename}")
    
    def create_deviation_analysis_plot(self, 
                                     training_deviations: Dict,
                                     test_assignments: List[Dict],
                                     output_filename: str = "deviation_analysis.html"):
        """
        Create plots showing deviation analysis.
        
        Args:
            training_deviations (Dict): Training deviation data
            test_assignments (List[Dict]): Test data assignments
            output_filename (str): Output HTML filename
        """
        output_file(output_filename)
        
        plots = []
        
        # Plot 1: Training deviations by dataset
        p1 = figure(width=500, height=400, 
                   title="Training Dataset Deviations",
                   x_axis_label="Training Dataset",
                   y_axis_label="Total Squared Deviation")
        
        datasets = list(training_deviations.keys())
        total_devs = [training_deviations[d]['total_deviation'] for d in datasets]
        max_devs = [training_deviations[d]['max_point_deviation'] for d in datasets]
        
        p1.vbar(x=datasets, top=total_devs, width=0.4, color='blue', alpha=0.7,
               legend_label="Total Deviation")
        
        plots.append(p1)
        
        # Plot 2: Test point deviations
        p2 = figure(width=500, height=400,
                   title="Test Point Deviations",
                   x_axis_label="Test Point Index",
                   y_axis_label="Deviation")
        
        assigned_indices = []
        assigned_devs = []
        
        for i, assignment in enumerate(test_assignments):
            if assignment['deviation'] is not None:
                assigned_indices.append(i)
                assigned_devs.append(assignment['deviation'])
        
        if assigned_indices:
            p2.circle(assigned_indices, assigned_devs, size=6, color='green', alpha=0.7)
        
        plots.append(p2)
        
        # Combine plots
        layout = row(*plots)
        save(layout)
        print(f"Deviation analysis saved as {output_filename}")
    
    def create_summary_statistics(self, 
                                training_summary: Dict,
                                test_assignments: List[Dict]) -> Dict:
        """
        Create summary statistics for the analysis.
        
        Args:
            training_summary (Dict): Training summary data
            test_assignments (List[Dict]): Test assignments
            
        Returns:
            Dict: Summary statistics
        """
        assigned_count = len([a for a in test_assignments if a['assigned_ideal_function'] is not None])
        total_test_points = len(test_assignments)
        
        assigned_deviations = [a['deviation'] for a in test_assignments if a['deviation'] is not None]
        
        stats = {
            'training_results': training_summary,
            'test_assignment_rate': assigned_count / total_test_points if total_test_points > 0 else 0,
            'total_test_points': total_test_points,
            'assigned_test_points': assigned_count,
            'unassigned_test_points': total_test_points - assigned_count,
            'test_deviation_stats': {
                'mean': np.mean(assigned_deviations) if assigned_deviations else 0,
                'std': np.std(assigned_deviations) if assigned_deviations else 0,
                'min': np.min(assigned_deviations) if assigned_deviations else 0,
                'max': np.max(assigned_deviations) if assigned_deviations else 0
            }
        }
        
        return stats
    
    def print_results_summary(self, stats: Dict):
        """
        Print a formatted summary of results.
        
        Args:
            stats (Dict): Summary statistics
        """
        print("\n" + "="*60)
        print("IDEAL FUNCTION SELECTION RESULTS SUMMARY")
        print("="*60)
        
        print("\nTRAINING RESULTS:")
        for train_num, match_info in stats['training_results']['best_matches'].items():
            print(f"  Dataset {train_num} → Ideal Function {match_info['ideal_function_number']}")
            print(f"    Total Deviation: {match_info['total_deviation']:.4f}")
            print(f"    Max Point Deviation: {match_info['max_point_deviation']:.4f}")
        
        print(f"\nTEST DATA ASSIGNMENT:")
        print(f"  Total Test Points: {stats['total_test_points']}")
        print(f"  Assigned Points: {stats['assigned_test_points']}")
        print(f"  Unassigned Points: {stats['unassigned_test_points']}")
        print(f"  Assignment Rate: {stats['test_assignment_rate']:.2%}")
        
        if stats['test_deviation_stats']['mean'] > 0:
            print(f"\nTEST DEVIATION STATISTICS:")
            print(f"  Mean Deviation: {stats['test_deviation_stats']['mean']:.4f}")
            print(f"  Std Deviation: {stats['test_deviation_stats']['std']:.4f}")
            print(f"  Min Deviation: {stats['test_deviation_stats']['min']:.4f}")
            print(f"  Max Deviation: {stats['test_deviation_stats']['max']:.4f}")
        
        print("="*60)
