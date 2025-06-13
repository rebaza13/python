"""
Custom exceptions for the ideal function selection system.
"""


class DataProcessingError(Exception):
    """Raised when there are issues with data processing or validation."""
    
    def __init__(self, message: str, data_type: str = None):
        """
        Initialize the DataProcessingError.
        
        Args:
            message (str): Error message describing the issue
            data_type (str, optional): Type of data that caused the error
        """
        self.data_type = data_type
        super().__init__(message)
    
    def __str__(self):
        if self.data_type:
            return f"DataProcessingError in {self.data_type}: {super().__str__()}"
        return f"DataProcessingError: {super().__str__()}"


class DatabaseError(Exception):
    """Raised when there are database-related issues."""
    pass


class ValidationError(Exception):
    """Raised when data validation fails."""
    pass
