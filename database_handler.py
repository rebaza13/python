"""
Database models and setup for the ideal function selection system.
"""

from sqlalchemy import create_engine, Column, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from exceptions import DatabaseError

Base = declarative_base()

class TrainingData(Base):
    __tablename__ = 'training_data'
    id = Column(Integer, primary_key=True)
    x = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    y3 = Column(Float, nullable=False)
    y4 = Column(Float, nullable=False)
    def __repr__(self):
        return f"<TrainingData(x={self.x}, y1={self.y1}, y2={self.y2}, y3={self.y3}, y4={self.y4})>"

class IdealFunctions(Base):
    __tablename__ = 'ideal_functions'
    id = Column(Integer, primary_key=True)
    x = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
    y3 = Column(Float, nullable=False)
    y4 = Column(Float, nullable=False)
    y5 = Column(Float, nullable=False)
    y6 = Column(Float, nullable=False)
    y7 = Column(Float, nullable=False)
    y8 = Column(Float, nullable=False)
    y9 = Column(Float, nullable=False)
    y10 = Column(Float, nullable=False)
    y11 = Column(Float, nullable=False)
    y12 = Column(Float, nullable=False)
    y13 = Column(Float, nullable=False)
    y14 = Column(Float, nullable=False)
    y15 = Column(Float, nullable=False)
    y16 = Column(Float, nullable=False)
    y17 = Column(Float, nullable=False)
    y18 = Column(Float, nullable=False)
    y19 = Column(Float, nullable=False)
    y20 = Column(Float, nullable=False)
    y21 = Column(Float, nullable=False)
    y22 = Column(Float, nullable=False)
    y23 = Column(Float, nullable=False)
    y24 = Column(Float, nullable=False)
    y25 = Column(Float, nullable=False)
    y26 = Column(Float, nullable=False)
    y27 = Column(Float, nullable=False)
    y28 = Column(Float, nullable=False)
    y29 = Column(Float, nullable=False)
    y30 = Column(Float, nullable=False)
    y31 = Column(Float, nullable=False)
    y32 = Column(Float, nullable=False)
    y33 = Column(Float, nullable=False)
    y34 = Column(Float, nullable=False)
    y35 = Column(Float, nullable=False)
    y36 = Column(Float, nullable=False)
    y37 = Column(Float, nullable=False)
    y38 = Column(Float, nullable=False)
    y39 = Column(Float, nullable=False)
    y40 = Column(Float, nullable=False)
    y41 = Column(Float, nullable=False)
    y42 = Column(Float, nullable=False)
    y43 = Column(Float, nullable=False)
    y44 = Column(Float, nullable=False)
    y45 = Column(Float, nullable=False)
    y46 = Column(Float, nullable=False)
    y47 = Column(Float, nullable=False)
    y48 = Column(Float, nullable=False)
    y49 = Column(Float, nullable=False)
    y50 = Column(Float, nullable=False)

class TestDataMappings(Base):
    __tablename__ = 'test_data_mappings'
    id = Column(Integer, primary_key=True)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    assigned_ideal_function = Column(Integer, nullable=True)
    deviation = Column(Float, nullable=True)
    def __repr__(self):
        return f"<TestDataMapping(x={self.x}, y={self.y}, function={self.assigned_ideal_function}, deviation={self.deviation})>"

class DatabaseManager:
    def __init__(self, db_path: str = "ideal_functions.db"):
        self.db_path = db_path
        self.engine = None
        self.Session = None
    def create_database(self):
        try:
            self.engine = create_engine(f'sqlite:///{self.db_path}')
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            print(f"Database created successfully at {self.db_path}")
        except Exception as e:
            raise DatabaseError(f"Failed to create database: {str(e)}")
    def get_session(self):
        if self.Session is None:
            raise DatabaseError("Database not initialized. Call create_database() first.")
        return self.Session()
    def close(self):
        if self.engine:
            self.engine.dispose()
