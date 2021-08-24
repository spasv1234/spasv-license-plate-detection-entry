import pandas as pd
import sqlalchemy
from sqlalchemy import and_
from sqlalchemy import Column, Integer, String, Boolean,LargeBinary, DateTime, ForeignKey, Table
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
import config
from datetime import datetime

#If writing to database is enabled, connect to database, and create session
if config.enable_write_to_database:
    engine = sqlalchemy.create_engine(config.database_connection)

    Session = sessionmaker(bind=engine)
    session = Session()

Base = declarative_base()

#Association tables
vehicle_entry_permission = Table('vehicle_entry_permission',
                                 Base.metadata,
                                 Column('vehicle',String(10), ForeignKey('vehicle.license_plate'), nullable=False),
                                 Column('entry_permission',Integer,ForeignKey('entry_permission.entry_permission_id'), nullable=False)
                                 )

vehicle_parking_permission = Table('vehicle_parking_permission',
                                   Base.metadata,
                                   Column('vehicle',String(10), ForeignKey('vehicle.license_plate'), nullable=False),
                                   Column('parking_permission',Integer,ForeignKey('parking_permission.parking_permission_id'), nullable=False)
                                   )

class Location(Base):

    __tablename__ = "location"
    
    location_id = Column(Integer, primary_key=True)
    name = Column(String(50))

    #Relationships
    vehicle_parking_record = relationship('Vehicle_parking_record', backref='parking_location', lazy=True)
    vehicle_entry_record = relationship('Vehicle_entry_record', backref='entry_location', lazy=True)

class Vehicle(Base):
    
    __tablename__ = "vehicle"
    
    vehicle_id = Column(Integer, primary_key=True)
    license_plate = Column(String(10), nullable=False, unique=True)

    #Relationships
    vehicle_entry_permission = relationship('Entry_permission',secondary=vehicle_entry_permission, backref='vehicle_entry_permission', lazy='dynamic')            #Many to Many
    vehicle_parking_permission = relationship('Parking_permission',secondary=vehicle_parking_permission,backref='vehicle_parking_permission', lazy='dynamic')     #Many to Many
    vehicle_entry_record = relationship('Vehicle_entry_record', backref='entry_record_of', lazy=True)                                                                        #One to Many
    vehicle_parking_record = relationship('Vehicle_parking_record', backref='parking_record_of', lazy=True)                                                                    #One to Many

class Entry_permission(Base):

    __tablename__ = "entry_permission"
    
    entry_permission_id = Column(Integer, primary_key=True)
    min_time = Column(DateTime, nullable=True)
    max_time = Column(DateTime, nullable=True)
    location_id = Column(Integer, ForeignKey('location.location_id'), nullable=False)
    authorization = Column(Boolean, nullable=False)
    vehicle = relationship('Vehicle',secondary=vehicle_entry_permission,backref='entry_permission', lazy='dynamic')


class Parking_permission(Base):

    __tablename__ = "parking_permission"
    
    parking_permission_id = Column(Integer, primary_key=True)
    min_time = Column(DateTime, nullable=True)
    max_time = Column(DateTime, nullable=True)
    location_id = Column(Integer, ForeignKey('location.location_id'), nullable=False)
    authorization = Column(Boolean, nullable=False)
    vehicle = relationship('Vehicle',secondary=vehicle_parking_permission,backref='parking_permission', lazy='dynamic')

class Vehicle_entry_record(Base):

    __tablename__ = "vehicle_entry_record"
    
    entry_record_id = Column(Integer, primary_key=True)
    vehicle = Column(String(10), ForeignKey('vehicle.license_plate'), nullable=False)
    entry_time = Column(DateTime, nullable=False)
    location_id = Column(Integer, ForeignKey('location.location_id'), nullable=True)
    image = Column(LargeBinary,nullable=True)
    
class Vehicle_parking_record(Base):

    __tablename__ = "vehicle_parking_record"
    
    parking_record_id = Column(Integer, primary_key=True)
    vehicle = Column(String(10), ForeignKey('vehicle.license_plate'), nullable=False)
    first_detected = Column(DateTime, nullable=False)
    last_detected = Column(DateTime, nullable=True)
    location_id = Column(Integer, ForeignKey('location.location_id'), nullable=False)
    image = Column(LargeBinary,nullable=True)
    
#---------------------------------------------------------------------------------------
#CRUD Functions
def add_vehicle_to_database(license_plate):
    if session.query(Vehicle).filter(Vehicle.license_plate==license_plate).first() is None:
        vehicle = Vehicle(license_plate=license_plate)
        session.add(vehicle)
        session.commit()
    else:
        print("License Plate Already Exist In Database")

def add_location_to_database(name):
    if session.query(Location).filter(Location.name==name).first() is None:
        location = Location(name=name)
        session.add(location)
        session.commit()
    else:
        print("Location Already Exist In Database")

def add_vehicle_parking_record(license_plate,first_detected,last_detected=None,location_id=None):
    vehicle=session.query(Vehicle).filter(Vehicle.license_plate==license_plate).first()
    location=session.query(Location).filter(Location.location_id==location_id).first()
    if vehicle is not None:
        vehicle_parking_record = Vehicle_parking_record(parking_record_of=vehicle,first_detected=first_detected,last_detected=last_detected,parking_location=location)
        session.add(vehicle_parking_record)
        session.commit()
    else:
        print("Vehicle Does Not Exist In Database")

def update_vehicle_parking_record(license_plate,first_detected,last_detected=None,location_id=None):
    vehicle_parking_record = session.query(Vehicle_parking_record).filter(and_(Vehicle_parking_record.vehicle==license_plate,Vehicle_parking_record.first_detected==first_detected)).first()
    if vehicle_parking_record is not None:
        vehicle_parking_record.last_detected = last_detected
        session.commit()
        print("RECORD UPDATED")
    else:
        print("FAIL TO UPDATE")

def add_vehicle_entry_record(license_plate,time_detected,location_id):
    vehicle_entry_record = Vehicle_entry_record(vehicle=license_plate,entry_time=time_detected,location_id=location_id)
    session.add(vehicle_entry_record)
    session.commit()

add_location_to_database("Singapore Polytechnic")
#add_vehicle_to_database("FW8720K")
#add_vehicle_entry_record("FW8720K",datetime.now(),location_id=1)



"""
def add_license_plate(license_plate):
    if session.query(License_plate).filter(License_plate.license_plate==license_plate).first() is None:
        license_plate = License_plate(license_plate=license_plate)
        session.add(license_plate)
        session.commit()
    else:
        print("License plate not added to database because it is already in the database")
"""
"""
#Add vehicle entry record table to database
#Used when a vehicle entered for the first time
def add_vehicle_entry_record(license_plate,first_detected,last_detected=None,location=None):
    vehicle_entry_record = Vehicle_entry_record(license_plate_id=license_plate,first_detected=first_detected,last_detected=last_detected,location=location)                                     #Create object for vehicle entry record to be mapped to the database table
    session.add(vehicle_entry_record)                                                                                                                                                           #Add to session
    session.commit()                                                                                                                                                                            #Make changes to database

#Update vehicle entry record table from database
#Used when a vehicle is leaving to update the last_detected field
def update_vehicle_entry_record(license_plate,first_detected,last_detected=None,location=None):
    first_detected = first_detected.replace(microsecond=0)                                                                                                                                      #The format of datetime object in python includes ms which have to be removed to be compared with datetime object of sql
    vehicle_entry_record = session.query(Vehicle_entry_record).filter(and_(Vehicle_entry_record.license_plate_id==license_plate,Vehicle_entry_record.first_detected==first_detected)).first()   #Get the first instance of vehicle with the license plate that enter at a certain time 
    if vehicle_entry_record is not None:
        vehicle_entry_record.last_detected = last_detected                                                                                                                                      #Update vehicle entry with time of vehicle leaving                                                                                            
        session.commit()
    else:
        print("Update vehicle entry record fail. No result found from query")
"""
"""
#Test
#add_vehicle_entry_record("test",datetime.now(),None,"Singapore Poly")
print(not session.query(License_plate).filter(License_plate.license_plate==license_plate).first())
print(type(session.query(License_plate).filter(License_plate.license_plate==license_plate).first()))
"""
