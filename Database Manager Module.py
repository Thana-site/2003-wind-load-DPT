"""
Database Manager Module
Handles SQLite database operations for storing and retrieving sections
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
import os

class DatabaseManager:
    """Manages SQLite database for section storage"""
    
    def __init__(self, db_path='data/sections.db'):
        """
        Initialize database manager
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create sections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    properties TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index on name and type for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_section_name 
                ON sections(name)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_section_type 
                ON sections(type)
            ''')
            
            conn.commit()
    
    def save_section(self, section_data):
        """
        Save a section to the database
        
        Args:
            section_data (dict): Dictionary containing section information
                - name: Section name
                - type: Section type
                - parameters: JSON string of parameters
                - properties: JSON string of calculated properties
        
        Returns:
            int: ID of the saved section
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if section with same name exists
            cursor.execute(
                "SELECT id FROM sections WHERE name = ?",
                (section_data['name'],)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing section
                cursor.execute('''
                    UPDATE sections 
                    SET type = ?, parameters = ?, properties = ?, 
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (
                    section_data['type'],
                    section_data['parameters'],
                    section_data['properties'],
                    existing[0]
                ))
                section_id = existing[0]
            else:
                # Insert new section
                cursor.execute('''
                    INSERT INTO sections (name, type, parameters, properties)
                    VALUES (?, ?, ?, ?)
                ''', (
                    section_data['name'],
                    section_data['type'],
                    section_data['parameters'],
                    section_data['properties']
                ))
                section_id = cursor.lastrowid
            
            conn.commit()
            return section_id
    
    def get_section(self, section_id):
        """
        Retrieve a section by ID
        
        Args:
            section_id (int): Section ID
            
        Returns:
            dict: Section data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM sections WHERE id = ?",
                (section_id,)
            )
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_section_by_name(self, name):
        """
        Retrieve a section by name
        
        Args:
            name (str): Section name
            
        Returns:
            dict: Section data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM sections WHERE name = ?",
                (name,)
            )
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
    
    def get_all_sections(self, section_type=None):
        """
        Retrieve all sections, optionally filtered by type
        
        Args:
            section_type (str): Optional filter by section type
            
        Returns:
            list: List of section dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if section_type:
                cursor.execute(
                    "SELECT * FROM sections WHERE type = ? ORDER BY created_at DESC",
                    (section_type,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM sections ORDER BY created_at DESC"
                )
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def delete_section(self, section_id):
        """
        Delete a section by ID
        
        Args:
            section_id (int): Section ID to delete
            
        Returns:
            bool: True if deleted, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute(
                "DELETE FROM sections WHERE id = ?",
                (section_id,)
            )
            
            conn.commit()
            return cursor.rowcount > 0
    
    def search_sections(self, query):
        """
        Search sections by name or type
        
        Args:
            query (str): Search query
            
        Returns:
            list: List of matching sections
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM sections 
                WHERE name LIKE ? OR type LIKE ?
                ORDER BY created_at DESC
            ''', (f'%{query}%', f'%{query}%'))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def export_all_to_dataframe(self):
        """
        Export all sections to a pandas DataFrame
        
        Returns:
            pd.DataFrame: DataFrame containing all sections
        """
        sections = self.get_all_sections()
        
        if not sections:
            return pd.DataFrame()
        
        # Expand properties into columns
        expanded_data = []
        for section in sections:
            row = {
                'id': section['id'],
                'name': section['name'],
                'type': section['type'],
                'created_at': section['created_at'],
                'updated_at': section['updated_at']
            }
            
            # Add parameters
            if section['parameters']:
                params = json.loads(section['parameters'])
                for key, value in params.items():
                    row[f'param_{key}'] = value
            
            # Add properties
            if section['properties']:
                props = json.loads(section['properties'])
                for key, value in props.items():
                    row[f'prop_{key}'] = value
            
            expanded_data.append(row)
        
        return pd.DataFrame(expanded_data)
    
    def import_from_csv(self, csv_path):
        """
        Import sections from a CSV file
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            int: Number of sections imported
        """
        df = pd.read_csv(csv_path)
        count = 0
        
        for _, row in df.iterrows():
            # Extract parameters and properties
            params = {}
            props = {}
            
            for col in df.columns:
                if col.startswith('param_'):
                    params[col.replace('param_', '')] = row[col]
                elif col.startswith('prop_'):
                    props[col.replace('prop_', '')] = row[col]
            
            section_data = {
                'name': row.get('name', f"Imported_{count}"),
                'type': row.get('type', 'Unknown'),
                'parameters': json.dumps(params),
                'properties': json.dumps(props)
            }
            
            self.save_section(section_data)
            count += 1
        
        return count
    
    def get_statistics(self):
        """
        Get database statistics
        
        Returns:
            dict: Statistics about stored sections
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total sections
            cursor.execute("SELECT COUNT(*) FROM sections")
            total = cursor.fetchone()[0]
            
            # Sections by type
            cursor.execute("""
                SELECT type, COUNT(*) as count 
                FROM sections 
                GROUP BY type
            """)
            by_type = dict(cursor.fetchall())
            
            # Recent sections
            cursor.execute("""
                SELECT COUNT(*) FROM sections 
                WHERE datetime(created_at) > datetime('now', '-7 days')
            """)
            recent = cursor.fetchone()[0]
            
            return {
                'total': total,
                'by_type': by_type,
                'recent_7_days': recent
            }
