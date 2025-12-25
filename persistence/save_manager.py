"""
The Village - Save Manager
Persistence system for saving and loading world state.
"""

import json
import os
from datetime import datetime
from typing import Optional
from config import CONFIG


class SaveManager:
    """Handles saving and loading world state."""
    
    def __init__(self, save_dir: str = None):
        self.save_dir = save_dir or CONFIG.save_directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.last_save_time = 0.0
        self.save_interval = CONFIG.auto_save_interval_seconds
    
    def save(self, world, filename: str = None) -> str:
        """
        Save world state to file.
        Returns the filename used.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"village_{timestamp}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Serialize world
        data = world.to_dict()
        
        # Add metadata
        data['_meta'] = {
            'saved_at': datetime.now().isoformat(),
            'version': '1.0.0',
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.last_save_time = 0.0  # Reset timer
        
        return filepath
    
    def load(self, filepath: str, config=None):
        """Load world state from file."""
        from simulation.world import World
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Remove metadata before loading
        data.pop('_meta', None)
        
        return World.from_dict(data, config=config)
    
    def load_latest(self, config=None):
        """Load the most recent save file."""
        saves = self.list_saves()
        
        if not saves:
            return None
        
        # Sort by modification time
        saves.sort(key=lambda x: x['modified'], reverse=True)
        
        return self.load(saves[0]['path'], config=config)
    
    def list_saves(self) -> list[dict]:
        """List all save files."""
        saves = []
        
        for filename in os.listdir(self.save_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.save_dir, filename)
                stat = os.stat(filepath)
                saves.append({
                    'filename': filename,
                    'path': filepath,
                    'modified': stat.st_mtime,
                    'size': stat.st_size
                })
        
        return saves
    
    def should_auto_save(self, elapsed_time: float) -> bool:
        """Check if it's time for an auto-save."""
        self.last_save_time += elapsed_time
        return self.last_save_time >= self.save_interval
    
    def auto_save(self, world) -> Optional[str]:
        """
        Perform auto-save if interval has passed.
        Returns filepath if saved, None otherwise.
        """
        if self.should_auto_save(0):
            return self.save(world, "autosave.json")
        return None
    
    def delete_save(self, filepath: str):
        """Delete a save file."""
        if os.path.exists(filepath):
            os.remove(filepath)
    
    def clean_old_saves(self, keep_count: int = 10):
        """Delete old saves, keeping only the most recent ones."""
        saves = self.list_saves()
        
        if len(saves) <= keep_count:
            return
        
        # Sort by modification time
        saves.sort(key=lambda x: x['modified'], reverse=True)
        
        # Delete old ones (except autosave)
        for save in saves[keep_count:]:
            if 'autosave' not in save['filename']:
                self.delete_save(save['path'])
