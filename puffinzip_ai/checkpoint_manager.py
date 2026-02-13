# PuffinZipAI - Checkpoint Management System
"""
Manages saving, loading, and comparing evolution checkpoints with metadata.
Enables efficient experiment tracking and checkpoint comparison for Evolution/Learning System.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path


class CheckpointMetadata:
    """Metadata for a single checkpoint."""
    
    def __init__(self, name: str, generation: int = 0, best_fitness: float = 0.0,
                 avg_fitness: float = 0.0, dataset_size: int = 0, dataset_name: str = "",
                 population_size: int = 0, timestamp: Optional[str] = None):
        self.name = name
        self.generation = generation
        self.best_fitness = best_fitness
        self.avg_fitness = avg_fitness
        self.dataset_size = dataset_size
        self.dataset_name = dataset_name
        self.population_size = population_size
        self.timestamp = timestamp or datetime.now().isoformat()
        self.compression_score = 0.0  # Calculated later
        self.baseline_comparison = {}  # Comparison vs standard methods
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
            'dataset_size': self.dataset_size,
            'dataset_name': self.dataset_name,
            'population_size': self.population_size,
            'timestamp': self.timestamp,
            'compression_score': self.compression_score,
            'baseline_comparison': self.baseline_comparison,
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'CheckpointMetadata':
        """Create from dictionary."""
        meta = CheckpointMetadata(
            name=data.get('name', ''),
            generation=data.get('generation', 0),
            best_fitness=data.get('best_fitness', 0.0),
            avg_fitness=data.get('avg_fitness', 0.0),
            dataset_size=data.get('dataset_size', 0),
            dataset_name=data.get('dataset_name', ''),
            population_size=data.get('population_size', 0),
            timestamp=data.get('timestamp', None),
        )
        meta.compression_score = data.get('compression_score', 0.0)
        meta.baseline_comparison = data.get('baseline_comparison', {})
        return meta


class CompressionScoreCalculator:
    """
    Calculates a generalized compression score based on multiple metrics.
    Normalizes different metrics into a 0-100 score.
    """
    
    @staticmethod
    def calculate_score(best_fitness: float, avg_fitness: float, 
                       fitness_improvement: float = 0.0,
                       convergence_rate: float = 0.0,
                       baseline_comparison: Dict = None) -> float:
        """
        Calculate an overall compression score (0-100).
        
        Args:
            best_fitness: Best fitness achieved in population
            avg_fitness: Average fitness of population
            fitness_improvement: Improvement from previous generation (0-1)
            convergence_rate: How well population is converging (0-1)
            baseline_comparison: Dict with comparison to baseline methods
        
        Returns:
            Score from 0 to 100
        """
        score_components = []
        weights = []
        
        # Component 1: Best fitness (0-100) - 40% weight
        if best_fitness is not None:
            # Assume best_fitness is 0-1 range (compression ratio)
            normalizedBestFit = min(100, max(0, best_fitness * 100))
            score_components.append(normalizedBestFit)
            weights.append(0.40)
        
        # Component 2: Population avg fitness - 25% weight
        if avg_fitness is not None:
            normalizedAvgFit = min(100, max(0, avg_fitness * 100))
            score_components.append(normalizedAvgFit)
            weights.append(0.25)
        
        # Component 3: Fitness improvement - 20% weight
        if fitness_improvement > 0:
            normalizedImprove = min(100, fitness_improvement * 100)
            score_components.append(normalizedImprove)
            weights.append(0.20)
        
        # Component 4: Convergence - 15% weight
        if convergence_rate > 0:
            normalizedConverge = min(100, convergence_rate * 100)
            score_components.append(normalizedConverge)
            weights.append(0.15)
        
        # Weighted average
        if not score_components:
            return 0.0
        
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        
        weighted_score = sum(c * w for c, w in zip(score_components, weights)) / total_weight
        return round(min(100.0, max(0.0, weighted_score)), 2)
    
    @staticmethod
    def compare_scores(checkpoint1_score: float, checkpoint2_score: float) -> Dict:
        """
        Compare two checkpoint scores.
        
        Returns:
            Dict with comparison metrics
        """
        improvement = checkpoint2_score - checkpoint1_score
        improvement_pct = (improvement / max(0.1, checkpoint1_score)) * 100 if checkpoint1_score > 0 else 0
        
        return {
            'score1': checkpoint1_score,
            'score2': checkpoint2_score,
            'difference': round(improvement, 2),
            'improvement_percent': round(improvement_pct, 2),
            'better': 'checkpoint2' if improvement > 0 else 'checkpoint1' if improvement < 0 else 'equal',
        }


class CheckpointManager:
    """Manages saving, loading, and comparing evolution checkpoints."""
    
    DEFAULT_CHECKPOINT_DIR = "checkpoints"
    
    def __init__(self, checkpoint_dir: str = None, logger: logging.Logger = None):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints (default: ./checkpoints)
            logger: Logger instance for messaging
        """
        self.checkpoint_dir = checkpoint_dir or self.DEFAULT_CHECKPOINT_DIR
        self.logger = logger or self._setup_default_logger()
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.checkpoints_metadata: Dict[str, CheckpointMetadata] = {}
        self._load_checkpoint_index()
    
    @staticmethod
    def _setup_default_logger():
        """Create a basic logger if none provided."""
        logger = logging.getLogger('CheckpointManager')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_checkpoint_index(self):
        """Load checkpoint metadata index from disk."""
        index_file = os.path.join(self.checkpoint_dir, "checkpoint_index.json")
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    for name, meta_dict in data.items():
                        self.checkpoints_metadata[name] = CheckpointMetadata.from_dict(meta_dict)
                self.logger.info(f"Loaded {len(self.checkpoints_metadata)} checkpoints from index.")
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint index: {e}")
    
    def _save_checkpoint_index(self):
        """Save checkpoint metadata index to disk."""
        index_file = os.path.join(self.checkpoint_dir, "checkpoint_index.json")
        try:
            data = {name: meta.to_dict() for name, meta in self.checkpoints_metadata.items()}
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint index: {e}")
    
    def save_checkpoint(self, checkpoint_name: str, optimizer_state: Dict,
                       best_fitness: float = 0.0, avg_fitness: float = 0.0,
                       generation: int = 0, dataset_size: int = 0,
                       dataset_name: str = "", population_size: int = 0) -> bool:
        """
        Save a checkpoint with metadata.
        
        Args:
            checkpoint_name: Name for this checkpoint
            optimizer_state: State dict from evolutionary optimizer
            best_fitness: Best fitness in current population
            avg_fitness: Average fitness in current population
            generation: Current generation number
            dataset_size: Size of training dataset in bytes
            dataset_name: Name/path of training dataset
            population_size: Population size during training
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create checkpoint filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_filename = f"checkpoint_{checkpoint_name}_{timestamp}.pkl"
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            
            # Save the optimizer state
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(optimizer_state, f)
            
            # Create and store metadata
            metadata = CheckpointMetadata(
                name=checkpoint_name,
                generation=generation,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
                dataset_size=dataset_size,
                dataset_name=dataset_name,
                population_size=population_size,
            )
            
            # Calculate compression score
            metadata.compression_score = CompressionScoreCalculator.calculate_score(
                best_fitness=best_fitness,
                avg_fitness=avg_fitness,
            )
            
            # Store metadata with timestamped key
            meta_key = f"{checkpoint_name}_{timestamp}"
            self.checkpoints_metadata[meta_key] = metadata
            self._save_checkpoint_index()
            
            self.logger.info(
                f"Checkpoint '{checkpoint_name}' saved successfully. "
                f"Gen: {generation}, Fitness: {best_fitness:.4f}, Score: {metadata.compression_score:.2f}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint '{checkpoint_name}': {e}", exc_info=True)
            return False
    
    def load_checkpoint(self, checkpoint_key: str) -> Tuple[bool, Optional[Dict]]:
        """
        Load a checkpoint by key.
        
        Args:
            checkpoint_key: Key from checkpoints_metadata (name_timestamp format)
        
        Returns:
            Tuple of (success: bool, optimizer_state: Optional[Dict])
        """
        try:
            if checkpoint_key not in self.checkpoints_metadata:
                self.logger.error(f"Checkpoint '{checkpoint_key}' not found in metadata.")
                return False, None
            
            # Find the checkpoint file
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir)
                              if f.startswith(f"checkpoint_{checkpoint_key}") and f.endswith('.pkl')]
            
            if not checkpoint_files:
                self.logger.error(f"No checkpoint file found for key '{checkpoint_key}'.")
                return False, None
            
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_files[0])
            
            with open(checkpoint_path, 'rb') as f:
                optimizer_state = pickle.load(f)
            
            self.logger.info(f"Checkpoint '{checkpoint_key}' loaded successfully.")
            return True, optimizer_state
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint '{checkpoint_key}': {e}", exc_info=True)
            return False, None
    
    def list_checkpoints(self) -> List[Dict]:
        """
        List all available checkpoints with their metadata.
        
        Returns:
            List of checkpoint info dicts, sorted by timestamp (newest first)
        """
        checkpoints_list = []
        for key, metadata in self.checkpoints_metadata.items():
            checkpoints_list.append({
                'key': key,
                'name': metadata.name,
                'generation': metadata.generation,
                'best_fitness': metadata.best_fitness,
                'avg_fitness': metadata.avg_fitness,
                'score': metadata.compression_score,
                'dataset_size': metadata.dataset_size,
                'dataset_name': metadata.dataset_name,
                'timestamp': metadata.timestamp,
            })
        
        # Sort by timestamp (newest first)
        checkpoints_list.sort(key=lambda x: x['timestamp'], reverse=True)
        return checkpoints_list
    
    def compare_checkpoints(self, checkpoint_key1: str, checkpoint_key2: str) -> Optional[Dict]:
        """
        Compare two checkpoints and return differences.
        
        Args:
            checkpoint_key1: First checkpoint key
            checkpoint_key2: Second checkpoint key
        
        Returns:
            Dict with comparison data or None if comparison failed
        """
        if checkpoint_key1 not in self.checkpoints_metadata or checkpoint_key2 not in self.checkpoints_metadata:
            self.logger.error("One or both checkpoints not found.")
            return None
        
        meta1 = self.checkpoints_metadata[checkpoint_key1]
        meta2 = self.checkpoints_metadata[checkpoint_key2]
        
        # Calculate differences
        fitness_diff = meta2.best_fitness - meta1.best_fitness
        avg_fitness_diff = meta2.avg_fitness - meta1.avg_fitness
        gen_diff = meta2.generation - meta1.generation
        
        # Compare scores
        score_comparison = CompressionScoreCalculator.compare_scores(
            meta1.compression_score,
            meta2.compression_score
        )
        
        return {
            'checkpoint1': {
                'key': checkpoint_key1,
                'name': meta1.name,
                'generation': meta1.generation,
                'best_fitness': meta1.best_fitness,
                'avg_fitness': meta1.avg_fitness,
                'score': meta1.compression_score,
                'dataset_size': meta1.dataset_size,
                'timestamp': meta1.timestamp,
            },
            'checkpoint2': {
                'key': checkpoint_key2,
                'name': meta2.name,
                'generation': meta2.generation,
                'best_fitness': meta2.best_fitness,
                'avg_fitness': meta2.avg_fitness,
                'score': meta2.compression_score,
                'dataset_size': meta2.dataset_size,
                'timestamp': meta2.timestamp,
            },
            'differences': {
                'best_fitness_diff': round(fitness_diff, 4),
                'avg_fitness_diff': round(avg_fitness_diff, 4),
                'generation_diff': gen_diff,
                'score_comparison': score_comparison,
            },
            'better_checkpoint': 'checkpoint2' if fitness_diff > 0 else 'checkpoint1' if fitness_diff < 0 else 'equal'
        }
    
    def delete_checkpoint(self, checkpoint_key: str) -> bool:
        """Delete a checkpoint."""
        try:
            if checkpoint_key not in self.checkpoints_metadata:
                self.logger.error(f"Checkpoint '{checkpoint_key}' not found.")
                return False
            
            # Find and delete the checkpoint file
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir)
                              if f.startswith(f"checkpoint_{checkpoint_key}") and f.endswith('.pkl')]
            
            for f in checkpoint_files:
                os.remove(os.path.join(self.checkpoint_dir, f))
            
            del self.checkpoints_metadata[checkpoint_key]
            self._save_checkpoint_index()
            
            self.logger.info(f"Checkpoint '{checkpoint_key}' deleted successfully.")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint '{checkpoint_key}': {e}", exc_info=True)
            return False
    
    def get_checkpoint_metadata(self, checkpoint_key: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint."""
        return self.checkpoints_metadata.get(checkpoint_key)
