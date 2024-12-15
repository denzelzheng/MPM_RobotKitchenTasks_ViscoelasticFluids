import os
import torch
import numpy as np
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class TaskConfig:
    """Task-related configuration information"""
    initial_mass: float
    additive_unit_mass: float
    initial_volume: float
    task_type: int
    target_viscosity: float
    
    def to_dict(self):
        return {
            'initial_mass': self.initial_mass,
            'additive_unit_mass': self.additive_unit_mass,
            'initial_volume': self.initial_volume,
            'task_type': self.task_type,
            'target_viscosity': self.target_viscosity
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            initial_mass=config_dict['initial_mass'],
            additive_unit_mass=config_dict['additive_unit_mass'],
            initial_volume=config_dict['initial_volume'],
            task_type=config_dict['task_type'],
            target_viscosity=config_dict['target_viscosity']
        )

class ModelSaver:
    def __init__(self, save_dir: str = './checkpoints'):
        self.save_dir = save_dir
        self.best_loss = float('inf')
        self.best_policy_seq = None
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def select_best_checkpoint(self, task_config: TaskConfig) -> str:
        """
        Select the best checkpoint based on loss values
        
        Args:
            task_config: Task configuration
            
        Returns:
            Path to the best checkpoint
        """
        best_model_path = os.path.join(self.save_dir, f'task{task_config.task_type}_best_distilled_model.pt')
        
        # If best model exists, return its path
        if os.path.exists(best_model_path):
            return best_model_path
            
        # Get all checkpoint files for this task
        checkpoint_files = []
        for f in os.listdir(self.save_dir):
            if f.startswith(f'task{task_config.task_type}_checkpoint_episode_') and f.endswith('.pt'):
                checkpoint_files.append(os.path.join(self.save_dir, f))
                
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found for task {task_config.task_type}")
            
        # Find the best checkpoint based on loss
        best_loss = float('inf')
        best_checkpoint = None
        
        for checkpoint_path in checkpoint_files:
            checkpoint = torch.load(checkpoint_path)
            if checkpoint['loss'] < best_loss:
                best_loss = checkpoint['loss']
                best_checkpoint = checkpoint_path
                
        return best_checkpoint

    def save_model_and_policy(
        self,
        student_policy: nn.Module,
        fusion_module: nn.Module,
        episode: int,
        execution_history: list,
        current_loss: float,
        task_config: TaskConfig,
        additional_info: Dict = None
    ) -> Tuple[float, list]:
        """
        Save model and optimal policy sequence
        """
        # Save checkpoint periodically
        if (episode + 1) % 10 == 0:
            checkpoint = {
                'episode': episode,
                'student_policy_state_dict': student_policy.state_dict(),
                'fusion_module_state_dict': fusion_module.state_dict(),
                'loss': current_loss,
                'task_config': task_config.to_dict(),
                'additional_info': additional_info
            }
            torch.save(
                checkpoint,
                os.path.join(self.save_dir, f'task{task_config.task_type}_checkpoint_episode_{episode+1}.pt')
            )
        
        # Update best results if current loss is better
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            successful_sequences = [
                {'policy': h['policy'], 'step': h['step']}
                for h in execution_history if h['success']
            ]
            
            if successful_sequences:
                self.best_policy_seq = successful_sequences
                best_model = {
                    'episode': episode,
                    'student_policy_state_dict': student_policy.state_dict(),
                    'fusion_module_state_dict': fusion_module.state_dict(),
                    'best_loss': self.best_loss,
                    'best_policy_seq': self.best_policy_seq,
                    'task_config': task_config.to_dict(),
                    'additional_info': additional_info
                }
                torch.save(
                    best_model,
                    os.path.join(self.save_dir, f'task{task_config.task_type}_best_distilled_model.pt')
                )
                
                detailed_info = {
                    'task_config': task_config.to_dict(),
                    'policy_sequences': self.best_policy_seq,
                    'additional_info': additional_info
                }
                np.save(
                    os.path.join(self.save_dir, f'task{task_config.task_type}_best_policy_detailed.npy'),
                    detailed_info
                )
        
        return self.best_loss, self.best_policy_seq
    
    def load_best_model(self, task_config: TaskConfig) -> Dict:
        """
        Load the best model and policy sequence
        
        Args:
            task_config: Task configuration
            
        Returns:
            Dict containing model state dict, task config, policy sequences and additional info
            
        Raises:
            FileNotFoundError: If no saved models exist
        """
        try:
            best_model_path = self.select_best_checkpoint(task_config)
            checkpoint = torch.load(best_model_path)
            
            if 'best_policy_seq' in checkpoint:  # If loading from best_distilled_model
                return {
                    'model_state_dict': {
                        'student_policy': checkpoint['student_policy_state_dict'],
                        'fusion_module': checkpoint['fusion_module_state_dict']
                    },
                    'task_config': TaskConfig.from_dict(checkpoint['task_config']),
                    'policy_sequences': checkpoint['best_policy_seq'],
                    'loss': checkpoint['best_loss'],
                    'episode': checkpoint['episode'],
                    'additional_info': checkpoint.get('additional_info', None)
                }
            else:  # If loading from checkpoint
                return {
                    'model_state_dict': {
                        'student_policy': checkpoint['student_policy_state_dict'],
                        'fusion_module': checkpoint['fusion_module_state_dict']
                    },
                    'task_config': TaskConfig.from_dict(checkpoint['task_config']),
                    'loss': checkpoint['loss'],
                    'episode': checkpoint['episode'],
                    'additional_info': checkpoint.get('additional_info', None)
                }
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"No saved models found for task {task_config.task_type}")