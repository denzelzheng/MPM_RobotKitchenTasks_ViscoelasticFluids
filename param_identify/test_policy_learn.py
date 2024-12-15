import numpy as np
import torch
import torch.nn as nn
from physical_feature_extractor import PhysicalFeatureExtractor
from visual_feature_extractor import VisualFeatureExtractor
from feature_fusion import TaskFusion
from action_executor import ActionExecutor
from typing import Optional, Tuple, Dict
from training_module import ModelSaver, TaskConfig
import logging
from datetime import datetime

class PolicyNet(nn.Module):
    """
    Policy network that outputs action values, now accepting target viscosity as input
    """
    def __init__(self, feature_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x, target_viscosity):
        # Convert inputs to tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        if isinstance(target_viscosity, (float, int)):
            target_viscosity = torch.FloatTensor([[target_viscosity]]).to(next(self.parameters()).device)
        
        # Concatenate features with target viscosity
        combined_input = torch.cat([x, target_viscosity], dim=1)
        out = self.net(combined_input)
        return out * 2
    
    def sample_action(self, x, target_viscosity):
        # Sample discrete actions from continuous action values
        action_values = self.forward(x, target_viscosity)
        return torch.round(action_values).int()

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def initialize_models(device):
    """Initialize all required models and components"""
    # Initialize task configuration
    task_config = TaskConfig(
        initial_mass=1.0,
        additive_unit_mass=0.1, 
        initial_volume=0.001,
        task_type=1,
        target_viscosity=1e6
    )
    
    # Initialize feature extractors and modules
    physical_extractor = PhysicalFeatureExtractor(
        initial_mass=task_config.initial_mass,
        additive_unit_mass=task_config.additive_unit_mass,
        initial_volume=task_config.initial_volume,
        task_type=task_config.task_type
    )
    visual_extractor = VisualFeatureExtractor(use_dinov2=False)
    fusion_module = TaskFusion(encoded_dim=128, device=device)
    action_executor = ActionExecutor()
    
    # Initialize policy networks
    teacher_policy = PolicyNet().to(device)
    student_policy = PolicyNet().to(device)
    
    # Load pretrained teacher policy if available
    try:
        teacher_policy.load_state_dict(torch.load('pretrained_policy.pt'))
        student_policy.load_state_dict(torch.load('pretrained_policy.pt'))
        logging.info("Loaded pretrained teacher policy model")
    except:
        logging.warning("No pretrained teacher policy found, using initialized weights")
    
    return (task_config, physical_extractor, visual_extractor, fusion_module,
            action_executor, teacher_policy, student_policy)

def train_step(batch_idx, dummy_images, models, device, steps_per_episode, target_viscosity):
    """Execute one training step"""
    # Unpack models
    (physical_extractor, visual_extractor, fusion_module,
     action_executor, teacher_policy, student_policy) = models
    
    current_policy = np.array([1, 1, 1])
    physical_features = []
    visual_features = []
    student_policies = []
    teacher_policies = []
    execution_history = []
    
    for step in range(steps_per_episode):
        # Feature extraction
        visual_feat = visual_extractor.extract_features(dummy_images[batch_idx])
        if step == 0:
            physical_feat = physical_extractor.extract_features(current_policy.tolist())
            
        # Convert features to tensors
        visual_tensor = torch.FloatTensor(visual_feat).unsqueeze(0).to(device)
        physical_tensor = torch.FloatTensor(physical_feat).unsqueeze(0).to(device)
        
        # Feature fusion and policy prediction
        fused_features = fusion_module.encode_and_fuse(visual_tensor, physical_tensor)
        if isinstance(fused_features, np.ndarray):
            fused_features = torch.FloatTensor(fused_features).to(device)
            
        # Get teacher and student policies with target viscosity
        with torch.no_grad():
            teacher_action_probs = teacher_policy(fused_features, target_viscosity)
        student_action_probs = student_policy(fused_features, target_viscosity)
        
        # Execute action and update state
        student_action = student_policy.sample_action(fused_features, target_viscosity)
        current_policy = student_action.squeeze().cpu().numpy()
        
        success = action_executor.execute_action(current_policy)
        execution_history.append({
            'step': step,
            'success': success,
            'policy': current_policy.tolist()
        })
        
        if success:
            next_physical_feat = physical_extractor.extract_features(current_policy.tolist())
        else:
            logging.warning(f"Action execution failed at batch {batch_idx}, step {step}")
            next_physical_feat = physical_feat
        
        current_viscosity = next_physical_feat[5]
        
        # Store features and policies for loss computation
        physical_features.append(physical_feat)
        visual_features.append(visual_feat)
        student_policies.append(student_action_probs)
        teacher_policies.append(teacher_action_probs)
    
    features = (physical_features, visual_features)
    policies = (student_policies, teacher_policies)
    
    return features, policies, execution_history, current_viscosity


def compute_losses(features, policies, current_viscosity, target_viscosity, fusion_module, device):
    """Compute all training losses"""
    # Prepare tensors
    physical_tensor = torch.FloatTensor(np.stack(features[0])).to(device)
    visual_tensor = torch.FloatTensor(np.stack(features[1])).to(device)
    student_policies = torch.stack(policies[0])
    teacher_policies = torch.stack(policies[1])
    
    # Compute viscosity loss
    final_viscosity = torch.tensor(current_viscosity).to(device)
    target_viscosity_tensor = torch.tensor(target_viscosity).to(device)
    viscosity_loss = nn.MSELoss()(final_viscosity, target_viscosity_tensor)
    
    # Compute contrastive loss
    contrast_loss = fusion_module.contrastive_loss(
        fusion_module.visual_encoder(visual_tensor),
        fusion_module.physical_encoder(physical_tensor)
    )
    
    # Compute distillation loss
    distillation_loss = nn.MSELoss()(student_policies, teacher_policies)
    
    # Combine losses
    total_loss = contrast_loss + 0.3 * distillation_loss + 0.5 * viscosity_loss
    
    return total_loss

def test_policy_learning():
    """Main training function"""
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize models and components
    (task_config, physical_extractor, visual_extractor, fusion_module,
     action_executor, teacher_policy, student_policy) = initialize_models(device)
    
    # Setup optimizer and model saver
    optimizer = torch.optim.Adam(
        list(fusion_module.parameters()) + list(student_policy.parameters()),
        lr=1e-4
    )
    model_saver = ModelSaver(save_dir='./checkpoints')
    
    # Training parameters
    num_episodes = 100
    steps_per_episode = 1
    batch_size = 1
    target_viscosity = task_config.target_viscosity
    
    # Additional info for logging
    additional_info = {
        'experiment_name': 'viscosity_control',
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'training_params': {
            'num_episodes': num_episodes,
            'steps_per_episode': steps_per_episode,
            'batch_size': batch_size
        }
    }
    
    # Training loop
    for episode in range(num_episodes):

        print("\nEpisode:", episode)

        dummy_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) 
                       for _ in range(batch_size)]
        
        episode_loss = 0
        all_execution_history = []
        
        for batch_idx in range(batch_size):
            models = (physical_extractor, visual_extractor, fusion_module,
                     action_executor, teacher_policy, student_policy)
            
            features, policies, execution_history, current_viscosity = train_step(
                batch_idx, dummy_images, models, device, steps_per_episode, target_viscosity
            )
            
            # Compute and backpropagate losses
            total_loss = compute_losses(
                features, policies, current_viscosity,
                target_viscosity, fusion_module, device
            )
            
            episode_loss += total_loss.item()
            all_execution_history.extend(execution_history)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        # Save model and policy
        best_loss, best_policy_seq = model_saver.save_model_and_policy(
            student_policy=student_policy,
            fusion_module=fusion_module,
            episode=episode,
            execution_history=all_execution_history,
            current_loss=episode_loss/batch_size,
            task_config=task_config,
            additional_info=additional_info
        )
        
        # Log training progress
        if (episode + 1) % 10 == 0:
            success_rate = sum(1 for h in all_execution_history if h['success']) / len(all_execution_history)
            logging.info(
                f"Episode {episode+1}/{num_episodes}, "
                f"Avg Loss: {episode_loss/batch_size:.4f}, "
                f"Current Viscosity: {current_viscosity:.4f}, "
                f"Target Viscosity: {target_viscosity:.4f}, "
                f"Action Success Rate: {success_rate:.2f}"
            )

if __name__ == "__main__":
    test_policy_learning()