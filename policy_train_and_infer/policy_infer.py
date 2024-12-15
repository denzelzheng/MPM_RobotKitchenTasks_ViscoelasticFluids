import numpy as np
import torch
import torch.nn as nn
from physical_feature_extractor import PhysicalFeatureExtractor
from visual_feature_extractor import VisualFeatureExtractor
from feature_fusion import TaskFusion
from action_executor import ActionExecutor
from training_module import ModelSaver, TaskConfig
import logging
from datetime import datetime

class PolicyNet(nn.Module):
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
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        if isinstance(target_viscosity, (float, int)):
            target_viscosity = torch.FloatTensor([[target_viscosity]]).to(next(self.parameters()).device)
        
        combined_input = torch.cat([x, target_viscosity], dim=1)
        out = self.net(combined_input)
        return out * 2
    
    def sample_action(self, x, target_viscosity):
        action_values = self.forward(x, target_viscosity)
        return torch.round(action_values).int()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def load_models(device):
    # Initialize task configuration
    task_config = TaskConfig(
        initial_mass=1.0,
        additive_unit_mass=0.1,
        initial_volume=0.001,
        task_type=1,
        target_viscosity=1e6
    )
    
    # Initialize components
    physical_extractor = PhysicalFeatureExtractor(
        initial_mass=task_config.initial_mass,
        additive_unit_mass=task_config.additive_unit_mass,
        initial_volume=task_config.initial_volume,
        task_type=task_config.task_type
    )
    visual_extractor = VisualFeatureExtractor(use_dinov2=False)
    fusion_module = TaskFusion(encoded_dim=128, device=device)
    action_executor = ActionExecutor()
    policy_net = PolicyNet().to(device)
    
    # Load trained models
    model_saver = ModelSaver(save_dir='./checkpoints')
    try:
        checkpoint = model_saver.load_best_model(task_config)
        policy_net.load_state_dict(checkpoint['model_state_dict']['student_policy'])
        fusion_module.load_state_dict(checkpoint['model_state_dict']['fusion_module'])
        logging.info(f"Successfully loaded model from episode {checkpoint['episode']} with loss {checkpoint['loss']}")
    except FileNotFoundError:
        logging.error("No model found")
        raise
        
    return (task_config, physical_extractor, visual_extractor, 
            fusion_module, action_executor, policy_net)

def infer_policy_sequence(num_steps, save_path=None):
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    (task_config, physical_extractor, visual_extractor,
     fusion_module, action_executor, policy_net) = load_models(device)
    
    # Generate dummy image for testing
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    current_policy = np.array([1, 1, 1])
    policy_sequence = []
    execution_history = []
    
    for step in range(num_steps):
        # Extract features
        visual_feat = visual_extractor.extract_features(dummy_image)
        if step == 0:
            physical_feat = physical_extractor.extract_features(current_policy.tolist())
            
        # Prepare tensors
        visual_tensor = torch.FloatTensor(visual_feat).unsqueeze(0).to(device)
        physical_tensor = torch.FloatTensor(physical_feat).unsqueeze(0).to(device)
        
        # Feature fusion
        fused_features = fusion_module.encode_and_fuse(visual_tensor, physical_tensor)
        if isinstance(fused_features, np.ndarray):
            fused_features = torch.FloatTensor(fused_features).to(device)
        
        # Get policy prediction
        with torch.no_grad():
            action = policy_net.sample_action(fused_features, task_config.target_viscosity)
        current_policy = action.squeeze().cpu().numpy()
        
        # Execute action
        success = action_executor.execute_action(current_policy)
        execution_info = {
            'step': step,
            'policy': current_policy.tolist(),
            'success': success
        }
        execution_history.append(execution_info)
        policy_sequence.append(current_policy.tolist())
        
        if success:
            physical_feat = physical_extractor.extract_features(current_policy.tolist())
            current_viscosity = physical_feat[5]
            logging.info(f"Step {step}: Current viscosity = {current_viscosity:.4f}")
        else:
            logging.warning(f"Action execution failed at step {step}")
    
    # Save results
    if save_path:
        result_data = {
            'policy_sequence': policy_sequence,
            'execution_history': execution_history,
            'task_config': task_config.to_dict(),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        np.save(save_path, result_data)
        logging.info(f"Saved inference results to {save_path}")
    
    return policy_sequence, execution_history

if __name__ == "__main__":
    num_steps = 5
    save_path = "inference_results.npy"
    
    try:
        policy_sequence, execution_history = infer_policy_sequence(num_steps, save_path)
        print("\nInferred Policy Sequence:")
        for step, policy in enumerate(policy_sequence):
            print(f"Step {step}: {policy}")
    except Exception as e:
        logging.error(f"Inference failed: {str(e)}")