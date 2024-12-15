import numpy as np
import torch
import torch.nn as nn
from physical_feature_extractor import PhysicalFeatureExtractor
from visual_feature_extractor import VisualFeatureExtractor
from feature_fusion import TaskFusion
from action_executor import ActionExecutor
from typing import Optional, Tuple, Dict

class PolicyNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(next(self.parameters()).device)
        out = self.net(x)
        return out * 2
    
    def sample_action(self, x):
        action_values = self.forward(x)
        return torch.round(action_values).int()

def test_policy_learning():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    physical_extractor = PhysicalFeatureExtractor(
        initial_mass=1.0, additive_unit_mass=0.1,
        initial_volume=0.001, task_type=1
    )
    visual_extractor = VisualFeatureExtractor(use_dinov2=False)
    fusion_module = TaskFusion(encoded_dim=128, device=device)
    action_executor = ActionExecutor()
    
    teacher_policy = PolicyNet().to(device)
    student_policy = PolicyNet().to(device)
    try:
        teacher_policy.load_state_dict(torch.load('pretrained_policy.pt'))
        student_policy.load_state_dict(torch.load('pretrained_policy.pt'))
        print("Loaded pretrained teacher policy model")
    except:
        print("No pretrained teacher policy found, using initialized weights")

    optimizer = torch.optim.Adam(list(fusion_module.parameters()) + 
                               list(student_policy.parameters()), lr=1e-4)

    num_episodes = 100
    steps_per_episode = 3
    batch_size = 8
    target_viscosity = 1.0

    for episode in range(num_episodes):
        dummy_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) 
                       for _ in range(batch_size)]
        
        episode_loss = 0
        for batch_idx in range(batch_size):
            current_policy = np.array([1, 1, 1])
            
            physical_features = []
            visual_features = []
            student_policies = []
            teacher_policies = []
            execution_history = []

            for step in range(steps_per_episode):
                visual_feat = visual_extractor.extract_features(dummy_images[batch_idx])

                if step == 0:
                    physical_feat = physical_extractor.extract_features(
                        current_policy.tolist()
                    )
                
                visual_tensor = torch.FloatTensor(visual_feat).unsqueeze(0).to(device)
                physical_tensor = torch.FloatTensor(physical_feat).unsqueeze(0).to(device)

                fused_features = fusion_module.encode_and_fuse(visual_tensor, physical_tensor)
                if isinstance(fused_features, np.ndarray):
                    fused_features = torch.FloatTensor(fused_features).to(device)

                with torch.no_grad():
                    teacher_action_probs = teacher_policy(fused_features)
                student_action_probs = student_policy(fused_features)
                
                student_action = student_policy.sample_action(fused_features)
                current_policy = student_action.squeeze().cpu().numpy()
                
                success = action_executor.execute_action(current_policy)
                execution_history.append({
                    'step': step,
                    'success': success,
                    'policy': current_policy.tolist()
                })
                print(execution_history)
                if success:
                    next_physical_feat = physical_extractor.extract_features(
                        current_policy.tolist()
                    )
                else:
                    print(f"Action execution failed at episode {episode}, batch {batch_idx}, step {step}")
                    next_physical_feat = physical_feat
                
                current_viscosity = next_physical_feat[5]
                
                physical_features.append(physical_feat)
                visual_features.append(visual_feat)
                student_policies.append(student_action_probs)
                teacher_policies.append(teacher_action_probs)

            physical_tensor = torch.FloatTensor(np.stack(physical_features)).to(device)
            visual_tensor = torch.FloatTensor(np.stack(visual_features)).to(device)
            student_policies = torch.stack(student_policies)
            teacher_policies = torch.stack(teacher_policies)

            final_viscosity = torch.tensor(current_viscosity).to(device)
            target_viscosity_tensor = torch.tensor(target_viscosity).to(device)
            viscosity_loss = nn.MSELoss()(final_viscosity, target_viscosity_tensor)

            contrast_loss = fusion_module.contrastive_loss(
                fusion_module.visual_encoder(visual_tensor),
                fusion_module.physical_encoder(physical_tensor)
            )
            
            distillation_loss = nn.MSELoss()(student_policies, teacher_policies)

            total_loss = contrast_loss + 0.3 * distillation_loss + 0.5 * viscosity_loss
            episode_loss += total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (episode + 1) % 10 == 0 and batch_idx == 0:
                success_rate = sum(1 for h in execution_history if h['success']) / len(execution_history)
                print(f"Episode {episode+1}/{num_episodes}, "
                      f"Avg Loss: {episode_loss/batch_size:.4f}, "
                      f"Current Viscosity: {current_viscosity:.4f}, "
                      f"Target Viscosity: {target_viscosity:.4f}, "
                      f"Action Success Rate: {success_rate:.2f}")

if __name__ == "__main__":
    test_policy_learning()