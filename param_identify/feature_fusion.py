import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import copy

class BaseEncoder(nn.Module):
    """Base encoder"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def get_weights(self) -> Dict:
        """Get current weights"""
        return self.state_dict()

class VisualEncoder(BaseEncoder):
    """Visual feature encoder"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]  # Default architecture for visual feature dimension reduction
        super().__init__(input_dim, output_dim, hidden_dims)

class PhysicalEncoder(BaseEncoder):
    """Physical feature encoder"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]  # Default architecture for physical feature dimension increase
        super().__init__(input_dim, output_dim, hidden_dims)

    def load_from_pretrained(self, pretrained_encoder: nn.Module):
        """Load weights from pretrained encoder"""
        pretrained_state = pretrained_encoder.state_dict()
        current_state = self.state_dict()
        
        # Create weight mapping
        state_dict_map = {}
        for (name, param), (pretrained_name, pretrained_param) in zip(
            current_state.items(), pretrained_state.items()):
            if param.shape == pretrained_param.shape:
                state_dict_map[pretrained_name] = name
        
        # Load matched weights
        for pretrained_name, name in state_dict_map.items():
            current_state[name].copy_(pretrained_state[pretrained_name])
        
        self.load_state_dict(current_state)

class ContrastiveLoss(nn.Module):
    """Contrastive loss"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, visual_features: torch.Tensor, physical_features: torch.Tensor) -> torch.Tensor:
        # Normalize features
        visual_features = nn.functional.normalize(visual_features, dim=1)
        physical_features = nn.functional.normalize(physical_features, dim=1)
        
        # Calculate similarity matrix
        logits = torch.mm(visual_features, physical_features.t()) / self.temperature
        
        # Positive pairs should be on diagonal
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        # Calculate cross entropy loss
        loss_i = nn.functional.cross_entropy(logits, labels)
        loss_t = nn.functional.cross_entropy(logits.t(), labels)
        
        return (loss_i + loss_t) / 2

        

class TaskFusion(nn.Module):
    def __init__(self, 
                 encoded_dim: int = 128,
                 device: Optional[torch.device] = None,
                 learning_rate: float = 1e-4):
        """
        Initialize task fusion module
        
        Args:
            encoded_dim: Dimension of shared embedding space
            device: Computation device
            learning_rate: Learning rate
        """
        super().__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoded_dim = encoded_dim
        self.learning_rate = learning_rate
        
        # Feature dimensions
        self.visual_feature_dim = 2048
        self.physical_feature_dim = 6
        
        # Initialize encoders
        self.initialize_encoders()
        self.contrastive_loss = ContrastiveLoss().to(self.device)
        
    def initialize_encoders(self):
        """Initialize encoders"""
        self.visual_encoder = VisualEncoder(
            input_dim=self.visual_feature_dim,
            output_dim=self.encoded_dim
        ).to(self.device)
        
        self.physical_encoder = PhysicalEncoder(
            input_dim=self.physical_feature_dim,
            output_dim=self.encoded_dim
        ).to(self.device)
        
        self.setup_optimizers()
        
    def setup_optimizers(self):
        """Setup optimizers"""
        self.visual_optimizer = torch.optim.AdamW(
            self.visual_encoder.parameters(),
            lr=self.learning_rate
        )
        self.physical_optimizer = torch.optim.AdamW(
            self.physical_encoder.parameters(),
            lr=self.learning_rate
        )

    def parameters(self):
        """返回所有可训练参数"""
        return list(self.visual_encoder.parameters()) + list(self.physical_encoder.parameters())
        

    def load_pretrained_physical_encoder(self, pretrained_encoder: nn.Module):
        """Load pretrained physical feature encoder"""
        self.physical_encoder.load_from_pretrained(pretrained_encoder)
    
    def prepare_batch(self, 
                     visual_features: torch.Tensor,
                     physical_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare batch data"""
        # Ensure input is 2D tensor
        if visual_features.dim() == 1:
            visual_features = visual_features.unsqueeze(0)
        if physical_features.dim() == 1:
            physical_features = physical_features.unsqueeze(0)
            
        # Convert data type and device
        visual_features = visual_features.to(dtype=torch.float32, device=self.device)
        physical_features = physical_features.to(dtype=torch.float32, device=self.device)
        
        return visual_features, physical_features

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Feature normalization"""
        return nn.functional.normalize(features, p=2, dim=1)

    def train_encoders(self,
                      visual_features: np.ndarray,
                      physical_features: np.ndarray,
                      num_epochs: int = 100,
                      batch_size: int = 32,
                      verbose: bool = True) -> List[float]:
        """
        Train encoders
        
        Args:
            visual_features: Visual feature array (N, 2048)
            physical_features: Physical feature array (N, 6)
            num_epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print training progress
        """
        # Convert to tensor
        visual_tensor = torch.from_numpy(visual_features).float()
        physical_tensor = torch.from_numpy(physical_features).float()
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(visual_tensor, physical_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_visual, batch_physical in dataloader:
                batch_visual, batch_physical = self.prepare_batch(
                    batch_visual, batch_physical
                )
                
                # Forward pass
                encoded_visual = self.visual_encoder(batch_visual)
                encoded_physical = self.physical_encoder(batch_physical)
                
                # Calculate loss
                loss = self.contrastive_loss(encoded_visual, encoded_physical)
                
                # Backward pass
                self.visual_optimizer.zero_grad()
                self.physical_optimizer.zero_grad()
                loss.backward()
                self.visual_optimizer.step()
                self.physical_optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return losses

    @torch.no_grad()
    def encode_and_fuse(self,
                       visual_features: Union[np.ndarray, torch.Tensor],
                       physical_features: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Encode and fuse features
        
        Args:
            visual_features: Visual features
            physical_features: Physical features
            
        Returns:
            Fused feature array
        """
        self.visual_encoder.eval()
        self.physical_encoder.eval()
        
        # Convert to tensor
        if isinstance(visual_features, np.ndarray):
            visual_features = torch.from_numpy(visual_features)
        if isinstance(physical_features, np.ndarray):
            physical_features = torch.from_numpy(physical_features)
        
        # Prepare data
        visual_features, physical_features = self.prepare_batch(
            visual_features, physical_features
        )
        
        # Encode
        encoded_visual = self.visual_encoder(visual_features)
        encoded_physical = self.physical_encoder(physical_features)
        
        # Normalize
        encoded_visual = self.normalize_features(encoded_visual)
        encoded_physical = self.normalize_features(encoded_physical)
        
        # Fuse (weighted average)
        fused_features = 0.5 * encoded_visual + 0.5 * encoded_physical
        fused_features = self.normalize_features(fused_features)
        
        return fused_features.cpu().numpy()

    def save_model(self, save_path: str):
        """Save model"""
        torch.save({
            'visual_encoder': self.visual_encoder.state_dict(),
            'physical_encoder': self.physical_encoder.state_dict(),
            'encoded_dim': self.encoded_dim,
            'visual_feature_dim': self.visual_feature_dim,
            'physical_feature_dim': self.physical_feature_dim
        }, save_path)

    def load_model(self, load_path: str):
        """Load model"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.encoded_dim = checkpoint['encoded_dim']
        self.visual_feature_dim = checkpoint['visual_feature_dim']
        self.physical_feature_dim = checkpoint['physical_feature_dim']
        
        self.initialize_encoders()
        self.visual_encoder.load_state_dict(checkpoint['visual_encoder'])
        self.physical_encoder.load_state_dict(checkpoint['physical_encoder'])

