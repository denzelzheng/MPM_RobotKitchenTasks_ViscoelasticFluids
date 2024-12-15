import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

class VisualFeatureExtractor:
    def __init__(self, use_dinov2=False, model_path=None):
        """
        Initialize the visual feature extractor
        
        Args:
            use_dinov2: Boolean to choose between DINOv2 and ResNet
            model_path: Path to DINOv2 model files (if use_dinov2 is True)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_dinov2 = use_dinov2
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        if use_dinov2 and model_path:
            try:
                from transformers import AutoConfig, AutoModel
                # Load DINOv2 model
                config = AutoConfig.from_pretrained(f"{model_path}/config.json")
                self.model = AutoModel.from_config(config)
                
                # Load weights
                checkpoint = torch.load(f"{model_path}/dinov2_vitg14_reg4_pretrain.pth", 
                                     map_location=self.device)
                self.model.load_state_dict(checkpoint, strict=False)
                
            except Exception as e:
                print(f"Failed to load DINOv2 model: {e}")
                print("Falling back to ResNet-50")
                self.use_dinov2 = False
                
        if not self.use_dinov2:
            # Load pre-trained ResNet50
            self.model = models.resnet50(pretrained=True)
            # Remove the final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def preprocess_image(self, image):
        """
        Preprocess the input image for the model
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be PIL Image or numpy array")
            
        img_tensor = self.transform(image)
        return img_tensor.unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def extract_features(self, image):
        """
        Extract visual features from the image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            numpy array: Visual feature vector B
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Extract features
        if self.use_dinov2:
            features = self.model(img_tensor).last_hidden_state.mean(dim=1)
        else:
            features = self.model(img_tensor)
            features = features.squeeze()
            
        # Convert to numpy array and flatten
        features = features.cpu().numpy().flatten()
        
        # Normalize features
        features = features / np.linalg.norm(features)
        
        return features
    
    def extract_batch_features(self, images):
        """
        Extract features from a batch of images
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            numpy array: Batch of visual feature vectors
        """
        batch_features = []
        for image in images:
            features = self.extract_features(image)
            batch_features.append(features)
            
        return np.array(batch_features)

# Example usage
if __name__ == "__main__":
    # Initialize extractor
    extractor = VisualFeatureExtractor(
        use_dinov2=False  # Set to True if DINOv2 model files are available
    )
    
    # Create dummy image (3 channels, RGB)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Extract features
    features = extractor.extract_features(dummy_image)
    
    print("Feature vector shape:", features.shape)
    print("Feature vector norm:", np.linalg.norm(features))
    
    # Test batch processing
    dummy_batch = [dummy_image] * 3
    batch_features = extractor.extract_batch_features(dummy_batch)
    
    print("\nBatch features shape:", batch_features.shape)