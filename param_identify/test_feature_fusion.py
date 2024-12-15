import numpy as np
import torch
from physical_feature_extractor import PhysicalFeatureExtractor
from visual_feature_extractor import VisualFeatureExtractor
from feature_fusion import TaskFusion

def test_fusion():
    # Initialize extractors and generate test data
    physical_extractor = PhysicalFeatureExtractor(
        initial_mass=1.0, additive_unit_mass=0.1,
        initial_volume=0.001, task_type=1
    )
    visual_extractor = VisualFeatureExtractor(use_dinov2=False)
    
    np.random.seed(42)
    initial_cloud = np.random.rand(1000, 3) * 0.01
    policies = [[0, 2, 2], [1, 1, 1], [2, 0, 2]]
    dummy_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(3)]

    # Extract features
    physical_features = []
    for policy in policies:
        features = physical_extractor.extract_features(policy)
        physical_features.append(features)
    physical_features = np.array(physical_features)
    visual_features = visual_extractor.extract_batch_features(dummy_images)

    # Initialize fusion module with pretrained physical encoder
    fusion_module = TaskFusion(encoded_dim=128)
    try:
        pretrained_physical_encoder = torch.load('pretrained_physical_encoder.pt')
        fusion_module.load_pretrained_physical_encoder(pretrained_physical_encoder)
    except:
        print("No pretrained physical encoder found, using initialized weights")

    # Train and fuse
    fusion_module.train_encoders(
        visual_features=visual_features,
        physical_features=physical_features,
        num_epochs=50,
        verbose=False
    )
    
    fused_features = fusion_module.encode_and_fuse(visual_features, physical_features)
    print(f"\nFused feature shape: {fused_features.shape}")

if __name__ == "__main__":
    test_fusion()