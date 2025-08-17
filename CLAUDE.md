# Fashion-MNIST Sparse Autoencoder Interpretability Project

## Project Overview

Build a k-sparse autoencoder on Fashion-MNIST to learn interpretable features, then analyze what the model learns through visualization and robustness testing.

**Key Goal**: Understand which internal features the model uses for classification through mechanistic interpretability.

## Architecture

- **Encoder**: 784 → 1024 neurons (ReLU activation)
- **Sparsity**: Only k=50 units active per sample (~5% sparsity)
- **Decoder**: 1024 → 784 (Linear reconstruction)
- **Loss**: MSE reconstruction loss

## Implementation Steps

### 1. Setup
```python
# Install dependencies
!pip install captum

# Load Fashion-MNIST
from torchvision.datasets import FashionMNIST
# Normalize to [0,1], create train/val/test loaders
```

### 2. Model Definition
- Create k-sparse autoencoder with top-k selection
- Implement sparsity mask that keeps only top-k activations

### 3. Training
- **Optimizer**: Adam (lr=1e-3)
- **Batch size**: 256
- **Epochs**: ~30
- Track train/validation loss and active units

### 4. Save Models
```python
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(decoder.state_dict(), "decoder.pth")
```

### 5. Linear Probe Classification
- Freeze encoder weights
- Extract sparse codes for all images
- Train linear classifier (1024 → 10 classes)
- Measure test accuracy

### 6. Feature Importance Analysis
For each class c and unit i:
```
importance[i,c] = |W[c,i]| × mean(z_i for images in class c)
```
Rank top contributing units per class.

### 7. Visualizations

**Decoder Atoms**:
- Reshape decoder weights to 28×28 images
- Show top contributing units as image patterns

**Feature Overlays**:
- Select top-K units for a specific image
- Multiply decoder atoms by contributions
- Overlay heatmap on original image

### 8. Robustness Testing

**Unit Ablation**:
- Zero out top contributing units
- Measure accuracy/logit drops

**Occlusion Mapping**:
- Slide patch across image
- Track prediction changes

## Expected Outputs

### Performance Metrics
- SAE reconstruction loss curves
- Linear probe test accuracy


### Interpretability Results
- Top latent units per Fashion-MNIST class
- Grid visualization of decoder atoms
- Feature overlay heatmaps on sample images
- Ablation study results

### Saved Artifacts
- `encoder.pth` and `decoder.pth` model files and classifier as `classifier.pth`
- Complete Jupyter notebook with embedded visualizations
- don't create virtual environment instead make a requirements.txt file only

## Tech Stack

- **Core**: Python 3, PyTorch, torchvision
- **Visualization**: matplotlib, seaborn
- **Utils**: numpy, tqdm
- **Interpretability**: Captum (optional)

## Best Practices

1. **Reproducibility**: Set random seeds for all libraries
2. **Data Safety**: Use only official Fashion-MNIST dataset
3. **Model Persistence**: Save trained models for future analysis  
4. **Validation**: Use early stopping to prevent overfitting
5. **Evidence-Based**: Support visual interpretations with ablation studies

## Fashion-MNIST Classes
0. T-shirt/top, 1. Trouser, 2. Pullover, 3. Dress, 4. Coat, 5. Sandal, 6. Shirt, 7. Sneaker, 8. Bag, 9. Ankle boot

## Success Criteria

- Reconstruction quality with sparse representation
- Meaningful classification performance on sparse codes
- Interpretable decoder atoms that correspond to clothing features
- Robust feature importance rankings validated by ablation