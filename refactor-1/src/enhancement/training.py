from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class ArtifactDataset(Dataset):
    def __init__(self, 
                 clean_paths: List[str],
                 separated_paths: List[str],
                 preprocessor: HighFrequencyArtifactPreprocessor):
        self.clean_paths = clean_paths
        self.separated_paths = separated_paths
        self.preprocessor = preprocessor
        
    def __len__(self):
        return len(self.clean_paths)
        
    def __getitem__(self, idx):
        clean = load_audio(self.clean_paths[idx])
        separated = load_audio(self.separated_paths[idx])
        
        condition, input_spec, target_spec = generate_training_pair(
            clean, separated, self.preprocessor
        )
        
        return {
            'condition': condition,
            'input': input_spec,
            'target': target_spec
        }

class ControlNetTrainer:
    def __init__(self,
                 model: PhaseAwareControlNet,
                 preprocessor: HighFrequencyArtifactPreprocessor,
                 learning_rate: float = 1e-4,
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.preprocessor = preprocessor.to(device)
        self.device = device
        
        # Freeze base model, train only ControlNet components
        for param in self.model.base_model.parameters():
            param.requires_grad = False
            
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad()
        
        # Move data to device
        condition = batch['condition'].to(self.device)
        input_spec = batch['input'].to(self.device)
        target_spec = batch['target'].to(self.device)
        
        # Forward pass
        output = self.model(input_spec, control=condition)
        
        # Compute losses
        losses = {
            'magnitude': F.mse_loss(output[:, :-1], target_spec[:, :-1]),
            'phase': F.mse_loss(output[:, -1:], target_spec[:, -1:]),
            'frequency': self.frequency_loss(output, target_spec)
        }
        
        # Combined loss
        total_loss = sum(losses.values())
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            max_norm=1.0
        )
        
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def frequency_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Frequency-domain loss focusing on high frequencies"""
        # Convert to frequency domain
        pred_fft = torch.fft.rfft2(pred[:, :-1])  # Exclude phase channel
        target_fft = torch.fft.rfft2(target[:, :-1])
        
        # Weight higher frequencies more
        freq_weights = torch.linspace(1.0, 2.0, pred_fft.size(-1))
        freq_weights = freq_weights.to(self.device)
        
        return F.mse_loss(
            pred_fft * freq_weights,
            target_fft * freq_weights
        )
    
    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             epochs: int = 100,
             save_dir: str = 'checkpoints'):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            
            for batch in train_loader:
                losses = self.train_step(batch)
                train_losses.append(losses)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(f'{save_dir}/best.pt')
            
            if epoch % 10 == 0:
                self.save_checkpoint(f'{save_dir}/epoch_{epoch}.pt')
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                condition = batch['condition'].to(self.device)
                input_spec = batch['input'].to(self.device)
                target_spec = batch['target'].to(self.device)
                
                output = self.model(input_spec, control=condition)
                loss = F.mse_loss(output, target_spec)
                val_losses.append(loss.item())
        
        return sum(val_losses) / len(val_losses)
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def prepare_training(
    clean_dir: str,
    separated_dir: str,
    batch_size: int = 8,
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader]:
    # Get audio paths
    clean_paths = sorted(glob.glob(f'{clean_dir}/*.wav'))
    separated_paths = sorted(glob.glob(f'{separated_dir}/*.wav'))
    
    # Split train/val
    split_idx = int(len(clean_paths) * (1 - val_split))
    
    # Create datasets
    preprocessor = HighFrequencyArtifactPreprocessor()
    train_dataset = ArtifactDataset(
        clean_paths[:split_idx],
        separated_paths[:split_idx],
        preprocessor
    )
    val_dataset = ArtifactDataset(
        clean_paths[split_idx:],
        separated_paths[split_idx:],
        preprocessor
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader