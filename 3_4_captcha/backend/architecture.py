import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=256):
        super(CRNN, self).__init__()
        
        # Input: 1 x 80 x 400 (GrayScale)
        # CNN pour extraire les features visuelles
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 64 x 40 x 200
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 128 x 20 x 100
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)), # -> 256 x 10 x 100 (On garde la largeur pour la séquence temporelle)
            
            # Layer 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)), # -> 512 x 5 x 100
        )
        
        # RNN pour la séquence
        # Input features calculation: 512 channels * 5 height = 2560
        self.rnn = nn.LSTM(input_size=512 * 5, 
                           hidden_size=hidden_size, 
                           bidirectional=True, 
                           batch_first=True, 
                           num_layers=2)
        
        # Output layer
        # hidden_size * 2 because bidirectional
        self.output = nn.Linear(hidden_size * 2, num_chars + 1) # +1 for CTC Blank

    def forward(self, x):
        # x: [Batch, 1, 80, 400]
        features = self.cnn(x)
        
        # Prepare for RNN
        b, c, h, w = features.size()
        # On veut: [Batch, TimeSteps, Features]
        # Ici TimeSteps = w (100), Features = c * h (512 * 5 = 2560)
        features = features.permute(0, 3, 1, 2) # [b, w, c, h]
        features = features.reshape(b, w, c * h) # [b, w, c*h]
        
        rnn_out, _ = self.rnn(features)
        
        out = self.output(rnn_out)
        
        # Pour CTC Loss, on a souvent besoin de log_softmax
        # Shape: [Batch, TimeSteps, NumClasses]
        return out.log_softmax(2)
