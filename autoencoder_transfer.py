# Parte 2: Transferencia con Autoencoder Convolucional
# ---------------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# Configuración general
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Dataset CIFAR10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# Definición del Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 16 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64 x 8 x 8
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)

# Entrenamiento del autoencoder
for epoch in range(EPOCHS):
    autoencoder.train()
    loss_total = 0
    for imgs, _ in trainloader:
        imgs = imgs.to(device)
        outputs = autoencoder(imgs)
        loss = criterion(outputs, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    print(f"[Epoch {epoch+1}] Reconstruction Loss: {loss_total / len(trainloader):.4f}")

torch.save(autoencoder.encoder.state_dict(), "encoder_cifar10.pth")

# Parte 2.2: Transferencia a clasificación (Cats vs Dogs)
if os.path.exists('./cats_vs_dogs'):
    cd_dataset = ImageFolder(root='./cats_vs_dogs', transform=transform)
    cd_loader = DataLoader(cd_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Clasificador simple sobre el encoder congelado
    class TransferModel(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )

        def forward(self, x):
            with torch.no_grad():
                x = self.encoder(x)
            x = self.classifier(x)
            return x

    encoder = ConvAutoencoder().encoder
    encoder.load_state_dict(torch.load("encoder_cifar10.pth"))
    encoder = encoder.to(device)
    model_transfer = TransferModel(encoder).to(device)

    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cls = torch.optim.Adam(model_transfer.classifier.parameters(), lr=LR)

    # Entrenamiento del clasificador
    for epoch in range(5):
        model_transfer.train()
        loss_total = 0
        for imgs, labels in cd_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model_transfer(imgs)
            loss = criterion_cls(outputs, labels)
            optimizer_cls.zero_grad()
            loss.backward()
            optimizer_cls.step()
            loss_total += loss.item()
        print(f"[Transfer Epoch {epoch+1}] Loss: {loss_total / len(cd_loader):.4f}")

    # Evaluación
    model_transfer.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in cd_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model_transfer(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy con encoder del autoencoder: {100 * correct / total:.2f}%")
else:
    print("Dataset Cats vs Dogs no encontrado en './cats_vs_dogs'.")