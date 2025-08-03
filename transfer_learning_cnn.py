# Parte 1: Transfer Learning con CNNs
# Autor: Pontificia Universidad Católica de Chile - Deep Learning
# ---------------------------------------------

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import os

# --------- Configuración ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001

# --------- Transformaciones ---------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --------- Parte 1.1: Entrenamiento en CIFAR10 ---------
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# Usamos una CNN pequeña (ResNet18)
model_cifar = models.resnet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_cifar.parameters(), lr=LR)

# Entrenamiento
for epoch in range(EPOCHS):
    model_cifar.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_cifar(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {running_loss / len(trainloader):.4f}")

torch.save(model_cifar.state_dict(), "cnn_cifar10.pth")

# --------- Parte 1.2: Evaluación en Cats vs Dogs ---------
# Debe descargarse el dataset de Cats vs Dogs de Kaggle y colocarse en ./cats_vs_dogs
catsdogs_path = './cats_vs_dogs'  # Estructura esperada: cats_vs_dogs/train/cat.jpg, cats_vs_dogs/train/dog.jpg

if os.path.exists(catsdogs_path):
    dataset_cd = ImageFolder(root=catsdogs_path, transform=transform_test)
    cd_loader = DataLoader(dataset_cd, batch_size=BATCH_SIZE, shuffle=False)
    
    model_cifar.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in cd_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_cifar(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy modelo entrenado en CIFAR10 sobre Cats vs Dogs: {100 * correct / total:.2f}%")
else:
    print("Cats vs Dogs dataset no encontrado en './cats_vs_dogs'.")

# --------- Parte 1.3: Comparación con ResNet18 preentrenado ---------
model_pretrained = models.resnet18(pretrained=True)
model_pretrained.fc = nn.Linear(model_pretrained.fc.in_features, 2)
model_pretrained = model_pretrained.to(device)

# Solo evaluación (no fine-tuning en esta versión)
if os.path.exists(catsdogs_path):
    model_pretrained.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in cd_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_pretrained(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Accuracy modelo preentrenado en ImageNet sobre Cats vs Dogs: {100 * correct / total:.2f}%")