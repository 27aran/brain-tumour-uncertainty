import torch
from src.model import model
from src.data_loader import (
    train_data_loader,
    test_data_loader,
    val_data_loader,
    cal_data_loader
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_metric = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_data_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_metric(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_data_loader)

    #Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_data_loader:

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = loss_metric(outputs, labels)
            value_max, index_max = torch.max(outputs, 1)
            val_loss += loss.item()

            correct += (index_max == labels).sum().item()
            total = total + labels.size(0)

        avg_val_loss = val_loss / len(val_data_loader)
        avg_val_accuracy = correct / total

    print(f"Epochs: {epoch+1}/{num_epochs}", f"Train Loss: {avg_train_loss: .4f}", f"Val Loss: {avg_val_loss: .4f}", f"Val Acc: {avg_val_accuracy: .4f}")