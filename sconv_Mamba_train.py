import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
from data import DealDataset
from sconv_cnn import ScConv
from sconv_Mamba import ScConv_mamba

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 128
    lr = 0.001  # Try a smaller learning rate if needed
    num_epochs = 40
    label_num = 12

    folder_path_list = [
        "/mnt/PyCharm_Project_1/senet_cnn1/try/dataset/12class/FlowAllLayers",
        "/mnt/PyCharm_Project_1/senet_cnn1/try/dataset/12class/FlowL7",
        "/mnt/PyCharm_Project_1/senet_cnn1/try/dataset/12class/SessionAllLayers",
        "/mnt/PyCharm_Project_1/senet_cnn1/try/dataset/12class/SessionL7",
    ]

    task_index = 2
    folder_path = folder_path_list[task_index]
    train_data_path = "train-images-idx3-ubyte.gz"
    train_label_path = "train-labels-idx1-ubyte.gz"
    test_data_path = "t10k-images-idx3-ubyte.gz"
    test_label_path = "t10k-labels-idx1-ubyte.gz"

    trainDataset = DealDataset(folder_path, train_data_path, train_label_path)
    testDataset = DealDataset(folder_path, test_data_path, test_label_path)

    train_loader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=False)

    model = ScConv_mamba(label_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调度器

    loss_values = []
    accuracy_values = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device).to(torch.float32)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            if torch.isnan(loss).any():  # Check for NaN loss
                print("Loss is NaN!")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            loss_values.append(loss.item())
            epoch_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.6f}')

        scheduler.step()  # 更新学习率

        # 测试集评估
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            predictions = []
            true_labels = []

            for images, labels in test_loader:
                images = images.to(device).to(torch.float32)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.extend(predicted.tolist())
                true_labels.extend(labels.tolist())

            accuracy = 100 * correct / total
            accuracy_values.append(accuracy)
            print(f'Test Accuracy of the model on the test set: {accuracy:.2f} %')

            # 计算精度和召回率
            precision = precision_score(true_labels, predictions, average=None, zero_division=0)
            recall = recall_score(true_labels, predictions, average=None, zero_division=0)

            print('Precision:', precision)
            print('Recall:', recall)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss_values)
    plt.title("Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_values)
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Save model checkpoint
    torch.save(model.state_dict(), 'model_scconv.ckpt')

if __name__ == '__main__':
    main()