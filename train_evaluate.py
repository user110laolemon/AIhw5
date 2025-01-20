import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataprocess import *
from model import MultiModalModel
from config import *
config = Config()


def train_and_test(train_dataloader, dev_dataloader):
    '''训练模型并在验证集上评估性能
        初始化模型、优化器和损失函数
        训练模型并在验证集上评估性能
        使用早停法保存最佳模型并绘制损失曲线图
    '''
    model = MultiModalModel(config).to(device=config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_func = nn.CrossEntropyLoss()

    best_dev_loss = float('inf')
    best_model_state = None
    patience = config.patience
    num_bad_epochs = 0
    best_epoch = 0

    train_losses = []
    dev_losses = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        for step, batch in enumerate(train_dataloader):
            ids, text, image, labels = batch
            text = text.to(device=config.device)
            image = image.to(device=config.device)
            labels = labels.to(device=config.device)
            optimizer.zero_grad()
            output = model(text=text, image=image)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += torch.sum(torch.argmax(output, dim=1) == labels).item()
            total_samples += labels.size(0)

        train_loss /= total_samples
        train_accuracy = train_correct / total_samples
        train_losses.append(train_loss)

        print(f"Epoch {epoch}/{config.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        dev_loss, dev_accuracy = evaluate(model, dev_dataloader, epoch)
        dev_losses.append(dev_loss)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_model_state = model.state_dict()
            best_epoch = epoch
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1

        if num_bad_epochs >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    if best_model_state is not None:
        torch.save(best_model_state, config.model_path + f'/model_{config.lr}_{config.dropout}.pt')
        print(f"Best model saved with dev loss: {best_dev_loss:.4f}")
    plot_loss_curve(train_losses, dev_losses, best_epoch)


def evaluate(model, dev_dataloader, epoch=None):
    '''评估模型在验证集上的性能
        计算模型在验证集上的损失和准确率
        返回验证集上的损失和准确率
    '''
    model.eval()
    dev_loss = 0.0
    dev_correct = 0
    total_samples = 0
    loss_func = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dev_dataloader:
            ids, text, image, labels = batch
            text = text.to(device=config.device)
            image = image.to(device=config.device)
            labels = labels.to(device=config.device)

            outputs = model(text=text, image=image)
            loss = loss_func(outputs, labels)

            dev_loss += loss.item() * labels.size(0)
            dev_correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            total_samples += labels.size(0)

    dev_loss /= total_samples
    dev_accuracy = dev_correct / total_samples

    if epoch:
        print(f"  Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f} (Epoch {epoch})")
    else:
        print(f"  Dev Loss: {dev_loss:.4f}, Dev Accuracy: {dev_accuracy:.4f}")

    return dev_loss, dev_accuracy

def get_test(model, test_dataloader, scenario=""):
    '''测试模型在不同场景下的准确率
        计算模型在测试集上的准确率
        保存测试结果打印测试准确率
    '''
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            ids, text, image, _ = batch
            text = text.to(device=config.device)
            image = image.to(device=config.device)

            if scenario == "text_only":
                outputs = model(text=text, image=None)
            elif scenario == "image_only":
                outputs = model(text=None, image=image)
            elif scenario == "text_and_image":
                outputs = model(text=text, image=image)
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            predicted_labels = torch.argmax(outputs, dim=1)

            for i in range(len(ids)):
                item_id = ids[i]
                tag = test_dataloader.dataset.label_dict_str[int(predicted_labels[i])]
                prediction = {
                    'guid': item_id,
                    'tag': tag,
                }
                predictions.append(prediction)

    save_data(config.test_output_file + f'/test_result_{config.lr}_{config.dropout}_{scenario}.txt', predictions)

    accuracy = calculate_accuracy(predictions, test_dataloader.dataset.data)
    print(f"  {scenario.capitalize()} Test Accuracy: {accuracy:.4f}")

def calculate_accuracy(predictions, data):
    '''计算预测的准确率'''
    correct_predictions = 0
    total_samples = len(predictions)

    for pred in predictions:
        try:
            guid_int = int(pred['guid'])
            if guid_int < len(data) and pred['tag'] == data[guid_int]['label']:
                correct_predictions += 1
        except (ValueError, IndexError):
            continue

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    return accuracy


def plot_loss_curve(train_losses, dev_losses, best_epoch, save_path=config.plt_path):
    '''绘制训练和评估损失曲线图
        在最佳模型处绘制红色虚线
        保存损失曲线图
        显示损失曲线图
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(dev_losses, label='Dev Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.savefig(save_path + f'/loss_curve_{config.lr}_{config.dropout}.png')
    plt.show()


if __name__ == '__main__':
    print(f'Device: {config.device}')

    train_set, dev_set = load_data(config)
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=config.batch_size)
    dev_dataloader = DataLoader(dev_set, shuffle=False, batch_size=config.batch_size)

    if config.do_train:
        print('训练中...')
        train_and_test(train_dataloader, dev_dataloader)
        print('训练完成.')
    elif config.do_test:
        print('预测中...')
        test_set, _ = load_data(config)
        test_dataloader = DataLoader(test_set, shuffle=False, batch_size=config.batch_size)
        model = MultiModalModel(config).to(device=config.device)
        model.load_state_dict(torch.load(config.model_path + f'/model_{config.lr}_{config.dropout}.pt'))
        model.to(config.device)
        
        print("\nTesting the model for different scenarios:")
        print("\nScenario: Text Input Only")
        get_test(model, test_dataloader, scenario="text_only")
        print("\nScenario: Image Input Only")
        get_test(model, test_dataloader, scenario="image_only")
        print("\nScenario: Text and Image Input")
        get_test(model, test_dataloader, scenario="text_and_image")
        print('预测完成.')