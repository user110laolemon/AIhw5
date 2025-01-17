import torch
from torch import nn
from torch.utils.data import DataLoader
from config import *
from dataprocess import *
from model import MultiModalModel


def train(args, train_dataloader, dev_dataloader):
    model = MultiModalModel(args).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        for step, batch in enumerate(train_dataloader):
            ids, text, image, labels = batch
            text = text.to(device=args.device)
            image = image.to(device=args.device)
            labels = labels.to(device=args.device)
            optimizer.zero_grad()
            # Forward pass
            text_out, img_out, multi_out = model(text=text, image=image)
            # Choose the appropriate output based on the available data
            output = text_out if text is not None else img_out if image is not None else multi_out
            # Calculate loss
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += torch.sum(torch.argmax(output, dim=1) == labels).item()
            total_samples += labels.size(0)

        train_loss /= total_samples
        train_accuracy = train_correct / total_samples

        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        evaluate(args, model, dev_dataloader, epoch)



def evaluate(args, model, dev_dataloader, epoch=None):
    model.eval()
    dev_loss = 0.0
    dev_correct = 0
    total_samples = 0
    loss_func = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dev_dataloader:
            ids, text, image, labels = batch
            text = text.to(device=args.device)
            image = image.to(device=args.device)
            labels = labels.to(device=args.device)

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



def train_and_test(args, train_dataloader, dev_dataloader, test_dataloader):
    model = MultiModalModel(args).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0
        for step, batch in enumerate(train_dataloader):
          ids, text, image, labels = batch
          text = text.to(device=args.device)
          image = image.to(device=args.device)
          labels = labels.to(device=args.device)
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

        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        evaluate(args, model, dev_dataloader, epoch)


    print("\nTesting the model for different scenarios:")
    print("\nScenario: Text Input Only")
    get_test(args, model, test_dataloader, scenario="text_only")
    print("\nScenario: Image Input Only")
    get_test(args, model, test_dataloader, scenario="image_only")
    print("\nScenario: Text and Image Input")
    get_test(args, model, test_dataloader, scenario="text_and_image")


def get_test(args, model, test_dataloader, scenario=""):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            ids, text, image, _ = batch
            text = text.to(device=args.device)
            image = image.to(device=args.device)

            outputs = model(text=text, image=image)
            predicted_labels = torch.argmax(outputs, dim=1)

            for i in range(len(ids)):
                item_id = ids[i]
                tag = test_dataloader.dataset.label_dict_str[int(predicted_labels[i])]
                prediction = {
                    'guid': item_id,
                    'tag': tag,
                }
                predictions.append(prediction)

    save_data(args.test_output_file, predictions)

    accuracy = calculate_accuracy(predictions, test_dataloader.dataset.data)
    print(f"  {scenario.capitalize()} Test Accuracy: {accuracy:.4f}")

def calculate_accuracy(predictions, data):
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


if __name__ == '__main__':
    config = Config()
    print(f'Device: {config.device}')

    train_set, dev_set = load_data(config)
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=config.batch_size)
    dev_dataloader = DataLoader(dev_set, shuffle=False, batch_size=config.batch_size)

    if config.do_train:
        print('训练中...')
        test_set, _ = load_data(config)
        test_dataloader = DataLoader(test_set, shuffle=False, batch_size=config.batch_size)
        train_and_test(config, train_dataloader, dev_dataloader, test_dataloader)