import torch
from torch.utils.data import DataLoader

def train_classifier(model, train_loader, test_loader, optimizer, criterion, epoch):
    device = next(model.parameters()).device
    criterion = criterion.to(device)
    
    train_loss = 0
    test_loss = 0
    
    model.train()
    for batch_idx, (inp, label) in enumerate(train_loader):
        optimizer.zero_grad()
        inp, label = inp.to(device), label.to(device)
        output = model(inp)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (inp, label) in enumerate(test_loader):
            inp, label = inp.to(device), label.to(device)
            output = model(inp)
            loss = criterion(output, label)
            test_loss += loss.item()
    return train_loss / len(train_loader), test_loss / len(test_loader)


def eval_accuracy(model, dataset, batch_size=100):
    device = next(model.parameters()).device
    model.eval()
    
    dataloader = DataLoader(dataset=testset, batch_size=batch_size)
    
    count = 0
    with torch.no_grad():
        for batch_idx, (inp, label) in enumerate(dataloader):
            inp, label = inp.to(device), label.to(device)
            output = model(inp)
            output = torch.argmax(output, dim=1)
            count += (output == label).sum()
    accuracy = count / len(dataset)
    return accuracy