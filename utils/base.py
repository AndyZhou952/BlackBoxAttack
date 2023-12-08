import torch
from torch.utils.data import DataLoader, Subset
import time
import copy
import torch.nn.functional as F

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

def train_classifier_imagenet(model, dataloaders, criterion, optimizer, num_epochs=50, use_auxiliary=True):
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train']: #, 'val']: # Each epoch has a training and validation phase
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]: # Iterate over data
                
                inputs = inputs.to(device)

                labels = labels.to(device)

                optimizer.zero_grad() # Zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'): # Forward. Track history if only in train
                    
                    if phase == 'train': # Backward + optimize only if in training phase
                        if use_auxiliary:
                            outputs, aux1, aux2 = model(inputs)
                            loss = criterion(outputs, labels) + 0.3 * criterion(aux1, labels) + 0.3 * criterion(aux2, labels)
                        else:
                            outputs, _, _ = model(inputs)
                            loss = criterion(outputs, labels)
                            
                        _, preds = torch.max(outputs, 1)
                        loss.backward()
                        optimizer.step()
                    
                    if phase == 'val':
                        outputs, _, _ = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            if phase == 'val': # Adjust learning rate based on val loss
                lr_scheduler.step(epoch_loss)
                
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def eval_accuracy(model, dataset, batch_size=100):
    device = next(model.parameters()).device
    model.eval()
    
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    
    count = 0
    with torch.no_grad():
        for batch_idx, (inp, label) in enumerate(dataloader):
            inp, label = inp.to(device), label.to(device)
            output = model(inp)[0] if isinstance(model(inp), tuple) else model(inp)
            output = torch.argmax(output, dim=1)
            count += (output == label).sum()
    accuracy = count / len(dataset)
    return accuracy

def get_correct_predictions_subset(model, dataset, batch_size=100):
    device = next(model.parameters()).device
    model.eval()

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    correct_indices = []
    count = 0
    with torch.no_grad():
        for batch_idx, (inp, label) in enumerate(dataloader):
            inp, label = inp.to(device), label.to(device)
            output = model(inp)
            pred = torch.argmax(output, dim=1)
            correct = (pred == label)
            count += correct.sum()

            # Add indices of correct predictions to the list
            if correct.any():
                indices = batch_idx * batch_size + torch.where(correct)[0]
                correct_indices.extend(indices.cpu().numpy())

    accuracy = count / len(dataset)
    correct_subset = Subset(dataset, correct_indices)

    return accuracy, correct_subset



def quick_predict(model, img):
    model.eval()
    device = next(model.parameters()).device
    return torch.argmax(F.softmax(model(img.to(device)), dim=1), dim=1)