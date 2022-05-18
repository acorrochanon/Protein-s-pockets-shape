import sys
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
abs_path = '/'.join([p for p in current_folder.split('/')[:-1]])+'/'
sys.path.insert(0, abs_path + 'funcs')

# from importlib import reload
# reload(data_funcs)

from datetime import datetime
from sklearn.metrics import roc_auc_score
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from data_funcs import *
from materials import Network, Convolution, EarlyStopping


def train_one_epoch(train_loader, net, criterion, optimizer, device):
    running_tloss = 0.0
    for idx, batch in enumerate(train_loader):
        inputs, labels = batch
        labels = labels.to(device)
        
        # Zero gradients for every batch
        optimizer.zero_grad()

        # forward + backward + optimize.
        pred = net(inputs, False)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        # Accumulate batch losses
        running_tloss += loss.item()

    # Loss per batch
    avg_loss = running_tloss / (idx + 1) 

    return avg_loss


def validate_one_epoch(val_loader, net, criterion, device):
    running_vloss = 0.0
    for idx, val_batch in enumerate(val_loader):
        val_inputs, val_labels = val_batch
        val_labels = val_labels.to(device)

        val_preds = net(val_inputs)
        val_loss = criterion(val_preds, val_labels)
        
        running_vloss += val_loss.item()
    
    # Loss per batch
    avg_loss = running_vloss / (idx + 1)

    return avg_loss


def train(net, train_loader, val_loader, device):
    print(f"Training on {device}.")
    timestamp = datetime.now().strftime('%Y_%m_%d_%H%M%S')
    writer = SummaryWriter(f'runs/{timestamp}')
    
    #Set the number of epochs + get number of batches
    NUM_EPOCHS = 50
    
    # Set the optimizer, criterion, and early stopping criteria. 
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    early_stopping = EarlyStopping(net)
    
    for epoch in range(NUM_EPOCHS):
        net.train()
        train_avg_loss = train_one_epoch(train_loader, net, criterion, optimizer, device)
        net.train(False)
        val_avg_loss = validate_one_epoch(val_loader, net, criterion, device)
        early_stopping(train_avg_loss, val_avg_loss, net)
        
        if early_stopping.early_stop:
            print(f'Training finished at epoch {epoch+1}')
            break
        
        # Tensorboard stats 
        writer.add_scalars('Train vs. Val Loss',
                            {'Training':train_avg_loss, 'Validation': val_avg_loss}, epoch + 1)
        writer.flush()

        # Print stats every N epochs
        if (epoch+1)%1 == 0:
            print(f'Epoch: {epoch+1} | Training Loss: {train_avg_loss:.3f} | Validation Loss: {val_avg_loss:.3f}')
    
    return net


def test(net, test_loader, device):
    predictions, true_labels = [], []
    criterion = torch.nn.BCELoss()
    net.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            inputs, y_true = batch
            y_true = y_true.to(device)

            y_pred = net(inputs)
            loss = criterion(y_pred, y_true)

            # Save predictions and true labels for later comparison
            predictions.append(y_pred.cpu().numpy())
            true_labels.append(y_true.cpu().numpy())

            accuracy = y_pred.round().eq(y_true).all(dim=1).double().mean(dim=0).item()*100
            auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
            print(f'Batch {idx}. Accuracy: {accuracy:.3f}% | AUC:{auc} | Loss: {loss:.2f}')

    return predictions, true_labels


if __name__ == '__main__':
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
    # Load training data
    cavs_train = torch.load(abs_path + 'data/Cavities/train_cavities.pt')
    labels_train = torch.FloatTensor(torch.load(abs_path + 'data/Labels/train_labels.pt')).unsqueeze(1)
    atypes_train = torch.load(abs_path + 'data/Features/train_atypes.pt')
    train_fp_features = torch.load(abs_path + 'data/Features/train_features.pt')

    # Load test data 
    cavs_test = torch.load(abs_path + 'data/Cavities/test_cavities.pt')
    labels_test = torch.FloatTensor(torch.load(abs_path + 'data/Labels/test_labels.pt')).unsqueeze(1)
    atypes_test = torch.load(abs_path + 'data/Features/test_atypes.pt')
    test_fp_features = torch.load(abs_path + 'data/Features/test_features.pt')

    # Temporary dataset is generated to later split in training and validation splits 
    #temp_dataset = buildDataset(cavs_train[:100], labels_train[:100], atypes_train[:100], train_fp_features[:100])
    temp_dataset = buildDataset(cavs_train[:100], labels_train[:100])
    test_dataset = buildDataset(cavs_test, labels_test, atypes_test, test_fp_features)
    
    train_size = int(len(temp_dataset)*0.9)
    train_dataset, val_dataset = random_split(temp_dataset, [train_size, len(temp_dataset)-train_size])

    # Initialize the model
    model = Network()
    model.to(device)

    # Generate train and validation dataloaders
    train_loader = makeLoader(train_dataset, 256)
    val_loader = makeLoader(val_dataset, 128)

    # Train the model
    model = train(model, train_loader, val_loader, device)

    #Save the model
    torch.save(model.state_dict(), abs_path + 'data/Models/1a_model.pt')

    # Test the model 
    test_loader = makeLoader(test_dataset, 128)
    predictions, true_labels = test(model, test_loader, device)

    # Save predictions and true labels
    torch.save(predictions, 'predictions.pt')
    torch.save(true_labels, 'true_labels.pt')

