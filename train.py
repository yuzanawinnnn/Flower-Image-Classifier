#importing modules
import argparse
from utils import Util

#Argument Parser
argpa = argparse.ArgumentParser(description='Train.py')
argpa.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
argpa.add_argument('data_dir', action="store", default="flowers/")
argpa.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
argpa.add_argument('--epochs', dest="epochs", action="store", type=int, default=10)
argpa.add_argument('--gpu', dest="gpu", action='store', default= 'gpu')
argpa.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=100)
argpa.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
argpa.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")


#Parse Argument
parsearg = argpa.parse_args()
arch = parsearg.arch
data_dir = parsearg.data_dir
dropout = parsearg.dropout
epochs = parsearg.epochs
hardware = "gpu" if parsearg.gpu else "cpu"
hidden_units = parsearg.hidden_units
learning_rate = parsearg.learning_rate
save_dir = parsearg.save_dir

#Model Training
train_loader, validation_loader, test_loader, train_dataset = Util.load_data(data_dir)

model, criterion, optimizer = Util.model_setup(arch, dropout, hidden_units, learning_rate, hardware)

Util.train_network(train_loader, validation_loader, model, criterion, optimizer, epochs, 10, hardware)

Util.test_accuracy(model, test_loader, hardware)

Util.save_checkpoint(model, train_dataset.class_to_idx, save_dir, arch, hidden_units, dropout, learning_rate)

print("Done Training the Model!")