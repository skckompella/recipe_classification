from torch.autograd import Variable
import cPickle as pickle
import torch.optim as optim
from torch.utils.data import DataLoader

from nn_model import *
import utils


def main():
    #Load informatiom from preprocessing step
    with open(constants.WORD_TO_IDX_FILE, "rb") as fp:
        word_to_idx = pickle.load(fp)
    with open(constants.IDX_TO_LABEL_FILE, "rb") as fp:
        idx_to_labels = pickle.load(fp)

    #Create Dataset object
    data = RecipeDataset(constants.X_TRAIN_FILE, constants.Y_TRAIN_FILE,
                         constants.X_DEV_FILE, constants.Y_DEV_FILE)

    #Instantiate model
    model = RecipeNet2(len(word_to_idx), len(idx_to_labels), data.get_max_len())
    if constants.ONGPU:
        model.cuda()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=constants.LEARNING_RATE)

    #DataLoader helps load data batches easily
    dloader = DataLoader(data, batch_size=constants.BATCH_SIZE, shuffle=True, num_workers=1)

    #Load dev set
    x_dev, y_dev = data.get_dev_set()
    if constants.ONGPU:
        x_dev, y_dev = Variable(x_dev.cuda(), requires_grad=False), Variable(y_dev.cuda(), requires_grad=False)
    else:
        x_dev, y_dev = Variable(x_dev, requires_grad=False), Variable(y_dev, requires_grad=False)

    for iter in range(constants.NUM_EPOCHS):
        running_loss = 0.0
        running_acc = 0.0
        for i_batch, (x, y) in enumerate(dloader):
            if constants.ONGPU:
                x, y = Variable(x.cuda(), requires_grad=False), Variable(y.cuda(), requires_grad=False)
            else:
                x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)
            optimizer.zero_grad()
            output = model.forward(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            running_acc += utils.get_accuracy(output.max(1)[1], y)
        print("Train: Epoch: %d Loss: %.3f Accuracy: %.3f" % (
                iter + 1, running_loss / len(dloader), running_acc / len(dloader)))
        print("----------------------------------------------")

        if(iter % 2 == 0):
            out = model.forward(x_dev)
            _, preds = out.max(1)
            print ("Validation Accuracy: %f" % (utils.get_accuracy(preds, y_dev)))
            print("----------------------------------------------")


if __name__ == "__main__":
    main()