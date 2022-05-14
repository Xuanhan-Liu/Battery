import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from utils.callbacks import LossHistory
from models.resnet import resnet50
from models.normal_cnn import CNN
from utils.utils_fit import fit_one_epoch
from data.CNNdata import get_train_val_test_loader, CNNpolygrain
from data_loader import get_dataLoader, get_cnn_dataLoader

model = resnet50()
# model = CNN()
model_train = model.train()
if torch.cuda.is_available():
    Cuda = True
else:
    Cuda = False

if Cuda:
    model_train = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model_train = model_train.cuda()

lr = 1e-5
end_epoch = 150
optimizer = optim.Adam(model_train.parameters(), lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
batch_size = 8

# train_loader, test_loader, mean, std = get_dataLoader(batch_size=batch_size, file_number=50, norm=False)
train_loader, test_loader, _ = get_cnn_dataLoader(batch_size=batch_size)
mean = None
std = None
epoch_step = len(train_loader)
epoch_step_val = len(test_loader)
loss_history = LossHistory("logs/", lr, batch_size, epoch_step, epoch_step_val, end_epoch)
for epoch in range(0, end_epoch):
    fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, train_loader,
                  test_loader,
                  end_epoch, Cuda, mean, std, False)
    lr_scheduler.step()
