from tqdm import tqdm
import torch
from torch import nn


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, data, data_test,
                  Epoch, cuda):
    total_loss = 0
    val_loss = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(data):
            if iteration >= epoch_step:
                break
            # imgs, pngs, labels = batch
            x, y = batch
            x = x.permute(0, 4, 1, 2, 3)
            with torch.no_grad():
                x = x.type(torch.FloatTensor)
                y = y.type(torch.FloatTensor)
                if cuda:
                    x = x.cuda()
                    y = y.cuda()

            optimizer.zero_grad()
            outputs = model_train(x)
            loss = nn.MSELoss()(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                # 'f_score'   : total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(data_test):
            if iteration >= epoch_step_val:
                break
            x, y = batch
            x = x.permute(0, 4, 1, 2, 3)
            with torch.no_grad():
                x = x.type(torch.FloatTensor)
                y = y.type(torch.FloatTensor)
                if cuda:
                    x = x.cuda()
                    y = y.cuda()
                outputs = model_train(x)
                loss = nn.HuberLoss()(outputs, y)
                val_loss += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                # 'f_score'   : val_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    loss_history.append_loss(total_loss / epoch_step, val_loss / epoch_step_val)
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    torch.save(model.state_dict(), loss_history.save_path + '/ep%03d-loss%.3f-val_loss%.3f.pth' % (
        (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
