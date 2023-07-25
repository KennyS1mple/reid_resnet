# CACIOUS CODING
# Data     : 7/24/23  8:00 PM
# File name: weight_util
# Desc     :
import torch


def save_model(path, epoch, model, optimizer=None):
    state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)


def load_model(model, model_path, optimizer=None, resume=False):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('Resumed optimizer with start lr',
                  optimizer.param_groups[0]['lr'])
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model
