import os, torch

def save_ckpt(path, epoch, model, optimizer, scaler=None, best_metric=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_metric': best_metric,
    }
    if scaler is not None:
        payload['scaler'] = scaler.state_dict()
    torch.save(payload, path)

def load_ckpt(path, model, optimizer=None, scaler=None, map_location='cpu'):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state['model'])
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    if scaler is not None and 'scaler' in state:
        scaler.load_state_dict(state['scaler'])
    return state
