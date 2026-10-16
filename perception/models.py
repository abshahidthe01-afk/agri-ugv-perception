"""Networks for crop/weed segmentation."""

import torch

MODELS = ('segformer_b0',)


def build(name='segformer_b0', classes=3, pretrained=True):
    """
    Return a segmentation network with 'classes' outputs per pixel, at the input's size.

    segformer_b0: SegFormer (Xie et al. 2021) with the MiT-B0 encoder, 3.7 M parameters
    (the PhenoBench ERFNet baseline has 2.1 M), from segmentation-models-pytorch; pretrained:
    the encoder starts from its ImageNet weights (downloaded once).
    """
    import segmentation_models_pytorch as smp
    if name == 'segformer_b0':
        return smp.Segformer(encoder_name='mit_b0',
                             encoder_weights='imagenet' if pretrained else None,
                             classes=classes)
    raise ValueError(f'unknown model {name!r}; known: {", ".join(MODELS)}')


def parameter_groups(model, lr, head_factor=10.0):
    """The encoder at 'lr', the newly made decoder and head 'head_factor' times faster."""
    encoder = [p for p in model.encoder.parameters() if p.requires_grad]
    known = {id(p) for p in encoder}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in known]
    return [{'params': encoder, 'lr': lr}, {'params': rest, 'lr': lr * head_factor}]


def save(path, model, name, epoch, metrics, settings):
    """Save the weights with what made them."""
    torch.save({'model': name, 'state_dict': model.state_dict(), 'epoch': epoch,
                'metrics': metrics, 'settings': settings}, path)


def load(path, device='cpu'):
    """Return (network, checkpoint dict) from a file written by save()."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = build(checkpoint['model'], pretrained=False)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint
