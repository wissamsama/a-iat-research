import torch
import torch.nn.functional as F


def _stats_tensor(values, device):
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, -1, 1, 1)


def normalize(images, mean, std):
    mean_tensor = _stats_tensor(mean, images.device)
    std_tensor = _stats_tensor(std, images.device)
    return (images - mean_tensor) / std_tensor


def denormalize(images, mean, std):
    mean_tensor = _stats_tensor(mean, images.device)
    std_tensor = _stats_tensor(std, images.device)
    return images * std_tensor + mean_tensor


def fgsm_attack(model, images, labels, epsilon, mean, std):
    """Return FGSM adversarial examples for normalized images.

    epsilon is expressed in pixel space [0, 1]. The returned tensor is normalized
    with the same mean/std as the input model expects.
    """
    was_training = model.training
    model.eval()
    attack_images = images.detach().clone().requires_grad_(True)

    outputs = model(attack_images)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad(set_to_none=True)
    loss.backward()

    pixel_images = denormalize(attack_images.detach(), mean, std)
    perturbation = epsilon * attack_images.grad.detach().sign()
    adversarial_pixels = torch.clamp(pixel_images + perturbation, 0.0, 1.0)
    if was_training:
        model.train()
    return normalize(adversarial_pixels, mean, std).detach()
