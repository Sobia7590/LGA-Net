# ── Grad-CAM for LGA-Net ──────────────────────────────────────
import os, cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

GRAPH_DIR = os.path.join(BASE_DIR, "outputs_600dpi")
os.makedirs(GRAPH_DIR, exist_ok=True)

# ── LOAD MODEL ────────────────────────────────────────────────
model = LGANet(num_classes=5).to(device)
model.load_state_dict(torch.load(
    os.path.join(WORK_DIR, 'lganet_stage2_best_stage2_aptos_full_finetune.pth'),
    map_location=device))
model.eval()
print("Loaded model.")

# ── HELPERS ───────────────────────────────────────────────────
grade_names = {0:'No DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferative'}

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * std + mean, 0, 1)

def normalize_map(x):
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def apply_fov_mask(arr, radius_scale=0.48):
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    r = radius_scale * min(h, w)
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    out = arr.copy()
    out[~mask] = 0
    return out

# ── HOOKS ─────────────────────────────────────────────────────
cam_activations = None
cam_gradients   = None

def forward_hook_attn(module, inputs, output):
    global cam_activations
    attended = output[0]
    cam_activations = attended
    def save_gradients(grad):
        global cam_gradients
        cam_gradients = grad
    attended.register_hook(save_gradients)

hook_handle = model.attn_gate.register_forward_hook(forward_hook_attn)

# ── GRAD-CAM FUNCTION ─────────────────────────────────────────
def generate_gradcam(model, input_tensor, target_class=None, output_size=(380, 380)):
    global cam_activations, cam_gradients
    cam_activations = None; cam_gradients = None
    model.zero_grad()
    logits, attn_map = model(input_tensor)
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())
    logits[:, target_class].backward(retain_graph=True)
    activations = cam_activations.detach()
    gradients   = cam_gradients.detach()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=output_size, mode='bilinear', align_corners=False)
    cam = normalize_map(cam.squeeze().cpu().numpy())
    return cam, logits, attn_map

# ── VISUALISE ON APTOS VALIDATION SAMPLES ─────────────────────
sample_imgs, sample_labels = next(iter(aptos_val_loader))
sample_imgs = sample_imgs.to(device)
num_samples = min(4, sample_imgs.shape[0])

# FIX (file size): figsize was (18, 5*num_samples) at dpi=600, i.e. a
# 10,800px-wide canvas — roughly 3x BSPC's minimum required resolution
# for this figure type (combination line/halftone: >=500 dpi / ~3,740px
# full-page-width). Trimmed figsize to 14in wide (still >=8,400px wide
# at dpi=500, well over the requirement) and dropped dpi 600 -> 500,
# which alone is still fully compliant and meaningfully smaller.
fig, axes = plt.subplots(num_samples, 4, figsize=(14, 4 * num_samples))
if num_samples == 1:
    axes = np.expand_dims(axes, axis=0)

for i in range(num_samples):
    img_tensor = sample_imgs[i].unsqueeze(0)
    with torch.enable_grad():
        gradcam_map, logits, attn_map = generate_gradcam(
            model, img_tensor, output_size=(380, 380))
    gradcam_map = apply_fov_mask(gradcam_map)
    pred    = torch.argmax(logits, dim=1).item()
    true    = sample_labels[i].item()
    correct = "✓" if pred == true else "✗"

    img_show = denormalize(sample_imgs[i]).permute(1, 2, 0).numpy()
    attn_show = normalize_map(
        F.interpolate(attn_map[0].unsqueeze(0), size=(380, 380),
                      mode='bilinear', align_corners=False)
        .squeeze().detach().cpu().numpy())
    attn_show = apply_fov_mask(attn_show)

    axes[i, 0].imshow(img_show)
    axes[i, 0].set_title(f"Original\nTrue: {grade_names[true]}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(attn_show, cmap='hot')
    axes[i, 1].set_title(f"Attention Map\nPred: {grade_names[pred]}")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(gradcam_map, cmap='jet')
    axes[i, 2].set_title(f"Grad-CAM\nPred: {grade_names[pred]}")
    axes[i, 2].axis("off")

    axes[i, 3].imshow(img_show)
    axes[i, 3].imshow(gradcam_map, cmap='jet', alpha=0.40)
    axes[i, 3].set_title(f"Overlay {correct}\nTrue: {grade_names[true]} | Pred: {grade_names[pred]}")
    axes[i, 3].axis("off")

plt.tight_layout()

# FIX (file size): PNG is lossless, which compresses poorly on a
# photo + jet-colormap-heatmap composite like this. BSPC explicitly
# accepts JPG for both photographic and combination line/halftone
# figures at >=500 dpi (fetched directly from their Guide for Authors),
# so switching format + dropping dpi 600 -> 500 gives a large size
# reduction with no compliance issue and no visible quality loss at
# quality=95.
path = os.path.join(GRAPH_DIR, "lganet_gradcam_aptos.jpg")
plt.savefig(path, dpi=500, bbox_inches='tight', pil_kwargs={'quality': 95})
plt.show()

before_note = (
    "Previous version: PNG, dpi=600, figsize width 18in (~10,800px). "
    "This version: JPG q95, dpi=500, figsize width 14in (~7,000px) — "
    "still well above BSPC's 500 dpi / ~3,740px minimum for combination "
    "line/halftone artwork."
)
print(before_note)

# ── REMOVE HOOK ───────────────────────────────────────────────
hook_handle.remove()
print(f"Saved: {path}")
