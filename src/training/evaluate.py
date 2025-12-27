from sklearn.metrics import classification_report, confusion_matrix
import torch
from torchvision import transforms
from src.models.resnet import get_model
from src.datasets.mri_dataset import MRIDataset
from torch.utils.data import DataLoader

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = MRIDataset("data/processed/test", transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = get_model(num_classes=4)
model.load_state_dict(torch.load("experiments/model.pth", map_location="cpu"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = outputs.argmax(1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
print("Classification report:")
print(classification_report(all_labels, all_preds, target_names=classes))

print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds))
