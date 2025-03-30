import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model.flowv1 import flowNet
import torchvision
from torchvision import transforms
from dataset.datasetv2 import VideoFrameFlowDataset

# from your_dataset_module import VideoDataset
from sklearn.metrics import roc_auc_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = flowNet()
model.load_state_dict(torch.load('/model_weight_root'))
model.to(device)
model.eval()

aug_test = torchvision.transforms.Compose([
    transforms.Resize((224, 224)),
    torchvision.transforms.RandomCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
])

DatasetTest = VideoFrameFlowDataset('/your_frame_dataroot/test',
                                        '/your_flow_dataroot/testflow',
                                        num_frames=6,
                                        transform=aug_test)
dataloaderTest = DataLoader(DatasetTest, batch_size=6, shuffle=True, num_workers=4)

correct = 0
total = 0
correct_predictions = 0
test_all_preds = []
test_all_labels = []
with torch.no_grad():
    for frames, flows, label, frame_paths, flow_paths in dataloaderTest:
        frames = frames.to(device)
        flows = flows.to(device)
        label = label.to(device)
        outputs = model(frames, flows)
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == label).sum().item()
        test_all_preds.extend(outputs[:, 1].detach().cpu().numpy())  # 假设是二分类问题，取第1列的概率值
        test_all_labels.extend(label.cpu().numpy())
    test_acc = correct_predictions / len(DatasetTest)
    try:
        test_auc = roc_auc_score(test_all_labels, test_all_preds)
    except ValueError as e:
        print(f"Error calculating AUC: {e}")
        test_auc = 0.0

print(f'Accuracy of the model on the test images: {test_acc}')
print('Test Set AUC: {:.4f}'.format(test_auc))
