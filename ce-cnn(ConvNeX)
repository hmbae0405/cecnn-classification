import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
import os
import datetime
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import shutil
from collections import Counter
import gc
from torchvision import transforms
from sklearn.metrics import roc_curve

# 캐시 삭제
torch_cache_dir = os.path.expanduser('~/.cache/torch')
if os.path.exists(torch_cache_dir):
    shutil.rmtree(torch_cache_dir)
    print(f"Deleted PyTorch cache: {torch_cache_dir}")

# GPU 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 데이터 경로 설정
path_data = 
path_data2 =
path_train = os.path.join(path_data, 'train/')
path_val = os.path.join(path_data, 'val/')
path_test = os.path.join(path_data, 'test/')

# 결과 저장 경로 설정
results_folder = os.path.join(path_data2, 'results/')
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
current_results_folder = os.path.join(results_folder, f"run_{timestamp}")
os.makedirs(current_results_folder)

# 라벨 추출 
def extract_label(file_name):
    if 'nor' in file_name:
        return 0
    elif 'ab' in file_name:
        return 1
    else:
        print(f"Skipping file with unexpected name format: {file_name}")
        return None

# 데이터 증강 
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# 데이터 로드 
def load_data(base_path, mode='train'):
    images, labels = [], []
    '''
    #환자벌호별 폴더일때
    #for img_path in glob(base_path + "**/*.jpg", recursive=True):
    '''
    #트레인 밸리데이션 테스트 안에 바로 이미지 존재
    for img_path in glob(os.path.join(base_path, "*.jpg")):

        img = Image.open(img_path).resize((224, 224))
        label = extract_label(os.path.basename(img_path))
        if label is not None:
            if mode == 'train':
                img = train_transform(np.array(img)).permute(1, 2, 0).numpy()
            else:
                img = np.array(img) / 255.0
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# 데이터 로드 코드
train_images, train_labels = load_data(path_train, mode='train')
val_images, val_labels = load_data(path_val, mode='val')
test_images, test_labels = load_data(path_test, mode='test')

# 클래스 불균형 가중치 
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# ============================================================
# ★ 모델: ConvNeXt-Tiny + CBAM
# ============================================================
# 모델은 여기서 변경
class ConvNeXtTiny_CBAM(nn.Module):  
    def __init__(self, num_classes=2):
        super(ConvNeXtTiny_CBAM, self).__init__()
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # 모든 파라미터 고정
        for param in convnext.parameters():
            param.requires_grad = False

        # features: ConvNeXt의 4-stage feature extractor
        # 마지막 stage(stage 3) 출력 채널: 768
        self.features = convnext.features  # Sequential of 8 blocks

        # avgpool: AdaptiveAvgPool2d(1) → (B, 768, 1, 1)
        self.avgpool = convnext.avgpool

        # CBAM 삽입 (마지막 feature map 이후, avgpool 이전)
        # ConvNeXt-Tiny 마지막 stage 출력: 768 채널
        self.cbam = CBAM(in_planes=768)

        # 최종 분류기 (학습 대상)
        # convnext.classifier: Sequential(LayerNorm2d, Flatten, Linear(768, 1000))
        # LayerNorm2d는 (B, 768, 1, 1) → avgpool 이후에 적용되므로 그대로 재사용
        self.norm = convnext.classifier[0]   # LayerNorm2d(768)
        self.flatten = convnext.classifier[1]  # Flatten
        self.classifier = nn.Linear(768, num_classes)  # 학습 대상

    def forward(self, x):
        x = self.features(x)      # (B, 768, H, W)
        x = self.cbam(x)          # CBAM 삽입 지점
        x = self.avgpool(x)       # (B, 768, 1, 1)
        x = self.norm(x)          # LayerNorm2d
        x = self.flatten(x)       # (B, 768)
        x = self.classifier(x)    # (B, num_classes)
        return x
# ============================================================

model = ConvNeXtTiny_CBAM().to(device)

# 손실 함수 및 최적화 설정
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# validation 정확도 평가 함수
def evaluate_val_accuracy(model, val_images, val_labels, batch_size=32):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(val_images), batch_size):
            batch = val_images[i:i + batch_size]
            images = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            del images, outputs, preds
            gc.collect()
            torch.cuda.empty_cache()
    acc = accuracy_score(val_labels, all_preds)
    return acc

# 학습 
def train_model_manual(model, criterion, optimizer, train_images, train_labels, val_images, val_labels, epochs, batch_size=32, patience=10):
    best_val_acc = 0
    patience_counter = 0
    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        indices = np.random.permutation(len(train_images))
        train_images, train_labels = train_images[indices], train_labels[indices]

        for i in range(0, len(train_images), batch_size):
            batch = train_images[i:i + batch_size]
            images = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(device)
            labels = torch.tensor(train_labels[i:i + batch_size]).long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)   
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            del images, labels, outputs, preds
            gc.collect()
            torch.cuda.empty_cache()

        scheduler.step()
        train_acc = correct / total
        val_acc = evaluate_val_accuracy(model, val_images, val_labels, batch_size=32)

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(current_results_folder, 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # 학습 곡선 저장
    plt.figure()
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(current_results_folder, 'training_plot.png'))
    plt.close()

# 테스트 
def evaluate_model(model, test_images, test_labels, batch_size=64):
    model.eval()

    best_model_path = os.path.join(current_results_folder, 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Best model loaded from {best_model_path}")

    all_preds, all_probs = [], []
    with torch.no_grad():
        for i in range(0, len(test_images), batch_size):
            batch = test_images[i:i + batch_size]
            images = torch.from_numpy(batch).float().permute(0, 3, 1, 2).to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])

            del images, outputs, probs, preds
            gc.collect()
            torch.cuda.empty_cache()
    
    #저장할 요소들
    accuracy = accuracy_score(test_labels, all_preds)
    precision = precision_score(test_labels, all_preds)
    recall = recall_score(test_labels, all_preds)
    f1 = f1_score(test_labels, all_preds)
    auc_roc = roc_auc_score(test_labels, all_probs)
    
    # ROC 커브 계산 및 저장
    fpr, tpr, thresholds = roc_curve(test_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_roc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    roc_curve_path = os.path.join(current_results_folder, "roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()
    
    cm = confusion_matrix(test_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'], annot_kws={"size": 16})
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(current_results_folder, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    metrics = f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-score: {f1}\nAUC-ROC: {auc_roc}\n"
    with open(os.path.join(current_results_folder, "classification_metrics.txt"), "w") as f:
        f.write(metrics)

    print(metrics)
    print(f"Confusion matrix saved at: {cm_path}")

#%%

# 학습 실행
train_model_manual(model, criterion, optimizer, train_images, train_labels, val_images, val_labels, epochs=500, patience=20)

# 테스트 실행
evaluate_model(model, test_images, test_labels)
