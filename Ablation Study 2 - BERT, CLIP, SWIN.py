import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel, SwinForImageClassification
from torchvision import transforms
from tqdm import tqdm
import json
from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import gc
import pandas as pd
import random
import warnings
import torch.nn.functional as F
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

images_path = '/home/umera_p/Dataset/training/memes'

def display_images_grid(image_paths, title):
    fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(15, 30))
    fig.suptitle(title, fontsize=20)
    for ax, image_path in zip(axes.flatten(), image_paths):
        try:
            image = Image.open(image_path)
            ax.imshow(image)
            ax.axis('off')
        except FileNotFoundError:
            ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

spanish_ids = [str(i) for i in range(110001, 112035)]
english_ids = [str(i) for i in range(210001, 212011)]

spanish_images = []
for image_id in spanish_ids[:50]:
    for ext in ['jpeg', 'jpg', 'png']:
        image_path = os.path.join(images_path, f"{image_id}.{ext}")
        if os.path.exists(image_path):
            spanish_images.append(image_path)
            break

english_images = []
for image_id in english_ids[:50]:
    for ext in ['jpeg', 'jpg', 'png']:
        image_path = os.path.join(images_path, f"{image_id}.{ext}")
        if os.path.exists(image_path):
            english_images.append(image_path)
            break

display_images_grid(spanish_images, 'First 50 Spanish Images')
display_images_grid(english_images, 'First 50 English Images')

annotations_path = '/home/umera_p/Dataset/training/training.json'
with open(annotations_path, 'r') as file:
    annotations = json.load(file)

df = pd.DataFrame.from_dict(annotations, orient='index')
display(df.head())

sexist_count = sum(1 for value in annotations.values() if value['labels_task4'].count('YES') > value['labels_task4'].count('NO'))
non_sexist_count = len(annotations) - sexist_count
sexist_count_es = sum(1 for value in annotations.values() if value['lang'] == 'es' and value['labels_task4'].count('YES') > value['labels_task4'].count('NO'))
non_sexist_count_es = sum(1 for value in annotations.values() if value['lang'] == 'es' and value['labels_task4'].count('YES') <= value['labels_task4'].count('NO'))
sexist_count_en = sum(1 for value in annotations.values() if value['lang'] == 'en' and value['labels_task4'].count('YES') > value['labels_task4'].count('NO'))
non_sexist_count_en = sum(1 for value in annotations.values() if value['lang'] == 'en' and value['labels_task4'].count('YES') <= value['labels_task4'].count('NO'))

print(f"Total number of images: {len(annotations)}")
print(f"Number of sexist images: {sexist_count}")
print(f"Number of sexist images (Spanish): {sexist_count_es}")
print(f"Number of sexist images (English): {sexist_count_en}")
print(f"Number of non-sexist images: {non_sexist_count}")
print(f"Number of non-sexist images (Spanish): {non_sexist_count_es}")
print(f"Number of non-sexis images (English): {non_sexist_count_en}")

class MemeDataset(Dataset):
    def __init__(self, dataset, base_dir, transform=None, is_test=False):
        self.data = dataset
        self.keys = list(dataset.keys())
        self.base_dir = base_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.data[key]
        image_path = os.path.join(self.base_dir, item['path_memes'])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = item['text']
        
        if not self.is_test:
            labels = 1 if item['labels_task4'].count('YES') > item['labels_task4'].count('NO') else 0
            return image, text, labels
        else:
            return image, text, key
        
def collate_fn(batch):
    images, texts, labels_or_keys = zip(*batch)
    images = torch.stack(images, dim=0)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    bert_inputs = bert_tokenizer(list(texts), return_tensors='pt', padding=True, truncation=True)

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_text_inputs = clip_processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
    clip_image_inputs = images

    if isinstance(labels_or_keys[0], int):
        labels = torch.tensor(labels_or_keys).long()
        return images, bert_inputs, clip_text_inputs.input_ids, clip_image_inputs, labels
    else:
        keys = labels_or_keys
        return images, bert_inputs, clip_text_inputs.input_ids, clip_image_inputs, keys

def stratified_split(dataset, split_ratios=(0.8, 0.1, 0.1)):
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"

    non_sexist_es = [key for key, value in dataset.items() if value['lang'] == 'es' and value['labels_task4'].count('YES') <= value['labels_task4'].count('NO')]
    non_sexist_en = [key for key, value in dataset.items() if value['lang'] == 'en' and value['labels_task4'].count('YES') <= value['labels_task4'].count('NO')]
    sexist_es = [key for key, value in dataset.items() if value['lang'] == 'es' and value['labels_task4'].count('YES') > value['labels_task4'].count('NO')]
    sexist_en = [key for key, value in dataset.items() if value['lang'] == 'en' and value['labels_task4'].count('YES') > value['labels_task4'].count('NO')]

    random.shuffle(non_sexist_es)
    random.shuffle(non_sexist_en)
    random.shuffle(sexist_es)
    random.shuffle(sexist_en)

    def split_list(data_list, split_ratios):
        train_size = round(int(split_ratios[0] * len(data_list)))
        val_size = round(int(split_ratios[1] * len(data_list)))
        train_split = data_list[:train_size]
        val_split = data_list[train_size:train_size + val_size]
        test_split = data_list[train_size + val_size:]
        return train_split, val_split, test_split

    train_non_sexist_es, val_non_sexist_es, test_non_sexist_es = split_list(non_sexist_es, split_ratios)
    train_non_sexist_en, val_non_sexist_en, test_non_sexist_en = split_list(non_sexist_en, split_ratios)
    train_sexist_es, val_sexist_es, test_sexist_es = split_list(sexist_es, split_ratios)
    train_sexist_en, val_sexist_en, test_sexist_en = split_list(sexist_en, split_ratios)

    train_keys = train_non_sexist_es + train_non_sexist_en + train_sexist_es + train_sexist_en
    val_keys = val_non_sexist_es + val_non_sexist_en + val_sexist_es + val_sexist_en
    test_keys = test_non_sexist_es + test_non_sexist_en + test_sexist_es + test_sexist_en

    random.shuffle(train_keys)
    random.shuffle(val_keys)
    random.shuffle(test_keys)

    return train_keys, val_keys, test_keys

train_keys, val_keys, test_keys = stratified_split(annotations)

print(f"Training set size: {len(train_keys)}")
print(f"Validation set size: {len(val_keys)}")
print(f"Known test set size: {len(test_keys)}")

def create_split_dataset(keys, dataset, base_dir, transform=None):
    split_data = {key: dataset[key] for key in keys}
    return MemeDataset(split_data, base_dir, transform=transform)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

base_dir = '/home/umera_p/Dataset/training'

train_dataset = create_split_dataset(train_keys, annotations, base_dir, transform=transform)
val_dataset = create_split_dataset(val_keys, annotations, base_dir, transform=transform)
known_test_dataset = create_split_dataset(test_keys, annotations, base_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
known_test_loader = DataLoader(known_test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

class MultimodalModel(nn.Module):
    def __init__(self, bert_model, clip_model, swin_model, hidden_dim=512, num_classes=2):
        super(MultimodalModel, self).__init__()
        self.bert_model = bert_model
        self.clip_model = clip_model
        self.swin_model = swin_model

        self.bert_fc = nn.Linear(bert_model.config.hidden_size, hidden_dim)
        self.clip_fc = nn.Linear(clip_model.config.text_config.hidden_size, hidden_dim)
        self.swin_fc = nn.Linear(swin_model.num_labels, hidden_dim)

        self.text_fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.visual_fusion_fc = nn.Linear(hidden_dim, hidden_dim)

        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.classifier = nn.Linear(hidden_dim // 4, num_classes)

    def forward(self, images, bert_input, clip_text_input, clip_image_input):
        bert_features = self.bert_fc(self.bert_model(**bert_input).last_hidden_state[:, 0, :])
        clip_text_features = self.clip_fc(self.clip_model.get_text_features(clip_text_input))
        clip_image_features = self.swin_fc(self.swin_model(images).logits)

        text_features = torch.cat((bert_features, clip_text_features), dim=1)
        visual_fusion = self.visual_fusion_fc(clip_image_features)

        text_fusion = self.text_fusion_fc(text_features)
        fused_features = torch.cat((text_fusion, visual_fusion), dim=1)

        output = self.classifier(F.relu(self.linear3(F.relu(self.linear2(F.relu(self.linear1(fused_features)))))))

        return output
    
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
swin_model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224')

for param in bert_model.parameters():
    param.requires_grad = False

for param in clip_model.parameters():
    param.requires_grad = False

for param in swin_model.parameters():
    param.requires_grad = False

bert_model.to(device)
clip_model.to(device)
swin_model.to(device)

model = nn.DataParallel(MultimodalModel(bert_model, clip_model, swin_model), device_ids=[1]).to(device)

def validate_model(model, val_loader, criterion):
    model.eval()
    val_running_loss = 0.0
    val_all_labels = []
    val_all_preds = []

    with torch.no_grad():
        for images, bert_input, clip_text_input, clip_image_input, labels in val_loader:
            images = images.to(device)
            bert_input = {key: val.to(device) for key, val in bert_input.items()}
            clip_text_input = clip_text_input.to(device)
            clip_image_input = clip_image_input.to(device)
            labels = labels.to(device)

            outputs = model(images, bert_input, clip_text_input, clip_image_input)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            val_all_preds.extend(preds.cpu().numpy())
            val_all_labels.extend(labels.cpu().numpy())

    avg_val_loss = val_running_loss / len(val_loader)
    val_accuracy = accuracy_score(val_all_labels, val_all_preds)
    val_f1 = f1_score(val_all_labels, val_all_preds)

    val_conf_matrix = confusion_matrix(val_all_labels, val_all_preds)

    return avg_val_loss, val_accuracy, val_f1, val_all_labels, val_all_preds

def train_model(model, train_loader, val_loader, known_test_loader, num_epochs=20):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        train_all_preds = []
        train_all_labels = []

        for images, bert_input, clip_text_input, clip_image_input, labels in train_loader:
            images = images.to(device)
            bert_input = {key: val.to(device) for key, val in bert_input.items()}
            clip_text_input = clip_text_input.to(device)
            clip_image_input = clip_image_input.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images, bert_input, clip_text_input, clip_image_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            train_all_preds.extend(preds.cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(train_loader)
        accuracy = accuracy_score(train_all_labels, train_all_preds)
        f1 = f1_score(train_all_labels, train_all_preds)

        conf_matrix = confusion_matrix(train_all_labels, train_all_preds)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {avg_loss:.4f} | Training Accuracy: {accuracy:.4f}")
        print("Training Confusion Matrix:")
        print(conf_matrix)
        print(f"Training F1 Score: {f1}")

        val_loss, val_accuracy, val_f1, val_all_labels, val_all_preds = validate_model(model, val_loader, criterion)
        scheduler.step(val_loss)
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")
        print("Validation Confusion Matrix:")
        print(confusion_matrix(val_all_labels, val_all_preds))
        print(f"Validation F1 Score: {val_f1:.4f}")

        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved for epoch {epoch+1}")
          
        gc.collect()

    print("\nFinal Training Classification Report:")
    print(classification_report(train_all_labels, train_all_preds, target_names=['Non-Sexist', 'Sexist']))

    print("\nFinal Validation Classification Report:")
    print(classification_report(val_all_labels, val_all_preds, target_names=['Non-Sexist', 'Sexist']))

    test_loss, test_accuracy, test_f1, test_labels, test_preds = validate_model(model, known_test_loader, criterion)
    print(f"Known Test Loss: {test_loss:.4f} | Known Test Accuracy: {test_accuracy:.4f}")
    print("Known Test Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    print(f"Known Test F1 Score: {test_f1:.4f}")

    return model

model = train_model(model, train_loader, val_loader, known_test_loader)

test_annotations_path = '/home/umera_p/Dataset/test/test_clean.json'
with open(test_annotations_path, 'r') as file:
    test_annotations = json.load(file)

test_base_dir = '/home/umera_p/Dataset/test'
test_dataset = MemeDataset(
    dataset=test_annotations, 
    base_dir=test_base_dir,
    transform=transform,
    is_test=True
)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

def predict_model(model, test_loader):
    model.eval()
    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, bert_input, clip_text_input, clip_image_input, keys in tqdm(test_loader):
            images = images.to(device)
            bert_input = {key: val.to(device) for key, val in bert_input.items()}
            clip_text_input = clip_text_input.to(device)
            clip_image_input = clip_image_input.to(device)

            outputs = model(images, bert_input, clip_text_input, clip_image_input)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds.cpu().numpy())
            image_ids.extend(keys)

    return image_ids, predictions

image_ids, predictions = predict_model(model, test_loader)

def save_predictions_json(image_ids, predictions, overall_path, sexist_path, non_sexist_path):
    results = []
    for img_id, pred in zip(image_ids, predictions):
        results.append({'image_id': img_id, 'prediction': 'Sexist' if pred == 1 else 'Non-Sexist'})
    
    with open(overall_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    sexist_results = [r for r in results if r['prediction'] == 'Sexist']
    non_sexist_results = [r for r in results if r['prediction'] == 'Non-Sexist']
    
    with open(sexist_path, 'w') as f:
        json.dump(sexist_results, f, indent=4)
    
    with open(non_sexist_path, 'w') as f:
        json.dump(non_sexist_results, f, indent=4)

save_predictions_json(image_ids, predictions, 'Task_4_overall_predictions.json', 'Task_4_sexist_predictions.json', 'Task_4_non_sexist_predictions.json')

def load_predictions_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

overall_results = load_predictions_json('Task_4_overall_predictions.json')
print("Overall Predictions :")
print(overall_results)

sexist_results = load_predictions_json('Task_4_sexist_predictions.json')
print("Sexist Predictions :")
print(sexist_results)

non_sexist_results = load_predictions_json('Task_4_non_sexist_predictions.json')
print("Non-Sexist Predictions :")
print(non_sexist_results)