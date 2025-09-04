import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import copy

# --- 1. 實驗設定 (Settings) ---
# 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用裝置 (Using device): {device}")

# 超參數
N_SAMPLES = 500
NOISE = 0.1
LEARNING_RATE = 0.01
EPOCHS = 2000
HIDDEN_DIM = 32
# 定義一個旋轉角度，用來模擬不同來源的數據（例如，不同醫院的儀器角度）
ROTATION_ANGLE = np.pi / 4  # 45 度

# --- 2. 數據準備 (Data Preparation) ---

def get_data():
    """生成並分割 '兩個月亮' 資料集"""
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return (
        torch.from_numpy(X_train).float().to(device),
        torch.from_numpy(y_train).long().to(device),
        torch.from_numpy(X_test).float().to(device),
        torch.from_numpy(y_test).long().to(device)
    )

def rotate_data(X, angle):
    """對 2D 數據進行旋轉"""
    rotation_matrix = torch.tensor([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ], dtype=torch.float32, device=device)
    return X @ rotation_matrix.T

# 生成原始數據和旋轉後的數據
X_train, y_train, X_test, y_test = get_data()
X_train_rot, X_test_rot = rotate_data(X_train, ROTATION_ANGLE), rotate_data(X_test, ROTATION_ANGLE)

# --- 3. 模型定義 (Model Definition) ---

class SimpleMLP(nn.Module):
    """一個簡單的多層感知機"""
    def __init__(self, input_dim=2, hidden_dim=HIDDEN_DIM, output_dim=2):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

# --- 4. 訓練與評估 (Training & Evaluation) ---

def train_model(model, X_data, y_data):
    """訓練模型的通用函數"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_data)
        loss = criterion(outputs, y_data)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

def evaluate_model(model, X_test, y_test):
    """評估模型準確率"""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / len(y_test)
    return accuracy

# --- 5. 核心實驗流程 (Core Experiment) ---

# 分別訓練兩個模型
# Model A: 在原始數據上訓練
print("--- 訓練模型 A (原始數據) ---")
model_A = SimpleMLP().to(device)
train_model(model_A, X_train, y_train)

# Model B: 在旋轉後的數據上訓練
print("\n--- 訓練模型 B (旋轉後數據) ---")
model_B = SimpleMLP().to(device)
train_model(model_B, X_train_rot, y_train)


# --- 【修正版】實現隱藏層空間的正交參數對位 (OPM in Hidden Space) ---
# 核心思想：我們不再對齊 2D 輸入空間，而是對齊 32D 的隱藏層神經元。
# 我們要找到一個正交矩陣 R (可以看作是廣義的置換/旋轉)，
# 使得模型 B 的神經元重新排列後，與模型 A 的盡可能一致。
# min ||W1_A - R * W1_B||_F, R* = UV^T for SVD(W1_A * W1_B^T = UΣV^T)
print("\n--- 執行隱藏層空間的正交參數對位 (OPM in Hidden Space) ---")
with torch.no_grad():
    W1_A = model_A.layer1.weight.data
    W1_B = model_B.layer1.weight.data
    
    # 計算對位矩陣 M
    M = W1_A.mm(W1_B.T) # (hidden, in) @ (in, hidden) -> (hidden, hidden)
    
    # 執行 SVD 分解
    U, _, Vh = torch.linalg.svd(M)
    
    # 找到最佳的正交矩陣 R
    R = U.mm(Vh) # R 的維度是 (hidden, hidden)
    
print(f"找到 {R.shape} 維的最佳隱藏層對位矩陣 R")

# --- 創建融合模型 (Model Fusion) ---
print("\n--- 創建融合模型 ---")
# 1. 樸素融合 (Naive Soup): 直接平均權重
model_naive_soup = SimpleMLP().to(device)
with torch.no_grad():
    for p_naive, p_A, p_B in zip(model_naive_soup.parameters(), model_A.parameters(), model_B.parameters()):
        p_naive.data.copy_((p_A.data + p_B.data) / 2.0)

# 2. POPMI 融合 (POPMI Soup): 在隱藏層對齊後再平均
model_popmi_soup = SimpleMLP().to(device)

# 先複製一個對齊後的模型 B
model_B_aligned = copy.deepcopy(model_B)

with torch.no_grad():
    # 對齊模型 B 的第一層 (輸入 -> 隱藏)
    # W1_B_aligned = R @ W1_B
    model_B_aligned.layer1.weight.data = R @ model_B.layer1.weight.data
    model_B_aligned.layer1.bias.data = R @ model_B.layer1.bias.data
    
    # 對齊模型 B 的第二層 (隱藏 -> 輸出)
    # 為了保持數學等價性，第二層的權重需要乘以 R 的轉置
    # W2_B_aligned = W2_B @ R^T
    model_B_aligned.layer2.weight.data = model_B.layer2.weight.data @ R.T

# 現在 model_B_aligned 和 model_A 的隱藏層是「對齊」的了，可以進行平均
with torch.no_grad():
    for p_popmi, p_A, p_B_aligned in zip(model_popmi_soup.parameters(), model_A.parameters(), model_B_aligned.parameters()):
        p_popmi.data.copy_((p_A.data + p_B_aligned.data) / 2.0)


# --- 6. 結果評估與視覺化 (Results & Visualization) ---

# 在原始測試集和旋轉後測試集上評估所有模型
print("\n--- 模型評估 ---")
models = {
    "模型 A": model_A,
    "模型 B": model_B,
    "樸素融合 (Naive Soup)": model_naive_soup,
    "POPMI 融合 (修正版)": model_popmi_soup,
}

results = {}
for name, model in models.items():
    acc_original = evaluate_model(model, X_test, y_test)
    acc_rotated = evaluate_model(model, X_test_rot, y_test)
    results[name] = {"原始數據準確率": acc_original, "旋轉數據準確率": acc_rotated}
    print(f"[{name}]")
    print(f"  - 在原始測試集上的準確率: {acc_original:.2%}")
    print(f"  - 在旋轉測試集上的準確率: {acc_rotated:.2%}")


def plot_decision_boundary(model, X, y, title):
    """繪製決策邊界"""
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    with torch.no_grad():
        Z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
        Z = torch.max(Z, 1)[1]
        Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

# 視覺化
# 將所有數據合併，以更好地觀察融合模型的泛化能力
X_combined_test = torch.cat([X_test, X_test_rot], dim=0).cpu()
y_combined_test = torch.cat([y_test, y_test], dim=0).cpu()

plt.figure(figsize=(16, 10))
plt.suptitle("POPMI Toy Experiment: Alignment and Fusion Effects (Revised)", fontsize=16)

# Plot Model A on original data
plt.subplot(2, 3, 1)
plot_decision_boundary(model_A, X_test.cpu(), y_test.cpu(), "Model A (Good on Original Data)")

# Plot Model B on rotated data
plt.subplot(2, 3, 2)
plot_decision_boundary(model_B, X_test_rot.cpu(), y_test.cpu(), "Model B (Good on Rotated Data)")

# Plot Naive Soup on combined data
plt.subplot(2, 3, 4)
plot_decision_boundary(model_naive_soup, X_combined_test, y_combined_test, "Naive Fusion (Poor Performance)")

# Plot POPMI Soup on combined data
plt.subplot(2, 3, 5)
plot_decision_boundary(model_popmi_soup, X_combined_test, y_combined_test, "POPMI Fusion (Strong Generalization)")

# Plot data distributions
plt.subplot(2, 3, 3)
plt.scatter(X_train.cpu()[:,0], X_train.cpu()[:,1], c=y_train.cpu(), cmap=plt.cm.RdYlBu, alpha=0.5, label='Original Data')
plt.scatter(X_train_rot.cpu()[:,0], X_train_rot.cpu()[:,1], c=y_train.cpu(), cmap=plt.cm.viridis, alpha=0.5, label='Rotated Data')
plt.title("Data Distributions")
plt.legend()


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()