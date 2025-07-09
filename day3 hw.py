import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import sys
import time

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
try:
    # 使用原始字符串格式指定文件路径
    file_path = r"D:\course source\aiSummerCamp2025-master\day3\assignment\data\household_power_consumption\household_power_consumption.txt"
    df = pd.read_csv(file_path, sep=";")
    print("数据加载成功，共有", len(df), "条记录")
except FileNotFoundError:
    print(f"错误：找不到数据文件，请检查路径是否正确: {file_path}")
    sys.exit(1)

# 数据基本信息
print("\n数据基本信息：")
df.info()

# 数据预处理
print("\n开始数据预处理...")
# 转换日期时间格式
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis=1, inplace=True)

# 处理缺失值
print("缺失值统计：")
print(df.isnull().sum())

# 将无法转换为数值的'?'替换为NaN
for col in df.select_dtypes(include=[object]).columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)
print(f"删除缺失值后剩余记录: {len(df)}")

# 打印数据时间范围
print("\n数据时间范围：")
print("开始日期: ", df['datetime'].min())
print("结束日期: ", df['datetime'].max())

# 分割训练集和测试集
print("\n分割训练集和测试集...")
# 确保 datetime 列是 datetime 类型
df['datetime'] = pd.to_datetime(df['datetime'])

# 使用明确的日期时间格式
train = df[df['datetime'] <= pd.Timestamp('2009-12-31 23:59:59')]
test = df[df['datetime'] > pd.Timestamp('2009-12-31 23:59:59')]

print(f"训练集样本数: {len(train)}")
print(f"测试集样本数: {len(test)}")
print(f"训练集日期范围: {train['datetime'].min()} 到 {train['datetime'].max()}")
print(f"测试集日期范围: {test['datetime'].min()} 到 {test['datetime'].max()}")

# 数据标准化
print("\n开始数据标准化...")
# 选择需要的特征列
feature_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                   'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# 验证特征列是否存在
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    print(f"错误：以下特征列不存在于数据中: {missing_cols}")
    sys.exit(1)

# 对训练集和测试集分别进行标准化
scalers = {}
for col in feature_columns:
    scaler = MinMaxScaler(feature_range=(0, 1))
    train[col] = scaler.fit_transform(train[[col]])
    test[col] = scaler.transform(test[[col]])
    scalers[col] = scaler

# 创建序列数据
def create_sequences(data, seq_length, target_col='Global_active_power'):
    """创建用于LSTM的序列数据"""
    print(f"开始创建序列数据，序列长度: {seq_length}")
    print(f"数据形状: {data.shape}")
    
    # 验证目标列是否存在
    if target_col not in data.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在于数据中")
    
    # 检查是否有缺失值
    if data[feature_columns].isnull().any().any():
        print("警告: 数据中存在缺失值，这可能导致序列创建失败")
        print(data[feature_columns].isnull().sum())
        data = data.dropna(subset=feature_columns)
        print(f"删除缺失值后数据形状: {data.shape}")
    
    # 计算所需内存
    rows = len(data) - seq_length
    if rows <= 0:
        raise ValueError(f"序列长度 {seq_length} 大于或等于数据行数 {len(data)}")
    
    # 估算所需内存 (GB)
    memory_estimate = (rows * seq_length * len(feature_columns) * 8) / (1024**3)
    print(f"估算所需内存: {memory_estimate:.2f} GB")
    
    # 创建序列
    print("正在创建序列...")
    X, y = [], []
    start_time = time.time()
    for i in range(rows):
        if i % 10000 == 0 and i > 0:
            elapsed = time.time() - start_time
            progress = i / rows * 100
            estimated_total = elapsed / progress * 100
            remaining = estimated_total - elapsed
            print(f"已处理 {i}/{rows} 条序列 ({progress:.1f}%), 已用时: {elapsed:.1f}s, 预计还需: {remaining:.1f}s")
        X.append(data.iloc[i:i+seq_length][feature_columns].values)
        y.append(data.iloc[i+seq_length][target_col])
    
    print(f"序列创建完成，X形状: ({len(X)}, {seq_length}, {len(feature_columns)})")
    return np.array(X), np.array(y)

# 设置序列长度
seq_length = 24  # 使用前24小时的数据预测下一小时
print(f"\n创建序列数据，序列长度: {seq_length}")

# 创建训练和测试序列
try:
    X_train, y_train = create_sequences(train, seq_length)
    X_test, y_test = create_sequences(test, seq_length)
except Exception as e:
    print(f"序列创建失败: {e}")
    sys.exit(1)

print(f"训练序列形状: X={X_train.shape}, y={y_train.shape}")
print(f"测试序列形状: X={X_test.shape}, y={y_test.shape}")

# 创建数据加载器
class PowerConsumptionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建数据集和数据加载器
print("\n创建数据加载器...")
train_dataset = PowerConsumptionDataset(X_train, y_train)
test_dataset = PowerConsumptionDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(f"训练批次数量: {len(train_loader)}")
print(f"测试批次数量: {len(test_loader)}")

# 构建LSTM模型 (提供简化版选项)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, simplified=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.simplified = simplified
        
        if simplified:
            # 简化版模型，减少参数数量
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, output_size)
        else:
            # 原始模型
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 50),
                nn.ReLU(),
                nn.Linear(50, output_size)
            )
        
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取序列的最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 实例化模型
print("\n初始化模型...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

input_size = len(feature_columns)
hidden_size = 64  # 减少隐藏层大小以提高速度
num_layers = 1    # 减少LSTM层数以提高速度
output_size = 1
simplified_model = True  # 设置为True使用简化版模型

model = LSTMModel(input_size, hidden_size, num_layers, output_size, simplified=simplified_model).to(device)

# 打印模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数数量: {total_params:,}")

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模型训练 (带进度显示和时间估计)
def train_model(model, train_loader, criterion, optimizer, epochs, device):
    print(f"\n开始训练模型，共 {epochs} 轮...")
    total_batches = len(train_loader)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            batch_start_time = time.time()
            
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 打印批次进度
            batch_time = time.time() - batch_start_time
            batches_left = total_batches - (batch_idx + 1)
            estimated_time = batches_left * batch_time
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{total_batches}] ({progress:.1f}%), "
                      f"Loss: {loss.item():.4f}, 批次耗时: {batch_time:.2f}s, 预计剩余时间: {estimated_time/60:.2f}m")
        
        # 打印轮次信息
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / total_batches
        print(f"Epoch [{epoch+1}/{epochs}] 完成, 平均 Loss: {avg_loss:.4f}, 轮次耗时: {epoch_time:.2f}s, "
              f"预计剩余时间: {(epochs - (epoch+1)) * epoch_time/60:.2f}m")

# 训练模型
epochs = 20  # 减少训练轮数以提高速度
train_model(model, train_loader, criterion, optimizer, epochs, device)

# 模型评估
def evaluate_model(model, test_loader, criterion, device, scaler):
    print("\n开始评估模型...")
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            
            # 转换为numpy数组
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())
    
    # 反标准化
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    # 计算评估指标
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    print(f'评估结果:')
    print(f'Test MSE: {mse:.4f}')
    print(f'Test RMSE: {rmse:.4f}')
    print(f'Test MAE: {mae:.4f}')
    
    return predictions, actuals

# 评估模型
target_scaler = scalers['Global_active_power']
predictions, actuals = evaluate_model(model, test_loader, criterion, device, target_scaler)

# 可视化预测结果
def plot_predictions(actuals, predictions, n_points=200, title="电力消耗预测结果"):
    plt.figure(figsize=(14, 7))
    plt.plot(actuals[:n_points], label='实际值')
    plt.plot(predictions[:n_points], label='预测值', alpha=0.7)
    plt.title(title)
    plt.xlabel('时间')
    plt.ylabel('全局有功功率 (千瓦)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制预测结果
plot_predictions(actuals, predictions, title="电力消耗预测结果")

# 绘制更长时间范围的预测结果
plot_predictions(actuals, predictions, n_points=1000, title="电力消耗预测结果（更长时间范围）")