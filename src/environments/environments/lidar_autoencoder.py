import scipy.signal
import torch
import torch.utils
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import scipy

# class LidarAE(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
         

#         self.encoder = torch.nn.Sequential(
#             torch.nn.Linear(682, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 16),
#         )
         
#         self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(16, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 256),
#             torch.nn.ReLU(),
#             torch.nn.Linear(256, 682)
#         )
 
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

class LidarConvAE(torch.nn.Module):
    def __init__(self):
        super(LidarConvAE, self).__init__()

        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=4, stride=4),  # [B, 1, 512] -> [B, 32, 128]
            torch.nn.ReLU(True),
            torch.nn.Conv1d(32, 64, kernel_size=4, stride=4),  # [B, 32, 128] -> [B, 64, 32]
            torch.nn.ReLU(True),
            torch.nn.Conv1d(64, 128, kernel_size=4, stride=4),  # [B, 64, 32] -> [B, 128, 8]
            torch.nn.ReLU(True),
            torch.nn.Flatten(), # Flatten [B, 128, 8] -> [B, 1024]             
            torch.nn.Linear(1024, 10)
        )

        # Decoder
        self.decoder = torch.nn.Sequential(  
            torch.nn.Linear(10,1024),
            torch.nn.Unflatten(-1,(128,8)),       
            torch.nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4),  # [B, 128, 8] -> [B, 64, 32]
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose1d(64, 32, kernel_size=4, stride=4),  # [B, 64, 32] -> [B, 32, 128]
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose1d(32, 1, kernel_size=4, stride=4),  # [B, 32, 128] -> [B, 1, 512]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class LidarDataSet(Dataset):
    def __init__(self,csv_file_path):
        print('loading csv')
        self.lidar_data = pd.read_csv(csv_file_path, na_values=['inf', '-inf'])
        self.lidar_data = self.lidar_data.replace(np.nan, -5)
        self.lidar_data = self.lidar_data.astype('float32')
        
        # self.lidar_data = np.genfromtxt('my_file.csv', delimiter=',', missing_values=["inf"], filling_values=[np.Infinity])
 
    
    def __len__(self):
        return len(self.lidar_data)

    def __getitem__(self, index):
        # return np.array(self.lidar_data.iloc[index].values)
        return np.array(scipy.signal.resample(self.lidar_data.iloc[index].values,512))

def main():

    BATCH_SIZE = 300
    LR = 1e-3
    WEIGHT_DECAY = 1e-8
    EPOCH = 5
    DATASET_PATH = "lidar_record.csv"
    MODEL_SAVE_PATH = "lidara_ae_18_1.pt"
 
    print("INIT")
    model = LidarConvAE()

    # from torchsummary import summary
    # summary(model,(1,512))
    # return

    dataset = LidarDataSet(DATASET_PATH)

    dataset_size = len(dataset)
    train_size = int(0.8*dataset_size)
    val_size = dataset_size - train_size

    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=True)

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Adam optimizer: learning rate:0.001, weight decay: 1e-8
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = LR,
                                weight_decay = WEIGHT_DECAY)

    training_loss = []
    validation_loss = []


    print("Loaded. Training.")


    for epoch_cnt in range(EPOCH):
        
        for scan_batch in train_loader:
            
            scan_batch:torch.TensorType= scan_batch
            
            # train batch
            model.train()

            scan_batch = scan_batch.unsqueeze(1)
            reconstructed = model(scan_batch)
            loss = loss_function(reconstructed, scan_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

            # evaluate model
            model.eval()
            validation_batch = next(iter(val_loader))
            validation_batch = validation_batch.unsqueeze(1)
            reconstructed = model(validation_batch)
            loss = loss_function(reconstructed, validation_batch)
            validation_loss.append(loss.item())
        
        print(f"Epoch {epoch_cnt+1} complete.")


    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(training_loss, label="Train")
    plt.plot(validation_loss, label="Validation")

    plt.show()

if __name__ == '__main__':
    main()