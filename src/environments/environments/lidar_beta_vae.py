import scipy.signal
import torch
import torch.utils
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import scipy
from torch import nn
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple

class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()
    def encode(self, input: torch.tensor) -> List[torch.tensor]:
        raise NotImplementedError
    def decode(self, input: torch.tensor) -> Any:
        raise NotImplementedError
    def sample(self, batch_size:int, current_device: int, **kwargs) -> torch.tensor:
        raise NotImplementedError
    def generate(self, x: torch.tensor, **kwargs) -> torch.tensor:
        raise NotImplementedError
    @abstractmethod
    def forward(self, *inputs: torch.tensor) -> torch.tensor:
        pass
    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> torch.tensor:
        pass

class BetaVAE1D(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 num_of_filters: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'H',
                 **kwargs) -> None:
        super(BetaVAE1D, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if num_of_filters is None:
            # num_of_filters = [32, 64, 128, 256, 512]
            num_of_filters = [32, 64, 128, 256]
        if in_channels is None:
            in_channels = 1

        # Build Encoder
        for filter_num in num_of_filters:
            # B,1,512 -> B,32,128 -> B,64,32 -> B,128,8 -> B,256,2
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=filter_num,
                              kernel_size=4, stride= 4), #kernel:3, stride:2, padding:1
                    nn.BatchNorm1d(filter_num),
                    nn.LeakyReLU())
            )
            in_channels = filter_num

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, 512)

        num_of_filters.reverse()

        for i in range(len(num_of_filters) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(num_of_filters[i],
                                       num_of_filters[i + 1],
                                       kernel_size=4,
                                       stride = 4,),
                    nn.BatchNorm1d(num_of_filters[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose1d(num_of_filters[-1],
                                               num_of_filters[-1],
                                               kernel_size=4,
                                               stride=4,
                                               output_padding=1),
                            nn.BatchNorm1d(num_of_filters[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(num_of_filters[-1], out_channels= 1,
                                      kernel_size= 4, padding=1)
                            # nn.Sigmoid()
                            #nn.Tanh()
                            )

    def encode(self, input: torch.tensor) -> List[torch.tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x 1 x features]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        '''
        :param x: (Tensor) [B x C x features]
        :return: (Tensor) [B x C x features]
        '''
        result = self.decoder_input(z) # (... ,512) latent_dim -> 512
        # result = result.view(-1, 256,2)
        result = torch.unflatten(result,-1, (256,2))
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.Tensor:
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.tensor, **kwargs) -> torch.tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]
    
    def get_latent(self, input:torch.Tensor):
        ''':param input: (Tensor) Input tensor to encoder [B x 1 x features]'''
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z.tolist()

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = 1  #self.latent_dim / 512 / 300 #kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =torch.nn.functional.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> torch.tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.tensor, **kwargs) -> torch.tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x features]
        :return: (Tensor) [B x C x features]
        """

        return self.forward(x)[0]

########################################################################################
############### TRAINING UTILITIES #####################################################
########################################################################################

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
    LR = 1e-4
    WEIGHT_DECAY = 1e-8
    EPOCH = 5
    DATASET_PATH = "lidar_record_mix_ftg-rand.csv"
    MODEL_SAVE_PATH = "lidar_ae_ftg_rand.pt"
 
    print("INIT")
    model = BetaVAE1D(1,10,beta=0.01) #0.00002

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

            scan_batch = scan_batch.unsqueeze(1) # [B, 512] -> [B, 1, 512]
            reconstructed = model(scan_batch) #[self.decode(z), input, mu, log_var]
            loss = model.loss_function(*reconstructed)
            optimizer.zero_grad()
            loss['loss'].backward()
            print(f"{round(loss['Reconstruction_Loss'].item(),3)} ; {round(loss['KLD'].item(),3)}")
            optimizer.step()
            # training_loss.append(loss['loss'].item())

            # evaluate model
            model.eval()
            validation_batch = next(iter(val_loader))
            validation_batch = validation_batch.unsqueeze(1) # [B, 512] -> [B, 1, 512]
            reconstructed = model(validation_batch) # [self.decode(z), input, mu, log_var]
            loss = model.loss_function(*reconstructed)
            # validation_loss.append(loss['loss'])
        
        print(f"Epoch {epoch_cnt+1} complete.")


    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(training_loss, label="Train")
    plt.plot(validation_loss, label="Validation")

    plt.show()

def main2():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.widgets import Button, Slider

    MODEL_SAVE_PATH = "lidar_ae_ftg_rand.pt"
    model = BetaVAE1D(1,10,beta=0.01) #0.00002
    model.load_state_dict(torch.load("/home/anyone/autonomous_f1tenth/lidar_ae_ftg_rand.pt"))
    model.eval()

    theta = np.linspace(-120,120,512)
    theta = np.deg2rad(theta)

    global r
    r = np.linspace(0.6,5,512)
    z = [-0.09051799774169922, 0.33830809593200684, -0.6161627769470215, 0.08786570280790329, -0.0008583962917327881, -0.05810356140136719, -0.6939213275909424, 0.27182549238204956, 1.030595064163208, 1.5578203201293945]

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax = fig.add_subplot(projection='polar')
    scatter = ax.scatter(theta, r, s=1, cmap='hsv', alpha=0.75)

    plt.subplots_adjust(left=0.25, bottom=0.6)

    def get_slider_values():
        return [slider.val for slider in sliders]
    
    def update(val):
        global r
        z = get_slider_values()
        r=model.decode(torch.tensor(z,dtype=torch.float32).unsqueeze(0)).tolist()[0][0]
        print(r)
        scatter.set_offsets(np.c_[theta,r])
        # fig.canvas.draw_idle()
        fig.canvas.draw()

    # Create 10 sliders
    sliders = []
    for i in range(10):
        ax_slider = plt.axes([0.25, 0.05 + i * 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, f'Slider {i+1}', -1.9, 1.9, valinit=z[i])
        sliders.append(slider)
        slider.on_changed(update)

    

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main2()