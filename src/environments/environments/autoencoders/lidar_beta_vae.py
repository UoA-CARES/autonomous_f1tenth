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
        return z.tolist()[0]

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

        #========= REPLACE NON HITTING RAY ==========
        self.lidar_data = self.lidar_data.replace(np.nan, -10)
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
    EPOCH = 15
    DATASET_PATH = "lidar_record_mix_ftg-rand.csv"
    MODEL_SAVE_PATH = "lidar_ae_ftg_rand.pt"
 
    # initialize model and set beta
    print("INIT")
    model = BetaVAE1D(1,10,beta=5) #0.00002

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
    model = BetaVAE1D(1,10) 
    model.load_state_dict(torch.load("/home/anyone/autonomous_f1tenth/lidar_ae_ftg_rand.pt"))
    model.eval()

    theta = np.linspace(-120,120,512)
    theta = np.deg2rad(theta)

    global r
    r = np.linspace(6,6,512)

    sample_non_hit = -10
    example_in = [1.49526345729828,1.4823591709137,1.47173547744751,1.47202587127686,1.47797513008118,1.4761198759079,1.46837615966797,1.45316922664642,1.43776881694794,1.43689095973969,1.4214586019516,1.43239748477936,1.40211737155914,1.39478147029877,1.42662858963013,1.42528975009918,1.41674101352692,1.42280042171478,1.40411198139191,1.40746831893921,1.37503695487976,1.38143658638,1.38090300559998,1.37146699428558,1.39131760597229,1.3757518529892,1.35846602916718,1.36436820030212,1.35873281955719,1.34672391414642,1.33636069297791,1.35249149799347,1.35538411140442,1.33621156215668,1.34596395492554,1.35637128353119,1.34588098526001,1.32197701931,1.32585191726685,1.34662771224976,1.32827389240265,1.31406807899475,1.33942234516144,1.31314146518707,1.33348774909973,1.31997525691986,1.3084557056427,1.3179703950882,1.30070865154266,1.3235992193222,1.31651258468628,1.30870306491852,1.29691541194916,1.31798112392426,1.27927887439728,1.31239247322083,1.2925295829773,1.28996860980988,1.28759062290192,1.28042602539063,1.29998791217804,1.29372584819794,1.2902809381485,1.28821325302124,1.28216350078583,1.27794277667999,1.27953410148621,1.2795135974884,1.2839949131012,1.2821456193924,1.27052652835846,1.2766056060791,1.25835812091827,1.26961350440979,1.25974357128143,1.2831859588623,1.27136588096619,1.26444482803345,1.2698769569397,1.25562989711761,1.26449930667877,1.26999115943909,1.27302503585815,1.26920962333679,1.2572512626648,1.26580250263214,1.26958572864532,1.2677104473114,1.26675164699554,1.26930820941925,1.25023591518402,1.27993524074554,1.25836670398712,1.26163184642792,1.26144599914551,1.25555384159088,1.25843143463135,1.29203295707703,1.29187774658203,1.28365206718445,1.26794278621674,1.27274596691132,1.25784420967102,1.26945447921753,1.25285363197327,1.27795743942261,1.28350377082825,1.26474511623383,1.26517570018768,1.27466762065887,1.26387548446655,1.28026914596558,1.28155553340912,1.29362690448761,1.27361762523651,1.28735327720642,1.27843034267426,1.28759467601776,1.28593897819519,1.27785444259644,1.29297626018524,1.27557826042175,1.30616426467896,1.30474364757538,1.30535614490509,1.29046738147736,1.30335867404938,1.29062187671661,1.30779767036438,1.3089371919632,1.32630085945129,1.31374621391296,1.30192506313324,1.30272054672241,1.30273711681366,1.33553636074066,1.31170296669006,1.31407630443573,1.33232164382935,1.32439541816711,1.31993842124939,1.33610904216766,1.34580862522125,1.33151078224182,1.34018921852112,1.35947477817535,1.3327454328537,1.33713209629059,1.34920883178711,1.35484826564789,1.34095287322998,1.35963261127472,1.35507309436798,1.3632618188858,1.35101163387299,1.36893105506897,1.36559092998505,1.3671863079071,1.37300157546997,1.37353265285492,1.38381624221802,1.38271367549896,1.39419257640839,1.37247848510742,1.38933634757996,1.37187230587006,1.39634811878204,1.3834764957428,1.41726732254028,1.39389824867249,1.43153405189514,1.42084503173828,1.43273937702179,1.44349431991577,1.42405164241791,1.44468462467194,1.44010102748871,1.44707441329956,1.46363174915314,1.44273138046265,1.4505740404129,1.47224235534668,1.47325086593628,1.46238124370575,1.46295869350433,1.49065780639648,1.48021352291107,1.49798321723938,1.50681602954865,1.51024949550629,1.50591313838959,1.53576755523682,1.55032360553741,1.53693068027496,1.54680562019348,1.55107927322388,1.55686390399933,1.55307745933533,1.56333363056183,1.5722119808197,1.55514419078827,1.54698121547699,1.59358048439026,1.57834303379059,1.57411503791809,1.61476314067841,1.60051262378693,1.64529180526733,1.6336327791214,1.63699686527252,1.63027358055115,1.63378894329071,1.65617024898529,1.67237818241119,1.68033409118652,1.6981680393219,1.6927752494812,1.71071743965149,1.73388350009918,1.7092547416687,1.71905958652496,1.73506414890289,1.73974657058716,1.77489924430847,1.77347457408905,1.78583526611328,1.78653120994568,1.80356168746948,1.83282387256622,1.8291277885437,1.83425760269165,1.83768332004547,1.84813964366913,1.88450229167938,1.88374269008636,1.90680468082428,1.89866518974304,1.91149938106537,1.92370200157166,1.94886386394501,1.94837439060211,1.96648228168488,1.98422884941101,2.00843501091003,2.00000476837158,2.03153347969055,2.03905916213989,2.07301735877991,2.07369780540466,2.0837562084198,2.13233780860901,2.12228775024414,2.15736746788025,2.16671180725098,2.18446373939514,2.20700812339783,2.21379899978638,2.23736763000488,2.24473643302917,2.25844120979309,2.2874231338501,2.30762243270874,2.30840039253235,2.348149061203,2.35723805427551,2.39777588844299,2.38308048248291,2.4119086265564,2.43375372886658,2.4674026966095,2.48277831077576,2.5171377658844,2.53805303573608,2.56681561470032,2.577312707901,2.61600780487061,2.6543698310852,2.66264247894287,2.70949959754944,2.71909761428833,2.76518821716309,2.77052712440491,2.81799364089966,2.83580160140991,2.86198806762695,2.89820671081543,2.92380166053772,2.97313117980957,3.02592921257019,3.0682258605957,3.06234049797058,3.09479188919067,3.15004634857178,3.2207350730896,3.23427748680115,3.30335927009583,3.33277750015259,3.35491919517517,3.39710354804993,3.45014762878418,3.5193464756012,3.58290457725525,3.65323972702026,3.65040755271912,3.71982073783875,3.79947209358215,3.89404797554016,3.98074436187744,4.05454921722412,4.04988098144531,4.16533660888672,4.24923896789551,4.35132169723511,4.46564722061157,4.44332075119019,4.59479808807373,4.72775650024414,4.8536548614502,4.98721408843994,4.97459125518799,5.11502027511597,5.29878902435303,5.46503591537476,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,sample_non_hit,5.47749662399292,5.25104665756226,5.04421663284302,4.87958431243897,4.87986326217651,4.72998380661011,4.56512355804443,4.43510723114014,4.31963014602661,4.18603849411011,4.04726362228394,3.926349401474,3.80534505844116,3.81275010108948,3.68644738197327,3.57590198516846,3.47956871986389,3.38459515571594,3.2954409122467,3.23705649375916,3.14235782623291,3.07424092292786,3.02215456962585,2.92536640167236,2.86646318435669,2.81069993972778,2.76818776130676,2.72882127761841,2.66902923583984,2.60956692695618,2.54555606842041,2.52418828010559,2.46717262268066,2.43987202644348,2.39277768135071,2.36316323280334,2.32025527954102,2.28605318069458,2.26730823516846,2.23683214187622,2.19598603248596,2.14360857009888,2.13269782066345,2.12056422233582,2.08789920806885,2.03146004676819,2.04348969459534,2.00431203842163,1.94253957271576,1.96023666858673,1.91998136043549,1.91714942455292,1.88745951652527,1.87153005599976,1.82949709892273,1.82301509380341,1.79920828342438,1.76370096206665,1.76779401302338,1.72515857219696,1.70189762115479,1.68580412864685,1.67381823062897,1.66429328918457,1.63062942028046,1.64186573028564,1.60507225990295,1.58989334106445,1.58084833621979,1.56362414360046,1.55339455604553,1.52504336833954,1.51230359077454,1.50337839126587,1.49167740345001,1.50308358669281,1.46236455440521,1.46331131458282,1.45525777339935,1.42971956729889,1.41570425033569,1.41101205348969,1.41739785671234,1.39254629611969,1.38318729400635,1.37521433830261,1.3695592880249,1.36164021492004,1.33871448040009,1.3212902545929,1.3131000995636,1.31383037567139,1.28843986988068,1.30349969863892,1.28511583805084,1.2742931842804,1.26830863952637,1.25823581218719,1.25074350833893,1.2555114030838,1.25632834434509,1.22964513301849,1.23250842094421,1.2341206073761,1.22839295864105,1.22040557861328,1.22391176223755,1.18344068527222,1.17762100696564,1.17765378952026,1.19031763076782,1.16365838050842,1.16937935352325,1.13551568984985,1.16447162628174,1.14946305751801,1.14500713348389,1.14861309528351,1.12098276615143,1.13036000728607,1.12842655181885,1.13849711418152,1.12206935882568,1.10883414745331,1.11911725997925,1.12046456336975,1.09268522262573,1.07481420040131,1.09639549255371,1.08793067932129,1.08779168128967,1.07928621768951,1.08232820034027,1.08780312538147,1.07588958740234,1.06809961795807,1.05240905284882,1.0696074962616,1.04350531101227,1.05172395706177,1.03428411483765,1.04434263706207,1.04139614105225,1.04684197902679,1.04681575298309,1.01796567440033,1.03181779384613,1.02468800544739,1.02860748767853,1.04080629348755,1.0179877281189,1.03095412254334,1.00511133670807,1.01785862445831,0.99866795539856,1.01578629016876,1.01013994216919,0.999418139457703,1.00316047668457,0.990642547607422,0.998843491077423,1.00399196147919,0.99013352394104,0.963823080062866,0.972705781459808,0.974676251411438,0.991172254085541,0.984543383121491,0.976840794086456,0.972028613090515,0.979664742946625,0.96641331911087,0.987993061542511,0.971717655658722,0.971064627170563,0.947444975376129,0.961677670478821,0.969362437725067,0.95258903503418,0.969769239425659,0.959586322307587,0.958484470844269,0.960296273231506,0.950118362903595,0.948909640312195,0.931155204772949,0.956961870193481,0.953459143638611,0.954306066036224,0.952421009540558,0.924330651760101,0.950810432434082,0.945943832397461,0.951357901096344,0.95478230714798,0.927935600280762,0.931201040744782,0.925460636615753,0.935665667057037,0.941974341869354,0.958449304103851,0.952643275260925,0.9432652592659,0.922052264213562,0.937949240207672,0.938230872154236,0.935936391353607,0.927850008010864,0.937244236469269,0.918153584003449,0.924954831600189,0.94192761182785,0.9355309009552,0.935993671417236,0.919784665107727,0.940152406692505,0.938108682632446,0.945896208286285,0.940702676773071,0.924436032772064,0.931939542293549,0.930129647254944,0.923890292644501,0.926809012889862,0.946922659873962,0.947165668010712,0.92166006565094,0.949771523475647,0.947425007820129,0.926158785820007,0.940540492534637,0.941286385059357,0.947126269340515,0.935090720653534,0.928309261798859,0.950974762439728,0.937844157218933,0.944136381149292,0.938311219215393,0.928596794605255,0.944227278232574,0.95948988199234,0.946469187736511,0.948937654495239,0.938698828220367,0.947250723838806,0.948572635650635,0.952847123146057,0.951454520225525,0.93832004070282,0.948463976383209,0.978358089923859,0.960850298404694,0.97469025850296,0.940527260303497,0.947890102863312,0.962229430675507,0.973747372627258,0.957586526870728,0.986732959747314,0.96146947145462,0.973540484905243,0.97641533613205,0.986945390701294,0.961026430130005,0.968965113162994,0.98871123790741,0.971937835216522,0.989520728588104,0.986837029457092,0.981703221797943,0.987387835979462,0.986851930618286,1.00387763977051,0.997894108295441,1.00789678096771,1.01997494697571,0.997544527053833,1.00592684745789,1.01184332370758,1.00069308280945,1.01571238040924,1.01648116111755,1.01492691040039,1.03750586509705,1.03140532970428,1.05738878250122,1.0191068649292,1.02965462207794,1.04322302341461,1.0358110666275,1.04125809669495,1.0350900888443,1.07065010070801]
    example_in = scipy.signal.resample(example_in,512)
    #z = [-0.09051799774169922, 0.33830809593200684, -0.6161627769470215, 0.08786570280790329, -0.0008583962917327881, -0.05810356140136719, -0.6939213275909424, 0.27182549238204956, 1.030595064163208, 1.5578203201293945]
    z = model.get_latent(torch.tensor(example_in, dtype=torch.float32).unsqueeze(0).unsqueeze(0))
    print(z)

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
        scatter.set_offsets(np.c_[theta,r])
        # fig.canvas.draw_idle()
        fig.canvas.draw()

    # Create 10 sliders
    sliders = []
    for i in range(10):
        ax_slider = plt.axes([0.25, 0.05 + i * 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, f'Slider {i+1}', -10, 10, valinit=z[i])
        sliders.append(slider)
        slider.on_changed(update)

    

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main2()