import torch
from torch.utils.data import Dataset
from torch.utils import data
from torch.optim import Adam
from model import Diffusion,p_losses,sample
import numpy as np
model = Diffusion()
optimizer = Adam(model.parameters(), lr=2e-4)

timesteps =1000
import soundfile as sf
filename_cargo_15 = r"DeepShip-main\Cargo\15.wav"
wav_cargo_15,sr = sf.read(filename_cargo_15)
filename_Passengership_1 = r"DeepShip-main\Passengership\1.wav"
wav_Passengership_1,sr = sf.read(filename_Passengership_1)
filename_Tanker_2 = r"DeepShip-main\Tanker\2.wav"
wav_Tanker_2,sr = sf.read(filename_Tanker_2)
filename_Tug_2 = r"DeepShip-main\Tug\9.wav"
wav_Tug_9,sr = sf.read(filename_Tug_2)


ns = np.linspace(0, 1024, 1024)
#])#,wav_Tanker_2[:1024],wav_Tug_9[:1024]]
data_single =np.array([wav_cargo_15[:1024],wav_Passengership_1[:1024],wav_Tanker_2[:1024],wav_Tug_9[:1024]])

class dataset_without_label(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx])

        # 在这里可以添加预处理步骤，比如将文本转换为ID列表等
        return text


dataset = dataset_without_label(10 * data_single[:, None, None, ...])

from torchvision.utils import save_image

epochs = 100000
device = "cuda"
model = model.to(device)
for epoch in range(epochs):
    for step, batch in enumerate(dataset):
        optimizer.zero_grad()
        images = batch.to(dtype=torch.float32)

        images = images.to(device)
        print(images.shape)
        t = torch.randint(0, timesteps, (len(images),), device=device).long()

        loss = p_losses(model, images, t)

        if step % 100 == 0:
            print(f"Epoch: {epoch}, step: {step} -- Loss: {loss.item():.3f}")

        loss.backward()
        optimizer.step()

device = "cuda"
model=model.to(device)
samples = sample(model, image_size=1024, batch_size=1, channels=1)

# Get the last sample and normalize it in [0,1]
# last_sample = (samples[-1] - samples[-1].min())/(samples[-1].max()-samples[-1].min())
last_sample = samples[-1]
print(len(samples))
print(last_sample.shape)

model.load_state_dict(torch.load('Signal1024_4_ship_diffusion_model_epoch_100000_dataset_1_sample_1000_time.pth'))