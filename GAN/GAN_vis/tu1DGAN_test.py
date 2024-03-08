import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 参数设置
num_epochs = 5000
batch_size = 64
noise_dim = 10
lr = 0.001
snapshot_interval = 100

# 真实数据分布
mu, sigma = 0, 1

# 生成器和判别器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

generator = Generator()
discriminator = Discriminator()

# 优化器
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)

# 损失函数
loss = nn.BCELoss()

# 训练GAN，并存储生成数据的快照
snapshots = []

for epoch in range(num_epochs):
    # Train the discriminator
    real_data = torch.tensor(np.random.normal(mu, sigma, (batch_size, 1)), dtype=torch.float32)
    real_labels = torch.ones(batch_size, 1)

    noise = torch.randn(batch_size, noise_dim)
    generated_data = generator(noise)
    fake_labels = torch.zeros(batch_size, 1)

    dis_real_out = discriminator(real_data)
    dis_real_loss = loss(dis_real_out, real_labels)
    dis_fake_out = discriminator(generated_data.detach())
    dis_fake_loss = loss(dis_fake_out, fake_labels)
    dis_loss = dis_real_loss + dis_fake_loss

    dis_optimizer.zero_grad()
    dis_loss.backward()
    dis_optimizer.step()

    # Train the generator
    noise = torch.randn(batch_size, noise_dim)
    generated_data = generator(noise)
    dis_out = discriminator(generated_data)
    gen_loss = loss(dis_out, real_labels)

    gen_optimizer.zero_grad()
    gen_loss.backward()
    gen_optimizer.step()

    if epoch % snapshot_interval == 0:
        num_samples = 1000
        noise = torch.randn(num_samples, noise_dim)
        generated_data = generator(noise).detach().numpy()
        snapshots.append(generated_data)

# 动画展示
fig, ax = plt.subplots()
x = np.linspace(-5, 5, 1000)
y = (1 / (np.sqrt(2 * np.pi * sigma**2))) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def update(frame):
    plt.cla()
    ax.hist(snapshots[frame], bins=50, density=True, alpha=0.6, label=f'Epoch {frame * snapshot_interval}')
    ax.plot(x, y, linewidth=2, label='Real Data')
    ax.legend()
    ax.set_ylim(0, 0.8)
    ax.set_xlabel('Data')
    ax.set_ylabel('Probability Density')

ani = FuncAnimation(fig, update, frames=len(snapshots), interval=200, repeat=True)
ani.save('gan_training_animation.gif', writer='imagemagick')

plt.title('1D GAN Visualization During Training')
plt.show()

