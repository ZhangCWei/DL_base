# 导入Python的时间模块，用于记录程序运行时间
import time

# 导入PyTorch库、PyTorch神经网络模块、PyTorch数据加载器、PyTorch视觉模型、以及数据集变换模块等
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms

# 导入Matplotlib绘图库和动画模块
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 导入IPython显示模块，用于在Jupyter Notebook中显示动画
from IPython.display import HTML

dataroot = "../CNN/MINIST/data/"  # 数据集所在的路径，我们事先下载下来的

#--- 定义了一些超参数 ---#
workers = 0         # 数据加载器使用的工作线程数, windows下需要置为0
batch_size = 100    # 数据加载器每批加载的样本数
image_size = 64     # 训练图像的大小
nc = 1              # 图像的通道数，因为MNIST数据集是灰度图像，所以通道数为1，彩色图像的话就是3
nz = 100            # 输入是100维的随机噪声z，看作是100个channel，每个特征图宽高是1*1
ngf = 64            # 生成器网络中特征图的大小
ndf = 64            # 判别器网络中特征图的大小
num_epochs = 10     # GAN模型的训练轮数
lr = 0.0002         # Adam优化器的学习率大小
beta1 = 0.5         # Adam优化器的beta1参数
ngpu = 0            # 可用 GPU的个数，0代表使用 CPU

#--- 定义训练集train_data---#
# 训练集加载并进行归一化等操作，包括训练集的根目录（root）、是否为训练集（train）、数据集变换（transform）
train_data = datasets.MNIST(
    root=dataroot,
    train=True,
    transform=transforms.Compose([
        transforms.Resize(image_size),          # 对数据进行了大小调整
        transforms.ToTensor(),                  # 转换为张量
        transforms.Normalize((0.5, ), (0.5, ))  # 标准化处理
    ]),
    download=True
)

#--- 定义测试集test_data ---#
test_data = datasets.MNIST(
    root=dataroot,
    train=False,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
)

#--- 训练数据集和测试数据集合并成一个数据集 ---#
dataset = train_data + test_data
print(f'Total Size of Dataset: {len(dataset)}') # Total Size of Dataset: 70000

#--- 数据集dataset加载到数据加载器dataloader中 ---#
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,  # 批量大小
    shuffle=True,           # 是否打乱数据
    num_workers=workers     # 使用的线程数量
)

#--- 选择计算设备为GPU(cuda:0)或CPU ---#
device = torch.device('cuda:0' if (
    torch.cuda.is_available() and ngpu > 0) else 'cpu')

#--- 定义权重初始化函数 ---#
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)   # 卷积层(Conv)的权重初始化为均值为0、标准差为0.02的正态分布随机数
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)   # 将批归一化层(BatchNorm)的权重初始化为均值为1、标准差为0.02的正态分布随机数
        nn.init.constant_(m.bias.data, 0)           # 将偏置初始化为0

# 这里的dataloader是之前创建的数据加载器，通过next(iter(dataloader))，可以从数据加载器中获取一个批次的数据。
# [0]表示只获取批次中的图像部分。inputs变量就包含了一个批次的图像数据
inputs = next(iter(dataloader))[0]
plt.figure(figsize=(10, 10))
plt.title("Training Images")
plt.axis('off') # 关闭坐标轴的显示，即不显示x轴和y轴的刻度和标签

# utils.make_grid函数将批次中的图像拼接成一个大的网格图像，nrow=10表示每行显示10张图像。inputs[:100]表示从整个批次中选择前100张图像，
# 然后乘以0.5并加上0.5的操作是将像素值从范围[-1, 1]转换为范围[0, 1]，以便正确显示图像
inputs = utils.make_grid(inputs[:100] * 0.5 + 0.5, nrow=10)

# inputs.permute(1, 2, 0)是将图像的维度顺序从(C, H, W)转换为(H, W, C)，以符合imshow函数的要求
plt.imshow(inputs.permute(1, 2, 0))

# 生成器
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 转置卷积层(反卷积层)，将输入的噪声向量z转换为特征图
            nn.ConvTranspose2d(
                in_channels=nz,     # 输入的通道数
                out_channels=ngf*8, # 输出的通道数
                kernel_size=4,      # 卷积核的大小
                stride=1,           # 步长
                padding=0,          # 填充大小
                bias=False          # 是否使用偏置项
            ),
            # 批归一化层
            # 用于对特征图进行归一化处理，增强网络的训练稳定性和生成图像的质量
            nn.BatchNorm2d(ngf*8),
            # ReLU激活函数层，用于引入非线性特性到生成器网络中
            nn.ReLU(True),
            # 以下代码将特征图逐渐上采样至更高的分辨率

            # 当前特征图大小(ngf*8)x4x4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            # 当前特征图大小 (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            # 当前特征图大小 (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 当前特征图大小 (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()

            # 当前特征图大小 (nc) x 64 x 64
        )

    # 前向传播方法，接收一个输入input(z)，通过main网络层生成与输入相同尺寸的图像
    def forward(self, input):
        return self.main(input)

# 判别器
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 当前特征图大小 (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # 当前特征图大小 (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            # 当前特征图大小 (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 当前特征图大小 (ndf*8) x 4 x 4
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),

            # 当前特征图大小 (1) x 1 x 1
            nn.Sigmoid()
        )

    # 前向传播方法，接收输入input(图像),通过main网络层对图像进行判别,输出0或1
    def forward(self, input):
        return self.main(input)


# 生成器初始化
# 创建生成器netG
netG = Generator(ngpu).to(device)
# 判断是否有多个GPU可用，如果有，则将生成器放到多卡并行模式中
if device.type == 'cuda' and ngpu > 1:
    netG = nn.DataParallel(netG, list(range(ngpu)))
# 对生成器的权重进行初始化,apply函数令weights_init函数自动遍历整个netG
netG.apply(weights_init)

# 判别器初始化
# 创建判别器netD
netD = Discriminator(ngpu).to(device)
# 多卡并行，如果有多卡的话
if device.type == 'cuda' and ngpu > 1:
    netD = nn.DataParallel(netD, list(range(ngpu)))
# 初始化权重
netD.apply(weights_init)


# 定义二元交叉熵损失函数,用于衡量生成器和判别器之间的误差
criterion = nn.BCELoss()

# 创建一批潜在向量,我们将使用它们来可视化生成器的生成过程
fixed_noise = torch.randn(100, nz, 1, 1, device=device)

real_label = 1.  # “真”标签
fake_label = 0.  # “假”标签

# 分别创建了生成器和判别器的优化器,使用Adam优化算法来更新生成器和判别器的参数,lr参数表示学习率，betas参数是Adam算法中的系数
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

# 定义变量, 存储每轮相关值
img_list = []   # 存储生成器在固定噪声上生成的图像
G_losses = []   # 存储生成器的损失值
D_losses = []   # 存储判别器的损失值
D_x_list = []   # 存储判别器在真实样本上的输出值
D_z_list = []   # 存储判别器在生成样本上的输出值
loss_tep = 10   # 用于保存最佳模型的阈值

print("Starting Training Loop...")

# 开始训练循环,循环迭代num_epochs次
for epoch in range(num_epochs):
    beg_time = time.time()
    # 循环迭代dataloader中每批次数据, data包含了图像数据和标签
    for i, data in enumerate(dataloader):
        # (1) 判别器训练: maximize log(D(x)) + log(1 - D(G(z)))

        # 判别器在真数据真标签上进行训练
        netD.zero_grad()                # 清零判别器梯度，准备开始一次反向传播
        real_cpu = data[0].to(device)   # 将真实图像数据移动到指定的设备上
        b_size = real_cpu.size(0)       # 获取当前批次的大小

        # 创建张量label，用于存放标签
        label = torch.full(
            (b_size, ),
            real_label,
            dtype=torch.float,
            device=device
        )

        output = netD(real_cpu).view(-1)        # 判别器推理
        errD_real = criterion(output, label)    # 使用二元交叉熵损失函数计算损失
        errD_real.backward()                    # 进行一次反向传播，计算判别器在真实数据上的梯度
        D_x = output.mean().item()              # 计算判别器在真实数据上的平均输出

        # 判别器在假数据假标签上进行训练
        # 利用生成器生成图像数据与对应标签
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)                      # 将噪声输入到生成器，生成假图像
        label.fill_(fake_label)                 # 使用假图片对应的假标签

        output = netD(fake.detach()).view(-1)   # 将生成的假图像输入到判别器，得到判别器的输出
        errD_fake = criterion(output, label)    # 计算判别器在假数据上的损失
        errD_fake.backward()                    # 进行一次反向传播，计算判别器在假数据上的梯度
        D_G_z1 = output.mean().item()           # 计算判别器在假数据上的平均输出

        errD = errD_real + errD_fake            # 计算判别器的总损失
        optimizerD.step()                       # 更新判别器的参数


        # (2) 生成器训练: maximize log(D(G(z)))
        # 真标签假图片,目的是使判别器能够返回假图像在变成真图像所需要拟合的变化
        netG.zero_grad()                        # 清零生成器的梯度，准备开始一次反向传播
        label.fill_(real_label)                 # 将标签中的"假"标签替换为"真"标签
        output = netD(fake).view(-1)            # 将生成的假图像再次输入到判别器，得到判别器的输出
        errG = criterion(output, label)         # 计算生成器在假数据上的损失函数
        errG.backward()                         # 反向传播，计算生成器在假数据上的梯度
        D_G_z2 = output.mean().item()           # 计算生成器对假数据的平均输出值
        optimizerG.step()                       # 更新生成器的参数

        end_time = time.time()                  # 记录当前epoch的结束时间
        run_time = round(end_time-beg_time)     # 计算当前epoch的训练时间

        # 打印每轮训练步骤的训练状态
        print(
            f'Epoch: [{epoch+1:0>{len(str(num_epochs))}}/{num_epochs}]',        # 当前训练的轮数
            f'Step: [{i+1:0>{len(str(len(dataloader)))}}/{len(dataloader)}]',   # 当前训练的步骤数
            f'Loss-D: {errD.item():.4f}',               # 判别器的损失值
            f'Loss-G: {errG.item():.4f}',               # 判生成器的损失值
            f'D(x): {D_x:.4f}',                         # 判别器在真实图像上的输出值
            f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]',    # 判别器在生成图像上的输出值
            f'Time: {run_time}s',                       # 每个步骤的运行时间
            end='\r'
        )

        # 将当前步骤的生成器和判别器的损失值添加到相应的列表中，以便后续绘制损失曲线
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # 将判别器在真实图像和生成图像上的输出值添加到相应的列表中，以便后续绘制判别器输出曲线
        D_x_list.append(D_x)
        D_z_list.append(D_G_z2)

        # 用于保存损失值低于阈值loss_tep的生成器模型。如果当前生成器的损失值较低，则将生成器的状态字典保存到文件model.pt中
        if errG < loss_tep:
            torch.save(netG.state_dict(), 'model.pt')
            temp = errG

    # 用于生成固定噪声输入上的生成图像，并将其添加到img_list中，以便后续绘制生成图像
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()  # 用生成器生成假数据，并将结果移动到CPU上
    img_list.append(utils.make_grid(fake * 0.5 + 0.5, nrow=10))  # 将生成器生成的假数据进行可视化，将可视化结果添加到img_list列表中
    print()  # 最后打印空行，用于格式化输出，使得每个epoch的训练统计信息在一行显示

# 绘制损失函数曲线
plt.title("Generator and Discriminator Loss During Training")
# 绘制生成器损失值和判别器损失值，其中[::100]表示每100个iterations绘制一个数据点，以便更清楚地观察损失函数的变化
plt.plot(G_losses[::100], label="G")
plt.plot(D_losses[::100], label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.axhline(y=0, label="0", c="g")  # 表示一个水平的渐近线。label="0"用于标识该渐近线，c="g"表示渐近线的颜色为绿色
plt.legend() # 显示图例，分别表示生成器和判别器的损失


fig = plt.figure(figsize=(10, 10)) # 设置图像尺寸10*10英寸
plt.axis("off") # 关闭坐标轴
# 循环遍历img_list中的每个图像，使用plt.imshow将其显示在图表中，用于生成动画。item.permute(1, 2, 0)用于调整图像的维度顺序
ims = [[plt.imshow(item.permute(1, 2, 0), animated=True)] for item in img_list]
# 创建一个艺术家动画对象，将ims中的图像转换为动画
ani = animation.ArtistAnimation(
    fig, # 图表对象
    ims, # 上方保存的图像列表
    interval=1000, # 每隔1000毫秒切换一次图像
    repeat_delay=1000, # 动画循环的延迟时间
    blit=True) # 提高动画的性能

HTML(ani.to_jshtml()) # 将动画对象转换为HTML格式，并使用HTML()进行显示

