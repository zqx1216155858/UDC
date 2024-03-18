import time
import torch.nn as nn
from tqdm import tqdm  # 进度条
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
# 自定义
from utils import torchPSNR, PSNRLoss
from model import DLGNet
from datasets import *
from options import Options
from DiscriminatorSN import UNetDiscriminatorSN, weight_init
from basicsr.losses.MPRNetloss import LS_D_Loss, LS_G_Loss, Pair_Loss
from basicsr.train.DLGNet_train import train_D, train_G



if __name__ == '__main__':  # 只有在 main 中才能开多线程

    opt = Options()  # 超参数配置
    cudnn.benchmark = True
    cnt = 0  # 一个用来保存参数文件的计数
    best_psnr = 0  # 训练过程在验证集上最好的 psnr

    random.seed(opt.Seed)  # 随机种子
    torch.manual_seed(opt.Seed)
    torch.cuda.manual_seed(opt.Seed)
    torch.manual_seed(opt.Seed)
    EPOCH = opt.Epoch  # 训练次数
    best_epoch = 0  # 效果最好的 epoch
    # 阶段训练策略
    BATCH_SIZE_TRAIN = opt.Batch_Size_Train
    BATCH_SIZE_VAL = opt.Batch_Size_Val
    PATCH_SIZE_TRAIN = opt.Patch_Size_Train
    PATCH_SIZE_VAL = opt.Patch_Size_Val
    LEARNING_RATE = opt.Learning_Rate  # 学习率

    inputPathTrain = opt.Input_Path_Train  # 训练输入图片路径
    targetPathTrain = opt.Target_Path_Train  # 训练目标图片路径
    inputPathVal = opt.Input_Path_Val  # 测试输入图片路径

    targetPathVal = opt.Target_Path_Val  # 测试目标图片路径

    criterion_psnr = PSNRLoss()  # PSNR 损失函数
    if opt.CUDA_USE:
        criterion_psnr = criterion_psnr.cuda()

    Net_G = DLGNet()  # 实例化网络
    Net_D = UNetDiscriminatorSN()
    Net_G.apply(weight_init)
    Net_D.apply(weight_init)
    # 多卡训练，自行判断有几张显卡可以用
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    Net_G = nn.DataParallel(Net_G, device_ids)
    Net_D = nn.DataParallel(Net_D, device_ids)
    if opt.CUDA_USE:
        Net_G = Net_G.cuda()  # 网络放入GPU中
        Net_D = Net_D.cuda()
    #LOSS
    D_Loss = LS_D_Loss().cuda()
    G_Loss = LS_G_Loss().cuda()
    pair_loss = Pair_Loss().cuda()

    # 网络参数优化算法
    optimizerG = torch.optim.Adam(Net_G.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(Net_D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    # 学习率调整策略
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, 50, 1e-6)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, 50, 1e-5)


    # 训练数据
    datasetTrain = MyTrainDataSet(inputPathTrain, targetPathTrain, patch_size=PATCH_SIZE_TRAIN)  # 实例化训练数据集类
    # 可迭代数据加载器加载训练数据

    trainLoader = DataLoader(dataset=datasetTrain, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                             drop_last=True, num_workers=opt.Num_Works, pin_memory=True)

    # 评估数据
    datasetValue = MyValueDataSet(inputPathVal, targetPathVal, patch_size=PATCH_SIZE_VAL)  # 实例化评估数据集类
    valueLoader = DataLoader(dataset=datasetValue, batch_size=BATCH_SIZE_VAL, shuffle=True,
                             drop_last=True, num_workers=opt.Num_Works, pin_memory=True)

    # 开始训练
    print('-------------------------------------------------------------------------------------------------------')
    if os.path.exists(opt.MODEL_SAVE_PATH):  # 判断是否预训练
        if opt.CUDA_USE:  # 加载 CUDA 下训练的模型
            Net_G.load_state_dict(torch.load(opt.MODEL_SAVE_PATH))
        else:  # 转到 CPU 模型加载
            Net_G.load_state_dict(torch.load(opt.MODEL_SAVE_PATH, map_location=torch.device('cpu')))







    #---->train
    for epoch in tqdm(range(1,2001)):
        timeStart = time.time()  # 每次训练开始时间
        for data in tqdm(trainLoader):
            train_D(data,Net_G,Net_D,D_Loss,optimizerD)
            train_D(data,Net_G,Net_D,D_Loss,optimizerD)
            train_D(data,Net_G,Net_D,D_Loss,optimizerD)
            train_D(data,Net_G,Net_D,D_Loss,optimizerD)
            train_D(data,Net_G,Net_D,D_Loss,optimizerD)
            train_D(data,Net_G,Net_D,D_Loss,optimizerD)
            train_D(data,Net_G,Net_D,D_Loss,optimizerD)
            train_D(data,Net_G,Net_D,D_Loss,optimizerD)
            train_D(data,Net_G,Net_D,D_Loss,optimizerD)
            train_D(data,Net_G,Net_D,D_Loss,optimizerD)
            train_G(data,Net_G,Net_D,G_Loss,pair_loss,optimizerG)

        if epoch%3==0: # 每三步在验证集上评估一下，目的是能知道中间训练结果，但又不至于每次训练都验证来增加训练时间
            Net_G.eval()  # 指定网络模型验证状态
            psnr_val_rgb = []
            for val in tqdm(valueLoader):
                input_, target_value = (val[0].cuda(), val[1].cuda()) if opt.CUDA_USE else (val[0], val[1])
                with torch.no_grad():
                    output_value = Net_G(input_)
                for output_value, target_value in zip(output_value, target_value):

                    psnr_val_rgb.append(torchPSNR(output_value, target_value))
            # print(psnr_val_rgb)
            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb >= best_psnr:  # 保存训练最好的模型参数
                best_psnr = psnr_val_rgb
                torch.save(Net_G.state_dict(), opt.MODEL_SAVE_PATH)  # 保存验证集上效果最好的一步参数文件
            schedulerD.step()
            schedulerG.step()
            timeEnd = time.time()  # 每次训练结束时
            print("------------------------------------------------------------")
            print("Epoch:  {}  Finished,  Time:  {:.4f} s,   current psnr:  {:.3f}, best psnr:  {:.3f}.".format(
                    epoch + 1, timeEnd - timeStart, psnr_val_rgb, best_psnr))

        
