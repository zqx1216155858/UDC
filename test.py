import sys
import time
from tqdm import tqdm  # 进度条
from torch.utils.data import DataLoader
from torchvision.utils import save_image
# 自定义
from model import DLGNet
from datasets import *
from utils import *
from options import Options
if __name__ == '__main__':

    opt = Options()  # 超参数配置

    inputPathTest = opt.Input_Path_Test  # 测试输入图片路径
    resultPathTest = opt.Result_Path_Test  # 测试目标图片路径

    myNet = DLGNet()
    if opt.CUDA_USE:
        myNet = myNet.cuda()  # 网络放入GPU中
    myNet = nn.DataParallel(myNet)
    # 测试数据
    datasetTest = MyTestDataSet(inputPathTest)  # 实例化测试数据集类
    # 可迭代数据加载器加载测试数据
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False,
                            num_workers=opt.Num_Works, pin_memory=True)

    # 测试;
    print('--------------------------------------------------------------')
    # 加载已经训练好的模型参数
    if opt.CUDA_USE:
        myNet.load_state_dict(torch.load(opt.MODEL_SAVE_PATH))
    else:
        myNet.load_state_dict(torch.load(opt.MODEL_SAVE_PATH, map_location=torch.device('cpu')))
    myNet.eval()  # 指定网络模型测试状态

    with torch.no_grad():  # 测试阶段不需要梯度
        timeStart = time.time()  # 测试开始时间
        for index, (x, name) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()  # 释放显存

            input_test = x.cuda() if opt.CUDA_USE else x   # 放入GPU

            # input_test, pad_size = pad(input_test, factor=16)  # 将输入补成 16 的倍数
            output_test = myNet(input_test)  # 输入网络，得到输出
            # output_test = unpad(output_test, pad_size)  # 将补上的像素去掉，保持输出输出大小一致

            save_image(output_test, resultPathTest + name[0])  # 保存网络输出结果
        timeEnd = time.time()  # 测试结束时间
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))
