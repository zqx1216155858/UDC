
class Options():
    def __init__(self):
        super().__init__()
        # 超参数
        self.Seed = 1234  # 随机种子
        self.Epoch = 1000  # 训练次数
        self.Learning_Rate = 2e-4  # 每个阶段学习率从头开始
        self.Batch_Size_Train = 20  # 每次训练加载的图片数量
        self.Batch_Size_Val = 10
        self.Patch_Size_Train = 256  # 训练时裁剪的图片大小
        self.Patch_Size_Val = 256
        # 训练集路径
        self.Input_Path_Train = "/storage/public/home/2022124023/project/DLGNet/datasets/training/Poled/train/input"
        self.Target_Path_Train = "/storage/public/home/2022124023/project/DLGNet/datasets/training/Poled/train/target"
        # 验证集路径，有时数据集没有划分验证集，可以直接拿测试集做验证集,但在论文里不能说这个
        self.Input_Path_Val = "/storage/public/home/2022124023/project/DLGNet/datasets/training/Poled/test/input"
        self.Target_Path_Val = "/storage/public/home/2022124023/project/DLGNet/datasets/training/Poled/test/target"
        # 测试集路径
        self.Input_Path_Test = "/storage/public/home/2022124023/project/DLGNet/datasets/training/Poled/test/input"
        self.Target_Path_Test = "/storage/public/home/2022124023/project/DLGNet/datasets/training/Poled/test/target"
        self.Result_Path_Test = "/storage/public/home/2022124023/project/DLGNet/datasets/training/Poled/test/result/"
        # 权重参数保存路径
        self.MODEL_SAVE_PATH = '/storage/public/home/2022124023/project/DLGNet/pth/DLGNet_poled.pth'

        # 线程数，如果报错页面文件太小，把这个值设为 0
        self.Num_Works = 2
        # 是否使用 CUDA
        self.CUDA_USE = True