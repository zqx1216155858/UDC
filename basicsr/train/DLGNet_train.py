import torch


def train_G(data,netG,netD,G_Loss,Pair_Loss,optimizerG):
    netG.train()
    netD.train()
    clear_img = data[0].cuda()
    degraded_img = data[1].cuda()
    generated_img = netG(clear_img)
    f_img = netD(torch.cat((generated_img,clear_img),dim=1))
    optimizerG.zero_grad()
    g_loss = G_Loss(f_img)
    pair_loss = Pair_Loss(generated_img,degraded_img)
    total_loss = g_loss + pair_loss * 10
    torch.autograd.set_detect_anomaly(True)
    total_loss.backward()
    optimizerG.step()
    return 0

def train_D(data,netG,netD,D_Loss,optimizerD):
    netG.train()
    netD.train()
    clear_img = data[0].cuda()
    degraded_img = data[1].cuda()
    generated_img = netG(clear_img)
    generated_img = generated_img.detach()
    f_img = netD(torch.cat((generated_img,clear_img),dim=1))
    r_img = netD(torch.cat((degraded_img,clear_img),dim=1))
    optimizerD.zero_grad()
    d_loss = D_Loss(r_img,f_img)
    d_loss.backward()
    optimizerD.step()
    return 0