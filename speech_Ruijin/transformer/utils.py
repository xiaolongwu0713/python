from common_dl import device
from tqdm import tqdm
import torch
import numpy as np
from pre_all import computer
from speech_Ruijin.transformer.lib.train import output_prediction2


def test_transformer_model(sid,model,opt,dataloader_test,result_dir):
    # loop through many test data
    encoder_only=False
    predictions=[]
    truths=[]

    model.eval()
    for i,(test_x, test_y) in tqdm(enumerate(dataloader_test)): # total 48 iterates
        # batch inference
        pred, tgt = output_prediction2(model, test_x, test_y, max_len=opt['out_len'], start_symbol=1, output_d=opt['tgt_d'],
                                       device=device, encoder_only=encoder_only)

        ''' individual inference
        test_x, test_y = next(iter(dataloader_test))
        test_x, test_y = test_x.numpy(), test_y.numpy()
        test_x, test_y = test_x[0], test_y[0]
        pred, tgt = output_prediction(model, test_x, test_y, max_len=opt['out_len'], start_symbol=1, output_d=opt['tgt_d'],
                                       device=device,encoder_only=encoder_only)
        '''
        predictions.append(pred.numpy())
        truths.append(test_y.numpy())

        #ax[0].imshow(tgt[-1, :, :].squeeze(), cmap='RdBu_r')
        #ax[1].imshow(pred[-1, :, :].squeeze(), cmap='RdBu_r')
        #filename = path_file[:-4]+'/'+str(i)+'.png'
        #fig.savefig(filename)

        #if i>400:
        #    break
    predictions2=np.concatenate(predictions,axis=0)
    truths2=np.concatenate(truths,axis=0)

    return predictions2, truths2


def test_seq2seq_model(net,dataloader_test,result_dir):
    loss_fun = torch.nn.MSELoss()
    predictions=[]
    truths=[]
    net.eval()
    with torch.no_grad():
        for i, (test_x, test_y) in enumerate(dataloader_test):
            #src = test_x.float().permute(1,0,2).to(device) # (batch,time,feature)-->#(time,batch,feature)
            #tgt = test_y.float().permute(1,0,2).to(device)
            src = test_x.float().permute(2,0,1).to(device)  # (batch,time,feature)-->#(time,batch,feature)
            tgt = test_y.float().permute(2,0,1).to(device)
            # src,tgt=src.permute(1,0,2),tgt.permute(1,0,2) # ECoG_seq2seq
            out_len = tgt.shape[0]
            # no teacher force during validation
            output, attention_weights = net.predict_step(src, tgt[1:, :, :], out_len)  # torch.Size([1, 194, 80])
            loss = loss_fun(output, tgt[1:, :, :])
            predictions.append(output.cpu().numpy())
            truths.append(tgt[1:, :, :].cpu().numpy())
    predictions2 = np.concatenate(predictions, axis=1).transpose(1,0,2)
    truths2 = np.concatenate(truths, axis=1).transpose(1,0,2)

    #filename1 = result_dir + 'predictions.npy'
    #np.save(filename1, predictions2)
    #filename2 = result_dir + 'truths.npy'
    #np.save(filename2, truths2)

    return predictions2,truths2

def averaging(pred,win,stride):
    h=pred.shape[0] # 3066
    d=pred.shape[2] # 40
    w=win+(h-1)*stride # 30750
    summary=np.zeros((w,d)) # (30750, 40)
    for i in range(len(pred)):
        start=i*stride
        end=start+win
        summary[start:end,:]=summary[start:end,:]+pred[i]

    stairs=np.ones(win)

    layers=int(win/stride)#10
    layers=10
    sublist=[1,]*layers
    steps=int(win/layers)
    for i in range(steps):
        start=i*layers
        stop=start+layers
        stairs[start:stop]=[j*(i+1) for j in sublist]

    stairs_reverse=[stairs[len(stairs)-i-1] for i in range(len(stairs))]
    middle=[10,]*(summary.shape[0]-len(stairs)-len(stairs_reverse))
    repeat=list(stairs)+middle+stairs_reverse # 30750

    average=np.zeros(summary.shape) # (30750, 40)
    for i in range(summary.shape[0]):
        average[i,:]=summary[i,:]/repeat[i]

    return average
