from pathlib import Path
from gesture.config import  *

sid=10
project_dir=data_dir+'preprocessing'+'/P'+str(sid)+'/'
result_dir=project_dir + 'result' + '/'
results_path = [pth for pth in Path(result_dir).iterdir() if pth.suffix == '.npy']
result={}
score=[]
for file_path in results_path:
    lr_com = file_path.name[12:-4]
    file_path=str(file_path)
    tmp_data=np.load(file_path)
    score.append(tmp_data)
    result[lr_com]=tmp_data
test_acc=[scorei[:,1] for scorei in score]
test_acc=np.asarray(test_acc)
test_acc.max()


a='/Users/long/Documents/data/gesture/preprocessing/P10/result1/ZZ0.0001_0.001.npy'
a=torch.load(a,map_location='cpu')











