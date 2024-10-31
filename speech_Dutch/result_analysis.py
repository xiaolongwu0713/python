import matplotlib.pyplot as plt
import numpy as np
from speech_Dutch.config import data_dir
from common_plot import color_codes as colors
result_dir=data_dir+'result/'


## compare the decoding result in MSE and correlation coefficient (LR, seq2seq, transformer)
# 23 mel bins
LR_cc=[0.501,0.628,0.834,0.779,0.532,0.859,0.686,0.709,0.651,0.684]
LR_cc_err=[0.051,0.042,0.103,0.104,0.095,0.062,0.109,0.111,0.081,0.068]
LR_mse=[2.075,2.657,1.808,1.909,3.697,1.078,2.741,2.163,2.225,1.059]
LR_mse_err=[0.121,0.222,0.123,0.104,0.155,0.112,0.109,0.111,0.141,0.108]
seq2seq_cc=[0.497,0.593,0.838,0.773,0.533,0.907,0.751,0.577,0.658,0.639]
seq2seq_cc_err=[0.101,0.092,0.103,0.084,0.088,0.101,0.109,0.056,0.077,0.108]
seq2seq_mse=[0.133,0.143,0.073,0.076,0.183,0.029,0.094,0.132,0.078,0.045]
seq2seq_mse_err=[0.021,0.022,0.023,0.023,0.085,0.052,0.019,0.011,0.044,0.032]
transformer_cc=[0.579,0.649,0.821,0.717,0.515,0.870,0.805,0.682,0.550,0.693]
transformer_cc_err=[0.021,0.034,0.023,0.056,0.077,0.063,0.043,0.025,0.054,0.063]
transformer_mse=[0.112,0.220,0.089,0.103,0.193,0.044,0.083,0.163,0.140,0.068]
transformer_mse_err=[0.061,0.032,0.023,0.084,0.065,0.072,0.059,0.041,0.064,0.066]

fig,ax=plt.subplots()
ax.clear()
sids=np.arange(1,11)
ax.plot(sids, LR_cc, linestyle="-.", color=colors[0])
ax.plot(sids, seq2seq_cc, linestyle="-.", color=colors[1])
ax.plot(sids, transformer_cc, linestyle="-.", color=colors[2])
ax.plot(sids,LR_mse, linestyle="-", color=colors[3])
ax.plot(sids,seq2seq_mse, linestyle="-", color=colors[4])
ax.plot(sids, transformer_mse, linestyle="-", color=colors[6])
ax.legend(['LR','RNN','Transformmer','LR','RNN','transformer'])

ax.fill_between(sids, [LR_cc[i-1]-LR_cc_err[i-1] for i in sids], [LR_cc[i-1]+LR_cc_err[i-1] for i in sids], alpha=0.5,color=colors[0])
ax.fill_between(sids, [seq2seq_cc[i-1]-seq2seq_cc_err[i-1] for i in sids], [seq2seq_cc[i-1]+seq2seq_cc_err[i-1] for i in sids], alpha=0.5, color=colors[1])
ax.fill_between(sids, [transformer_cc[i-1]-transformer_cc_err[i-1] for i in sids], [transformer_cc[i-1]+transformer_cc_err[i-1] for i in sids], alpha=0.5, color=colors[2])
ax.fill_between(sids, [LR_mse[i-1]-LR_mse_err[i-1] for i in sids], [LR_mse[i-1]+LR_mse_err[i-1] for i in sids], alpha=0.5, color=colors[3])
ax.fill_between(sids, [seq2seq_mse[i-1]-seq2seq_mse_err[i-1] for i in sids], [seq2seq_mse[i-1]+seq2seq_mse_err[i-1] for i in sids], alpha=0.5, color=colors[4])
ax.fill_between(sids, [transformer_mse[i-1]-transformer_mse_err[i-1] for i in sids], [transformer_mse[i-1]+transformer_mse_err[i-1] for i in sids], alpha=0.5, color=colors[6])

filename=result_dir+'compare_cc_mse_filled.pdf'
fig.savefig(filename)


## channel selection using LR
cc1=[0.448,0.493,0.542,0.570,0.583,0.598,0.605,0.608,0.613,0.616,0.618,0.619,0.623,0.624,0.625]
mse1=[3.174,3.023,2.769,2.682,2.624,2.630,2.595,2.579,2.558,2.526,2.506,2.521,2.487,2.479,2.447]
cc2=[0.537,0.596,0.613,0.625,0.634,0.640,0.646,0.650,0.653,0.658,0.660,0.662,0.666,0.667,0.668]
mse2=[3.485,3.150,3.046,3.054,2.964,2.916,2.891,2.856,2.828,2.817,2.806,2.791,2.736,2.719,2.694]
cc3=[0.779,0.798,0.804,0.810,0.815,0.818,0.820,0.821,0.822,0.822,0.824,0.825,0.826,0.827,0.828]
mse3=[2.850,2.836,2.756,2.677,2.580,2.525,2.466,2.435,2.406,2.413,2.411,2.415,2.424,2.422,2.403]
cc4=[0.684,0.718,0.738,0.749,0.756,0.784,0.789,0.794,0.798,0.800,0.803,0.804,0.805,0.806,0.808]
mse4=[3.041,2.725,2.648,2.453,2.372,2.076,2.000,1.934,1.896,1.866,1.865,1.851,1.844,1.848,1.858]
cc5=[0.461,0.545,0.563,0.578,0.598,0.607,0.616,0.617,0.621,0.624,0.624,0.626,0.628,0.628,0.628]
mse5=[4.688,4.696,4.225,4.035,3.803,3.709,3.650,3.662,3.557,3.579,3.578,3.586,3.531,3.554,3.554]
cc6=[0.743,0.804,0.826,0.835,0.843,0.846,0.849,0.851,0.853,0.856,0.794,0.815,0.827,0.835,0.840]
mse6=[2.071,1.707,1.547,1.476,1.452,1.391,1.341,1.318,1.303,1.273,1.642,1.471,1.390,1.355,1.295,]
cc7=[0.582,0.645,0.694,0.711,0.719,0.729,0.740,0.749,0.750,0.752,0.649,0.689,0.711,0.715,0.723,]
mse7=[2.994,2.616,2.330,2.213,2.206,1.981,1.970,1.982,1.968,2.077,2.433,2.302,2.276,2.288,2.217,]
cc8=[0.615,0.674,0.692,0.700,0.705,0.709,0.712,0.716,0.718,0.718,0.719,0.720,0.721,0.722,0.722,]
mse8=[3.059,2.641,2.507,2.553,2.535,2.546,2.541,2.522,2.511,2.538,2.459,2.488,2.472,2.449,2.434,]
cc9=[0.511,0.693,0.717,0.728,0.738,0.741,0.744,0.744,0.747,0.748,0.748,0.749,0.749,0.749,0.749,]
mse9=[2.751,2.053,1.814,1.728,1.659,1.619,1.606,1.616,1.591,1.599,1.592,1.586,1.586,1.586,1.571,]
cc10=[0.762,0.765,0.770,0.772,0.773,0.775,0.776,0.777,0.778,0.778,0.778,0.777,0.777,0.777,0.776,]
mse10=[1.182,1.153,1.130,1.129,1.117,1.106,1.105,1.082,1.076,1.078,1.078,1.084,1.083,1.086,1.095,]
ccs=[cc1,cc2,cc3,cc4,cc5,cc6,cc7,cc8,cc9,cc10]
mses=[mse1,mse2,mse3,mse4,mse5,mse6,mse7,mse8,mse9,mse10]
labels=['sid'+str(i+1) for i in range(10)]
fig,ax=plt.subplots()
ax.clear()
for i,cc in enumerate(ccs):
    ax.plot(cc, label=labels[i] , linestyle="-.")
ax.legend()
filename=result_dir+'channel_selection_cc.pdf'
fig.savefig(filename)

ax.clear()
for i,mse in enumerate(mses):
    ax.plot(mse, label=labels[i] , linestyle="-")
ax.legend()
filename=result_dir+'channel_selection_mse.pdf'
fig.savefig(filename)


## attention matrix
file='/Volumes/Samsung_T5/data/speech_RuiJin/seq2seq_transformer/SingleWordProductionDutch/mel_23/sid_3/2023_07_25_18_52/attention_matrix.npy'
att=np.load(file,allow_pickle=True).item()
att_enc=att['att_encoders']
att_dec=att['att_decoders']
att_enc_dec=att['att_enc_decs']

fig,ax=plt.subplots()
im=ax.imshow(att_dec[0,0,:,:])
fig.colorbar(im, orientation="vertical",fraction=0.046, pad=0.02,ax=ax)
filename=result_dir+'att_matrix_decoder.pdf'
fig.savefig(filename)

'''
array([[1.0069445 , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0. ],
       [0.16799495, 0.8633002 , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0. ],
       [0.14757024, 0.42108834, 0.4502    , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0. ],
       [0.        , 0.1       , 0.4       , 0.5       , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0. ],
       [0.15021014, 0.18649812, 0.20708731, 0.2350481 , 0.197036  ,
        0.        , 0.        , 0.        , 0.        , 0. ],
       [0.        , 0.        , 0.1       , 0.2       , 0.2       ,
        0.6       , 0.        , 0.        , 0.        , 0. ],
       [0.        , 0.        , 0.        , 0.2       , 0.2       ,
        0.1       , 0.5       , 0.        , 0.        , 0. ],
       [0.        , 0.        , 0.        , 0.        , 0.2       ,
        0.1       , 0.5       , 0.3       , 0.        , 0. ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.1       , 0.2       , 0.3       , 0.4       , 0. ],
        [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.1       , 0.3       , 0.2       , 0.2       , 0.2 ]], dtype=float32)
        '''


