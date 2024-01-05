opt={
'model_name':'linear_regression',
'mel_bins':23,
'step_size':5,
'model_order':4,
'winL':0.05, # feature extraction
'frameshift':0.01, # feature extraction
}

opt['sf_EEG']=1024

'''
selected_channels={
'1':[56,49,6,91,79,47,84,46,67,26,86,58,55,22,51],
'2':[108,101,116,26,57,84,15,48,81,79,89,119,120,87,27],
'3':[48,47,87,97,63,113,9,111,106,41,70,99,42,53,45],
'4':[30,20,36,16,7,108,112,102,54,100,0,4,84,97,86],
'5':[47,13,24,44,40,37,54,45,31,39,58,11,28,41,38],
'6':[30,43,35,47,52,98,61,22,48,102,50,8,74,51,100],
'7':[8,100,51,31,74,85,50,52,101,25,94,11,0,102,30],
'8':[47,7,31,3,13,26,0,29,22,46,9,18,35,49,12],
'9':[89,26,116,9,90,81,112,108,98,80,109,39,110,114,87],
'10':[33,37,61,55,39,112,0,35,57,38,56,58,97,60,53],
}
'''
selected_channels={
'1':[],
'2':[],
'3':[],
'4':[],
'5':[],
'6':[],
'7':[],
'8':[],
'9':[],
'10':[],
}

