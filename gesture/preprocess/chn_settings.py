def get_active_channel(sid):
    # not implemented yet.
    return sid

def get_channel_setting(pn):
    if pn==2:
        #Session_num=[1:2]
        #UseChn=[1:19,21:37,43:44,47:129]
        #EmgChn=[145:146]
        #TrigChn=[38:42]
        Session_num=[0,1]
        UseChn=[*range(0,19)] + [*range(20,37)] + [42,43] +[*range(46,129)]
        EmgChn=[144,145]
        TrigChn=[37,38,39,40,41]
    elif pn==3:
        Session_num=[0,2]
        UseChn=[*range(0,19)] + [*range(20,37)] + [43,44] + [*range(47,189)]
        EmgChn=[191,192]
        TrigChn=[37,38,39,40,41]
        #Session_num=[1,3];
        #UseChn=[1:19,21:37,44:45,48:189];
        #EmgChn=[192:193];
        #TrigChn=[38:42];
    elif pn==4:
        Session_num = [1,2]
        UseChn = [*range(0, 19)] + [*range(20, 37)] + [42,43] + [*range(46, 68)]
        EmgChn = [74,75]
        TrigChn = [37, 38, 39, 40, 41]
        #Session_num=[2,3];
        #UseChn=[1:19,21:37,43:44,47:68];
        #EmgChn=[75:76];
        #TrigChn=[38:42];
    elif pn==5:
        # N1 has some uring the entire session, and need to be removed in the future
        Session_num = [0,2]
        UseChn = [*range(0, 19)] + [*range(20, 37)] + [42, 43] + [*range(46, 186)]
        EmgChn = [186,187]
        TrigChn = [37, 38, 39, 40, 41]
        #Session_num=[1,3];
        #UseChn=[1:19,21:37,43:44,47:150,151:166,167:186];
        #EmgChn=[187:188];
        #TrigChn=[38:42];
    elif pn==7:
        Session_num = [0, 1]
        UseChn = [*range(0, 19)] + [*range(20, 37)] + [43, 44] + [*range(47, 126)] + [*range(127,153)]
        EmgChn = [186, 187]
        TrigChn = [37, 38, 39, 40, 41]

        #Session_num=[1,2];
        #UseChn=[1:19,21:37,44:45,48:126,128:153];
        #EmgChn=[162:163];
        #TrigChn=[38:42];
    elif pn==8:
        Session_num = [0, 1]
        UseChn = [*range(0, 36)] + [*range(52, 141)] + [*range(142, 186)]
        EmgChn = [192,193]
        TrigChn = [37, 38, 39, 40, 41]

        #Session_num=[1,2];
        #UseChn=[1:36,53:141,143:186];
        # N1 has some drifts during the ession, and need to be removed in the future
        #EmgChn=[193:194];
        #TrigChn=[38:42];
    elif pn==9:
        Session_num = [0, 1]
        UseChn = [*range(0, 19)] + [*range(20, 37)] + [43,44] + [*range(47, 123)]
        EmgChn = [123,124]
        TrigChn = [37, 38, 39, 40, 41]

        #Session_num=[1:2];
        #UseChn=[1:19,21:37,44:45,48:123];
        #EmgChn=[124:125];
        #TrigChn=[38:42];
    elif pn==10:
        Session_num = [0, 1]
        UseChn = [*range(0, 19)] + [*range(20, 33)] + [*range(38, 60)] + [*range(62, 216)]
        EmgChn = [60,61]
        TrigChn = [33,34,35,36,37]
        activeChn=[57,65,66,67,70,71,72]+[*range(143,150)]+[158,]+[*range(162,168)]+[180,183]
        #Session_num=[1,2];
        #UseChn=[1:19,21:33,39:60,63:216];
        #EmgChn=[61:62];
        #TrigChn=[34:38];
    elif pn==11:
        Session_num = [0, 1]
        UseChn = [*range(0, 19)] + [*range(20, 35)] + [43,44] + [*range(47, 143)] + [*range(145, 151)] + [*range(153, 209)]
        EmgChn = [215,216]
        TrigChn = [37, 38, 39, 40, 41]

        #Session_num=[1,2];
        #UseChn=[1:19,21:35,44:45,48:143,146:151,154:209];
        #EmgChn=[216:217];
        #TrigChn=[36:40];
    elif pn==12:
        Session_num = [0, 1]
        UseChn = [*range(0, 19)] + [*range(20, 37)] + [42,43] + [*range(46, 102)]
        EmgChn = [108,109]
        TrigChn = [37, 38, 39, 40, 41]

        #Session_num=[1,2];
        #UseChn=[1:19,21:37,43:44,47:102];
        #EmgChn=[109:110];
        #TrigChn=[38:42];
    elif pn==13:
        Session_num = [0, 1]
        UseChn = [*range(0, 19)] + [*range(20, 35)] + [*range(51, 119)]
        EmgChn = [125,126]
        TrigChn = [43,44,45,46,47]

        #Session_num=[1,2];
        #UseChn=[1:19,21:35,52:119];
        #EmgChn=[126:127];
        #TrigChn=[44:48];
    elif pn==14:
        Session_num = [0, 1]
        UseChn = [*range(0, 17)] + [*range(18, 35)] + [*range(43, 139)]
        EmgChn = [141,142] # % not the correct EMG
        TrigChn = [35,36,37,38,39]

        #Session_num=[1,2];
        #UseChn=[1:17,19:35,44:139];
        #EmgChn=[142:143]; % not the correct EMG
        #TrigChn=[36:40];
    elif pn==16:
        Session_num = [0, 1]
        UseChn = [*range(0, 18)] + [*range(20, 37)] + [*range(45, 179)]
        EmgChn = [185,186]
        TrigChn = [37, 38, 39,40,41]

        #Session_num=[1,2];
        #UseChn=[1:19,21:37,46:179];
        #EmgChn=[186:187];
        #TrigChn=[38:42];
    elif pn==17:
        Session_num = [0, 1]
        UseChn = [*range(0, 18)] + [*range(20, 37)] + [*range(45, 153)]
        EmgChn = [159,160]
        TrigChn = [37, 38, 39, 40, 41]

        #Session_num=[1,2];
        #UseChn=[1:19,21:37,46:153];
        #EmgChn=[160:161];
        #TrigChn=[38:42];
    elif pn==18:
        Session_num = [0, 1]
        UseChn = [*range(0, 18)] + [*range(20, 35)] + [*range(51, 161)]
        EmgChn = [181,182]
        TrigChn = [43,44,45,46,47]

        #Session_num=[1,2];
        #UseChn=[1:19,21:35,52:161];
        #EmgChn=[182:183];
        #TrigChn=[44:48];
    elif pn==19:
        Session_num = [0, 1]
        UseChn = [*range(0, 18)] + [*range(20, 37)] + [*range(45, 146)]
        EmgChn = [152,153]
        TrigChn = [37,38,39,40,41]

        #Session_num=[1,2];
        #UseChn=[1:19,21:37,46:146];
        #EmgChn=[153:154];
        #TrigChn=[38:42];
    elif pn==20:
        Session_num = [0, 1]
        UseChn = [*range(0, 18)] + [*range(20, 37)] + [46,47] + [*range(50, 120)]
        EmgChn = [126,127]
        TrigChn = [38, 39, 40, 41,42]

        #Session_num=[1,2];
        #UseChn=[1:19,21:37,47:48,51:120];
        #EmgChn=[127:128];
        #TrigChn=[39:43];
    elif pn==21:
        Session_num = [0, 1]
        UseChn = [*range(0, 18)] + [*range(20, 37)] + [45, 46] + [*range(49, 129)]
        EmgChn = [135,136]
        TrigChn = [37,38, 39, 40, 41]

        #Session_num=[1,2];
        #UseChn=[1:19,21:37,46:47,50:129];
        #EmgChn=[136:137];
        #TrigChn=[38:42];
    elif pn==22:
        Session_num = [0, 1]
        UseChn = [*range(0, 17)] + [*range(18, 33)] + [*range(41, 159)]
        EmgChn = [159,160]
        TrigChn = [33,34,35,36,37]

        #Session_num=[1,2];
        #UseChn=[1:17,19:33,42:159];
        #EmgChn=[160:161];
        #TrigChn=[34:38];
    elif pn==23:
        Session_num = [0, 1]
        UseChn = [*range(0, 16)] + [*range(17, 34)] + [*range(42, 161)] + [*range(167,213)]
        EmgChn = [213,214]
        TrigChn = [34, 35, 36, 37,38]

        #Session_num=[1,2];
        #UseChn=[1:16,18:34,43:161,168:213];
        #EmgChn=[214:215];
        #TrigChn=[35:39];
    elif pn == 24:
        Session_num = [0, 1]
        UseChn = [*range(0, 14)] + [*range(15, 30)] + [*range(46, 145)]
        EmgChn = [145,146]
        TrigChn = [38,39,40,41,42]

        #Session_num=[1,2];
        #UseChn=[1:14,16:30,47:145];  % miss K5 & A3 ,add two virtual  for P24, and should be defined as bad channels.
        #EmgChn=[146:147];
        #TrigChn=[39:43];
    elif pn == 25:
        Session_num = [0, 1]
        UseChn = [*range(0, 14)] + [*range(16, 30)] + [*range(39, 146)]
        EmgChn = [146,147]
        TrigChn = [31,32,33,34,35]

        #Session_num=[1,2];
        #UseChn=[1:15,17:30,40:146]; % H9 is missing, create one virtual H9 for P25
        #EmgChn=[147:148];
        #TrigChn=[32:36];
    elif pn == 26:
        Session_num = [0, 1]
        UseChn = [*range(0, 17)] + [*range(18, 33)] + [*range(42, 163)]
        EmgChn = [163,164]
        TrigChn = [34,35,36,37,38]

        #Session_num=[1,2];
        #UseChn=[1:17,19:33,43:163]; % L6 is missing for P26
        #EmgChn=[164:165];
        #TrigChn=[35:39];
    elif pn == 28:
        pass #Information Loss !')
    elif pn == 29:
        Session_num = [0, 1]
        UseChn = [*range(0, 15)] + [*range(16, 29)] + [*range(37, 119)]
        EmgChn = [119,120]
        TrigChn = [29,30,31,32,33, 34]

        #Session_num=[1,2];
        #UseChn=[1:15,17:29,38:119];
        #EmgChn=[120:121];
        #TrigChn=[30:34];
    elif pn == 30:
        Session_num = [0, 1]
        UseChn = [*range(0, 15)] + [*range(16, 29)] + [*range(37, 103)] + [*range(105, 119)]
        EmgChn = [119, 120]
        TrigChn = [29, 30, 31, 32, 33, 34]

        #Session_num=[1,2];
        #UseChn=[1:15,17:29,38:103,106:119];
        #EmgChn=[120:121];
        #TrigChn=[30:34];
    elif pn == 31:
        Session_num = [0, 1]
        UseChn = [*range(0, 16)] + [*range(18, 33)] + [*range(41, 81)]
        EmgChn = [81,82]
        TrigChn = [33, 34, 35,36,37]

        #Session_num=[1,2];
        #UseChn=[1:17,19:33,42:81];
        #EmgChn=[82:83];
        #TrigChn=[34:38];
    elif pn == 32:
        Session_num = [0, 1]
        UseChn = [*range(0, 18)] + [*range(20, 37)] + [45,46] + [*range(49, 67)]
        EmgChn = [67,68]
        TrigChn = [37, 38, 39, 40, 41]

        #Session_num=[1,2];
        #UseChn=[1:19,21:37,46:47,50:67];
        #EmgChn=[68:69];
        #TrigChn=[38:42];
    elif pn == 33:
        pass  # Information Loss !')
    elif pn == 34:
        Session_num = [0, 1]
        UseChn = [*range(0, 15)] + [*range(16, 31)] + [*range(42, 114)]
        EmgChn = [116,117]
        TrigChn = [36,37, 38, 39, 40]

        #Session_num=[1,2];
        #UseChn=[1:15,17:31,43:114];
        #EmgChn=[117:118];
        #TrigChn=[35:39];
    elif pn == 35:
        Session_num = [0, 1]
        UseChn = [*range(0, 18)] + [*range(20, 31)] + [39,40] + [*range(43, 57)] + [*range(59, 87)] + [*range(89, 151)]
        EmgChn = [87,88]
        TrigChn = [31,32,33,34,35]

        #Session_num=[1,2];
        #UseChn=[1:19,21:31,40:41,44:57,60:87,90:151];
        #EmgChn=[88:89];
        #TrigChn=[32:36];
    elif pn == 36:
        Session_num = [0, 1]
        UseChn = [*range(0, 15)] + [*range(16, 25)] + [*range(33, 126)]
        EmgChn = [126,127]
        TrigChn = [25,26,27,28,29]

        #Session_num=[1,2];
        #UseChn=[1:15,17:25,34:126];
        #EmgChn=[127:128];
        #TrigChn=[26:30];
    elif pn == 37:
        Session_num = [0, 1]
        UseChn = [*range(0, 15)] + [*range(16, 23)] + [*range(31, 73)]
        EmgChn = [73,74]
        TrigChn = [25, 26, 27, 28, 29]

        #Session_num=[1,2];
        #UseChn=[1:15,17:23,32:73];
        #EmgChn=[74:75];
        #TrigChn=[24:28];
    elif pn == 41:
        Session_num = [0, 1]
        UseChn = [*range(0, 18)] + [*range(20, 37)] + [*range(53, 207)]
        EmgChn = [209,210]
        TrigChn = [45,46,47,48,49]

        #Session_num=[1,2];
        #UseChn=[1:19,21:37,54:207];
        #EmgChn=[210:211];
        #TrigChn=[46:50];
    else:
        print("No participant ID found.")
    return Session_num,UseChn,EmgChn,TrigChn

