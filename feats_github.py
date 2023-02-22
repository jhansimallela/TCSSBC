# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:00:09 2023

@author: Jhansi
"""

from myfunctions import spectral_selection,  temporal_corr#,temp_vec_corr
from myfunctions import spectral_corr,statFunctions_Syl,statFunctions_Vwl,smooth,vocoder_func#,get_labels
import numpy as np
from scipy.signal import medfilt
import scikits.samplerate
from scipy.io import loadmat

wordsfile = loadmat('./Words.mat')

spurtWordTimes = wordsfile['spurtWordTimes']

words = wordsfile['words']

syllablefile = loadmat('./Syllable.mat')

spurtSyl = syllablefile['spurtSyl']

spurtSylTimes = syllablefile['spurtSylTimes']

vowelfile = loadmat('./Vowel.mat')

vowelStartTime = vowelfile['vowelStartTime']

vowelEndTime = vowelfile['vowelEndTime']

wavFile = './ISLE_SESS0006_BLOCKD01_06_sprt1.wav'

def compute_stress_features(wavFile,words,spurtWordTimes,spurtSyl,spurtSylTimes,vowelStartTime,vowelEndTime):
        # Compute features              ########################################################################################
    twin = 5
    t_sigma = 1.4
    swin = 7
    s_sigma = 1.5
    # mwin = 13
    # max_threshold = 25
    
    vwlSB_num= 4
    vowelSB= [1,2,4,5,6,7,8,13,14,15,16,17]
    sylSB_num= 5
    sylSB= [1,2,3,4,5,6,13,14,15,16,17,18]
    
    
    # startWordFrame_all = []; spurtStartFrame_all = []; spurtEndFrame_all=[]
    # vowelStartFrame_all = []; vowelEndFrame_all = []; eng_full_all = []
    # spurtStress_all = []
    
    # Execute the vocoder [MODIFICATION]: Get the audio file back so that it can be stored in a text file for C code.
    eng_full, xx = vocoder_func(wavFile)
    #eng_full = np.loadtxt('./ISLE_SESS0003_BLOCKD01_11_sprt1.e19' , delimiter=',')
    eng_full = eng_full.conj().transpose()
    
    
    # Processing word boundary file
    # FILE READ DELETED HERE
    a = spurtWordTimes
    b = words
    if(len(a) is not len(b)):
     return []
    else:
        wordData = np.hstack((a, np.array([b], dtype='S32').T))
        startWordTime = [row[0] for row in wordData]  # Extract first coloumn of wordData
        endWordTime = [row[1] for row in wordData]
        startWordFrame = np.round((np.subtract(np.array(startWordTime, dtype='float'), spurtSylTimes[0][0].astype(float))*100))
        endWordFrame = np.round((np.subtract(np.array(endWordTime, dtype='float'), spurtSylTimes[0][0].astype(float))*100) + 1)
        startWordFrame = np.append(startWordFrame,endWordFrame[-1])
        
        # Processing of stress and syllable boundary file
        spurtSylTime = spurtSylTimes
        spurtStartTime = spurtSylTime[:, 0]
        spurtEndTime = spurtSylTime[:, 1]
        spurtStartFrame = np.round((spurtStartTime - spurtStartTime[0]) * 100)
        spurtEndFrame = np.round((spurtEndTime - spurtStartTime[0]) * 100)
        
        # Processing of Vowel boundary file
        vowelStartFrame = np.round(vowelStartTime*100 - spurtStartTime[0] * 100)
        vowelEndFrame = np.round(vowelEndTime*100 - spurtStartTime[0] * 100)
        
        # TCSSBC computation
        if len(sylSB) > sylSB_num:
            eng = spectral_selection(eng_full[np.subtract(sylSB, 1), :], sylSB_num)
        else:
            eng = eng_full[sylSB, :]
        t_cor = temporal_corr(eng, twin, t_sigma)
        s_cor = spectral_corr(t_cor)
        sylTCSSBC = smooth(s_cor, swin, s_sigma)
        sylTCSSBC = np.array([sylTCSSBC])
        
        # Modify TCSSBC contour by clipping from the syllable start
        start_idx = np.round(spurtStartTime[0]*100).astype(int)
        sylTCSSBC = np.array([sylTCSSBC[0][start_idx:-1]])
        
        sylTCSSBC = np.divide(sylTCSSBC, max(sylTCSSBC[0]))
        
        if len(vowelSB) > vwlSB_num:
            eng = spectral_selection(eng_full[np.subtract(vowelSB, 1), :], vwlSB_num)
        else:
            eng = eng_full[vowelSB, :]
        t_cor = temporal_corr(eng, twin, t_sigma)
        s_cor = spectral_corr(t_cor)
        vwlTCSSBC = smooth(s_cor, swin, s_sigma)
        
        vwlTCSSBC = np.array([vwlTCSSBC])
        
        # Modify TCSSBC contour by clipping from the vowel start
        start_idx = np.round(vowelStartTime[0][0]*100).astype(int)
        vwlTCSSBC = np.array([vwlTCSSBC[0][start_idx:-1]])
        
        vwlTCSSBC = np.divide(vwlTCSSBC, max(vwlTCSSBC[0]))
        
        
        # Compute silence statistics
        # Preprocessing of the data
        word_duration = np.zeros((1, len(startWordFrame) - 1))
        word_Sylsum = np.zeros((1, len(startWordFrame) - 1))
        word_Vwlsum = np.zeros((1, len(startWordFrame) - 1))
        
        for j in range(0, len(startWordFrame) - 1):
            temp_start = startWordFrame[j].astype(int)
            temp_end = startWordFrame[j + 1].astype(int) - 1
            #jhansi
            if (temp_end >= sylTCSSBC.shape[1]):
                temp_end1 = sylTCSSBC.shape[1]-1
                sylTCSSBC[0, np.arange(temp_start, temp_end1)] = medfilt(sylTCSSBC[0, np.arange(temp_start, temp_end1)], 3)
                sylTCSSBC[0, temp_start] = sylTCSSBC[0, temp_start+1]
                sylTCSSBC[0, temp_end1] = sylTCSSBC[0, temp_end1 - 1]
                tempArr = sylTCSSBC[0, np.arange(temp_start, temp_end1)]
                word_Sylsum[0, j] = tempArr.sum(axis=0)
            else:
                sylTCSSBC[0, np.arange(temp_start, temp_end)] = medfilt(sylTCSSBC[0, np.arange(temp_start, temp_end)], 3)
                sylTCSSBC[0, temp_start] = sylTCSSBC[0, temp_start+1]
                sylTCSSBC[0, temp_end] = sylTCSSBC[0, temp_end - 1]
                tempArr = sylTCSSBC[0, np.arange(temp_start, temp_end)]
                word_Sylsum[0, j] = tempArr.sum(axis=0)
            if (temp_end >= vwlTCSSBC.shape[1]):
                temp_end = vwlTCSSBC.shape[1]-1
        #    temp_end = np.min([temp_end,len(vwlTCSSBC)])
            vwlTCSSBC[0, np.arange(temp_start, temp_end)] = medfilt(vwlTCSSBC[0, np.arange(temp_start, temp_end)], 3)
            vwlTCSSBC[0, temp_start] = vwlTCSSBC[0, temp_start+1]
            vwlTCSSBC[0, temp_end] = vwlTCSSBC[0, temp_end - 1]
        
            word_duration[0, j] = temp_end - temp_start + 1
    
            tempArr = vwlTCSSBC[0, np.arange(temp_start, temp_end)]
            word_Vwlsum[0, j] = tempArr.sum(axis=0)
        sylTCSSBC[np.isnan(sylTCSSBC)] = 0
        vwlTCSSBC[np.isnan(vwlTCSSBC)] = 0
        tempOut = np.array([[]])
        
        wordIndication = []; peakVals = []; avgVals = []
        
        # Generating the features
        for j in range(0, len(spurtSyl), 1):
            inds = (startWordFrame <= spurtStartFrame[j]).nonzero()
            word_ind = inds[0][-1]; wordIndication.append(word_ind)
    #        print([0, np.arange(spurtStartFrame[j], spurtEndFrame[j]-1, 1).astype(int)])
            currFtr1SylSeg = sylTCSSBC[0, np.arange(spurtStartFrame[j], spurtEndFrame[j]-1, 1).astype(int)]
            currFtr1SylSeg = np.array([currFtr1SylSeg])
            temp = np.multiply(currFtr1SylSeg, len(currFtr1SylSeg[0]) / word_duration[0, word_ind])
            arrResampled = np.array([scikits.samplerate.resample(temp[0], float(30) / len(temp[0]), 'sinc_best')])
            
            #To be put in the output file
            peakVals.append(np.amax(arrResampled))
            avgVals.append(np.average(arrResampled))
        
            currSylFtrs = statFunctions_Syl(arrResampled)
            arr1 = np.array([np.array([np.sum(currFtr1SylSeg) / word_Sylsum[0, word_ind]])]).T
            currSylFtrs = np.vstack((currSylFtrs, arr1))
            ##########jhansi
            if (j>= vowelEndFrame.shape[1]):
                break
            if (vowelEndFrame [0,j] >= vwlTCSSBC.shape[1]):
                vowelEndFrame[0,j] = vwlTCSSBC.shape[1]-1
        
            currFtr1VowelSeg = vwlTCSSBC[0, np.arange(vowelStartFrame[0, j], vowelEndFrame[0, j]-1, 1).astype(int)]
            currFtr1VowelSeg = np.array([currFtr1VowelSeg])
            temp = np.multiply(currFtr1VowelSeg, len(currFtr1VowelSeg[0]) / word_duration[0, word_ind])
            if (len(temp[0])==0):
                break
                
            arrResampled = np.array([scikits.samplerate.resample(temp[0], float(20) / len(temp[0]), 'sinc_best')])
            currVowelFtrs = statFunctions_Vwl(arrResampled)
            arr1 = np.array([np.array([np.sum(currFtr1VowelSeg) / word_Sylsum[0, word_ind]])]).T
            currVowelFtrs = np.vstack((currVowelFtrs, arr1))
            if j == 0:
                tempOut = np.vstack((currSylFtrs, currVowelFtrs, len(currFtr1VowelSeg[0]), len(currFtr1SylSeg[0])))
            else:
                tempOut = np.hstack((tempOut, np.vstack((currSylFtrs, currVowelFtrs,len(currFtr1VowelSeg[0]), len(currFtr1SylSeg[0])))))
        if (len(temp[0])==0):
               print('Length of temp = 0; Features cannot be computed')
               return []
        else:
            # sylDurations = spurtEndTime - spurtStartTime
            
            ftrs = tempOut
            
            wordLabls = np.unique(wordIndication)
            for iterWrd in range(0, len(wordLabls)):
                inds = [i for i, x in enumerate(wordIndication) if x == wordLabls[iterWrd]] #doing argwhere(wordIndication==wordLabls[iterWrd]
                if len(inds)>1 :
                    ftrs[-1, inds] = ftrs[-1, inds] / sum(ftrs[-1, inds])
                    ftrs[-2, inds] = ftrs[-2, inds] / sum(ftrs[-2, inds])
            # end=1
            print(ftrs.shape)
            feats = ftrs
            return feats
features = compute_stress_features(wavFile,words,spurtWordTimes,spurtSyl,spurtSylTimes,vowelStartTime,vowelEndTime)
    
#     mat = scipy.io.loadmat(stressLabelspath+fileName[0:-4]+'.mat')
#     lab = mat['spurtStress']
#     lab_list = lab.tolist()
#     if (fa.shape[1] is not len(lab_list)):
#         label_mismatch = label_mismatch+1
# #            is_looping = False
#         continue
#     else:
#         fb,filenm = get_labels(lab_list,fa,fileName)
#         feats = fb #features ,last row:labels
#         w=[];#polysyl_feat=[];
#         for w_l in range(len(words)):
#     #        cou = 0
#             w_st = spurtWordTimes[w_l][0]; w_ed = spurtWordTimes[w_l][1]
#             for s_l in range(len(w),len(spurtSyl)):
#                 sy_st = spurtSylTimes[s_l][0];sy_ed = spurtSylTimes[s_l][1]
#                 if (sy_ed <= w_ed):
#                     w.append(w_l+1)
# #                        w.append('W'+str(w_l+1))
#     #                cou=cou+1
#                 else:
#                     break
#         if (len(w)>np.shape(feats)[1]):
#             continue
        # feats = np.vstack((feats,w))#features ,last row:labels, word labels
        # AF_inform = filenm
        # CF_feats,CF_inform = contextFeats(spurtSyl,spurtSylTimes,spurtWordTimes,vowel);  
    # if fileN == 0:#411 or fileN ==412:
    #     AF = feats
    #     AF_info = AF_inform
    #     CF = CF_feats
    #     CF_info =CF_inform                
    # else:
    #     AF = np.hstack((AF,feats)) 
    #     AF_info = np.hstack((AF_info,AF_inform))
    #     CF = np.hstack((CF,CF_feats))
    #     CF_info = np.hstack((CF_info,CF_inform))
    #     done=done+1
    #     print('Done:::file number ' + str(done) + '  out of ' + str(len(files)))