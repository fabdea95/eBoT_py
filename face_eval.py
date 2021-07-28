# -*- coding: utf-8 -*-
import os
import subprocess
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from load_audio import reconstruct_wav_from_mag_phase, wav2filterbanks
from utils import (colorize, detect_peaks, extract_attended_features,
                   extract_face_crops, full_to_map, load_model_params,
                   logsoftmax_2d, my_unfold, run_func_in_parts,
                   calc_flow_on_vid_wrapper)
from viz_utils import VideoSaver, viz_avobjects, viz_source_separation
from RetinaFace.test_fddb import RetinaFddb
from image_features import Img2Vec
from eBoT_python.main import EBoT
import math
import pandas as pd
import re
import cv2
from PIL import Image
import time
import shutil

class FaceEval():

    def __init__(self, model, opts):
        self.model = model
        self.opts = opts

        self.device = torch.device('cuda:0')
        self.model.to(self.device)
        self.n_speak = 0
        # make log directories
        self.step = 0
        opts.output_dir = os.path.join(opts.output_dir)

        if os.path.exists(opts.output_dir):
            # Clean up old logs
            command = 'rm %s/* -rf' % (opts.output_dir)
            print(command)
            subprocess.call(command, shell=True, stdout=None)

        self.checkpoints_path = opts.output_dir + "/checkpoints"

        # set up tb saver
        os.makedirs(opts.output_dir, exist_ok=True)
        self.video_saver = VideoSaver(opts.output_dir)

        from warp_video import Warper
        self.warper = Warper(device=self.device)


    def get_n_speakers(self, index):
        print("index",index[0][9:])
        index=str(index[0][9:])
        index = re.sub("person_._*", "", index) #delete person_%_ if exixts in the string
        index = index[:index.rfind('_')]        #delete substring after last '_'
        print("new index: ", index) 
        df = pd.read_csv('../inputvideos/video_info.csv',index_col='conversation_id')
        #print(index)
        num_speaker = df['num_speakers']
        print(num_speaker[index])
        return num_speaker[0]

    def eval(self, dataloader):		#FOWARD?

        bs = self.opts.batch_size

        for batch_sample in dataloader:
            #print("batch: ", batch_sample)
            self.n_speak = 3 #self.get_n_speakers(batch_sample['sample']) 
            print("n_speak", self.n_speak)
            
            print("continuare? (Y/n): ")
            risp = input()
            if risp=='n':
                break
             
            self.model.zero_grad()
             
            video = batch_sample['video']  # b x T_v x H x W x 3
            audio = batch_sample['audio']  # b x T_a
            print("video shape 1:",video.shape)
            print("audio shape 1:",audio.shape)
            # extract log-mel filterbanks  -  # b*41 x T_a x C
            mel, _, _, _ = wav2filterbanks(audio.to(self.device))

            # -- 1. Forward audio and video through the model to get embeddings

            video = torch.squeeze(video)
            print("squeezed", video.shape)
            print(video[:,20,:,:])
            dim,frames,h,w = video.shape
            image = np.zeros([h,w,3])


            dataset = 'RetinaFace/faces'


            face_eval = RetinaFddb(dataset)

            vis_threshold = 0.6
            for fr in range(0,frames):
                image[:,:,0] = video[0,fr,:,:]
                image[:,:,1] = video[1,fr,:,:]
                image[:,:,2] = video[2,fr,:,:]    
                image = np.array(image)
                image = np.float32(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #print(image)
                cv2.imwrite("RetinaFace/faces/enc_img"+ str(fr) +".jpg", image)
                #enc_img = cv2.imread('enc_img.jpg') 
                #cv2.imshow("image",enc_img)
            print("frame_img_shape",image.shape)
            face_eval.Fddb()



            #new part    
            f = open(dataset +'/frames_dets.txt', 'r') 
            with open(dataset+'/thresh_dets.txt','w') as f_thr:
                 pass
            f_thr = open(dataset+ '/thresh_dets.txt', 'w')
            Lines = f.readlines()
            faces_feat = {}
            faces = {} 
            seed_score = 0.87
            flag = False
            print("extracting features...")
            f_extract = Img2Vec()
            subjects = {}
            for line in Lines:
                if line[0] == '/': 
                   name = line.split()[0]
                   f_thr.write(name + os.linesep)
                   f_count = 0
                elif(len(line) < 5):
                   continue
                elif float(line.split(' ')[4]) < vis_threshold:
                   continue
                else:
                   if f_count > self.n_speak-2:
                       continue
                   x,y,w,h, s, mlx, mly, mrx, mry = line.split(' ') 
                   f_thr.write(x + ' ' + y + ' ' + w + ' ' + h + ' ' + mlx + ' ' + mly + ' ' + mrx + ' ' + mry) 
                   x,y,w,h, s , mlx, mly, mrx, mry = int(x), int(y), int(w), int(h), float(s), int(mlx), int(mly), int(mrx), int(mry)
                   path = dataset + name
                   #print(x, " ", y, " ",w," ",h," ",s)
                   #print(path)
                   #print(len(name))
                   #img_raw = cv2.imread(dataset + '/' + name, cv2.IMREAD_COLOR)
                   img_raw = Image.open(dataset + '/' + name)
                   #img_raw.show()    
                 
                   #print(img_raw.shape)
                   #print(x+w)
                   xmax = x + w
                   ymax = y + h
                   face = img_raw.crop((x,y,xmax,ymax)) #crop(left,top,right,bottom)
                   fname = name[1:-4]+'_'+str(f_count)
                   face.save(dataset+'/crop/' + fname ,'JPEG')
                   
                   #face.show()
                   """                    
                   #:avobjects feeatures does not work for image
                   face = np.moveaxis(np.array(face),-1,0)
                   print(face.shape)
                   emb, feat = self.model.forward_vid(torch.unsqueeze(
                           torch.transpose(
                           torch.unsqueeze(
                               torch.tensor(face),0),0,1),0).to(self.device),
                               return_feats=True)
                   faces_feat.append(feat)
                   """
                   
                   feat = f_extract.get_vec(face,tensor=True)            
                   
                   #print(feat.shape)                
                   faces_feat[fname] = feat
                   faces[fname] = [x,y,w,h,mlx,mly,mrx,mry]     
                   
                   if s > seed_score and flag == False:
                       seed = len(faces_feat)-1
                       flag = True
                       print("SEED: ", seed)
                       with open(dataset+'/crop/dets.txt','w') as fw:
                           pass
                       fw = open(dataset+'/crop/dets.txt','w')
                       fw.write('SEED {:s} face1\n'.format(fname)) 
                   f_count += 1
            f_thr.close()
            f.close()
            #cosine similarity AND TRACKLET
                 
            triang_sim = np.zeros((len(list(faces.keys())),len(list(faces.keys()))))
            triang_score = np.zeros((len(list(faces.keys())),len(list(faces.keys()))))
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            print("calculating similarity")
            tracklet ={}
            print(faces)
            #max_face = ''

            def face_id(face):
                return int(face[0][7:-2]) 
            faces =sorted(faces.items(), key=face_id)
            faces = dict(faces)

            coord = []
            prev = list(faces.keys())[0][:-2]
            if os.path.exists("RetinaFace/mask"):
                shutil.rmtree("RetinaFace/mask")
            os.makedirs("RetinaFace/mask")
            fr = 0
            """
            video = video.numpy()
            video = np.zeros(video.shape)  

            for f in faces:  
                if f[:-2] == prev:
                     #print(f[:-2]," ", prev) 
                     coord.append(faces[f])
                     prev = f[:-2]
                else:
                    new_img = self.apply_mask(prev+'.jpg', coord)
                    #print("frame torch shape: ", (torch.from_numpy(new_img[:,:,0])).size())
                    
                    video[0,fr,:,:] = new_img[:,:,0]
                    video[1,fr,:,:] = new_img[:,:,1]
                    video[2,fr,:,:] = new_img[:,:,2]

                    cv2.imwrite("RetinaFace/mask/"+prev+'.jpg', new_img)
                    coord = []
                    coord.append(faces[f])
                    prev = f[:-2]
                    fr+=1
            video = torch.from_numpy(video)
            print("shape BEFORE: ", video.size()) 
            #video = torch.unsqueeze(video, 0)
            print("shape AFTER: ", video.size())
            print(video[0,50,:,:])
            print(torch.sum(video))
            b_id = 0
            #self.video_saver.save_mp4_from_vid_and_audio(video)
            """
            #BETTER:


















            #OLD ALGORITHM
            max_norm = 958.57 #diagonal size of frame 
            print("\nsortFACES:\n", faces) 
            """
            print("length: ", len(list(faces.keys())))
            for i in range(0,len(list(faces.keys()))):
                tr = {}
                track = [] 
                max_face = ''
                max_f_sim = 0
                distance = 0
                dist_seed = list(faces.keys())[i]
                #print("dist_seed: ",dist_seed)
                prev_f = i
                flag_dist = 0 
                #FOWARD PROPAGATION:
                for j in range (i+1,len(list(faces.keys()))):
                    if list(faces.keys())[i][:-2] == list(faces.keys())[j][:-2]:	#if same frame of the seed, skip
                        continue
                    else:                                                               #else
                        if max_face == '':
                            max_face =list(faces.keys())[j] #???
                        #print(list(faces.keys())[j])
                        

                        if flag_dist==0:	#initialize maxs
                            #max_face = list(faces.keys())[i] #???
                            max_f_sim = 0 #max_f_sim= j+1 ???
                            dist_seed = list(faces.keys())[i]
                            flag_dist = 1

                        print(list(faces.keys())[j], " - ", dist_seed, " = ",int(list(faces.keys())[j][7:-2]) - int(dist_seed[7:-2]) )
                        if int(list(faces.keys())[j][7:-2]) - int(dist_seed[7:-2]) >1:
                            distance = 0
                        else: 
                            
                            distance=math.sqrt(
                                 (list(faces.values())[j][0]-faces[dist_seed][0])**2
                                 + 
                                 (list(faces.values())[j][1]-faces[dist_seed][1])**2
                                 )
                        
                        if distance ==0:
                            triang_dist[i][j] = 0
                        else:
                            triang_dist[i][j] = 1/distance

                            
                            

                        #print(distance)   
                        #reminder: MAX DISTANCE is 958
                        if distance != 0:
                            #print(1/distance)
                            triang_sim[i][j] = (1/distance)*float(cos(
                                faces_feat[list(faces.keys())[i]].to(self.device), faces_feat[list(faces.keys())[j]].to(self.device)
                             ).to(self.device))
                            
                            #print("sim:", triang_sim[i][j])
                        else:                        
                             triang_sim[i][j] = float(cos(
                                faces_feat[list(faces.keys())[i]].to(self.device), faces_feat[list(faces.keys())[j]].to(self.device)
                                ).to(self.device))
                             #print("sim:", triang_sim[i][j])


                       
                        #now, choose the face to add to the tracklet:

                        if flag_dist==0:	#initialize maxs
                           #max_face = list(faces.keys())[i] #???
                           max_f_sim = 0 #max_f_sim= j+1 ???
                           dist_seed = list(faces.keys())[i]
                           flag_dist = 1
                        else:

                            if list(faces.keys())[j][:-2] == list(faces.keys())[j-1][:-2]:	#if same frame but different face
                                if triang_sim[i][j] > max_f_sim:
                                    max_f_sim = triang_sim[i][j]
                                    max_face = list(faces.keys())[j]
                            else:
                                if list(faces.keys())[j][:-2] != list(faces.keys())[i][:-2]:  #if NOT first frame after seed frame
                                    tr[max_face[:-2]+'.jpg'] = faces[max_face]                #add max_face from previous step 
                                dist_seed = max_face                                      #dist_seed = max face from previous frame       #controllare, (alternativa: faces[j-1]
                                #update vars:  
                                max_f_sim = triang_sim[i][j]                                   
                                max_face=list(faces.keys())[j]
 
                        #print("max_face: ", max_face)
                max_face = ''
                max_f_sim = 0
                dist_seed = list(faces.keys())[i]
                 
                #BACKWARD PROPAGATION: (scorro i frame "che ho alle spalle", per i quali ho già i valori di similiarità)
                for k in range(i-1,0,-1):

                     if list(faces.keys())[k][:-2] == list(faces.keys())[i][:-2]:	#if same frame of the seed, skip
                        continue

                     if max_face == '':
                         max_face = list(faces.keys())[k]
                     if int(dist_seed[7:-2]) - int(list(faces.keys())[k][7:-2]) > 1:
                         distance = 0
                     else:
                         distance=math.sqrt(
                                 (list(faces.values())[k][0]-faces[dist_seed][0])**2
                                 + 
                                 (list(faces.values())[k][1]-faces[dist_seed][1])**2
                                 )
                     if distance ==0:
                         triang_dist[i][j] = 0
                     else:
                         triang_dist[i][j] = 1/distance 
                     if distance != 0:
                         #print(1/distance)
                         triang_sim[i][j] = (1/distance)*float(cos(
                                faces_feat[list(faces.keys())[i]].to(self.device), faces_feat[list(faces.keys())[j]].to(self.device)
                             ).to(self.device))
                            
                         #print("sim:", triang_sim[i][j])
                     else:                        
                         triang_sim[i][j] = float(cos(
                                faces_feat[list(faces.keys())[i]].to(self.device), faces_feat[list(faces.keys())[j]].to(self.device)
                                ).to(self.device))
                     if flag_dist==0:	#initialize maxs
                           #max_face = list(faces.keys())[i] #???
                           max_f_sim = 0 #max_f_sim= j+1 ???
                           dist_seed = list(faces.keys())[i]
                           flag_dist = 1
                     if list(faces.keys())[k][:-2] == list(faces.keys())[k+1][:-2]:
                         if triang_sim[k][i] > max_f_sim:
                             max_f_sim = triang_sim[k][i] #i o j?
                             max_face = list(faces.keys())[k]
                     else:
                         #tr[max_face[:-2]+'.jpg'] = self.getFaceCoords(max_face)
                         if list(faces.keys())[k+1][:-2] != list(faces.keys())[i][:-2]:
                             tr[max_face[:-2]+'.jpg'] = faces[max_face] 
                             dist_seed = max_face 
                         max_f_sim = triang_sim[k][i]
                         max_face = list(faces.keys())[k]

                #print("tr:\n",tr)
                tracklet[list(faces.keys())[i]] = tr
                #print("tracklet["+list(faces.keys())[i]+"]:")  
                #print(tracklet[list(faces.keys())[i]])        
            print("triang_sim: ",triang_sim)
            print(triang_sim.shape)
            print("max_sim: ", triang_sim.max())
            print("example tracklet: (70_0) \n",tracklet["enc_img70_0"])
            
            #DEBUG:
            for i in faces.keys():
                for j in faces.keys():
                    if i in ['enc_img_0','enc_img0_1','enc_img92_0','enc_img92_1', 'enc_img91_0', 'enc_img91_1','enc_img65_0','enc_img65_1','enc_img87_0','enc_img87_1']:
                        if j in [ 'enc_img63_0', 'enc_img63_1',
                                      'enc_img64_0','enc_img64_1','enc_img65_0','enc_img65_1','enc_img66_0','enc_img66_1','enc_img67_0','enc_img67_1','enc_img68_0',
                                      'enc_img68_1','enc_img69_0','enc_img69_1','enc_img70_0','enc_img70_1','enc_img71_0','enc_img71_1','enc_img72_0','enc_img72_1','enc_img73_0','enc_img73_1',
                                       'enc_img74_0', 'enc_img74_1']:
                            if j>i:
                                print("sim[",i,"][",j,"]: ",triang_sim[list(faces.keys()).index(i)][list(faces.keys()).index(j)])
                                print("dist[",i,"][",j,"]: ",triang_dist[list(faces.keys()).index(i)][list(faces.keys()).index(j)])
                            else:
                                print("sim[",i,"][",j,"]: ",triang_sim[list(faces.keys()).index(j)][list(faces.keys()).index(i)])
                                print("dist[",i,"][",j,"]: ",triang_dist[list(faces.keys()).index(j)][list(faces.keys()).index(i)])
            
            """
            #print("faces_feat: ", faces_feat)
            for i in range(0,len(list(faces.keys()))):
                for j in range (i+1,len(list(faces.keys()))):
                    if list(faces.keys())[i][:-2] == list(faces.keys())[j][:-2]:	#if same frame of the seed, skip
                         continue    
                    triang_sim[i][j] = float(cos(
                                faces_feat[list(faces.keys())[i]].to(self.device), faces_feat[list(faces.keys())[j]].to(self.device)
                                ).to(self.device))
            print("triang_sim:\n",triang_sim) 
            #NEW DISTANCE:
            tracklet = {}
            for i in range(0,len(list(faces.keys()))):
                dist_seed = list(faces.keys())[i] 
                max_score = 0
                init_flag = 1
                if i+1 < len(list(faces.keys())):
                    max_el = list(faces.keys())[i+1] 
                tr = {}
                #FOWARD:    
                for j in range (i+1,len(list(faces.keys()))): 
                    if i == 0:
                        print("J: ", j)
                    if list(faces.keys())[j][:-2] == list(faces.keys())[i][:-2]: #if same frame of the seed:
                        print("continue: ",list(faces.keys())[j], " ",list(faces.keys())[i])
                        if j+1 < len(list(faces.keys())):
                            max_el = list(faces.keys())[j+1]    
                        continue
                    
                    if init_flag == 0: # if primo frame dopo il seed
                         
                        distance=math.sqrt(
                                 (list(faces.values())[j][0]-faces[dist_seed][0])**2
                                 + 
                                 (list(faces.values())[j][1]-faces[dist_seed][1])**2
                                 )
                        if distance == 0:
                            max_score = triang_sim[i][j]
                        else:
                            max_score = 1/distance * triang_sim[i][j]
                        max_el = list(faces.keys())[j]
                        init_flag = 1
                    else:
                        if i == 0:   
                            print("dist_seed: ", dist_seed," elem: ",list(faces.keys())[j], "max_el: ", max_el)  
                        if list(faces.keys())[j][:-2] != max_el[:-2]:
                            #CHECK MUST BE HERE
                            if (list(faces.keys())[j-1][:-2] != list(faces.keys())[j-2][:-2]) and (list(faces.keys())[j-1][:-2] != list(faces.keys())[j][:-2]): #if only one face in previous frame
                                if list(faces.keys())[i-1][:-2] == list(faces.keys())[i][:-2]:
                                    #print("FIRST IF")
                                    if triang_sim[i][j-1] >= triang_sim[i-1][j-1]:
                                        print(list(faces.keys())[j-1])
                                        print(list(faces.keys())[i],triang_sim[i][j-1],list(faces.keys())[i-1], triang_sim[i-1][j-1])
                                        dist_seed = max_el 			#update seed
                                        tr[dist_seed[:-2]+'.jpg'] = faces[dist_seed]
                                    """
                                    else:
                                    #?
                                    """
                                elif list(faces.keys())[i+1][:-2] == list(faces.keys())[i][:-2]:
                                    #print("second if")
                                    if triang_sim[i][j-1] >= triang_sim[i+1][j-1]:
                                        #print("inside if")
                                        dist_seed = max_el 			#update seed
                                        tr[dist_seed[:-2]+'.jpg'] = faces[dist_seed]
                                    """
                                    else:
                                    #?
                                    """
                                else:                                                       #if only one seed from this frame
                                    threshold = 0.86
                                    if triang_sim [i][j-1] > threshold:
                                        #print("triang sim >")
                                        dist_seed = max_el 			#update seed
                                        tr[dist_seed[:-2]+'.jpg'] = faces[dist_seed]
                            else: #more than 1 face                                    
                                dist_seed = max_el
                                tr[dist_seed[:-2]+'.jpg'] = faces[dist_seed]

                            #dist_seed = max_el 			#update seed
                            #tr[dist_seed[:-2]+'.jpg'] = faces[dist_seed]    #add new seed to tracklet
                            max_score = -1                       #re initialize max
                            max_el = list(faces.keys())[j]
                            if i == 0: 
                                print("NEW dist_seed: ", dist_seed, "elem: ",list(faces.keys())[j], "max_el: ", max_el)
     
                        distance=math.sqrt(
                                 (list(faces.values())[j][0]-faces[dist_seed][0])**2
                                 + 
                                 (list(faces.values())[j][1]-faces[dist_seed][1])**2
                                 )
                        if distance == 0 or ( int(list(faces.keys())[j][7:-2]) - int(dist_seed[7:-2]) ) > 1 : 
                            score = triang_sim[i][j]
                            triang_score[i][j] = score
                        else:
                            score = (1/distance)*triang_sim[i][j]
                            triang_score[i][j] = score
                        if i == 0:     
                            print("score: ", score, "max_score: ", max_score)               
                        if score > max_score:		#if new face score > previous
                            max_score = score
                            max_el = list(faces.keys())[j]
 		#BACKWARD:
                max_score = 0
                dist_seed = list(faces.keys())[i]
                if i == 50: 
                    print("\n\nBACKWARD")
                    print("dist_seed: ", dist_seed)   
                if i-1 > 0:
                    max_el = list(faces.keys())[i-1]   
                for k in range (i-1,0,-1): 
                    if i == 50:
                        print("K: ", k)
                    if list(faces.keys())[k][:-2] == list(faces.keys())[i][:-2]: #if same frame of the seed:
                        if i == 50:
                            print("continue: ",list(faces.keys())[k], " ",list(faces.keys())[i])
                        if k-1 > 0:
                            max_el = list(faces.keys())[k-1]    
                        continue
                    
                    if init_flag == 0: # if primo frame dopo il seed
                         
                        distance=math.sqrt(
                                 (list(faces.values())[k][0]-faces[dist_seed][0])**2
                                 + 
                                 (list(faces.values())[k][1]-faces[dist_seed][1])**2
                                 )
                        if distance == 0:
                            max_score = triang_sim[k][i]
                        else:
                            max_score = 1/distance * triang_sim[k][i]
                        max_el = list(faces.keys())[k]
                        init_flag = 1
                    else:
                        if i == 50:   
                            print("dist_seed: ", dist_seed," elem: ",list(faces.keys())[k], "max_el: ", max_el)  
                        if list(faces.keys())[k][:-2] != max_el[:-2]: #if frame changed

                            #CHECK MUST BE HERE
                            if list(faces.keys())[j-1][:-2] != list(faces.keys())[j-2][:-2]: #if only one face in previous frame
                                #print("FIRST IF")
                                if list(faces.keys())[i-1][:-2] == list(faces.keys())[i][:-2]:
                                    if triang_sim[i][k+1] >= triang_sim[i-1][k+1]:
                                        #print("INSIDE if")
                                        dist_seed = max_el 			#update seed
                                        tr[dist_seed[:-2]+'.jpg'] = faces[dist_seed]
                                    """
                                    else:
                                    #?
                                    """
                                elif list(faces.keys())[i+1][:-2] == list(faces.keys())[i][:-2]:
                                    #print("SECOND IF")
                                    if triang_sim[i][k+1] >= triang_sim[i+1][k+1]:
                                        #print("INSIDE IF")
                                        dist_seed = max_el 			#update seed
                                        tr[dist_seed[:-2]+'.jpg'] = faces[dist_seed]
                                    """
                                    else:
                                    #?
                                    """
                                else:
                                    #print("ELSE")                                                       #if only one seed from this frame
                                    threshold = 0.86
                                    if triang_sim [i][k+1] > threshold:
                                        #print("TRIANG SIM >")
                                        dist_seed = max_el 			#update seed
                                        tr[dist_seed[:-2]+'.jpg'] = faces[dist_seed]
                            else: #more than 1 face                                    
                                dist_seed = max_el
                                tr[dist_seed[:-2]+'.jpg'] = faces[dist_seed]

                            #dist_seed = max_el 			#update seed
                            #tr[dist_seed[:-2]+'.jpg'] = faces[dist_seed]    #add new seed to tracklet
                            max_score = -1                       #re initialize max
                            max_el = list(faces.keys())[k]
                            if i == 50: 
                                print("NEW dist_seed: ", dist_seed, "elem: ",list(faces.keys())[k], "max_el: ", max_el)
     
                        distance=math.sqrt(
                                 (list(faces.values())[k][0]-faces[dist_seed][0])**2
                                 + 
                                 (list(faces.values())[k][1]-faces[dist_seed][1])**2
                                 )
                        if distance == 0 or ( int(dist_seed[7:-2]) - int(list(faces.keys())[j][7:-2]) ) > 1: 
                            score = triang_sim[k][i]
                            triang_score[k][i] = score 
                        else:
                            score = (1/distance)*triang_sim[k][i]
                            triang_score[k][i] = score 
                        if i == 50:     
                            print("score: ", score, "max_score: ", max_score)               
                        if score > max_score:		#if new face score > previous
                            max_score = score
                            max_el = list(faces.keys())[k]
                   
                tracklet[list(faces.keys())[i]] = tr
            print(tracklet[list(faces.keys())[0]]) 
            """faces
            #shoe just ONE tracklet
            if os.path.exists("tracklet/enc_img70_0"):
                shutil.rmtree("tracklet/enc_img70_0")
            os.makedirs("tracklet/enc_img70_0")
            for key in tracklet["enc_img70_0"]:
                img_raw = Image.open(dataset+"/"+key)
                coords = tracklet["enc_img70_0"][key]
                xmax = coords[0]+coords[2]
                ymax = coords[1]+coords[3]
                face = img_raw.crop((coords[0],coords[1],xmax,ymax)) 
                face.save("tracklet/enc_img70_0"+"/"+key)  
            """
            """  
            #show tracklets
            for i in range (len(list(faces.keys()))):
                for key in tracklet[list(faces.keys())[i]]:
                    img_raw = Image.open(dataset+"/"+key)
                    coords = tracklet[list(faces.keys())[i]][key]
                    xmax = coords[0]+coords[2]
                    ymax = coords[1]+coords[3]
                    face = img_raw.crop((coords[0],coords[1],xmax,ymax))
                    if not os.path.exists("tracklet/"+list(faces.keys())[i]):
                        os.makedirs("tracklet/"+list(faces.keys())[i])
                    face.save("tracklet/"+list(faces.keys())[i]+"/"+key)  
            """ 
            eBoT = []
                 
            eBoT.append(self.BOT(tracklet))

            eBoT.append(self.BOT(self.removekey(tracklet, list(faces.keys())[0])))
            for i in range(len(eBoT)):             
                for key in eBoT[i]:
                    if not os.path.exists("BagOfTrack_"+str(i)):
                        os.makedirs("BagOfTrack_"+str(i))
                    if os.path.exists("BagOfTrack_"+str(i)+"/"+key):
                        shutil.rmtree("BagOfTrack_"+str(i)+"/"+key)
                    try:
                        shutil.copytree("tracklet/"+key, "BagOfTrack_"+str(i)+"/"+key)
                    except:
                        #print("skipped ", str(i))
                        continue 
            """                  
            print(list(faces.keys())[0])
            #print(tracklet[list(faces.keys())[0]])
            #REMOVE UNBOT:
            for b in range(len(eBoT)):
                 density=len(eBoT[b])/frames
                 print(density)
                 if density < 0.2:
                     shutil.rmtree("BagOfTrack_"+str(b))
                     print("BOT REMOVED")
        
            """        
            """
            if tracklet.keys()[1][:-2] == tracklet.keys()[0][:-2]:
               self.BOT(tracklet[1:])
            """


            imgdir = []
            for file in os.listdir(dataset):
                if file.endswith(".jpg"):
                    imgdir.append(file)

            #CALCULATE CONFIDENCE
            confidence = []
            for i in range(len(eBoT)): #i: number of bots
                conf = {}
                for f in imgdir:       #for every frame      
                    sum_c = 0  
                    for s in eBoT[i]:  #for every tracklet  
                        
                        if f not in tracklet[s]:
                            continue
                        index_s = list(faces.keys()).index(s) 
                        index_k =list(tracklet[s].values()).index(tracklet[s][f])
                        sim = triang_sim[index_s][index_k]
                        if sim == 0:
                            sim = triang_score[index_k][index_s]
                        sum_c += sim
                    conf[f] = sum_c/len(eBoT[i])
                confidence.append(conf)
                print("confidence["+str(i)+"]", conf)  
            #print("FRAMES CONFIDENCES: \n",confidence)
            """
            print("zeros\n")  
            for c in conf.keys():
                if conf[c] ==0:
                    print(c)
                    print("\n")
            """    
            #PROTOTYPE GENERATION
               
            prototype = []
            prot = {}
            kmax = '' 
            pr_flag = 0
            thresh_L = 0.13
            for i in range(len(eBoT)): #for each bag of trackl
                removed = 0
                prot = {}  
                for f in imgdir:       #for j=1:seqLength
                    #print("f: ", f)
                    prev_summ = 0
                    kmax = ''
                    for j in eBoT[i]:  #per ogni tracklet nella bag i
                        #print("J: ",j) #j scorre normalmente (già controllato)
                        if f not in tracklet[j].keys() or f == j[:-2]+'.jpg':
                            continue 
                        """
                        if confidence[i][f]<thresh_L:
                            print("occluded frame: ",f)
                            removed +=1
                            continue
                        """
                        area1 = tracklet[j][f][2]*tracklet[j][f][3]
                        a = tracklet[j][f]
                        #start = tracklet[j].keys().index(f)
                        inter = 0 
                        #kmax = ''
                        summ = 0
                        
                        #sommatoria inters tra j e gli altri trackl in f  
                        for kk in eBoT[i]: #bag of  tracklet i
                            inter = 0
                            dx = 0
                            dy = 0
                            if f not in tracklet[kk].keys() or j==kk:
                                continue
                            area2 = tracklet[kk][f][2]*tracklet[kk][f][3]    
                            b = tracklet[kk][f]
                            dx = min(a[0]+a[2], b[0]+b[2]) - max(a[0],b[0])
                            dy = min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1])
                            if dx >=0 and dy>=0:
                                inter=dx*dy #intersection of tracklet kk and tracklet j in frame f
                                #print("inter: ", inter)
                                summ += inter #/ (area1+area2-inter)
                        #print("SUMM: ", summ, "j: ", j)
                        if summ > prev_summ:
                            #print("maggiore")
                            #print("f: ", f, "J: ", j)
                            #print("summ: ",summ, "prev: ", prev_summ)
                            kmax = j
                            #print("kmax: ", kmax)    
                            prev_summ =summ
                    if kmax !='' :                    
                        prot[f]=kmax
                        r = tracklet[kmax][f] 
                        #print("tracklet[",kmax,"][",f,"]: ",tracklet[kmax][f])
                        img_raw = cv2.imread(dataset+"/"+f)
                        
                        cv2.rectangle(img_raw, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 0, 255), 2)
                        #img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                        if not os.path.exists("prototype"+str(i)):
                            os.makedirs("prototype"+str(i)) 
                        cv2.imwrite("prototype"+str(i)+'/'+f, img_raw)
                            
                print(j)        
                print("prot["+str(i)+"]:\n", prot)
                print("len: ",len(prot.keys()))
                prototype.append(prot)
                #print("num occluded frames: ", removed)
            trajectories = self.trajectories_from_faces(prototype,video,faces,tracklet)
            print(trajectories[0])
            print(trajectories[1])

            """
            #APPLY MASK FROM PROTOTYPES:
            video = video.numpy()
            video = np.zeros(video.shape)
            for i in range(len(video)):
                for p in prototype:
                    if 'enc_img'+str(i)+'.jpg' in p.keys():
                        coord.append(faces[p['enc_img'+str(i)+'.jpg']])
                        if len(coord)==0:
                            continue
                new_img = apply_mask('enc_img'+str(i)+'.jpg',coord)   
                video[0,fr,:,:] = new_img[:,:,0]
                video[1,fr,:,:] = new_img[:,:,1]
                video[2,fr,:,:] = new_img[:,:,2]
                video = torch.from_numpy(video)
            #self.video_saver.save_mp4_from_vid_and_audio(video)
            """        
        #return "", 0
        return np.array(trajectories)

    def removekey(self,d, key):
        r = dict(d)
        del r[key]
        return r

    def getFaceCoords(self, face):
        facename, facenum = face[:-2], int(face[-1])
        f = open(dataset+'/thresh_dets.txt', 'r') #only thresholded dets coords for every img
        Lines = f.readlines()
        #print(Lines)
        #print(len(Lines))
        i=Lines.index('/'+facename+'.jpg\n')+facenum+1
        #print(i)
        #print(Lines[i])
        coords = [int(s) for s in Lines[i].split(' ')]
        
        f.close()
        #print(coords)
        return coords

    def back(self, s_att_av, vid_emb, aud_emb, part_len=10, retard=3):
        """
        :backpropagation
	"""
	#s_att_v = max(s_av(x, y, t))
	#loss = -log(exp(s_att_av(v, a_i)/(exp(s_att_av(v,a_i)+sum_j(s_att_av(v,a_j)))
        exp_j = self.shifted(vid_emb,aud_emb)
        print("LOSSES:")
        for i in range(part_len,len(s_att_av),part_len):
            loss= -1*np.log(np.divide(np.exp(np.array(s_att_av[(i-part_len):i])),
                                   np.add(np.exp(np.array(s_att_av[(i-part_len):i])),exp_j)
                                  )
                           )
            print(loss)
        #loss[0].backward()
        return 1 

    def shifted(self, vid_emb, aud_emb, part_len=10):
        """
        :sum of exp of s_att for shifted audio
        """
        device = self.device
        # b x C x   1   x T x h x w
        vid_emb = vid_emb[:, :, None]
        print("new vid_emb shape: ", vid_emb.shape)
        # b x C x n_neg x T x 1 x 1
        aud_emb = aud_emb.transpose(1, 2)[..., None, None]
        print("new aud_emb shape: ", aud_emb.shape)
        vid_chunks = []
        aud_chunks = []
        dist_chunk = []
        vid_chunks = list(vid_emb.split(part_len, 3)) #(size, dim)
        aud_chunks = list(aud_emb.split(part_len, 3))

        #now exchange the audio
        last = len(aud_chunks) - 1
        print("chunks len: ", len(aud_chunks))
        aud_chunks.insert(0,aud_chunks[last])         #one chunk shift
        del aud_chunks[-1]
        #print(aud_chunks[:2])
        av = []
        for i in range(0, len(vid_chunks)):
            av.append(tuple((vid_chunks[i], aud_chunks[i])))
        #print(" av: ",av[0])
        dim = 3
        func = lambda x, y: (x*y).sum(1)
        i = 0
        for t in av:                                  # for each tuple (v_chunk, a_chunk)
            v_split = t[0]
            a_split = t[1]
            dist_chunk.append(func(v_split.to(device),a_split.to(device)))
        #dist = torch.cat(dist_chunk, dim - 1)
        #dist = self.model.logits_scale(dist[..., None]).squeeze(-1)
            dist_chunk[i] = self.model.logits_scale(dist_chunk[i][..., None]).squeeze(-1)
            frames,h_map,w_map = dist_chunk[i].shape[-3:]
            #print("DIST_chunk shape: ", dist_chunk[i].shape)
            s_att_av_J = []
            for fr in range(0,frames):
                s_att_av_J.append(torch.max(dist_chunk[i][:,:,fr,:,:]).item())

            if i==0:
                exp_j = np.exp(np.array(s_att_av_J))
            else:
                exp_j = np.add(exp_j, np.exp(np.array(s_att_av_J)))
            i=i+1
        print("exp_j: ", exp_j)
        #print("s_att_av lenght: ", len(s_att_av))
        #print(s_att_av)

        """
        dist = torch.cat(dist_chunk, dim - 1)
        #dist = self.model.logits_scale(dist[..., None]).squeeze(-1)
        frames,h_map, w_map = dist.shape[-3:]
        print("DIST shape: ", dist.shape)
        s_att_av = []
        for fr in range(0,frames):
            s_att_av.append(torch.max(dist[:,:,fr,:,:]).item())
        print("s_att_v lenght: ", len(s_att_av))
        """
        return exp_j

    def BOT(self,tracklets):
        print("\n\n\nEBOT GENERATION:")
        thresh = 0.8
        tlist = list(tracklets.keys())
        #print("tlist:\n",tlist)
        eBoT = [] 
        #remove 0 from list:
        eBoT.append(tlist.pop(0))
        #vectScore = []
        #vectIndex = []
        vectScore = {}
        y=0
        flag = 0
        sim = {}
        simil = []
        summ_ss = {}
        while flag == 0:
            #print("eBoT:\n",eBoT)   
            #vectScore = {}
            i=0
            #for t in eBoT: #for each tracklet seed name in ebot so [ Ti ]    #PROBABILMENTE NON SERVE BASTA t = eBoT[-1]           
            t = eBoT[-1]
            ss = 0
            s = 0
            sim = {}
            #print("TR = tracklets["+name+"]:\n")
            #print(tr)
            #summ_ss = {}
            while len(tlist) > 0: #while there are tracklets left to analyz (list of Tjs)                 
                #print("t: ",t)
                #print("updated eBoT:\n")
                #print(eBoT)
                summ = 0 
                name = tlist.pop(0)
                tr = tracklets[name] #trkName = Nametrks[1]
                #loop = (dic for dic in simil) 
                """              
                if name in [simil[i].keys() for i in len(simil)]:                
                    s = simil[i][name]
                    break
                else:
                """  
                for key in tracklets[t]: #foreach img in selected tracklet from ebot
                    if key[:-4] == name[:-2]: #if frame == seed
                        continue
                    if key not in tr.keys():
                        continue      
                    inter = 0
                    #area of intersect of tracklets[t][key] and tr[key] (intersect of values, which means coords)
                    area1 = tracklets[t][key][2] * tracklets[t][key][3] #w*h
                    area2 = tr[key][2]*tr[key][3]
                    a = tracklets[t][key] #x,y,w,h
                    b = tr[key]
                    dx = min(a[0]+a[2], b[0]+b[2]) - max(a[0],b[0])
                    dy = min(a[1]+a[3], b[1]+b[3]) - max(a[1],b[1])
                    if dx >= 0 and dy>=0:
                        inter = dx*dy
                
                    summ += inter / (area1+area2-inter)
                if len(tracklets[t]) == 0:
                    s = 0
                else:  
                    s = summ/len(tracklets[t]) #S(Ti,Tj), so S(tracklets[t], tr)
                #end else   
                sim[name] = s 
                                
            simil.append(sim)               
            if not summ_ss:
                summ_ss = sim
                for j in summ_ss.keys():
                    ss = summ_ss[j]/len(eBoT)
                    if ss > thresh:
                        vectScore[j] = ss
            else: 
                #per ogni traklet sommo la similarita di questa riga con summ di questo tracklet:                
                for j in list(summ_ss.keys()):
                    if j not in simil[-1].keys():
                        if j in vectScore.keys():
                            del vectScore[j]
                        continue 
                    summ_ss[j] =+simil[-1][j]
                    ss = summ_ss[j]/len(eBoT) #average S(eBoT,Tj) for every j
                                           # first time: summ_ss = s and len = 1 
                    if ss > thresh: #and isempty(ids)
                        vectScore[j] = ss #dict
            
    
            
                #print(vectScore) 
                #now sort
            import operator
            sortedv = sorted(vectScore.items(), key=operator.itemgetter(1))
            """ 
            if not sortedv:
                print("BREAK")
                break
            """          
            #print("SORTED") 
            #print(sortedv) # {(Tj, score), ...}
            #now I have all scores > thresh...
            """  
            for m in range(len(vectScore)):
                #maxIdx = vectScore.index(max(vectScore))
                vm = max(vectScore.values())
                for k,v in vectScore.items():
                    if v == vm:
                        maxIdx = k
                        break
               
                
                print("maxIdx: ",maxIdx)                
                y = y+1
                vectScore[maxIdx] = 0
                #vectIndex[maxIdx] = 0            
            """ 
            i +=1
            max_el = sortedv.pop(-1)
            eBoT.append(max_el[0]) 
            
            #print("tlist:\n",tlist)
            #tlist = sortedv[0]
            tlist = [x for x,_ in sortedv]
            #print("LEN_TLIST: ", len(tlist))
            if len(tlist)==0:
                flag=1
            #print("new tlist: \n",tlist)            
        print("init len: ", len(list(tracklets.keys())))  
        print("\nFINAL EBOT:\n",eBoT)
        print("len: ", len(eBoT))      
        return eBoT     




# ============ att maps & av scores ============

    def apply_mask(self, frame, c):
        #print("apply mask: ")  
        frame = cv2.imread("RetinaFace/faces/"+frame)
        frame = np.array(frame)       
        new = np.zeros(frame.shape)
        #print(new.shape)
        #print("LEN: ",len(c))
        #print(c)
        for i in range(len(c)):
            #print(c[i])
            new[c[i][1]:(c[i][1]+c[i][3]),c[i][0]:(c[i][0]+c[i][2]),:] = frame[c[i][1]:(c[i][1]+c[i][3]),c[i][0]:(c[i][0]+c[i][2]),:]
        return new

    def calc_av_scores(self, vid_emb, aud_emb):
        """
        :return: aggregated scores over T, h, w
        """

        scores = self.calc_att_map(vid_emb, aud_emb)
        att_map = logsoftmax_2d(scores)

        scores = torch.nn.MaxPool3d(kernel_size=(1, scores.shape[-2],
                                                 scores.shape[-1]))(scores)
        scores = scores.squeeze(-1).squeeze(-1).mean(2)

        return scores, att_map

    def calc_att_map(self, vid_emb, aud_emb):
        """
        :param vid_emb: b x C x T x h x w
        :param aud_emb: b x num_neg x C x T
        """

        # b x C x   1   x T x h x w
        vid_emb = vid_emb[:, :, None]
        print("new vid_emb shape: ", vid_emb.shape)
        # b x C x n_neg x T x 1 x 1
        aud_emb = aud_emb.transpose(1, 2)[..., None, None]
        print("new aud_emb shape: ", aud_emb.shape)
        scores = run_func_in_parts(lambda x, y: (x * y).sum(1),
                                   vid_emb,
                                   aud_emb,
                                   part_len=10,
                                   dim=3,
                                   device=self.device)

        # this is the learned logits scaling to move the input to the softmax
        # out of the [-1,1] (e.g what SimCLR sets to 0.07)
        scores = self.model.logits_scale(scores[..., None]).squeeze(-1)

        return scores

    # ============ online AV sync  ============

    def online_sync(self, vid_emb, aud_emb, video, audio, vid_feat):

        sync_offset = self.calc_optimal_av_offset(vid_emb, aud_emb)

        vid_emb_sync, aud_emb_sync = self.sync_av_with_offset(vid_emb,
                                                              aud_emb,
                                                              sync_offset,
                                                              dim_v=2,
                                                              dim_a=3)

        vid_feat_sync, _ = self.sync_av_with_offset(vid_feat,
                                                    aud_emb,
                                                    sync_offset,
                                                    dim_v=2,
                                                    dim_a=3)

        # e.g. a_mult = 640 for 16khz and 25 fps
        a_mult = int(self.opts.sample_rate / self.opts.fps)
        video, audio = self.sync_av_with_offset(video,
                                                audio,
                                                sync_offset,
                                                dim_v=2,
                                                dim_a=1,
                                                a_mult=a_mult)
        return video, audio, vid_emb_sync, aud_emb_sync, vid_feat_sync

    def create_online_sync_negatives(self, vid_emb, aud_emb):
        assert self.opts.n_negative_samples % 2 == 0
        ww = self.opts.n_negative_samples // 2

        fr_trunc, to_trunc = ww, aud_emb.shape[-1] - ww
        vid_emb_pos = vid_emb[:, :, fr_trunc:to_trunc]
        slice_size = to_trunc - fr_trunc

        aud_emb_posneg = aud_emb.squeeze(1).unfold(-1, slice_size, 1)
        aud_emb_posneg = aud_emb_posneg.permute([0, 2, 1, 3])

        # this is the index of the positive samples within the posneg bundle
        pos_idx = self.opts.n_negative_samples // 2
        aud_emb_pos = aud_emb[:, 0, :, fr_trunc:to_trunc]

        # make sure that we got the indices correctly
        assert torch.all(aud_emb_posneg[:, pos_idx] == aud_emb_pos)

        return vid_emb_pos, aud_emb_posneg, pos_idx

    def calc_optimal_av_offset(self, vid_emb, aud_emb):
        vid_emb, aud_emb, pos_idx = self.create_online_sync_negatives(
            vid_emb, aud_emb)
        scores, _ = self.calc_av_scores(vid_emb, aud_emb)
        offset = scores.argmax() - pos_idx
        return offset.item()

    def sync_av_with_offset(self,
                            vid_emb,
                            aud_emb,
                            offset,
                            dim_v,
                            dim_a,
                            a_mult=1):
        if vid_emb is not None:
            init_dim = vid_emb.shape[dim_v]
        else:
            init_dim = aud_emb.shape[dim_a] // a_mult

        length = init_dim - int(np.abs(offset))

        if vid_emb is not None:
            if offset < 0:
                vid_emb = vid_emb.narrow(dim_v, -offset, length)
            else:
                vid_emb = vid_emb.narrow(dim_v, 0, length)

            assert vid_emb.shape[dim_v] == init_dim - np.abs(offset)

        if aud_emb is not None:
            if offset < 0:
                aud_emb = aud_emb.narrow(dim_a, 0, length * a_mult)
            else:
                aud_emb = aud_emb.narrow(dim_a, offset * a_mult,
                                         length * a_mult)
            assert aud_emb.shape[dim_a] // a_mult == init_dim - np.abs(offset)

        return vid_emb, aud_emb

    # ============ avobjects tracking ============

    def avobject_trajectories_from_flow(self, att_map, video):
        """
        Use the av attention map to aggregate vid
        features corresponding to peaks
        """

        # - flow not provided, we need to calculate it on the fly
        flows = []
        for b_id in range(len(video)):
            flow_inp = video.permute([0, 2, 3, 4, 1])

            # NOTE: This is a workaround to the GPL license of PWCnet wrapper
            # We call it as an executable: The network is therefore initialized
            # again in each call and the input images and output flow are passed
            # by copying into shared memory (/dev/shm)
            # This is very suboptimal - not to be used for training
            flow = calc_flow_on_vid_wrapper(flow_inp[b_id].detach().cpu().numpy(), gpu_id=self.opts.gpu_id)
            flow = torch.from_numpy(flow).permute([0, 2, 3, 1])  # tchw -> thwc
            flows.append(flow)
        flow = np.stack(flows)
        flow = torch.from_numpy(flow)
        flow = torch.nn.ConstantPad3d([0, 0, 0, 0, 0, 0, 0, 1], 0)(flow)

        def smoothen_and_pad_att_map(av_att, pad_resid=2):
            map_t_avg = torch.nn.AvgPool3d(kernel_size=(5, 1, 3),
                                            stride=1,
                                            padding=(2, 0, 1),
                                            count_include_pad=False)(av_att)
            map_t_avg = torch.nn.ReplicationPad3d(
                (0, 0, 0, 0, pad_resid, pad_resid))(map_t_avg[None]).squeeze()
            return map_t_avg

        att_map_smooth = smoothen_and_pad_att_map(att_map[:, 0])

        if self.opts.batch_size == 1:
            att_map_smooth = att_map_smooth[None]

        map_for_peaks = att_map_smooth

        # -- 1. aggregate (sum) the attention map values over every pixel trajectory
        flow_inp_device = 'cpu'  # for handling large inputs
        agg_att_map, pixel_trajectories, _ = \
            self.warper.integrate_att_map_over_flow_trajectories(flow.to(flow_inp_device),
                                                                    map_for_peaks,
                                                                    int(self.model.start_offset),
                                                                    )
        agg_att_map = agg_att_map.detach().cpu().numpy()  # bs x T x h x w

        avobject_trajectories = []

        bs = len(pixel_trajectories)
        for b_id in range(bs):

            # -- 2. detect peaks on the aggregated attention map with NMS
            peaks, peak_coords = detect_peaks(
                agg_att_map[b_id], overlap_thresh=self.opts.nms_thresh)
            peak_sort_ids = np.argsort(-agg_att_map[b_id][peak_coords.T[0],
                                                          peak_coords.T[1]])

            selected_peaks = peak_coords[peak_sort_ids[:self.n_speak]]
            top_traj_map = np.zeros_like(agg_att_map[b_id])
            for peak_y, peak_x in selected_peaks:
                pw = 5 // 2
                top_traj_map[max(0, peak_y - pw):peak_y + pw + 1,
                             max(0, peak_x - pw):peak_x + pw + 1] = 1

            # -- 3. Select only the trajectories of the peaks
            peak_pixel_traj = torch.stack([
                pixel_trajectories[b_id][..., peak[0], peak[1]]
                for peak in selected_peaks
            ])
            peak_pixel_traj = peak_pixel_traj.detach().cpu().numpy(
            )  # bs x T x 2 x h x w

            avobject_trajectories.append(
                peak_pixel_traj[..., [1, 0]])  # x -> y and y -> x

        avobject_trajectories = np.stack(avobject_trajectories)

        time_dim = att_map_smooth.shape[1]
        agg_att_map = np.tile(agg_att_map[:, None], [1, time_dim, 1, 1])

        return avobject_trajectories, agg_att_map, att_map_smooth


    def trajectories_from_faces(self,prototypes,video,faces,tr):
        trajectories = []
        _, fr,_,_ = video.shape
        for p in prototypes:
            traj = [] 
            for i in range(fr):
                if 'enc_img'+str(i)+'.jpg' in p.keys():
                    print("p:", p['enc_img'+str(i)+'.jpg'])
                    print("f(p): ", faces[p['enc_img'+str(i)+'.jpg']][-4:])
                    coord = tr[p['enc_img'+str(i)+'.jpg']]['enc_img'+str(i)+'.jpg'][-4:]
                    coord = [(coord[0]+coord[2])//2,(coord[1]+coord[3])//2]
                    traj.append(coord)
                else:
                    traj.append([])
            trajectories.append([traj])
        return trajectories
