"""
EXTENDED BAG OF TRACKLET FROM FACE DETECTOR
"""
import shutil
import os
import subprocess
from collections import defaultdict
from RetinaFace.test_fddb import RetinaFddb
from image_features import Img2Vec
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import math
import pandas as pd
import re
import cv2
from PIL import Image

class FaceTrack:
    
    def __init__(self,video,n_speak):
        self.video = video
        self.n_speak = n_speak
        self.device = torch.device('cuda:0')

    def face_tracker(self):
        # -- 1. Forward audio and video through the model to get embeddings
        video = torch.squeeze(self.video)
        print("squeezed", video.shape)
        print(video[:,20,:,:])
        dim,frames,h,w = video.shape
        image = np.zeros([h,w,3])

        dataset = 'RetinaFace/faces'


        face_eval = RetinaFddb(dataset)

        vis_threshold = 0.7
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
        face_eval.Fddb(vis_threshold)



    
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
                x,y,w,h, s = line.split(' ') 
                f_thr.write(x + ' ' + y + ' ' + w + ' ' + h + os.linesep) 
                x,y,w,h, s = int(x),int(y),int(w),int(h), float(s)
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

                   
                feat = f_extract.get_vec(face,tensor=True)            
                   
                #print(feat.shape)                
                faces_feat[fname] = feat
                faces[fname] = [x,y,w,h]     
                  
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
            
        video = video.numpy()
        video = np.zeros(video.shape)

        max_norm = 958.57 #diagonal size of frame 
        print("\nsortFACES:\n", faces) 


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
                    if list(faces.keys())[j][:-2] != max_el[:-2]:#list(faces.keys())[j-1][:-2]: #if still in the same frame
                        dist_seed = max_el 			#update seed
                        tr[dist_seed[:-2]+'.jpg'] = faces[dist_seed]    #add new seed to tracklet
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
                        dist_seed = max_el 			#update seed
                        tr[dist_seed[:-2]+'.jpg'] = faces[dist_seed]    #add new seed to tracklet
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
        """
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
            prototype.append(prot.copy())
            #print("num occluded frames: ", removed)

        #APPLY MASK FROM PROTOTYPES:
        #video = video.numpy()
        print("num prototypes:", len(prototype))
        dim,frames,h,w = video.shape
        video = np.zeros(video.shape)
        for i in range(frames):
            #print("for ok")
            coord = []
            print("empty coord: ",coord)
            #print("enc_img"+str(i)+".jpg")
            for p in prototype:
                #print(prototype[p])
                if 'enc_img'+str(i)+'.jpg' in p.keys():
                    print("enc_img"+str(i)+".jpg")
                    print(p['enc_img'+str(i)+'.jpg'])
                    coord.append(tracklet[p['enc_img'+str(i)+'.jpg']]['enc_img'+str(i)+'.jpg'])
            if len(coord)==0:
                continue
            print("coord:",coord)
            new_img = self.apply_mask('enc_img'+str(i)+'.jpg',coord)
            cv2.imwrite("RetinaFace/mask/"+str(i)+'.jpg', new_img)   
            video[0,i,:,:] = new_img[:,:,2]
            video[1,i,:,:] = new_img[:,:,1]
            video[2,i,:,:] = new_img[:,:,0]
        video = torch.from_numpy(video)
        video = torch.unsqueeze(video,0)
        #self.video_saver.save_mp4_from_vid_and_audio(video)
        return video



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

            #now I have all scores > thresh...

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

    def removekey(self,d, key):
        r = dict(d)
        del r[key]
        return r

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
