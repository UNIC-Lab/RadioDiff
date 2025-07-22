from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
from itertools import product

warnings.filterwarnings("ignore")


                 #dir_gainDPM="gain/DPM/", 
                 #dir_gainDPMcars="gain/carsDPM/", 
                 #dir_gainIRT2="gain/IRT2/", 
                 #dir_gainIRT2cars="gain/carsIRT2/", 
                 #dir_buildings="png/", 
                 #dir_antenna= , 
                    
#using
class RadioUNet_c_sprseIRT4(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="/home/DataDisk/qmzhang/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.2,
                 simulation="IRT4",
                 carsSimul="no",
                 carsInput="no",
                 cityMap="complete",
                 missing=1,
                 num_samples=300,
                 enlarge_inputs="no",
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default = 2. Note that IRT4 works only with numTx<=2.                
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation: default="IRT4", with an option to "DPM", "IRT2".
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            num_samples: number of samples in the sparse IRT4 radio map. Default=300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
            
        Output:
            
        """
        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )

        self.transform_compose = transforms.Compose([
            transform_BZ
        ])


        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=600
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=600
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="IRT4":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT4/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT4/"
        
        elif simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"  
        
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        #self.num_samples=num_samples#这个似乎没有用到！！！！！！！！！！！！！！
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        self.enlarge_inputs = enlarge_inputs

    
        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        
        #Saprse IRT4 samples, determenistic and fixed samples per map
        # image_samples = np.zeros((self.width,self.height))
        # seed_map=np.sum(image_buildings) # Each map has its fixed samples, independent of the transmitter location.
        # np.random.seed(seed_map)       
        # x_samples=np.random.randint(0, 255, size=self.num_samples)
        # y_samples=np.random.randint(0, 255, size=self.num_samples)
        # image_samples[x_samples,y_samples]= 1
        
        #inputs to radioUNet
        if self.enlarge_inputs == "no":  #不进行扩容输入
            if self.carsInput=="no":
                inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
                #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
                #so we can use the same learning rate as RadioUNets
            elif self.carsInput=="K2": #K2
                inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            else: #cars
                #Normalization, so all settings can have the same learning rate
                image_buildings=image_buildings/256
                image_Tx=image_Tx/256
                img_name_cars = os.path.join(self.dir_cars, name1)
                image_cars = np.asarray(io.imread(img_name_cars))/256
                inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
                #note that ToTensor moves the channel from the last asix to the first!

        elif self.enlarge_inputs == "yes":    #进行扩容 输入改为4
            if self.carsInput=="no":
                inputs=np.stack([image_buildings, image_Tx, image_buildings, image_buildings], axis=2)
            elif self.carsInput=="K2": #K2
                inputs=np.stack([image_buildings, image_Tx, image_buildings, image_buildings], axis=2)
            else: #cars
                image_buildings=image_buildings/256
                image_Tx=image_Tx/256
                img_name_cars = os.path.join(self.dir_cars, name1)
                image_cars = np.asarray(io.imread(img_name_cars))/256
                inputs=np.stack([image_buildings, image_Tx, image_cars, image_cars], axis=2)
        
        

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #image_samples = self.transform(image_samples).type(torch.float32)

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['cond'] = self.transform_GY(inputs)
        out['img_name'] = name2
        return out

#using
class RadioUNet_c_sprseIRT4_K2(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="/home/DataDisk/qmzhang/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.2,
                 simulation="IRT4",
                 carsSimul="no",
                 carsInput="K2",
                 cityMap="complete",
                 missing=1,
                 num_samples=300,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default = 2. Note that IRT4 works only with numTx<=2.                
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation: default="IRT4", with an option to "DPM", "IRT2".
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            num_samples: number of samples in the sparse IRT4 radio map. Default=300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
            
        Output:
            
        """
        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )

        self.transform_compose = transforms.Compose([
            transform_BZ
        ])


        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=600
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=600
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="IRT4":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT4/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT4/"
        
        elif simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"  
        
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        #self.num_samples=num_samples#这个似乎没有用到！！！！！！！！！！！！！！
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        self.dir_k2_neg_norm=self.dir_dataset+"gain/IRT4_k2_neg_norm/"
        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256

            img_name_k2_neg_norm = os.path.join(self.dir_k2_neg_norm, name2)
            k2_neg_norm = np.asarray(io.imread(img_name_k2_neg_norm))/255
            
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        
        #Saprse IRT4 samples, determenistic and fixed samples per map
        # image_samples = np.zeros((self.width,self.height))
        # seed_map=np.sum(image_buildings) # Each map has its fixed samples, independent of the transmitter location.
        # np.random.seed(seed_map)       
        # x_samples=np.random.randint(0, 255, size=self.num_samples)
        # y_samples=np.random.randint(0, 255, size=self.num_samples)
        # image_samples[x_samples,y_samples]= 1
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        elif self.carsInput=="K2": #K2

            #保证了单一变量原则
            inputs=np.stack([image_buildings, image_Tx, k2_neg_norm], axis=2)
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!
        
        

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #image_samples = self.transform(image_samples).type(torch.float32)

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['cond'] = self.transform_GY(inputs)
        out['img_name'] = name2
        return out

#using
class RadioUNet_c_K2(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="/home/DataDisk/qmzhang/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """
       

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            # self.maps_inds=np.array([599])
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            self.ind1=600
            self.ind2=699
            # self.ind1=0
            # self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        self.dir_k2_neg_norm=self.dir_dataset+"gain/DPM_k2_neg_norm/"

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255

            img_name_k2_neg_norm = os.path.join(self.dir_k2_neg_norm, name2)
            k2_neg_norm = np.asarray(io.imread(img_name_k2_neg_norm))/255
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        elif self.carsInput=="K2": #K2
            #保证了单一变量原则
            inputs=np.stack([image_buildings, image_Tx, k2_neg_norm], axis=2)
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['cond'] = self.transform_GY(inputs)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        return out
    
class RadioUNet_c_WithCar_NOK_or_K(Dataset):
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="/home/DataDisk/qmzhang/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="yes",
                 carsInput="yes",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 have_K2="no",
                 transform= transforms.ToTensor()):
        
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
        """

        self.have_K2 = have_K2  #默认是no

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        self.transform_GYCAR = transforms.Normalize(
            mean=[0.5, 0.5, 0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5, 0.5, 0.5]
        )

        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            # self.maps_inds=np.array([599])
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=600
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            self.ind1=600
            self.ind2=699
            # self.ind1=0
            # self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"  #直接进入这个
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        self.dir_k2_neg_norm=self.dir_dataset+"gain/DPMCAR_k2_neg_norm/"

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255

            if self.have_K2 == "yes":
                img_name_k2_neg_norm = os.path.join(self.dir_k2_neg_norm, name2)
                k2_neg_norm = np.asarray(io.imread(img_name_k2_neg_norm))/255
                
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        elif self.carsInput=="K2": #K2
            #保证了单一变量原则
            inputs=np.stack([image_buildings, image_Tx, k2_neg_norm], axis=2)
        elif self.carsInput=="yes":  #默认进入这个
            if self.have_K2 == "no":  
                #保证了输入条件都没有除以256，进行了一次统一
                img_name_cars = os.path.join(self.dir_cars, name1)
                image_cars = np.asarray(io.imread(img_name_cars))
                inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            elif self.have_K2 == "yes":  
                #保证了输入条件都没有除以256，进行了一次统一
                img_name_cars = os.path.join(self.dir_cars, name1)
                image_cars = np.asarray(io.imread(img_name_cars))
                #我想强化k2_neg_norm的作用
                inputs=np.stack([image_buildings, image_Tx, image_cars, k2_neg_norm, k2_neg_norm], axis=2)
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['image'] = self.transform_compose(image_gain)
        if self.have_K2 == "no":  
            out['cond'] = self.transform_GY(inputs)
        elif self.have_K2 == "yes":
            out['cond'] = self.transform_GYCAR(inputs)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        return out

class RadioUNet_c(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="/home/DataDisk/qmzhang/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            # self.maps_inds=np.array([599])
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=600
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            self.ind1=600
            self.ind2=699
            # self.ind1=0
            # self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['cond'] = self.transform_GY(inputs)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        return out
    
#现在这个模块用来加载沿着建筑物边缘的mask，也可以应对同等采样数量的建筑物，关键变量在于mask=True还是False
class RadioUNet_s(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="/home/disk01/qmzhang/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 fix_samples=0,
                 num_samples_low= 10, 
                 num_samples_high= 300,
                 mask=False,
                 num_sample = 10,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            fix_samples: fixed or a random number of samples. If zero, fixed, else, fix_samples is the number of samples. Default = 0.
            num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10. 
            num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
        """
        

        
        #self.phase=phase


        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])


        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=200
        elif phase=="val":
            self.ind1=501
            self.ind2=520
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput

        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
         
        self.fix_samples= fix_samples
        self.num_samples_low= num_samples_low 
        self.num_samples_high= num_samples_high
                
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        self.mask = mask

        
        self.dir_mask = self.dir_dataset + "png/half_mask/"
        self.num_sample = num_sample

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))/256  
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))/256
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
            
        # image_gain=image_gain*256 # we use this normalization so all RadioUNet methods can have the same learning rate.
                                  # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
                                  # Important: when evaluating the accuracy, remember to devide the errors by 256!
                 
        #input measurements
        image_samples = np.zeros((256,256))

        sparse_mask_path = self.dir_mask + "mask_" + str(dataset_map_ind) + ".png"

        if self.mask == True:
            #对遍历image_gain 进行掩膜处理
            sparse_mask = np.asarray(io.imread(sparse_mask_path))
            y_coords, x_coords = np.nonzero(sparse_mask)
            image_samples[y_coords, x_coords]= image_gain[y_coords, x_coords,0]
            num_samplesa = len(np.nonzero(sparse_mask)[0])
            # print(num_samplesa)
            
        
        if self.mask == False:
            sparse_mask = np.asarray(io.imread(sparse_mask_path))
            y_coords, x_coords = np.nonzero(sparse_mask)
            num_samplesa = len(np.nonzero(sparse_mask)[0])
            # print(num_samplesa)
            x_samples=np.random.randint(0, 128, size=num_samplesa)
            y_samples=np.random.randint(0, 255, size=num_samplesa)
            image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]

        # if self.mask == False:
        #     sparse_mask = np.asarray(io.imread(sparse_mask_path))
        #     y_coords, x_coords = np.nonzero(sparse_mask)
        #     num_samplesa = max(1, len(np.nonzero(sparse_mask)[0]) // self.num_sample)
        #     # print(num_samplesa)
        #     x_samples=np.random.randint(0, 128, size=num_samplesa)
        #     y_samples=np.random.randint(0, 255, size=num_samplesa)
        #     image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]


            # if self.fix_samples==0:
            #     num_samples=np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
            # else:
            #     num_samples=np.floor(self.fix_samples).astype(int)               
            # x_samples=np.random.randint(0, 255, size=num_samples)
            # y_samples=np.random.randint(0, 255, size=num_samples)
            # image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_samples, image_samples], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_samples, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['cond'] = self.transform_GY(inputs)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        return out


        # return [inputs, image_gain]

#现在这个模块用来加载顶点边缘的mask，也可以应对同等采样数量的建筑物，关键变量在于mask=True还是False
class RadioUNet_s_vertex(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="/home/disk01/qmzhang/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 fix_samples=0,
                 num_samples_low= 10, 
                 num_samples_high= 300,
                 mask=False,
                 num_sample = 10,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathloss threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            fix_samples: fixed or a random number of samples. If zero, fixed, else, fix_samples is the number of samples. Default = 0.
            num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10. 
            num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
        """
        

        
        #self.phase=phase


        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])


        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=200
        elif phase=="val":
            self.ind1=501
            self.ind2=520
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput

        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
         
        self.fix_samples= fix_samples
        self.num_samples_low= num_samples_low 
        self.num_samples_high= num_samples_high
                
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        self.mask = mask

        
        self.dir_mask = self.dir_dataset + "png/half_mask_vertex/"
        self.num_sample = num_sample

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))/256  
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))/256
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
            
        # image_gain=image_gain*256 # we use this normalization so all RadioUNet methods can have the same learning rate.
                                  # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
                                  # Important: when evaluating the accuracy, remember to devide the errors by 256!
                 
        #input measurements
        image_samples = np.zeros((256,256))

        sparse_mask_path = self.dir_mask + "mask_" + str(dataset_map_ind) + ".png"

        if self.mask == True:
            #对遍历image_gain 进行掩膜处理
            sparse_mask = np.asarray(io.imread(sparse_mask_path))
            y_coords, x_coords = np.nonzero(sparse_mask)
            image_samples[y_coords, x_coords]= image_gain[y_coords, x_coords,0]
            num_samplesa = len(np.nonzero(sparse_mask)[0])
            # print(num_samplesa)
            
        if self.mask == False:
            sparse_mask = np.asarray(io.imread(sparse_mask_path))
            y_coords, x_coords = np.nonzero(sparse_mask)
            num_samplesa = len(np.nonzero(sparse_mask)[0])
            # print(num_samplesa)
            x_samples=np.random.randint(0, 128, size=num_samplesa)
            y_samples=np.random.randint(0, 255, size=num_samplesa)
            image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]

        # if self.mask == False:
        #     sparse_mask = np.asarray(io.imread(sparse_mask_path))
        #     y_coords, x_coords = np.nonzero(sparse_mask)
        #     num_samplesa = max(1, len(np.nonzero(sparse_mask)[0]) // self.num_sample)
        #     # print(num_samplesa)
        #     x_samples=np.random.randint(0, 128, size=num_samplesa)
        #     y_samples=np.random.randint(0, 255, size=num_samplesa)
        #     image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]


            # if self.fix_samples==0:
            #     num_samples=np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
            # else:
            #     num_samples=np.floor(self.fix_samples).astype(int)               
            # x_samples=np.random.randint(0, 255, size=num_samples)
            # y_samples=np.random.randint(0, 255, size=num_samples)
            # image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_samples, image_samples], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_samples, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['cond'] = self.transform_GY(inputs)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        return out

class RadioUNet_s_random(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="/home/disk01/qmzhang/RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 fix_samples=0,
                 num_samples_low= 10, 
                 num_samples_high= 300,
                 mask=False,
                 num_sample = 10,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathloss threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            fix_samples: fixed or a random number of samples. If zero, fixed, else, fix_samples is the number of samples. Default = 0.
            num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10. 
            num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
        """
        

        
        #self.phase=phase


        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])


        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=200
        elif phase=="val":
            self.ind1=501
            self.ind2=520
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput

        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
         
        self.fix_samples= fix_samples
        self.num_samples_low= num_samples_low 
        self.num_samples_high= num_samples_high
                
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        self.mask = mask

        # ============================================================================ #
        self.dir_mask = self.dir_dataset + "png/random_half_mask/"
        self.num_sample = num_sample

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))/256  
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))/256
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
            
        # image_gain=image_gain*256 # we use this normalization so all RadioUNet methods can have the same learning rate.
                                  # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
                                  # Important: when evaluating the accuracy, remember to devide the errors by 256!
                 
        #input measurements
        image_samples = np.zeros((256,256))

        sparse_mask_path = self.dir_mask + "mask_" + str(dataset_map_ind) + ".png"

        if self.mask == True:
            #对遍历image_gain 进行掩膜处理
            sparse_mask = np.asarray(io.imread(sparse_mask_path))
            y_coords, x_coords = np.nonzero(sparse_mask)
            image_samples[y_coords, x_coords]= image_gain[y_coords, x_coords,0]
            num_samplesa = len(np.nonzero(sparse_mask)[0])
            # print(num_samplesa)
        
        # if self.mask == True:
        #     # 读取掩膜图像
        #     sparse_mask = np.asarray(io.imread(sparse_mask_path))
            
        #     # 找出非零点坐标
        #     y_coords, x_coords = np.nonzero(sparse_mask)
            
        #     # 总数量
        #     num_points = len(y_coords)
            
        #     # 要选择的数量（约十分之一）
        #     sample_num = max(1, num_points // self.num_sample)  # 至少选一个点，防止为0
            
        #     # 随机抽样索引
        #     selected_indices = np.random.choice(num_points, size=sample_num, replace=False)
            
        #     # 选中的坐标
        #     selected_y = y_coords[selected_indices]
        #     selected_x = x_coords[selected_indices]
            
        #     # 应用 mask
        #     image_samples[selected_y, selected_x] = image_gain[selected_y, selected_x, 0]
            
        
        # if self.mask == False:
        #     sparse_mask = np.asarray(io.imread(sparse_mask_path))
        #     y_coords, x_coords = np.nonzero(sparse_mask)
        #     num_samplesa = len(np.nonzero(sparse_mask)[0])
        #     # print(num_samplesa)
        #     x_samples=np.random.randint(0, 128, size=num_samplesa)
        #     y_samples=np.random.randint(0, 255, size=num_samplesa)
        #     image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]

        # if self.mask == False:
        #     sparse_mask = np.asarray(io.imread(sparse_mask_path))
        #     y_coords, x_coords = np.nonzero(sparse_mask)
        #     num_samplesa = max(1, len(np.nonzero(sparse_mask)[0]) // self.num_sample)
        #     # print(num_samplesa)
        #     x_samples=np.random.randint(0, 128, size=num_samplesa)
        #     y_samples=np.random.randint(0, 255, size=num_samplesa)
        #     image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]


            # if self.fix_samples==0:
            #     num_samples=np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
            # else:
            #     num_samples=np.floor(self.fix_samples).astype(int)               
            # x_samples=np.random.randint(0, 255, size=num_samples)
            # y_samples=np.random.randint(0, 255, size=num_samples)
            # image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_samples, image_samples], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_samples, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['cond'] = self.transform_GY(inputs)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        return out

# class RadioUNet_c_sprseIRT4(Dataset):
#     """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
#     def __init__(self,maps_inds=np.zeros(1), phase="train",
#                  ind1=0,ind2=0, 
#                  dir_dataset="RadioMapSeer/",
#                  numTx=2,                  
#                  thresh=0.2,
#                  simulation="IRT4",
#                  carsSimul="no",
#                  carsInput="no",
#                  cityMap="complete",
#                  missing=1,
#                  num_samples=300,
#                  transform= transforms.ToTensor()):
#         """
#         Args:
#             maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
#             phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
#                   "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
#             ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
#             dir_dataset: directory of the RadioMapSeer dataset.
#             numTx: Number of transmitters per map. Default = 2. Note that IRT4 works only with numTx<=2.                
#             thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
#             simulation: default="IRT4", with an option to "DPM", "IRT2".
#             carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
#             carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
#             cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
#                       a random number of missing buildings.
#             missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
#             num_samples: number of samples in the sparse IRT4 radio map. Default=300.
#             transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
            
#         Output:
            
#         """
#         if maps_inds.size==1:
#             self.maps_inds=np.arange(0,700,1,dtype=np.int16)
#             #Determenistic "random" shuffle of the maps:
#             np.random.seed(42)
#             np.random.shuffle(self.maps_inds)
#         else:
#             self.maps_inds=maps_inds
            
#         if phase=="train":
#             self.ind1=0
#             self.ind2=500
#         elif phase=="val":
#             self.ind1=501
#             self.ind2=600
#         elif phase=="test":
#             self.ind1=601
#             self.ind2=699
#         else: # custom range
#             self.ind1=ind1
#             self.ind2=ind2
            
#         self.dir_dataset = dir_dataset
#         self.numTx=  numTx                
#         self.thresh=thresh
        
#         self.simulation=simulation
#         self.carsSimul=carsSimul
#         self.carsInput=carsInput
#         if simulation=="IRT4":
#             if carsSimul=="no":
#                 self.dir_gain=self.dir_dataset+"gain/IRT4/"
#             else:
#                 self.dir_gain=self.dir_dataset+"gain/carsIRT4/"
        
#         elif simulation=="DPM" :
#             if carsSimul=="no":
#                 self.dir_gain=self.dir_dataset+"gain/DPM/"
#             else:
#                 self.dir_gain=self.dir_dataset+"gain/carsDPM/"
#         elif simulation=="IRT2":
#             if carsSimul=="no":
#                 self.dir_gain=self.dir_dataset+"gain/IRT2/"
#             else:
#                 self.dir_gain=self.dir_dataset+"gain/carsIRT2/"  
        
        
#         self.cityMap=cityMap
#         self.missing=missing
#         if cityMap=="complete":
#             self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
#         else:
#             self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
#         #else:  #missing==number
#         #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
#         self.transform= transform
        
#         self.num_samples=num_samples
        
#         self.dir_Tx = self.dir_dataset+ "png/antennas/" 
#         #later check if reading the JSON file and creating antenna images on the fly is faster
#         if carsInput!="no":
#             self.dir_cars = self.dir_dataset+ "png/cars/" 
        
#         self.height = 256
#         self.width = 256

        
        
        
        
#     def __len__(self):
#         return (self.ind2-self.ind1+1)*self.numTx
    
#     def __getitem__(self, idx):
        
#         idxr=np.floor(idx/self.numTx).astype(int)
#         idxc=idx-idxr*self.numTx 
#         dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
#         #names of files that depend only on the map:
#         name1 = str(dataset_map_ind) + ".png"
#         #names of files that depend on the map and the Tx:
#         name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
#         #Load buildings:
#         if self.cityMap == "complete":
#             img_name_buildings = os.path.join(self.dir_buildings, name1)
#         else:
#             if self.cityMap == "rand":
#                 self.missing=np.random.randint(low=1, high=5)
#             version=np.random.randint(low=1, high=7)
#             img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
#             str(self.missing)
#         image_buildings = np.asarray(io.imread(img_name_buildings))   
        
#         #Load Tx (transmitter):
#         img_name_Tx = os.path.join(self.dir_Tx, name2)
#         image_Tx = np.asarray(io.imread(img_name_Tx))
        
#         #Load radio map:
#         if self.simulation!="rand":
#             img_name_gain = os.path.join(self.dir_gain, name2)  
#             image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
#         else: #random weighted average of DPM and IRT2
#             img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
#             img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
#             #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
#             #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
#             w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
#             image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
#                         + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
#         #pathloss threshold transform
#         if self.thresh>0:
#             mask = image_gain < self.thresh
#             image_gain[mask]=self.thresh
#             image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
#             image_gain=image_gain/(1-self.thresh)
        
#         #Saprse IRT4 samples, determenistic and fixed samples per map
#         image_samples = np.zeros((self.width,self.height))
#         seed_map=np.sum(image_buildings) # Each map has its fixed samples, independent of the transmitter location.
#         np.random.seed(seed_map)       
#         x_samples=np.random.randint(0, 255, size=self.num_samples)
#         y_samples=np.random.randint(0, 255, size=self.num_samples)
#         image_samples[x_samples,y_samples]= 1
        
#         #inputs to radioUNet
#         if self.carsInput=="no":
#             inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
#             #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
#             #so we can use the same learning rate as RadioUNets
#         else: #cars
#             #Normalization, so all settings can have the same learning rate
#             image_buildings=image_buildings/256
#             image_Tx=image_Tx/256
#             img_name_cars = os.path.join(self.dir_cars, name1)
#             image_cars = np.asarray(io.imread(img_name_cars))/256
#             inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
#             #note that ToTensor moves the channel from the last asix to the first!
        
        

        
#         if self.transform:
#             inputs = self.transform(inputs).type(torch.float32)
#             image_gain = self.transform(image_gain).type(torch.float32)
#             image_samples = self.transform(image_samples).type(torch.float32)


#         return [inputs, image_gain, image_samples]
    
    
    
    
    
    
    
# class RadioUNet_s(Dataset):
#     """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
#     def __init__(self,maps_inds=np.zeros(1), phase="train",
#                  ind1=0,ind2=0, 
#                  dir_dataset="RadioMapSeer/",
#                  numTx=80,                  
#                  thresh=0.2,
#                  simulation="DPM",
#                  carsSimul="no",
#                  carsInput="no",
#                  IRT2maxW=1,
#                  cityMap="complete",
#                  missing=1,
#                  fix_samples=0,
#                  num_samples_low= 10, 
#                  num_samples_high= 300,
#                  transform= transforms.ToTensor()):
#         """
#         Args:
#             maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
#             phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
#                   "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
#             ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
#             dir_dataset: directory of the RadioMapSeer dataset.
#             numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
#             thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
#             simulation:"DPM", "IRT2", "rand". Default= "DPM"
#             carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
#             carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
#             IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
#             cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
#                       a random number of missing buildings.
#             missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
#             fix_samples: fixed or a random number of samples. If zero, fixed, else, fix_samples is the number of samples. Default = 0.
#             num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10. 
#             num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
#             transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
#         Output:
#             inputs: The RadioUNet inputs.  
#             image_gain
            
#         """
        

        
#         #self.phase=phase
                
#         if maps_inds.size==1:
#             self.maps_inds=np.arange(0,700,1,dtype=np.int16)
#             #Determenistic "random" shuffle of the maps:
#             np.random.seed(42)
#             np.random.shuffle(self.maps_inds)
#         else:
#             self.maps_inds=maps_inds
            
#         if phase=="train":
#             self.ind1=0
#             self.ind2=600
#         elif phase=="val":
#             self.ind1=501
#             self.ind2=600
#         elif phase=="test":
#             self.ind1=601
#             self.ind2=650
#         else: # custom range
#             self.ind1=ind1
#             self.ind2=ind2
            
#         self.dir_dataset = dir_dataset
#         self.numTx=  numTx                
#         self.thresh=thresh
        
#         self.simulation=simulation
#         self.carsSimul=carsSimul
#         self.carsInput=carsInput
#         if simulation=="DPM" :
#             if carsSimul=="no":
#                 self.dir_gain=self.dir_dataset+"gain/DPM/"
#             else:
#                 self.dir_gain=self.dir_dataset+"gain/carsDPM/"
#         elif simulation=="IRT2":
#             if carsSimul=="no":
#                 self.dir_gain=self.dir_dataset+"gain/IRT2/"
#             else:
#                 self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
#         elif  simulation=="rand":
#             if carsSimul=="no":
#                 self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
#                 self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
#             else:
#                 self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
#                 self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
#         self.IRT2maxW=IRT2maxW
        
#         self.cityMap=cityMap
#         self.missing=missing
#         if cityMap=="complete":
#             self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
#         else:
#             self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
#         #else:  #missing==number
#         #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
         
#         self.fix_samples= fix_samples
#         self.num_samples_low= num_samples_low 
#         self.num_samples_high= num_samples_high
                
#         self.transform= transform
        
#         self.dir_Tx = self.dir_dataset+ "png/antennas/" 
#         #later check if reading the JSON file and creating antenna images on the fly is faster
#         if carsInput!="no":
#             self.dir_cars = self.dir_dataset+ "png/cars/" 
        
#         self.height = 256
#         self.width = 256

        
#     def __len__(self):
#         return (self.ind2-self.ind1+1)*self.numTx
    
#     def __getitem__(self, idx):
        
#         idxr=np.floor(idx/self.numTx).astype(int)
#         idxc=idx-idxr*self.numTx 
#         dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
#         #names of files that depend only on the map:
#         name1 = str(dataset_map_ind) + ".png"
#         #names of files that depend on the map and the Tx:
#         name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
#         #Load buildings:
#         if self.cityMap == "complete":
#             img_name_buildings = os.path.join(self.dir_buildings, name1)
#         else:
#             if self.cityMap == "rand":
#                 self.missing=np.random.randint(low=1, high=5)
#             version=np.random.randint(low=1, high=7)
#             img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
#             str(self.missing)
#         image_buildings = np.asarray(io.imread(img_name_buildings))/256  
        
#         #Load Tx (transmitter):
#         img_name_Tx = os.path.join(self.dir_Tx, name2)
#         image_Tx = np.asarray(io.imread(img_name_Tx))/256
        
#         #Load radio map:
#         if self.simulation!="rand":
#             img_name_gain = os.path.join(self.dir_gain, name2)  
#             image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
#         else: #random weighted average of DPM and IRT2
#             img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
#             img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
#             #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
#             #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
#             w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
#             image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
#                         + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
#         #pathloss threshold transform
#         if self.thresh>0:
#             mask = image_gain < self.thresh
#             image_gain[mask]=self.thresh
#             image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
#             image_gain=image_gain/(1-self.thresh)
            
#         image_gain=image_gain*256 # we use this normalization so all RadioUNet methods can have the same learning rate.
#                                   # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
#                                   # Important: when evaluating the accuracy, remember to devide the errors by 256!
                 
#         #input measurements
#         image_samples = np.zeros((256,256))
#         if self.fix_samples==0:
#             num_samples=np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
#         else:
#             num_samples=np.floor(self.fix_samples).astype(int)               
#         x_samples=np.random.randint(0, 255, size=num_samples)
#         y_samples=np.random.randint(0, 255, size=num_samples)
#         image_samples[x_samples,y_samples]= image_gain[x_samples,y_samples,0]
        
#         #inputs to radioUNet
#         if self.carsInput=="no":
#             inputs=np.stack([image_buildings, image_Tx, image_samples], axis=2)        
#             #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
#             #so we can use the same learning rate as RadioUNets
#         else: #cars
#             #Normalization, so all settings can have the same learning rate
#             img_name_cars = os.path.join(self.dir_cars, name1)
#             image_cars = np.asarray(io.imread(img_name_cars))/256
#             inputs=np.stack([image_buildings, image_Tx, image_samples, image_cars], axis=2)
#             #note that ToTensor moves the channel from the last asix to the first!

        
        
#         if self.transform:
#             inputs = self.transform(inputs).type(torch.float32)
#             image_gain = self.transform(image_gain).type(torch.float32)
#             #note that ToTensor moves the channel from the last asix to the first!


#         return [inputs, image_gain]
    
    
    
    

class RadioUNet_s_sprseIRT4(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=2,                  
                 thresh=0.2,
                 simulation="IRT4",
                 carsSimul="no",
                 carsInput="no",
                 cityMap="complete",
                 missing=1,
                 data_samples=300,
                 fix_samples=0,
                 num_samples_low= 10, 
                 num_samples_high= 299,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default = 2. Note that IRT4 works only with numTx<=2.                
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation: default="IRT4", with an option to "DPM", "IRT2".
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            data_samples: number of samples in the sparse IRT4 radio map. Default=300. All input samples are taken from the data_samples
            fix_samples: fixed or a random number of samples. If zero, fixed, else, fix_samples is the number of samples. Default = 0.
            num_samples_low: if random number of samples, this is the minimum number of samples. Default = 10. 
            num_samples_high: if random number of samples, this is the maximal number of samples. Default = 300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
            
        Output:
            
        """
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="IRT4":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT4/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT4/"
        
        elif simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"  
        
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
         
        self.data_samples=data_samples
        self.fix_samples= fix_samples
        self.num_samples_low= num_samples_low 
        self.num_samples_high= num_samples_high
        
        self.transform= transform
        
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
        
        
        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))  #Will be normalized later, after random seed is computed from it
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))/256 
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
        
        image_gain=image_gain*256 # we use this normalization so all RadioUNet methods can have the same learning rate.
                                  # Namely, the loss of RadioUNet_s is 256 the loss of RadioUNet_c
                                  # Important: when evaluating the accuracy, remember to devide the errors by 256!
                    
        #Saprse IRT4 samples, determenistic and fixed samples per map
        sparse_samples = np.zeros((self.width,self.height))
        seed_map=np.sum(image_buildings) # Each map has its fixed samples, independent of the transmitter location.
        np.random.seed(seed_map)       
        x_samples=np.random.randint(0, 255, size=self.data_samples)
        y_samples=np.random.randint(0, 255, size=self.data_samples)
        sparse_samples[x_samples,y_samples]= 1
        
        #input samples from the sparse gain samples
        input_samples = np.zeros((256,256))
        if self.fix_samples==0:
            num_in_samples=np.random.randint(self.num_samples_low, self.num_samples_high, size=1)
        else:
            num_in_samples=np.floor(self.fix_samples).astype(int)
            
        data_inds=range(self.data_samples)
        input_inds=np.random.permutation(data_inds)[0:num_in_samples[0]]      
        x_samples_in=x_samples[input_inds]
        y_samples_in=y_samples[input_inds]
        input_samples[x_samples_in,y_samples_in]= image_gain[x_samples_in,y_samples_in,0]
        
        #normalize image_buildings, after random seed computed from it as an int
        image_buildings=image_buildings/256
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, input_samples], axis=2)        
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, input_samples, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!
        
        

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            sparse_samples = self.transform(sparse_samples).type(torch.float32)
            


        return [inputs, image_gain, sparse_samples]
    

class RadioUNet_c_split_time(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            # self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            # self.Tx_inds=np.arange(0,80,1,dtype=np.int16)
            # self.maps_inds=np.array([2])
            # self.maps_inds=np.arrange(640,671,1,dtype=np.int16)
            # self.Tx_inds=np.array([9,10,11])
            self.maps_inds=np.arange(640,671,1,dtype=np.int16) # 600~699 total 100 maps
            print("len self.maps_inds : ",len(self.maps_inds))
            # self.numTx = 80
            # total 80 Tx per map
            self.Tx_inds=np.arange(35,56,1,dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            # np.random.seed(42)
            # np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            self.Tx_inds=None
            
        if phase=="train":
            self.ind1=0
            self.ind2=640
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            # self.ind1=600
            # self.ind2=699
            self.ind1=0
            self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=numTx if self.Tx_inds is None else len(self.Tx_inds)           
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int) # idxr为场景列表的第几个场景
        idxc=idx-idxr*self.numTx # idxc为该场景的第几个Tx
        idxc=self.Tx_inds[idxc] # idxc为该场景的第几个Tx的索引
        idxc_cond2 = (idxc + 1) % self.numTx # idxc_cond2为该场景的另一个Tx的索引
        idxc_cond2 = self.Tx_inds[idxc_cond2] # idxc_cond2为该场景的另一个Tx的索引
        dataset_map_ind=self.maps_inds[idxr+self.ind1] # 实际的场景索引
        
        # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        name2_cond2 = str(dataset_map_ind) + "_" + str(idxc_cond2) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        img_name_Tx_cond2 = os.path.join(self.dir_Tx, name2_cond2)
        image_Tx_cond2 = np.asarray(io.imread(img_name_Tx_cond2))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
            
            img_name_gain_cond2 = os.path.join(self.dir_gain, name2_cond2)  
            image_gain_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gain_cond2)),axis=2)/255
            
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256

            img_name_gainDPM_cond2 = os.path.join(self.dir_gainDPM, name2_cond2) 
            img_name_gainIRT2_cond2 = os.path.join(self.dir_gainIRT2, name2_cond2) 
            #image_gainDPM_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gainDPM_cond2)),axis=2)/255
            #image_gainIRT2_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2_cond2)),axis=2)/255
            w_cond2=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain_cond2= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2_cond2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM_cond2)),axis=2)/256    
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            inputs_cond2=np.stack([image_buildings, image_Tx_cond2, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            image_Tx_cond2=image_Tx_cond2/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            inputs_cond2=np.stack([image_buildings, image_Tx_cond2, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            inputs_cond2 = self.transform(inputs_cond2).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            image_gain_cond2 = self.transform(image_gain_cond2).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['image_cond2'] = self.transform_compose(image_gain_cond2)
        out['cond'] = self.transform_GY(inputs)
        out['cond2'] = self.transform_GY(inputs_cond2)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        out['img_name_cond2'] = name2_cond2
        return out


class RadioUNet_c_split_time_diff_maps(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            # self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            # self.Tx_inds=np.arange(0,80,1,dtype=np.int16)
            self.maps_inds=np.array([1, 2])
            self.Tx_inds=np.array([0,1,2])
            # Determenistic "random" shuffle of the maps:
            # np.random.seed(42)
            # np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            self.Tx_inds=None
            
        if phase=="train":
            self.ind1=0
            self.ind2=640
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            # self.ind1=600
            # self.ind2=699
            self.ind1=0
            self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=numTx if self.Tx_inds is None else len(self.Tx_inds)           
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int) # idxr为场景列表的第几个场景
        idxc=idx-idxr*self.numTx # idxc为该场景的第几个Tx
        idxc=self.Tx_inds[idxc] # idxc为该场景的第几个Tx的索引
        idxc_cond2 = (idxc + 1) % self.numTx # idxc_cond2为该场景的另一个Tx的索引
        idxc_cond2 = self.Tx_inds[idxc_cond2] # idxc_cond2为该场景的另一个Tx的索引
        dataset_map_ind=self.maps_inds[idxr+self.ind1] # 实际的场景索引
        dataset_map_ind2=self.maps_inds[(idxr+self.ind1+1) % len(self.maps_inds)] # 实际的下一个场景索引
        
        # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        name12 = str(dataset_map_ind2) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        name2_cond2 = str(dataset_map_ind2) + "_" + str(idxc_cond2) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
            img_name_buildings2 = os.path.join(self.dir_buildings, name12)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            img_name_buildings2 = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name12)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        image_buildings2 = np.asarray(io.imread(img_name_buildings2))   
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        img_name_Tx_cond2 = os.path.join(self.dir_Tx, name2_cond2)
        image_Tx_cond2 = np.asarray(io.imread(img_name_Tx_cond2))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
            
            img_name_gain_cond2 = os.path.join(self.dir_gain, name2_cond2)  
            image_gain_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gain_cond2)),axis=2)/255
            
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256

            img_name_gainDPM_cond2 = os.path.join(self.dir_gainDPM, name2_cond2) 
            img_name_gainIRT2_cond2 = os.path.join(self.dir_gainIRT2, name2_cond2) 
            #image_gainDPM_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gainDPM_cond2)),axis=2)/255
            #image_gainIRT2_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2_cond2)),axis=2)/255
            w_cond2=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain_cond2= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2_cond2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM_cond2)),axis=2)/256    
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            inputs_cond2=np.stack([image_buildings2, image_Tx_cond2, image_buildings2], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_buildings2=image_buildings2/256
            image_Tx=image_Tx/256
            image_Tx_cond2=image_Tx_cond2/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            img_name_cars2 = os.path.join(self.dir_cars, name12)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            image_cars2 = np.asarray(io.imread(img_name_cars2))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            inputs_cond2=np.stack([image_buildings2, image_Tx_cond2, image_cars2], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            inputs_cond2 = self.transform(inputs_cond2).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            image_gain_cond2 = self.transform(image_gain_cond2).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['image_cond2'] = self.transform_compose(image_gain_cond2)
        out['cond'] = self.transform_GY(inputs)
        out['cond2'] = self.transform_GY(inputs_cond2)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        out['img_name_cond2'] = name2_cond2
        return out
    

class RadioUNet_c_save_frames(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            # self.maps_inds=np.array([561])
            # Determenistic "random" shuffle of the maps:
            self.Tx_inds=np.array([0])
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=640
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            self.ind1=600
            self.ind2=699
            # self.ind1=0
            # self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx = numTx if self.Tx_inds is None else len(self.Tx_inds)          
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx
        idxc=self.Tx_inds[idxc] # idxc为该场景的第几个Tx的索引
        dataset_map_ind=self.maps_inds[idxr+self.ind1]
        # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['cond'] = self.transform_GY(inputs)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        return out
    
    

class RadioUNet_c_split_time_complete(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            # self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            # self.Tx_inds=np.arange(0,80,1,dtype=np.int16)
            self.maps_inds=np.arange(600,700,1,dtype=np.int16) # 600~699 total 100 maps
            print("len self.maps_inds : ",len(self.maps_inds))
            # self.numTx = 80
            # total 80 Tx per map
            self.Tx_inds=np.arange(0,80,1,dtype=np.int16)
            
            # 下列代码对self.Tx_inds进行配对((0,1)、（0，2） ...  (0,3)、(1,0) (1,2) ... (79,0) (79,78))
            self.Tx_inds_pairs=np.array([(i,j) for i in self.Tx_inds for j in self.Tx_inds if i!=j])
            print("len self.Tx_inds_pairs : ",len(self.Tx_inds_pairs))
            self.num_Tx_pairs = len(self.Tx_inds_pairs)
            # Determenistic "random" shuffle of the maps:
            # np.random.seed(42)
            # np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            self.Tx_inds=None
            
        if phase=="train":
            self.ind1=0
            self.ind2=640
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            # self.ind1=600
            # self.ind2=699
            self.ind1=0
            self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        # self.numTx=numTx if self.Tx_inds is None else len(self.Tx_inds)           
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.num_Tx_pairs

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.num_Tx_pairs).astype(int) # idxr为场景列表的第几个场景
        # idxc=idx-idxr*self.num_Tx_pairs # idxc为该场景的第几个Tx
        idx_pairs = self.Tx_inds_pairs[idx] # idxc为该场景的第几个Tx对
        idxc=idx_pairs[0] # 
        idxc_cond2 =  idx_pairs[1] # 
        dataset_map_ind=self.maps_inds[idxr+self.ind1] # 实际的场景索引
        
        # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        name2_cond2 = str(dataset_map_ind) + "_" + str(idxc_cond2) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        img_name_Tx_cond2 = os.path.join(self.dir_Tx, name2_cond2)
        image_Tx_cond2 = np.asarray(io.imread(img_name_Tx_cond2))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
            
            img_name_gain_cond2 = os.path.join(self.dir_gain, name2_cond2)  
            image_gain_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gain_cond2)),axis=2)/255
            
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256

            img_name_gainDPM_cond2 = os.path.join(self.dir_gainDPM, name2_cond2) 
            img_name_gainIRT2_cond2 = os.path.join(self.dir_gainIRT2, name2_cond2) 
            #image_gainDPM_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gainDPM_cond2)),axis=2)/255
            #image_gainIRT2_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2_cond2)),axis=2)/255
            w_cond2=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain_cond2= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2_cond2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM_cond2)),axis=2)/256    
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            inputs_cond2=np.stack([image_buildings, image_Tx_cond2, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            image_Tx_cond2=image_Tx_cond2/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            inputs_cond2=np.stack([image_buildings, image_Tx_cond2, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            inputs_cond2 = self.transform(inputs_cond2).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            image_gain_cond2 = self.transform(image_gain_cond2).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['image_cond2'] = self.transform_compose(image_gain_cond2)
        out['cond'] = self.transform_GY(inputs)
        out['cond2'] = self.transform_GY(inputs_cond2)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        out['img_name_cond2'] = name2_cond2
        return out



class RadioUNet_c_split_time_v2(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            # self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            # self.Tx_inds=np.arange(0,80,1,dtype=np.int16)
            self.maps_inds=np.arange(640,671,1,dtype=np.int16) # 600~699 total 100 maps
            print("len self.maps_inds : ",len(self.maps_inds))
            # self.numTx = 80
            # total 80 Tx per map
            self.Tx_inds=np.arange(35,56,1,dtype=np.int16)
            
            # 下列代码对self.Tx_inds进行配对((0,1)、（0，2） ...  (0,3)、(1,0) (1,2) ... (79,0) (79,78))
            # self.Tx_inds_pairs=np.array([(i,j) for i in self.Tx_inds for j in self.Tx_inds if i!=j])
            # print("len self.Tx_inds_pairs : ",len(self.Tx_inds_pairs))
            # self.num_Tx_pairs = len(self.Tx_inds_pairs)
            # Determenistic "random" shuffle of the maps:
            # np.random.seed(42)
            # np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            self.Tx_inds=None
            
        if phase=="train":
            self.ind1=0
            self.ind2=640
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            # self.ind1=600
            # self.ind2=699
            self.ind1=0
            self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=len(self.Tx_inds)           
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def input_transform(self, out, pair_idx):
        # ['name1', 'image_buildings', 'img_name_cond_pairs']
        # dict_keys(['name1', 'image_buildings', 'img_name_cond_pairs'])
        # name1 ->['600.png']
        # image_buildings.shape ->torch.Size([1, 256, 256])
        # type of img_name_cond_pairs -> <class 'list'>
        # length of img_name_cond_pairs -> 3160
        name1 = out['name1'][0]
        image_buildings = out['image_buildings'][0].cpu().numpy()
        
        image_name_cond_pair = out['img_name_cond_pairs'][pair_idx]
        name2_cond1 = image_name_cond_pair[0][0]
        # print("name2_cond1 : ",name2_cond1)
        name2_cond2 = image_name_cond_pair[1][0]
        
        # print("name2_cond1 : ",name2_cond1)
        # print("name2_cond2 : ",name2_cond2)
        # #Load Tx (transmitter):
        img_name_Tx_cond1 = os.path.join(self.dir_Tx, name2_cond1)
        img_name_Tx_cond2 = os.path.join(self.dir_Tx, name2_cond2)
        image_Tx_cond1 = np.asarray(io.imread(img_name_Tx_cond1))
        # print("image_Tx_cond1.shape : ",image_Tx_cond1.shape)
        image_Tx_cond2 = np.asarray(io.imread(img_name_Tx_cond2))
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain_cond1 = os.path.join(self.dir_gain, name2_cond1)
            img_name_gain_cond2 = os.path.join(self.dir_gain, name2_cond2)
            image_gain_cond1 = np.expand_dims(np.asarray(io.imread(img_name_gain_cond1)),axis=2)/255
            image_gain_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gain_cond2)),axis=2)/255
          
        else:
            raise NotImplementedError("rand simulation not implemented yet")
        
        # inputs to radioUNet
        if self.carsInput=="no":
            inputs_cond1=np.stack([image_buildings, image_Tx_cond1, image_buildings], axis=2)
            inputs_cond2=np.stack([image_buildings, image_Tx_cond2, image_buildings], axis=2)
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx_cond1=image_Tx_cond1/256
            image_Tx_cond2=image_Tx_cond2/256
            image_cars = out['image_cars'][0].cpu().numpy()
            # img_name_cars = os.path.join(self.dir_cars, name1)
            # image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs_cond1=np.stack([image_buildings, image_Tx_cond1, image_cars], axis=2)
            inputs_cond2=np.stack([image_buildings, image_Tx_cond2, image_cars], axis=2)
        
        if self.transform:
            inputs_cond1 = self.transform(inputs_cond1).type(torch.float32)
            inputs_cond2 = self.transform(inputs_cond2).type(torch.float32)
            image_gain_cond1 = self.transform(image_gain_cond1).type(torch.float32)
            image_gain_cond2 = self.transform(image_gain_cond2).type(torch.float32)
        
        real_out = {}
        real_out['image_cond1'] = self.transform_compose(image_gain_cond1)
        real_out['image_cond2'] = self.transform_compose(image_gain_cond2)
        real_out['cond1'] = self.transform_GY(inputs_cond1)
        real_out['cond2'] = self.transform_GY(inputs_cond2)
        
        real_out['img_name_cond1'] = name2_cond1
        real_out['img_name_cond2'] = name2_cond2
        return real_out
        
    def __getitem__(self, idx):
        # 现在idx就表示第idx个场景
        idxr = idx
        dataset_map_ind=self.maps_inds[idxr+self.ind1] # 实际的场景索引
        name1 = str(dataset_map_ind) + ".png" # get building and cars
        out={}
        out['name1'] = name1 # get building and cars
        
        # loading buildings
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   

        out['image_buildings'] = image_buildings # get building
        # # loading cars
        # img_name_cars = os.path.join(self.dir_cars, name1)
        # image_cars = np.asarray(io.imread(img_name_cars))/256
        # out['image_cars'] = image_cars # get cars
        if self.carsInput=="no":
            pass 
        else: #cars input
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            out['image_cars'] = image_cars # get cars
        
        
        # loading antennas
        out['img_name_cond_pairs'] = []
        
        for i in range(self.numTx):
            # name2 = str(dataset_map_ind) + "_" + str(i) + ".png" # get Radio map
            cond_name1 = str(dataset_map_ind) + "_" + str(i) + ".png"
            for j in range(self.numTx):
                cond_name2 = str(dataset_map_ind) + "_" + str(j) + ".png" # get Radio map for condition 2
                out['img_name_cond_pairs'].append((cond_name1, cond_name2)) # 
            
        return out  
        # idxr=np.floor(idx/self.num_Tx_pairs).astype(int) # idxr为场景列表的第几个场景
        # # idxc=idx-idxr*self.num_Tx_pairs # idxc为该场景的第几个Tx
        # idx_pairs = self.Tx_inds_pairs[idx] # idxc为该场景的第几个Tx对
        # idxc=idx_pairs[0] # 
        # idxc_cond2 =  idx_pairs[1] # 
        # dataset_map_ind=self.maps_inds[idxr+self.ind1] # 实际的场景索引
        
        # # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        # #names of files that depend only on the map:
        # name1 = str(dataset_map_ind) + ".png"
        # #names of files that depend on the map and the Tx:
        # name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        # name2_cond2 = str(dataset_map_ind) + "_" + str(idxc_cond2) + ".png"
        
        # #Load buildings:
        # if self.cityMap == "complete":
        #     img_name_buildings = os.path.join(self.dir_buildings, name1)
        # else:
        #     if self.cityMap == "rand":
        #         self.missing=np.random.randint(low=1, high=5)
        #     version=np.random.randint(low=1, high=7)
        #     img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
        #     str(self.missing)
        # image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        # #Load Tx (transmitter):
        # img_name_Tx = os.path.join(self.dir_Tx, name2)
        # image_Tx = np.asarray(io.imread(img_name_Tx))
        
        # img_name_Tx_cond2 = os.path.join(self.dir_Tx, name2_cond2)
        # image_Tx_cond2 = np.asarray(io.imread(img_name_Tx_cond2))
        
        # #Load radio map:
        # if self.simulation!="rand":
        #     img_name_gain = os.path.join(self.dir_gain, name2)  
        #     image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
            
        #     img_name_gain_cond2 = os.path.join(self.dir_gain, name2_cond2)  
        #     image_gain_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gain_cond2)),axis=2)/255
            
        # else: #random weighted average of DPM and IRT2
        #     img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
        #     img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
        #     #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
        #     #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
        #     w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
        #     image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
        #                 + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256

        #     img_name_gainDPM_cond2 = os.path.join(self.dir_gainDPM, name2_cond2) 
        #     img_name_gainIRT2_cond2 = os.path.join(self.dir_gainIRT2, name2_cond2) 
        #     #image_gainDPM_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gainDPM_cond2)),axis=2)/255
        #     #image_gainIRT2_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2_cond2)),axis=2)/255
        #     w_cond2=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
        #     image_gain_cond2= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2_cond2)),axis=2)/256  \
        #                 + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM_cond2)),axis=2)/256    
        # # #pathloss threshold transform
        # # if self.thresh>0:
        # #     mask = image_gain < self.thresh
        # #     image_gain[mask]=self.thresh
        # #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        # #     image_gain=image_gain/(1-self.thresh)
        # # ---------- important ----------


        # # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        # #
        # # image_gain[image_gain >= self.thresh] = 1
                 
        
        # #inputs to radioUNet
        # if self.carsInput=="no":
        #     inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
        #     inputs_cond2=np.stack([image_buildings, image_Tx_cond2, image_buildings], axis=2)
        #     #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
        #     #so we can use the same learning rate as RadioUNets
        # else: #cars
        #     #Normalization, so all settings can have the same learning rate
        #     image_buildings=image_buildings/256
        #     image_Tx=image_Tx/256
        #     image_Tx_cond2=image_Tx_cond2/256
        #     img_name_cars = os.path.join(self.dir_cars, name1)
        #     image_cars = np.asarray(io.imread(img_name_cars))/256
        #     inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
        #     inputs_cond2=np.stack([image_buildings, image_Tx_cond2, image_cars], axis=2)
        #     #note that ToTensor moves the channel from the last asix to the first!

        
        # if self.transform:
        #     inputs = self.transform(inputs).type(torch.float32)
        #     inputs_cond2 = self.transform(inputs_cond2).type(torch.float32)
        #     image_gain = self.transform(image_gain).type(torch.float32)
        #     image_gain_cond2 = self.transform(image_gain_cond2).type(torch.float32)
        #     #note that ToTensor moves the channel from the last asix to the first!

        # out={}
        # out['image'] = self.transform_compose(image_gain)
        # out['image_cond2'] = self.transform_compose(image_gain_cond2)
        # out['cond'] = self.transform_GY(inputs)
        # out['cond2'] = self.transform_GY(inputs_cond2)
        # # out['image'] = image_gain
        # # out['cond'] = inputs
        # out['img_name'] = name2
        # out['img_name_cond2'] = name2_cond2
        # return out
        
        

class RadioUNet_only_building(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            self.ind1=600
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_buildings, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_buildings, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['cond'] = self.transform_GY(inputs)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        return out
    
    

class RadioUNet_c_PINN(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            # self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            self.maps_inds=np.array([599])
            # Determenistic "random" shuffle of the maps:
            # np.random.seed(42)
            # np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=640
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            # self.ind1=600
            # self.ind2=699
            self.ind1=0
            self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        buildings_mask = image_buildings > 0.
        
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        Tx_mask = image_Tx > 0. # Tx mask == f
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!
        


        out={}
        out['image'] = self.transform_compose(image_gain)
        out['cond'] = self.transform_GY(inputs)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        out['buildings_mask'] = buildings_mask
        out['Tx_mask'] = Tx_mask
        return out
    

class RadioUNet_c_tsne(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            # self.maps_inds=np.array([599])
            # Determenistic "random" shuffle of the maps:
            np.random.seed(64)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=19
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            self.ind1=600
            self.ind2=699
            # self.ind1=0
            # self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['image'] = self.transform_compose(image_gain)
        out['cond'] = self.transform_GY(inputs)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        out['target_building'] = dataset_map_ind
        return out
    
    

class RadioUNet_c_sprseIRT4_k(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=2,                  
                 thresh=0.2,
                 simulation="IRT4",
                 carsSimul="no",
                 carsInput="no",
                 cityMap="complete",
                 missing=1,
                 num_samples=300,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default = 2. Note that IRT4 works only with numTx<=2.                
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation: default="IRT4", with an option to "DPM", "IRT2".
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            num_samples: number of samples in the sparse IRT4 radio map. Default=300.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
            
        Output:
            
        """
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=601
            self.ind2=699
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="IRT4":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT4/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT4/"
        
        elif simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"  
        
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.num_samples=num_samples
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
        
        
        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/256
        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        #pathloss threshold transform
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
        
        #Saprse IRT4 samples, determenistic and fixed samples per map
        image_samples = np.zeros((self.width,self.height))
        seed_map=np.sum(image_buildings) # Each map has its fixed samples, independent of the transmitter location.
        np.random.seed(seed_map)       
        x_samples=np.random.randint(0, 255, size=self.num_samples)
        y_samples=np.random.randint(0, 255, size=self.num_samples)
        image_samples[x_samples,y_samples]= 1
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!
        
        

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            image_samples = self.transform(image_samples).type(torch.float32)


        return [inputs, image_gain, image_samples]
    
    

class RadioUNet_c_IRT4_k2_neg_norm(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="IRT4_k2_neg_norm",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            # self.maps_inds=np.array([599])
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=501
            self.ind2=600
        elif phase=="test":
            self.ind1=600
            self.ind2=699
            # self.ind1=0
            # self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif simulation=="IRT4_k2_neg_norm":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT4/"
                self.dir_k2_neg_norm=self.dir_dataset+"gain/IRT4_k2_neg_norm/"
            else:
                raise NotImplementedError
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gain, name2)  
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
            
            img_name_k2_neg_norm = os.path.join(self.dir_k2_neg_norm, name2)
            k2_neg_norm = np.asarray(io.imread(img_name_k2_neg_norm))/255
        else: #random weighted average of DPM and IRT2
            # img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            # img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            # #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            # #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            # w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            # image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
            #             + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
            raise NotImplementedError
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            # 
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['k2_neg_norm'] = k2_neg_norm
        out['image'] = self.transform_compose(image_gain)
        out['cond'] = self.transform_GY(inputs)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        out['Tx_pos_mask'] = image_Tx
        return out


class RadioUNet_c_split_time_z_mean(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            self.Tx_inds=np.arange(0,80,1,dtype=np.int16)
            # self.maps_inds=np.array([1, 2])
            # self.Tx_inds=np.array([0,1,2])
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            self.Tx_inds=None
            
        if phase=="train":
            self.ind1=0
            self.ind2=640
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            # self.ind1=600
            # self.ind2=699
            self.ind1=0
            self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=numTx if self.Tx_inds is None else len(self.Tx_inds)           
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1) * self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int) # idxr为场景列表的第几个场景
        idxc=idx-idxr*self.numTx # idxc为该场景的第几个Tx
        idxc=self.Tx_inds[idxc] # idxc为该场景的第几个Tx的索引
        # idxc_cond2 = (idxc + 1) % self.numTx # idxc_cond2为该场景的另一个Tx的索引
        # idxc_cond2 = self.Tx_inds[idxc_cond2] # idxc_cond2为该场景的另一个Tx的索引
        dataset_map_ind=self.maps_inds[idxr+self.ind1] # 实际的场景索引
        # dataset_map_ind2=self.maps_inds[(idxr+self.ind1+1) % len(self.maps_inds)] # 实际的下一个场景索引
        
        # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        # name12 = str(dataset_map_ind2) + ".png"
        #names of files that depend on the map and the Tx:
        name2_list = [str(dataset_map_ind) + "_" + str(i) + ".png" for i in self.Tx_inds]
        
        # name2_cond = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
            # img_name_buildings2 = os.path.join(self.dir_buildings, name12)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            # img_name_buildings2 = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name12)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))
        image_buildings_list = [image_buildings for _ in range(len(name2_list))]
        # image_buildings2 = np.asarray(io.imread(img_name_buildings2))   
        #Load Tx (transmitter):
        # img_name_Tx = os.path.join(self.dir_Tx, name2)
        img_name_Tx_list = [os.path.join(self.dir_Tx, name2) for name2 in name2_list]
        # image_Tx = np.asarray(io.imread(img_name_Tx))
        image_Tx_list = [np.asarray(io.imread(img_name_Tx)) for img_name_Tx in img_name_Tx_list]
        # img_name_Tx_cond2 = os.path.join(self.dir_Tx, name2_cond2)
        # image_Tx_cond2 = np.asarray(io.imread(img_name_Tx_cond2))
        
        #Load radio map:
        if self.simulation!="rand":
            # img_name_gain = os.path.join(self.dir_gain, name2)  
            # image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255
            img_name_gain_list = [os.path.join(self.dir_gain, name2) for name2 in name2_list]
            image_gain_list = [np.expand_dims(np.asarray(io.imread(img_name_gain)),axis=2)/255 for img_name_gain in img_name_gain_list]
            
            # img_name_gain_cond2 = os.path.join(self.dir_gain, name2_cond2)  
            # image_gain_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gain_cond2)),axis=2)/255
           
            
        else: #random weighted average of DPM and IRT2
            # img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            # img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            # #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            # #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            # w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            # image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
            #             + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256

            # img_name_gainDPM_cond2 = os.path.join(self.dir_gainDPM, name2_cond2) 
            # img_name_gainIRT2_cond2 = os.path.join(self.dir_gainIRT2, name2_cond2) 
            # #image_gainDPM_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gainDPM_cond2)),axis=2)/255
            # #image_gainIRT2_cond2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2_cond2)),axis=2)/255
            # w_cond2=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            # image_gain_cond2= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2_cond2)),axis=2)/256  \
            #             + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM_cond2)),axis=2)/256    
            raise NotImplementedError
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            # inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            inputs_list = [np.stack([image_buildings, image_Tx, image_buildings], axis=2) for image_buildings, image_Tx in zip(image_buildings_list, image_Tx_list)]
            # inputs_cond2=np.stack([image_buildings2, image_Tx_cond2, image_buildings2], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            # image_buildings2=image_buildings2/256
            # image_Tx=image_Tx/256
            image_Tx_list = [image_Tx/256 for image_Tx in image_Tx_list]
            # image_Tx_cond2=image_Tx_cond2/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            # img_name_cars2 = os.path.join(self.dir_cars, name12)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            image_cars_list = [image_cars for _ in range(len(name2_list))]
            # image_cars2 = np.asarray(io.imread(img_name_cars2))/256
            # inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            # inputs_cond2=np.stack([image_buildings2, image_Tx_cond2, image_cars2], axis=2)
            inputs_list = [np.stack([image_buildings, image_Tx, image_cars], axis=2) for image_buildings, image_Tx, image_cars in zip(image_buildings_list, image_Tx_list, image_cars_list)]
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            # inputs = self.transform(inputs).type(torch.float32)
            inputs_list = [self.transform(inputs).type(torch.float32) for inputs in inputs_list]
            # inputs_cond2 = self.transform(inputs_cond2).type(torch.float32)
            image_gain_list = [self.transform(image_gain).type(torch.float32) for image_gain in image_gain_list]
            # image_gain_cond2 = self.transform(image_gain_cond2).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        # out['image'] = self.transform_compose(image_gain)
        # out['image_cond2'] = self.transform_compose(image_gain_cond2)
        # out['cond'] = self.transform_GY(inputs)
        # out['cond2'] = self.transform_GY(inputs_cond2)
        
        out['image_list'] = [self.transform_compose(image_gain) for image_gain in image_gain_list]
        out['cond2_image_list'] = [out['image_list'][idxc] for _ in range(len(out['image_list']))]
        out['cond_list'] = [self.transform_GY(inputs) for inputs in inputs_list]
        out['cond2_list'] = [out['cond_list'][idxc] for _ in range(len(out['cond_list']))]
        # out['image'] = image_gain
        # out['cond'] = inputs
        # out['img_name'] = name2
        out['img_name_list'] = name2_list
        out['cond2_img_name_list'] = [name2_list[idxc] for _ in range(len(out['cond_list']))]
        
        # out['idxc'] = idxc
        # out['img_name_cond2'] = name2_cond2
        return out


class RadioUNet_c_DPM2IRT4(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""
    def __init__(self,maps_inds=np.zeros(1), phase="train",
                 ind1=0,ind2=0, 
                 dir_dataset="RadioMapSeer/",
                 numTx=80,                  
                 thresh=0.05,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom". 
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.                 
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is 
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())
                 
        Output:
            inputs: The RadioUNet inputs.  
            image_gain
            
        """

        self.transform_GY = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        transform_BZ = transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
        self.transform_compose = transforms.Compose([
            transform_BZ
        ])
        #self.phase=phase
                
        if maps_inds.size==1:
            self.maps_inds=np.arange(0,700,1,dtype=np.int16)
            # self.maps_inds=np.array([599])
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
            
        if phase=="train":
            self.ind1=0
            self.ind2=640
        elif phase=="val":
            self.ind1=600
            self.ind2=650
        elif phase=="test":
            self.ind1=600
            self.ind2=699
            # self.ind1=0
            # self.ind2=len(self.maps_inds)-1
            # self.ind1= 0
            # self.ind2= 0
        else: # custom range
            self.ind1=ind1
            self.ind2=ind2
            
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput
        if simulation=="DPM" :
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/DPM/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsDPM/"
        elif simulation=="IRT2":
            if carsSimul=="no":
                self.dir_gain=self.dir_dataset+"gain/IRT2/"
            else:
                self.dir_gain=self.dir_dataset+"gain/carsIRT2/"
        elif  simulation=="rand":
            if carsSimul=="no":
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/IRT2/"
            else:
                # self.dir_gainDPM=self.dir_dataset+"gain/carsDPM/"
                self.dir_gainDPM=self.dir_dataset+"gain/DPM/"
                self.dir_gainIRT2=self.dir_dataset+"gain/carsIRT2/"
        elif simulation=="DPM2IRT4":
            if carsSimul=="no":
                self.dir_gainDPM="/home/Users_Work_Space/plzheng/RadioDiff_v2/results/dpm_gt/"
                self.dir_gainIRT4=self.dir_dataset+"gain/IRT4/"
                self.dir_k2_neg_norm=self.dir_dataset+"gain/IRT4_k2_neg_norm/"
            else:
                raise NotImplementedError
                
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing
        if cityMap=="complete":
            self.dir_buildings=self.dir_dataset+"png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset+"png/buildings_missing" # a random index will be concatenated in the code
        #else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"
            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "png/antennas/" 
        #later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput!="no":
            self.dir_cars = self.dir_dataset+ "png/cars/" 
        
        self.height = 256
        self.width = 256

        
    def __len__(self):
        return (self.ind2-self.ind1+1)*self.numTx

    def normalize_to_neg_one_to_one(self, img):
        return img * 2 - 1
    
    def __getitem__(self, idx):
        
        idxr=np.floor(idx/self.numTx).astype(int)
        idxc=idx-idxr*self.numTx 
        dataset_map_ind=self.maps_inds[idxr+self.ind1]+1
        # dataset_map_ind=self.maps_inds[idxr+self.ind1]
        #names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        #names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"
        
        #Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing=np.random.randint(low=1, high=5)
            version=np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings+str(self.missing)+"/"+str(version)+"/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))   
        
        #Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))
        
        #Load radio map:
        if self.simulation!="rand":
            img_name_gain = os.path.join(self.dir_gainDPM, name2)  
            img_name_gain_IRT4 = os.path.join(self.dir_gainIRT4, name2)
            img_name_k2_neg_norm = os.path.join(self.dir_k2_neg_norm, name2)
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain, as_gray=True)),axis=2)/255
            image_gain_IRT4 = np.expand_dims(np.asarray(io.imread(img_name_gain_IRT4)),axis=2)/255
            
            k2_neg_norm = np.asarray(io.imread(img_name_k2_neg_norm))/255

        else: #random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2) 
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2) 
            #image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            #image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w=np.random.uniform(0,self.IRT2maxW) # IRT2 weight of random average
            image_gain= w*np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/256  \
                        + (1-w)*np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/256
        
        # #pathloss threshold transform
        # if self.thresh>0:
        #     mask = image_gain < self.thresh
        #     image_gain[mask]=self.thresh
        #     image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
        #     image_gain=image_gain/(1-self.thresh)
        # ---------- important ----------


        # image_gain[np.logical_and(image_gain > 0, image_gain < self.thresh)] /= 255.
        #
        # image_gain[image_gain >= self.thresh] = 1
                 
        
        #inputs to radioUNet
        if self.carsInput=="no":
            inputs=np.stack([image_buildings, image_Tx, image_buildings], axis=2)
            #The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence, 
            #so we can use the same learning rate as RadioUNets
        else: #cars
            #Normalization, so all settings can have the same learning rate
            image_buildings=image_buildings/256
            image_Tx=image_Tx/256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars))/256
            inputs=np.stack([image_buildings, image_Tx, image_cars], axis=2)
            #note that ToTensor moves the channel from the last asix to the first!

        
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            # print("image gain shape: ", image_gain.shape)
            # print("image gain IRT4 shape: ", image_gain_IRT4.shape)
            image_gain = self.transform(image_gain).type(torch.float32)
            image_gain_IRT4 = self.transform(image_gain_IRT4).type(torch.float32)
            #note that ToTensor moves the channel from the last asix to the first!

        out={}
        out['k2_neg_norm'] = k2_neg_norm
        out['image_dpm'] = self.transform_compose(image_gain)
        out['image_irt4'] = self.transform_compose(image_gain_IRT4)
        out['cond'] = self.transform_GY(inputs)
        # out['image'] = image_gain
        # out['cond'] = inputs
        out['img_name'] = name2
        out['Tx_pos_mask'] = image_Tx
        return out