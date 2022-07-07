from pyfr.shapes import BaseShape
from pyfr.util import lazyprop, subclass_where
from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import sys

import h5py


from loadload import load_class


class spod_avg_z(object):

    def __init__(self):
        start  = 60.0  #60
        end    = 90.0   #90
        dt     = 0.1     #0.1
        tt = np.arange(start, end, dt)
        self.time = []
        for i in range(len(tt)):
            self.time.append("{:.4f}".format(tt[i]))

        self.name = ['naca0012.ini','../../Re5e4/mesh.pyfrm','./pyfrs/naca0012_60.0000.pyfrs']
        self.name[-1] = f'../../Re5e4/pyfrs/naca0012_{self.time[0]}.pyfrs'
        self.mesh = load_class(self.name).load_mesh()

        self.suffix_etype = ['hex','pri']
        self.order = 4+1


    def load(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        rankn = list()
        p = list()
        if rank == 0:
            for key in self.mesh.keys():
                for etype in self.suffix_etype:
                    if key.split('_')[0] == 'spt' and key.split('_')[1] == etype:
                        if key.split('_')[-1] not in p:
                            p.append(key.split('_')[-1])
            #print(p)
            if len(p) != size:
                raise RuntimeError(f'please use exactly {len(p)} fo ranks.')

        # scatter partitions into current availible ranks and name it as rankn i.e. p0
        rankn = comm.scatter(p, root = 0)
        #print(rank,rankn)
        # wait until check is finished
        comm.Barrier()




        f = h5py.File('con.zhenyang','r')
        # do some reordering to the data
        self.data= defaultdict(list)
        for key in f.keys():
            if key.split('_')[1] == rankn:
                self.data[key].append(np.array(f[key]))
            elif len(key.split('_')) > 2 and key.split('_')[-2] == rankn:

                etype = key.split('_')[0]
                eid = key.split('_')[1]
                rank_original = key.split('_')[2]

                if f'{etype}_{rankn}_{eid}_{rank_original}' in self.data.keys():
                    self.data[f'{etype}_{rankn}_{eid}_{rank_original}'] = np.concatenate((self.data[f'{etype}_{rankn}_{eid}_{rank_original}'],np.array(f[key])[0]),axis=None)
                else:
                    self.data[f'{etype}_{rankn}_{eid}_{rank_original}'] = np.array(f[key])[0]

        f.close()

        #print(rankn,list(self.data.keys()))

        self.average_z(rankn, comm)
        #print(str(rankn)+' finished')
        #comm.Barrier()
        #if rankn == 'p2':
        #    print('All finished')
        #self.collect_all_data(rankn,comm)

    def get_right_keys(self, mesh_info):
        ele = defaultdict()
        eid = defaultdict()
        cut_index = defaultdict()
        for key in mesh_info.keys():
            #get right element:
            if key.split('_')[0] == 'spt' and  key.split('_')[1] == 'tet':# or  key.split('_')[1] == 'pyr':
                part_name = f'con_{key.split("_")[-1]}'
                con = mesh_info[part_name]
                con_etype = con[['f0']].astype('U4')
                con_eid = con[['f1']].astype('i4')
                con_fid = con[['f2']].astype('i1')
                con_pid = con[['f3']].astype('i2')
                index1 = np.where(con_pid == -1)
                index2 = np.where(con_etype[index1] == key.split('_')[1])[0]

                index = (index1[0][index2],index1[1][index2])
                #print(index)

                keysoln = f'tet_{key.split("_")[-1]}'
                eid[keysoln] = con_eid[index]
                mesh = mesh_info[key]
                ele[keysoln] = mesh[:,eid[keysoln]]
                print(ele[keysoln].shape)

        for key in ele.keys():
            cut_index[key] = np.where(ele[key][...,2] < 10e-8)

        return eid, ele, cut_index


    def average_z(self,rankn, comm):
        self.avgfield = defaultdict()
        self.avgmesh = defaultdict()
        self.avgfield_t = defaultdict()
        self.avgmesh_t = defaultdict()
        self.length_spa = defaultdict(list)

        # for acoustics, it is 2D so just get one 2D slice is ok
        eindex, ac_mesh, cut_index = self.get_right_keys(self.mesh)
        acoustic_soln = defaultdict()
        acoustic_mesh = defaultdict()



        for time in self.time:
            # average in space each rank:
            self.name[-1] = f'../../Re5e4/pyfrs/naca0012_{time}.pyfrs'
            soln = load_class(self.name).load_soln()
            #print(list(soln.keys()))


            for key in self.data.keys():

                part0 = key.split('_')
                part = f'soln_{part0[0]}_{part0[1]}'
                partm = f'spt_{part0[0]}_{part0[1]}'

                if len(key.split('_')) > 2:


                    # average all element in span of this rank
                    sln = np.sum(soln[part][...,self.data[key]],axis=-1)
                    # average all nodes in span of this element
                    self.avgfield_t[key] = np.sum( sln.reshape( (int(sln.shape[0]/self.order),self.order,sln.shape[1]),order='F') ,axis=1)

                    if time == self.time[0]:
                        msh = np.sum(self.mesh[partm][:,self.data[key]],axis=-2)
                        self.avgmesh_t[key] = np.sum( msh.reshape( (int(msh.shape[0]/self.order),self.order,msh.shape[1]),order='F') ,axis=1)
                        self.length_spa[key].append(len(self.data[key]))

                    # at the end of each time step, exchange info try to collect all data of that element


                else:

                    #print(self.data[key][0].shape)

                    shape1 = self.data[key][0].reshape(self.data[key][0].shape[0]*self.data[key][0].shape[1])
                    sln = soln[part][...,shape1]
                    sln = np.sum(sln.reshape(sln.shape[0],sln.shape[1],self.data[key][0].shape[0],self.data[key][0].shape[1]),axis=-1)
                    self.avgfield[key] = np.sum( sln.reshape( (int(sln.shape[0]/self.order),self.order,sln.shape[1],sln.shape[2]),order='F') ,axis=1) / self.order / min(self.data[key][0].shape)

                    if time == self.time[0]:
                        msh = self.mesh[partm][:,shape1]
                        msh = np.sum(msh.reshape(msh.shape[0],self.data[key][0].shape[0],self.data[key][0].shape[1],msh.shape[-1]),axis=-2)
                        self.avgmesh[key] = np.sum( msh.reshape( (int(msh.shape[0]/self.order),self.order,msh.shape[1],msh.shape[2]),order='F') ,axis=1).swapaxes(1,2) / self.order / min(self.data[key][0].shape)
            self.exchange_info(comm, rankn, time)

            # acousitcs
            for key in soln.keys():
                if key.split('_')[0] == 'soln' and key.split('_')[1] == 'tet' and key.split('_')[-1] == rankn:
                    keysoln = f'tet_{key.split("_")[-1]}'
                    acoustic_soln[keysoln] = soln[key][...,eindex[keysoln]]
                    acoustic_soln[keysoln] = acoustic_soln[keysoln][cut_index[keysoln][0],:,cut_index[keysoln][1]]
                    if time == self.time[0]:
                        keymesh = f'spt_tet_{key.split("_")[-1]}'
                        acoustic_mesh[keysoln] = ac_mesh[keysoln][cut_index[keysoln]]



            for rank in range(comm.Get_size()):
                if f'p{rank}' == rankn:
                    dir = f'./series2/time_series_{time}.zhenyang'
                    self.write_to_file(self.avgfield, dir)
                    self.write_to_file(acoustic_soln, dir)
                    if time == self.time[0]:
                        dir = f'./series2/time_series_{time}_mesh.zhenyang'
                        self.write_to_file(self.avgmesh, dir)
                        self.write_to_file(acoustic_mesh, dir)

                comm.Barrier()
            print(list(self.avgfield.keys()))



    def exchange_info(self, comm, rankn, time):
        revbuff = defaultdict(list)
        revmshbuff = defaultdict(list)
        lengthbuff = defaultdict(list)

        for i in range(comm.Get_size()):
            if rankn == 'p0' and i == 0:
                print('communication between ranks')
            revbuff.update(comm.bcast(self.avgfield_t, root=i))
            if time == self.time[0]:
                revmshbuff.update(comm.bcast(self.avgmesh_t, root=i))
            lengthbuff.update(comm.bcast(self.length_spa, root=i))
            comm.Barrier()


        if rankn == 'p0':
            print('finish comm')
        for key1 in revbuff.keys():
            ksp1 = key1.split('_')
            if ksp1[1] == ksp1[-1]:
                length = np.array(self.length_spa[key1])
                for key2 in revbuff.keys():
                    ksp2 = key2.split('_')
                    if ksp2[2] == ksp1[2] and ksp2[1] != ksp2[-1] and ksp2[0] == ksp1[0] and ksp2[-1] == ksp1[-1]:
                        revbuff[key1] += revbuff[key2]
                        if time == self.time[0]:
                            revmshbuff[key1] += revmshbuff[key2]
                        length += np.array(lengthbuff[key2])
                if rankn == ksp1[-1]:
                    #print(self.length_spa[key1])
                    self.avgfield[f'{ksp1[0]}_{ksp1[1]}'] = np.dstack((self.avgfield[f'{ksp1[0]}_{ksp1[1]}'], revbuff[key1] / length / self.order))
                    if time == self.time[0]:
                        self.avgmesh[f'{ksp1[0]}_{ksp1[1]}'] = np.dstack((self.avgmesh[f'{ksp1[0]}_{ksp1[1]}'], revmshbuff[key1] / length / self.order))
                    #self.avgfield[f'{ksp1[0]}_{ksp1[1]}'] = np.concatenate((self.avgfield[f'{ksp1[0]}_{ksp1[1]}'],revbuff[key1] / self.length_spa[key1] / self.order),axis=-1)
                    #self.avgmesh[f'{ksp1[0]}_{ksp1[1]}'] = np.append(self.avgmesh[f'{ksp1[0]}_{ksp1[1]}'],revmshbuff[key1] / self.length_spa[key1] / self.order)


    def write_to_file(self,msh,dir):
        with h5py.File(dir,'a') as f:
            for key in msh.keys():
                if len(msh[key].shape) > 2:
                    msh[key] = msh[key].swapaxes(1,2)
                #f.create_dataset(f'{key}', data=msh[key])
                f.create_dataset(f'{key}', data=msh[key].reshape(-1, msh[key].shape[-1]))

            f.close()






















""" not working from now"""








class spod_time(spod_avg_z):
    def __init__(self):
        super().__init__()

    def load(self, dir):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        rankn = list()
        p = list()
        if rank == 0:
            for key in self.mesh.keys():
                for etype in self.suffix_etype:
                    if key.split('_')[0] == 'spt' and key.split('_')[1] == etype:
                        if key.split('_')[-1] not in p:
                            p.append(key.split('_')[-1])
            #print(p)
            if len(p) != size:
                raise RuntimeError(f'please use exactly {len(p)} fo ranks.')

        # scatter partitions into current availible ranks and name it as rankn i.e. p0
        rankn = comm.scatter(p, root = 0)
        #print(rank,rankn)
        # wait until check is finished
        comm.Barrier()



        Nt = len(self.time)
        NFFT = 64
        Novlp = int(NFFT / 2)
        Nblk = np.floor((Nt - Novlp)/(NFFT - Novlp))
        FreqofInterest = 5 # fs = 1/dt, fmax = fs/2

        self.data = defaultdict()
        for time in self.time:
            dir_time = f'./series/time_series_{time}.zhenyang'

            f = h5py.File(dir_time,'r')
            for key in f.keys():
                if key.split('_')[-1] == f'p{rank}':
                    if time == self.time[0]:
                        self.data[key] = np.array(f[key])[:,:,:,None]
                    else:
                        self.data[key] = np.append(self.data[key],np.array(f[key])[:,:,:,None],axis=-1)

            f.close()
        if rankn == 'p0':
            print(self.data['hex_p0'].shape)


        self.eigen = defaultdict()

        for key in self.data.keys():
            # reshape data to reduce matrix dimension
            shape = self.data[key].shape
            self.data[key] = self.data[key].reshape((shape[0]*shape[1]*shape[2],shape[3]))
            # create Creat_blocks
            self.data[key] = self.Creat_blocks(self.data[key],NFFT,Novlp,rankn)
            # foriour transform of data block
            self.data[key] = self.block_DFT(self.data[key],rankn)
            # collect data at each frequency
            self.data[key] = self.creat_freq_blocks(self.data[key],FreqofInterest,rankn)
            # do svd to these frequency blocks
            self.data[key],self.eigen[f'{key}_eigen'] = self.SPOD_SVD(self.data[key],rankn)
            # reshape to readable format
            self.data[key] = self.data[key].reshape((shape[0],shape[1],shape[2],self.data[key].shape[-2],self.data[key].shape[-1]))
        # wrtie to file
        for rank in range(comm.Get_size()):
            if f'p{rank}' == rankn:
                self.write_to_file(self.data,'SPOD.zhenyang')
                self.write_to_file(self.eigen,'SPOD.zhenyang')
            comm.Barrier()



    def Creat_blocks(self,var,NFFT,Novlp,rankn):
        # -----------------------------------------------------------------
        # generate blocks
        if rankn == 'p0':
            print('-------------------------------------------------')
            print('Divide the varible into blocks using \'NFFT\' and \'overlap\' parameters')
            print('-------------------------------------------------')

        Nblocks = int(np.floor((len(self.time) - Novlp)/(NFFT - Novlp)))
        print('Number of blocks: '+str(Nblocks))
        if Nblocks < 1:
            raise ValueError('No mate, input NFFT is larger than total time, it is not gonna work')

        blocks = np.zeros([var.shape[0],NFFT,Nblocks],dtype = 'complex_')   # blocks.shape = npts,nfft,blockid
        for i in range(Nblocks):
            blocks[:,:,i] = var[:,int(i*NFFT-i*Novlp):int((i+1)*NFFT-i*Novlp)]

        if rankn == 'p0':
            print('block shape: '+str(blocks.shape))
            print('-------------------------------------------------')
        return blocks

    def hamming_window(self, N):
        '''
            Standard Hamming window of length N
        '''
        x = np.arange(0,N,1)
        window = (0.54 - 0.46 * np.cos(2 * np.pi * x / (N-1)) ).T
        return window


    def block_DFT(self, blocks,rankn):
        if rankn == 'p0':
            print('-------------------------------------------------')
            print('Do DFT to each block')
            print('-------------------------------------------------')


        #blockHat = np.zeros([blocks.shape[0],blocks.shape[1],blocks.shape[2]],dtype = 'complex_')

        window = self.hamming_window(blocks.shape[1])

        if rankn == 'p0':
            print('Compeleted blocks: ')
        for i in tqdm(range(blocks.shape[2])):   #block.shape[2] == Nblocks
            #blockHat[:,:,i] = pyfftw.interfaces.numpy_fft.fftn(blocks[:,:,i],axes=1)*window  #this could be faster
            blocks[:,:,i] = np.fft.fft(blocks[:,:,i]*window ,axis=1)  #overwrite
        if rankn == 'p0':
            print('blockHat shape: '+str(blocks.shape))
            print('-------------------------------------------------')
        return blocks

    def creat_freq_blocks(self,blockHat,FreqofInterest,rankn):
        if rankn == 'p0':
            print('-------------------------------------------------')
            print('Assemble the data matrix in frequency domain: Qhat.shape = [npts,Nblocks,Nfrequencies]')
            print('-------------------------------------------------')
        Qhat = np.zeros([blockHat.shape[0],blockHat.shape[2],FreqofInterest],dtype = 'complex_')  # for memory problem FreqofInterest << blockHat.shape[1]
        for i in range(FreqofInterest):
            for j in range(blockHat.shape[2]):
                Qhat[:,j,i] = blockHat[:,i,j]
        if rankn == 'p0':
            print('Qhat block shape: '+str(Qhat.shape))
            print('-------------------------------------------------')
        return Qhat

    def SPOD_SVD(self,Qhat,rankn):
        if rankn == 'p0':
            print('-------------------------------------------------')
            print('Compute inner production to each block')
            print('-------------------------------------------------')

            print('Compeleted frequencies: ')

        A = np.zeros([Qhat.shape[1],Qhat.shape[1],Qhat.shape[2]],dtype = 'complex_')
        #Weight = BuildWieghtMatrix(TypeofWeight,Nvars,shape = Qhat.shape[0])
        for i in tqdm(range(Qhat.shape[2])):
            """
                a = blockHat[:,:,i]
                b = np.matmul(a.T,a)
            """
            #A[:,:,i] = np.dot(np.dot(Qhat[:,:,i].conj().T,Weight) , Qhat[:,:,i])/(Qhat.shape[1] - 1)    #using matmul could be super slow
            A[:,:,i] = np.dot(Qhat[:,:,i].conj().T , Qhat[:,:,i])#/(Qhat.shape[1] - 1)    #using matmul could be super slow

        if rankn == 'p0':
            print(A.shape)
            print('-------------------------------------------------')


            print('-------------------------------------------------')
            print('Compute the eigenvectors and engenvalues of A')
            print('-------------------------------------------------')

        w = np.zeros([A.shape[1],A.shape[2]],dtype='complex_')
        v = np.zeros([A.shape[1],A.shape[1],A.shape[2]],dtype='complex_')
        for i in range(A.shape[2]):
            w[:,i], v[:,:,i] = np.linalg.eig(A[:,:,i])    # w is eigenvalue, v[:,:,i] is normaöized eigenvector of w[:,i]

        del A ## clear the memory

        ## check routine
        for j in range(w.shape[1]):
            for i in w[:,j]:
                if i < 0:
                    print('Warning!: eignevalue is smaller than 0: eig = ' + str(i))
                    if abs(i) < 10e-5:
                        print('Warning!: eignevalue is significantly close to zero, take absolute value!')

                    else:
                        raise ValueError('Error!: Check with eigenvalue, programme exits...')


        if rankn == 'p0':
            print('-------------------------------------------------')
            print('Formulate eq9 in Andre\'s paper (performing the SVD)')
            print('-------------------------------------------------')


        sig = np.zeros([w.shape[0],w.shape[0],w.shape[1]],dtype='complex_')
        for i in range(w.shape[1]):
            for j in range(w.shape[0]):
                if w[j,i] == 0:
                    sig[j,j,i] = 0
                else:
                    sig[j,j,i] = 1/np.sqrt(w[j,i])

        Phi = np.zeros([Qhat.shape[0],Qhat.shape[1],Qhat.shape[2]],dtype='complex_')
        for i in range(Qhat.shape[2]):
            Phi[:,:,i] = np.dot(np.dot(Qhat[:,:,i],v[:,:,i]),sig[:,:,i])

        if rankn == 'p0':
            print('-------------------Done!------------------------')
        return Phi,w
