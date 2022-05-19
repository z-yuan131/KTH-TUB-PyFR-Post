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
        end    = 64.1   #90
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


    def average_z(self,rankn, comm):
        self.avgfield = defaultdict()
        self.avgmesh = defaultdict()
        self.avgfield_t = defaultdict()
        self.avgmesh_t = defaultdict()
        self.length_spa = defaultdict(list)

        # first average in span while loading each snapshot:


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
            self.exchange_info(comm, rankn)


            for rank in range(comm.Get_size()):
                if f'p{rank}' == rankn:
                    dir = f'./series/time_series_{time}.zhenyang'
                    self.write_to_file(self.avgfield, dir)
                    if time == self.time[0]:
                        dir = f'./series/time_series_{time}_mesh.zhenyang'
                        self.write_to_file(self.avgmesh, dir)

                comm.Barrier()
            print(list(self.avgmesh.keys()))



    def exchange_info(self, comm, rankn):
        revbuff = defaultdict(list)
        revmshbuff = defaultdict(list)
        lengthbuff = defaultdict(list)

        for i in range(comm.Get_size()):
            if rankn == 'p0' and i == 0:
                print('communication between ranks')
            revbuff.update(comm.bcast(self.avgfield_t, root=i))
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
                        revmshbuff[key1] += revmshbuff[key2]
                        length += np.array(lengthbuff[key2])
                if rankn == ksp1[-1]:
                    #print(self.length_spa[key1])
                    self.avgfield[f'{ksp1[0]}_{ksp1[1]}'] = np.dstack((self.avgfield[f'{ksp1[0]}_{ksp1[1]}'], revbuff[key1] / length / self.order))
                    self.avgmesh[f'{ksp1[0]}_{ksp1[1]}'] = np.dstack((self.avgmesh[f'{ksp1[0]}_{ksp1[1]}'], revmshbuff[key1] / length / self.order))
                    #self.avgfield[f'{ksp1[0]}_{ksp1[1]}'] = np.concatenate((self.avgfield[f'{ksp1[0]}_{ksp1[1]}'],revbuff[key1] / self.length_spa[key1] / self.order),axis=-1)
                    #self.avgmesh[f'{ksp1[0]}_{ksp1[1]}'] = np.append(self.avgmesh[f'{ksp1[0]}_{ksp1[1]}'],revmshbuff[key1] / self.length_spa[key1] / self.order)


    def write_to_file(self,msh,dir):
        with h5py.File(dir,'a') as f:
            for key in msh.keys():
                f.create_dataset(f'{key}', data=msh[key])
            f.close()




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
        NFFT = 8
        Novlp = NFFT / 2
        Nblk = np.floor((Nt - Novlp)/(NFFT - Novlp))

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
