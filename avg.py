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

class Average(object):

    def __init__(self):
        start  = 60.0  #60
        end    = 90.0    #90
        dt     = 0.1     #0.1
        tt = np.arange(start, end, dt)
        self.time = []
        for i in range(len(tt)):
            self.time.append("{:.4f}".format(tt[i]))

        self.name = ['naca0012.ini','../../Re5e4/mesh.pyfrm','./pyfrs/naca0012_60.0000.pyfrs']
        self.name[-1] = f'../../Re5e4/pyfrs/naca0012_{self.time[0]}.pyfrs'
        self.mesh = load_class(self.name).load_mesh()

        self.suffix_etype = ['hex','pri']


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




        f = h5py.File('random2.zhenyang','r')
        # do some reordering to the data
        self.data= defaultdict(list)
        for key in f.keys():
            if key.split('_')[1] == rankn:
                self.data[key].append(np.array(f[key]))
            elif len(key.split('_')) > 2 and key.split('_')[-2] == rankn:

                etype = key.split('_')[0]
                eid = key.split('_')[1]

                if f'{etype}_{rankn}_{eid}' in self.data.keys():
                    self.data[f'{etype}_{rankn}_{eid}'] = np.concatenate((self.data[f'{etype}_{rankn}_{eid}'],np.array(f[key])[0]),axis=None)
                else:
                    self.data[f'{etype}_{rankn}_{eid}'] = np.array(f[key])[0]
                    #print(self.data[f'{etype}_{rankn}_{eid}'][0])
                    #print(np.array(f[key])[0])
                    #print(np.concatenate((),axis=none))

        #if rankn == 'p5':
        #    eid = 1005
        #    etype = 'hex'
        #    print(rankn,f'{etype}_{rankn}_{eid}',self.data[f'{etype}_{rankn}_{eid}'])
        #return 0



        f.close()

        #print(rankn,list(self.data.keys()))

        self.average(rankn)
        print(str(rankn)+' finished')
        comm.Barrier()
        if rankn == 'p0':
            print('All finished')
        self.collect_all_data(rankn,comm)



    def average(self,rankn):
        self.avgfield = defaultdict()
        self.avgmesh = defaultdict()
        self.length_spa = defaultdict(list)

        # first average in space while loading each snapshot:
        for time in self.time:
            # average in space each rank:
            self.name[-1] = f'../../Re5e4/pyfrs/naca0012_{time}.pyfrs'
            soln = load_class(self.name).load_soln()
            #print(list(soln.keys()))

            part0 = list(self.data.keys())[0].split('_')
            part = f'soln_{part0[0]}_{part0[1]}'
            partm = f'spt_{part0[0]}_{part0[1]}'

            #print(part)


            for key in self.data.keys():
                if len(key.split('_')) > 2:

                    if time == self.time[0]:
                        self.avgfield[key] = np.sum(soln[part][...,self.data[key]],axis=-1)
                        self.avgmesh[key] = np.sum(self.mesh[partm][:,self.data[key]],axis=-2)
                        self.length_spa[key].append(len(self.data[key]))
                    else:
                        self.avgfield[key] += np.sum(soln[part][...,self.data[key]],axis=-1)

                else:

                    #print(self.data[key][0].shape)
                    if time == self.time[0]:
                        shape1 = self.data[key][0].reshape(self.data[key][0].shape[0]*self.data[key][0].shape[1])
                        sln = soln[part][...,shape1]
                        msh = self.mesh[partm][:,shape1]
                        self.avgfield[key] = np.sum(sln.reshape(sln.shape[0],sln.shape[1],self.data[key][0].shape[0],self.data[key][0].shape[1]),axis=-1)
                        self.avgmesh[key] = np.sum(msh.reshape(msh.shape[0],self.data[key][0].shape[0],self.data[key][0].shape[1],msh.shape[-1]),axis=-2)
                        self.length_spa[key].append(min(self.data[key][0].shape))
                    else:
                        shape1 = self.data[key][0].reshape(self.data[key][0].shape[0]*self.data[key][0].shape[1])
                        sln = soln[part][...,shape1]
                        self.avgfield[key] += np.sum(sln.reshape(sln.shape[0],sln.shape[1],self.data[key][0].shape[0],self.data[key][0].shape[1]),axis=-1)
                    #print(np.sum(sln.reshape(sln.shape[0],sln.shape[1],self.data[key][0].shape[0],self.data[key][0].shape[1]),axis=-1).shape)

            if rankn == 'p0':
                #print(key,self.avgfield['hex_p0_1151'][0].shape)
                #print(key,self.avgfield['hex_p0'][0].shape)
                #print(self.avgfield.keys())
                #print(self.length_spa.keys())


                print(time+' in '+self.time[-1])
                if time == self.time[-1]:
                    print('waiting other ranks to finish')




    def collect_all_data(self,rankn,comm):
        revbuff = defaultdict(list)
        revmshbuff = defaultdict(list)

        for i in range(comm.Get_size()):
            if rankn == 'p0' and i == 0:
                print('communication between ranks')
            revbuff.update(comm.bcast(self.avgfield, root=i))
            revmshbuff.update(comm.bcast(self.avgmesh, root=i))
            comm.Barrier()


        """
        #revbuff.update(comm.gather(self.avgfield, root=0))
        for key in self.avgfield.keys():
            if len(key.split('_')) > 2:
                if rankn == 'p0':
                    print(key)
                revbuff.update(comm.gather(self.avgfield[key], root=0))
            #else:
            #    self.write_to_file()
            comm.Barrier()
        """
        if rankn == f'p0':

            print('start to write to file')

            idlist = list()
            idsln = defaultdict(list)
            idmsh = defaultdict(list)
            #print(list(revbuff.keys()))
            for key in revbuff.keys():
                eid = key.split('_')[-1]
                if len(key.split('_')) > 2 and eid not in idlist:
                    #print(key)
                    #keyname = f'other_{eid}'
                    tele = revbuff[key]
                    mele = revmshbuff[key]
                    idlist.append(eid)
                    #print(idlist)
                    for key2 in revbuff.keys():
                        if key2.split('_')[-1] == eid and key2 != key:
                            #print(key,key2)
                            tele += revbuff[key]
                            mele += revmshbuff[key]

                    #print(tele.shape,mele.shape,self.length_spa['hex_p0'])

                    sln = np.sum(tele.reshape((25,5,tele.shape[1]),order='F'),axis=1)
                    msh = np.sum(mele.reshape((25,5,mele.shape[1]),order='F'),axis=1)
                    idsln['others'].append(sln/len(self.time)/self.length_spa['hex_p0']/5)
                    idmsh['others'].append(msh/self.length_spa['hex_p0']/5)

                elif len(key.split('_')) == 2:
                    sln = np.sum(revbuff[key].reshape((25,5,revbuff[key].shape[1],revbuff[key].shape[2]),order='F'),axis=1)
                    self.write_to_file(sln/len(self.time)/self.length_spa['hex_p0']/5,key)
                    self.write_to_file(revmshbuff[key][:25]/self.length_spa['hex_p0'],f'{key}mesh')
            # write keyname 'others'
            self.write_to_file(idsln['others'],'others')
            self.write_to_file(idmsh['others'],'othersmesh')





    def write_to_file(self,msh,key):
        #print(key,msh.shape)
        print(key)
        with h5py.File('random1.zhenyang','a') as f:
            f.create_dataset(f'{key}', data=msh)
            f.close()
