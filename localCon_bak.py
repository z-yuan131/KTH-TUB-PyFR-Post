
from pyfr.shapes import BaseShape
from pyfr.util import lazyprop, subclass_where
from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import sys

from loadload import load_class


class localConMain(object):

    def __init__(self):
        start  = 60.0  #60
        end    = 61    #90
        dt     = 1     #0.1
        self.suffix_etype = ['hex','pri']
        tt = np.arange(start, end, dt)
        time = []
        for i in range(len(tt)):
            time.append("{:.4f}".format(tt[i]))

        name = ['naca0012.ini','../../Re5e4/mesh.pyfrm','./pyfrs/naca0012_60.0000.pyfrs']
        name[-1] = f'../../Re5e4/pyfrs/naca0012_{time[0]}.pyfrs'
        self.mesh = load_class(name).load_mesh()


        self.layers = 20

    #---------------------------------------------------------
    def load_connectivity(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        self.cons = list()
        self.parts = list()

        self.face = {'hex': np.array([5,3,4,1,2,0],dtype='int8'), 'pri': np.array([1,0,2,3,4],dtype='int8')}
        rankn = list()

        # first there is a check routine to see if the number of
        # your mpi rank covers the number
        # of rank that your etype going to be avaerged covered.
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

        # second each rank do its own job for collecting elements to be averaged in z

        con_eid_bc = defaultdict(list)
        con_fid_bc = defaultdict(list)
        con_pid_bc = defaultdict(list)
        con_etype_bc = defaultdict(list)
        for key in self.mesh.keys():
            if key == f'con_{rankn}':
                #print(rankn,key)
                con = self.mesh[key]
                con_etype = con[['f0']].astype('U4')
                con_eid = con[['f1']].astype('i4')
                con_fid = con[['f2']].astype('i1')
                con_pid = con[['f3']].astype('i2')
            elif key[:-2] == f'con_{rankn}':
                con = self.mesh[key]
                con_etype_bc[key] = con[['f0']].astype('U4')
                con_eid_bc[key] = con[['f1']].astype('i4')
                con_fid_bc[key] = con[['f2']].astype('i1')
                con_pid_bc[key] = con[['f3']].astype('i2')

        # find the element taged with periodic boundary condition
        pindex = np.array(np.where(con_pid == -1))
        print(rankn, pindex.shape)

        # searching from p boundary mesh to another p boundary mesh
        self.extmesh = defaultdict(list)
        m = 0
        nz = 20
        for il, ir in pindex.T:
            if con_etype[il,ir] in self.suffix_etype:
                extmesh_temp = list()

                #print(rankn,con_etype[il,ir],con_eid[il,ir])

                #extmesh[f'{con_etype[il,ir]}_{con_eid[il,ir]}_{rankn}'].append(con_eid[il,ir])
                extmesh_temp.append(con_eid[il,ir])
                jl,jr,_f = self.find_neighbour(con_etype[il,ir],con_fid[il,ir],con_eid[il,ir],con_etype,con_eid,con_fid,con_pid)

                #print(rankn,con_etype[jl,jr],con_eid[jl,jr])
                while(True):
                    flag = False

                    if jl == -1 :
                        #print(rankn)
                        break
                    elif _f == None:
                        extmesh_temp.append(jr)
                        flag = True
                        break
                    else:
                        #extmesh[f'{con_etype[il,ir]}_{con_eid[il,ir]}_{rankn}'].append(con_eid[jl,jr])
                        extmesh_temp.append(con_eid[jl,jr])
                        jl,jr,_f = self.find_neighbour(con_etype[jl,jr],con_fid[jl,jr],con_eid[jl,jr],con_etype,con_eid,con_fid,con_pid)

                    #if rankn == 'p2':
                    #    print(rankn,con_etype[jl,jr],con_eid[jl,jr])
                    #    print(jl)

                #print(rankn,extmesh[f'{con_etype[il,ir]}_{con_eid[il,ir]}_{rankn}'])
                #print(m)
                #print(rankn,np.array(extmesh_temp))
                if flag:
                    self.extmesh[f'{con_etype[il,ir]}_{rankn}_{con_eid[il,ir]}_{jl}'].append(extmesh_temp)
                else:
                    self.extmesh[f'{con_etype[il,ir]}_{rankn}'].append(extmesh_temp)
                #print(extmesh[f'{con_etype[il,ir]}_{rankn}'])
            if m == 80:
                break
                #break
            #print(len(extmesh[f'{con_etype[il,ir]}_{rankn}']))
            #if m == 2:
            #    raise ValueError('stop1')

            m += 1
            if rankn == 'p0':
                print(m,m/pindex.shape[1])

        #print(extmesh.keys())
        comm.Barrier()

        self.extmeshMan(rankn,comm,con_eid_bc,con_fid_bc,con_pid_bc,con_etype_bc)

        #self.write(comm)




    def find_neighbour(self,etype,fid,eid,con_etype,con_eid,con_fid,con_pid):
        #print(self.face[etype])

        index = np.array(np.where(con_eid == eid))

        for il, ir in index.T:
            if con_fid[il,ir] == self.face[etype][fid] and con_etype[il,ir] == etype:
                #print(con_pid[il,ir])
                if con_pid[il,ir] == 1:
                    return -1,-1,True
                else:
                    #print(etype,eid,fid,con_etype[il,ir])
                    return 1-il,ir,True

        # cannot find periodic neighbour in this partition:
        # using connectivity between the partition to jump into another partition
        return fid,eid,None

    def extmeshMan(self,rankn,comm,con_eid_bc,con_fid_bc,con_pid_bc,con_etype_bc):
        for key in self.extmesh:
            if len(key.split('_')) > 2:
                etype = key.split('_')[0]
                rank = key.split('_')[1]
                #eid = key.split('_')[2]
                eid = self.extmesh[key][0][-1]
                fid = key.split('_')[3]

                print(rankn,eid)

                for i in con_eid_bc.keys():
                    npartid = np.array(np.where(con_eid_bc[i] == eid))
                    if npartid.size > 0:
                        npart = i.split('_')[1]
                        print(i,con_eid_bc[i],npartid)
                        break


    def write(self,comm):
        import h5py

        rank = comm.Get_rank()
        size = comm.Get_size()

        # serial flush to disk
        #newmesh = comm.gather(mesh, root=0)
        for i in range(size):
            if i == rank:

                with h5py.File('random3.zhenyang', 'a') as f:
                    for k in self.extmesh.keys():
                        print(k)
                        #print(k,mesh[k])
                        f.create_dataset(f'{k}', data=self.extmesh[k])

                    f.close()

            comm.Barrier()
