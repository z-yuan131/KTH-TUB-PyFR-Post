
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

        self.face = {'hex': np.array([5,3,4,1,2,0],dtype='int8'), 'pri': np.array([1,0],dtype='int8')}
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

                self.extmeshM0(con_etype[il,ir],con_fid[il,ir],con_eid[il,ir],rankn,con_etype,con_eid,con_fid,con_pid)
            #if m == 80:
            #    break


            m += 1
            if rankn == 'p0':
                print(m,m/pindex.shape[1])

        #print(self.extmesh.keys())

        comm.Barrier()

        for iter in range(100):
            flag = self.extmeshM1(rankn,comm,con_eid_bc,con_fid_bc,con_pid_bc,con_etype_bc,con_etype,con_eid,con_fid,con_pid,iter)
            if flag == None:
                break
            #break
        self.write(comm)




    def find_neighbour(self,etype,fid,eid,con_etype,con_eid,con_fid,con_pid,test=False):
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

    def extmeshM0(self,etype,fid,eid,rankn,con_etype,con_eid,con_fid,con_pid,origin=False,test=False):
        extmesh_temp = list()

        extmesh_temp.append(eid)
        jl,jr,_f = self.find_neighbour(etype,fid,eid,con_etype,con_eid,con_fid,con_pid)

        while(True):
            flag = False
            if jl == -1 :
                break
            elif _f == None:
                #extmesh_temp.append(jr)
                flag = True
                break
            else:
                extmesh_temp.append(con_eid[jl,jr])
                jl,jr,_f = self.find_neighbour(con_etype[jl,jr],con_fid[jl,jr],con_eid[jl,jr],con_etype,con_eid,con_fid,con_pid,test)

        if test and etype == 'pri':
            print(rankn,flag,extmesh_temp,origin)
        if flag and not origin:  # not finished but from the first interation
            self.extmesh[f'{etype}_{eid}_{rankn}_{jl}'].append(extmesh_temp)
        elif flag and origin:    # not finished from other iteration
            #print(origin)
            self.extmesh[f'{origin}_{rankn}_{jl}'].append(extmesh_temp)
        elif origin and not flag:  #finished with more than one rank in process
            self.extmesh[f'{origin}_{rankn}_{jl}'].append(extmesh_temp)
        else: #finished
            self.extmesh[f'{etype}_{rankn}'].append(extmesh_temp)



    def extmeshM1(self,rankn,comm,con_eid_bc,con_fid_bc,con_pid_bc,con_etype_bc,con_etype,con_eid,con_fid,con_pid,iter):
        sendbuff = defaultdict(list)
        revbuff = defaultdict(list)

        for key in self.extmesh:
            flag = None
            if len(key.split('_')) > 2*(iter+1) and int(key.split('_')[-1]) != -1:
                etype = key.split('_')[0]
                #rank = key.split('_')[2]
                #eid = key.split('_')[2]
                eid = self.extmesh[key][0][-1]
                fid = int(key.split('_')[-1])

                #print(rankn,etype,eid,type(fid))

                for i in con_eid_bc.keys():
                    npartid = np.where(con_eid_bc[i] == eid)[0]

                    if npartid.size > 0:
                        for j in npartid:

                            # test
                            if int(key.split('_')[1]) == 591:
                                print(fid,self.face[etype][fid],con_fid_bc[i][j],eid,con_eid_bc[i][j],i,self.extmesh[key][0][-1])


                            if con_fid_bc[i][j] == self.face[etype][fid] and con_etype_bc[i][j] == etype:

                                npart = i.split('_')[1]
                                #print(i,j,con_eid_bc[i][j],npart)
                                sendbuff[f'{npart[:2]}_{npart[2:]}'].append(list((j,key)))
                                flag = 1
                                break
                        if flag:
                            break



        for i in range(comm.Get_size()):
            revbuff.update(comm.bcast(sendbuff, root=i))
            comm.Barrier()
        #if rankn == 'p3':
        #print(rankn,revbuff.keys())
        if revbuff:
            for key in revbuff.keys():
                if rankn == key.split('_')[-1]:
                    #print(revbuff[key])
                    for eleid, origin in revbuff[key]:

                        pl = key.split('_')[0]
                        pr = key.split('_')[-1]
                        #print(eleid,f'{pr}{pl}',con_eid_bc.keys())

                        eid = con_eid_bc[f'con_{pr}{pl}'][eleid]
                        fid = con_fid_bc[f'con_{pr}{pl}'][eleid]
                        pid = con_pid_bc[f'con_{pr}{pl}'][eleid]
                        etype = con_etype_bc[f'con_{pr}{pl}'][eleid]



                        #print(eid,fid,pid)
                        #print(origin)
                        if etype != origin.split('_')[0]:
                            print('It is not extruded mesh')
                            raise ValueError('stop1')
                        self.extmeshM0(etype,fid,eid,rankn,con_etype,con_eid,con_fid,con_pid,origin,True)


                    #print(self.extmesh.keys())


            #print(2*(iter+1))
            comm.Barrier()
            return True
        else:
            comm.Barrier()
            return None







    def write(self,comm):
        import h5py

        rank = comm.Get_rank()
        size = comm.Get_size()

        # serial flush to disk
        #newmesh = comm.gather(mesh, root=0)
        for i in range(size):
            if i == rank:

                with h5py.File('te.zhenyang', 'a') as f:
                    for k in self.extmesh.keys():
                        print(k)
                        #print(k,mesh[k])
                        f.create_dataset(f'{k}', data=self.extmesh[k])

                    f.close()

            comm.Barrier()
