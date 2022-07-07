# -*- coding: utf-8 -*-
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from pyfr.shapes import BaseShape
from pyfr.util import lazyprop, subclass_where

from base import BaseAvg


class localConMain(BaseAvg):

    def __init__(self, argv):
        super().__init__(argv)

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

        # Check if rank infomation meets requirements
        for etype in self.mesh_part:
            if etype in self.suffix_etype:
                if size != len(self.mesh_part[etype]) and rank == 0:
                    raise RuntimeError(f'please use exactly {len(self.mesh_part[etype])} ranks.')
                break

        # Each rank do its own job for collecting elements to be averaged in z

        con = defaultdict(list)
        con_bc = defaultdict(list)

        for key in self.mesh.keys():
            if key == f'con_p{rank}':
                con['etype'] = self.mesh[key][['f0']].astype('U4')
                con['eid'] = self.mesh[key][['f1']].astype('i4')
                con['fid'] = self.mesh[key][['f2']].astype('i1')
                con['pid'] = self.mesh[key][['f3']].astype('i2')
            elif key[:-2] == f'con_p{rank}':
                con_bc[key] = {'etype': self.mesh[key][['f0']].astype('U4'),
                                          'eid': self.mesh[key][['f1']].astype('i4'),
                                          'fid': self.mesh[key][['f2']].astype('i1'),
                                          'pid': self.mesh[key][['f3']].astype('i2') }

        # find the element taged with periodic boundary condition
        pindex = np.array(np.where(con['pid'] == -1))
        print(rank, pindex.shape)


        # searching from p boundary mesh to another p boundary mesh
        self.extmesh = defaultdict(list)
        self.extmesh_send = defaultdict(list)

        # Step 1: get one periodic boundary and relevant neighbours
        m = 0
        for il, ir in pindex.T:
            if con['etype'][il,ir] in self.suffix_etype:
                #print(con['etype'][il,ir], con['fid'][il,ir], con['eid'][il,ir])

                self.extmeshM0(con['etype'][il,ir], con['fid'][il,ir], con['eid'][il,ir], rank, con)

            m += 1
            if rank == 0:
                print(m,m/pindex.shape[1])

        #print(rank,self.extmesh)

        comm.Barrier()

        """
        # for tests uses
        print(list(self.extmesh.keys()), list(self.extmesh_send.keys()))
        self.write(comm, self.extmesh)
        comm.Barrier()
        self.write(comm, self.extmesh_send)


        self.read(comm,rank)
        """

        # Step 2: get its neighbours in other ranks
        while(True):

            flag = self.extmeshM1(rank, comm, con_bc, con)
            #if rank == 0:
            #    print(rank,self.extmesh['hex_359_p0_p0'])
            #print(flag)
            if not flag:
                break

        self.write(comm, self.extmesh)




    def extmeshM0(self, etype, fid, eid, rank, con, multi = False):
        extmesh_temp = list()

        extmesh_temp.append(eid)
        jl,jr = self.find_neighbour(etype, fid, eid, con)

        if multi:
            etype, eid, original_rank, current_rank, fid = multi.split('_')


        while(True):
            if jl == -1 and jr == -1:
                if multi:
                    # For the case that it reaches another periodic boundary in another partition
                    #self.extmesh[f'{multi}'][0].update({f'p{rank}_-1': extmesh_temp})
                    """ f'{etype}_{eid}_{original_rank}_p{rank}'"""
                    self.extmesh[f'{etype}_{eid}_{original_rank}_p{rank}_{-1}'].append(extmesh_temp)
                else:
                    # For the case that it reaches another periodic boundary in current partition
                    self.extmesh[f'{etype}_p{rank}'].append(extmesh_temp)
                break
            elif jl != -1 and jr == -1:
                # For the case that this partition only has a part of this extrution
                if multi:
                    #self.extmesh[f'{multi}'][0].update({f'p{rank}_{fid}': extmesh_temp})
                    self.extmesh_send[f'{etype}_{eid}_{original_rank}_p{rank}_{fid}'].append(extmesh_temp)
                else:
                    #self.extmesh[f'{etype}_{eid}_p{rank}'].append({f'p{rank}_{fid}': extmesh_temp})
                    self.extmesh_send[f'{etype}_{eid}_p{rank}_p{rank}_{fid}'].append(extmesh_temp)
                break
            else:
                # For the case that accumulating other elements
                extmesh_temp.append(con['eid'][jl,jr])
                jl,jr = self.find_neighbour(con['etype'][jl,jr],con['fid'][jl,jr],con['eid'][jl,jr],con)




    def find_neighbour(self,etype,fid,eid,con):
        index = np.array(np.where(con['eid'] == eid))

        for il, ir in index.T:

            if con['fid'][il,ir] == self.face[etype][fid] and con['etype'][il,ir] == etype:
                # Check if we have reached another periodic boundary
                if con['pid'][il,ir] == 1:
                    return -1,-1
                else:
                    return 1-il,ir

        # cannot find periodic neighbour in this partition:
        # using connectivity between the partition to jump into another partition
        return fid,-1

    def extmeshM1(self, rank, comm, con_bc, con):
        sendbuff = defaultdict(list)
        revbuff = defaultdict(list)

        for key in self.extmesh_send:
            flag = None
            etype = key.split('_')[0]
            eid = self.extmesh_send[key][0][-1]
            fid = int(key.split('_')[-1])

            #print(rank,etype,eid,type(fid))

            for i in con_bc.keys():
                npartid = np.where(con_bc[i]['eid'] == eid)[0]

                if npartid.size > 0:
                    for j in npartid:

                        if con_bc[i]['fid'][j] == self.face[etype][fid] and con_bc[i]['etype'][j] == etype:

                            npart = i.split('_')[1]
                            #print(i,j,con_bc[i]['eid'][j],npart)
                            sendbuff[f'{npart[:2]}_{npart[2:]}'].append(list((j,key)))

                            etype, eid, porigin, pcurrent, fid = key.split('_')
                            """ f'{etype}_{eid}_{porigin}_{pcurrent}' """
                            self.extmesh[f'{etype}_{eid}_{porigin}_{pcurrent}_{fid}'].append(self.extmesh_send[key][0])
                            flag = 1
                            break
                    if flag:
                        break

        self.extmesh_send = defaultdict(list)


        for i in range(comm.Get_size()):
            revbuff.update(comm.bcast(sendbuff, root=i))
            comm.Barrier()

        if revbuff:
            for key in revbuff.keys():
                if f'p{rank}' == key.split('_')[-1]:
                    #print(revbuff[key])
                    for eleid, multi in revbuff[key]:

                        pl = key.split('_')[0]
                        pr = key.split('_')[-1]
                        #print(eleid,f'{pr}{pl}',con_eid_bc.keys())

                        eid = con_bc[f'con_{pr}{pl}']['eid'][eleid]
                        fid = con_bc[f'con_{pr}{pl}']['fid'][eleid]
                        pid = con_bc[f'con_{pr}{pl}']['pid'][eleid]
                        etype = con_bc[f'con_{pr}{pl}']['etype'][eleid]



                        #print(eid,fid,pid)
                        #print(origin)
                        if etype != multi.split('_')[0]:
                            print('It is not extruded mesh')
                            raise ValueError('stop1')


                        #print(rank,multi)
                        self.extmeshM0(etype,fid,eid,rank,con,multi)


                    #print(self.extmesh.keys())


            #print(2*(iter+1))
            comm.Barrier()
            return True
        else:
            comm.Barrier()
            return None





    def write(self, comm, datafile):
        import h5py

        rank = comm.Get_rank()
        size = comm.Get_size()

        # serial flush to disk
        for i in range(size):
            if i == rank:

                with h5py.File('midfile.zhenyang', 'a') as f:
                    for k in datafile.keys():
                        print(k)
                        f.create_dataset(f'{k}', data=datafile[k])
                    f.close()
            comm.Barrier()


    def read(self, comm, rank):
        import h5py
        f = h5py.File('midfile.zhenyang')
        for key in f.keys():
            if len(key.split('_')) == 2 and f'p{rank}' == key.split('_')[-1]:
                self.extmesh[key] = np.array(f[key])
            elif len(key.split('_')) > 2 and f'p{rank}' == key.split('_')[-2]:
                self.extmesh_send[key] = np.array(f[key])
