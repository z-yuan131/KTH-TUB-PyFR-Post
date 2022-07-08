from pyfr.shapes import BaseShape
from pyfr.util import lazyprop, subclass_where


from collections import defaultdict
import numpy as np
from tqdm import tqdm
import h5py

from base import BaseAvg

class avgcls(BaseAvg):

    def __init__(self,argv):
        super().__init__(argv)
        self.order = 4+1
        self.argv = argv


    def load(self):
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()


        f = h5py.File('./midfile.zhenyang','r')
        # do some reordering to the data
        self.data = defaultdict()
        self.mpiMap = list()
        for key in f.keys():
            if len(key.split('_')) > 2:
                etype, eid, origin_rank, current_rank, _fid = key.split('_')
                # construct mpi map
                if origin_rank != current_rank and origin_rank == f'p{rank}' and int(current_rank[-1]) not in self.mpiMap:
                    self.mpiMap.append(int(current_rank[-1]))

                # collect data which belongs to each rank
                if current_rank == f'p{rank}':
                    if f'{etype}_{eid}_{origin_rank}' in self.data.keys():
                        self.data[f'{etype}_{eid}_{origin_rank}'] = np.concatenate((self.data[f'{etype}_{eid}_{origin_rank}'],np.array(f[key])[0]),axis=None)
                    else:
                        self.data[f'{etype}_{eid}_{origin_rank}'] = np.array(f[key])[0]
            elif key.split('_')[-1] == f'p{rank}':
                self.data[key] = np.array(f[key])

        f.close()
        print(rank,self.mpiMap)


        self.average_main(rank, comm)
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



    def average_main(self,rank, comm):
        self.avgfield = defaultdict()
        self.avgmesh = defaultdict()
        #self.length_spa = defaultdict(list)


        # for acoustics, it is 2D so just get one 2D slice is ok
        eindex, ac_mesh, cut_index = self.get_right_keys(self.mesh)
        acoustic_soln = defaultdict()
        acoustic_mesh = defaultdict()



        for time in self.time:

            # first average in space while loading each snapshot:
            # Construct field we need to average in span
            Mysoln = self.soln_loader(time)
            soln = self.construct_soln_field(Mysoln, rank)
            #print(list(soln.keys()))

            # Average in space each rank:
            self.average_loc(soln, rank, time)


            self.exchange_info(comm, time)


            # Acousitcs, just to take a 2d slice
            for key in Mysoln.keys():
                if key.split('_')[0] == 'tavg' and key.split('_')[1] == 'tet' and key.split('_')[-1] == f'p{rank}':
                    keysoln = f'tet_{key.split("_")[-1]}'
                    acoustic_soln[keysoln] = Mysoln[key][...,eindex[keysoln]]
                    acoustic_soln[keysoln] = acoustic_soln[keysoln][cut_index[keysoln][0],:,cut_index[keysoln][1]]
                    if self.dataprefix == 'soln':
                        acoustic_soln[keysoln] = np.array(self.con_to_pri(acoustic_soln[keysoln])).swapaxes(0,1)
                    if time == self.time[0]:
                        keymesh = f'spt_tet_{key.split("_")[-1]}'
                        acoustic_mesh[keysoln] = ac_mesh[keysoln][cut_index[keysoln]]



            for i in range(comm.Get_size()):
                if i == rank:
                    dir = f'../series3/time_series_{time}.zhenyang'
                    self.write_to_file(self.avgfield_writeout, dir)
                    self.write_to_file(acoustic_soln, dir)
                    if time == self.time[0]:
                        dir = f'../series3/time_series_{time}_mesh.zhenyang'
                        self.write_to_file(self.avgmesh_writeout, dir)
                        self.write_to_file(acoustic_mesh, dir)

                comm.Barrier()



            if rank == 3:


                print(time+' in '+self.time[-1])
                if time == self.time[-1]:
                    print('waiting other ranks to finish')



    def average_loc(self, soln, rank, time):
        for key in self.data.keys():

            etype = key.split('_')[0]
            soln_name = f'{etype}_p{rank}'
            mesh_name = f'spt_{etype}_p{rank}'



            if len(key.split('_')) > 2:
                #print(rank, key)
                sln = soln[soln_name][...,self.data[key]]
                sln = sln.reshape(self.order, int(sln.shape[0]/self.order), sln.shape[1], sln.shape[2])


                if time == self.time[0]:
                    msh = self.mesh[mesh_name][:,self.data[key]]
                    msh = msh.reshape(self.order, int(msh.shape[0]/self.order), msh.shape[1], msh.shape[2])

                    self.avgmesh[key] = np.einsum('ijkl -> jl',msh)
                    #self.length_spa[key].append(len(self.data[key]))

                self.avgfield[key] = np.einsum('ijkl -> jk', sln)


            else:
                shape1 = self.data[key].reshape(-1)
                sln = soln[soln_name][...,shape1]
                sln = sln.reshape(self.order, int(sln.shape[0]/self.order), sln.shape[1], self.data[key].shape[0], self.data[key].shape[1])


                if time == self.time[0]:
                    msh = self.mesh[mesh_name][:,shape1]
                    msh = msh.reshape(self.order, int(msh.shape[0]/self.order), self.data[key].shape[0],self.data[key].shape[1],msh.shape[-1])

                    self.avgmesh[key] = np.einsum('ijklm -> jkm', msh)
                    #self.length_spa[key].append(min(self.data[key].shape))
                    length = msh.shape[3]

                self.avgfield[key] = np.einsum('ijklm -> jkl', sln)




        if time == self.time[0]:
            self.length = length
        print(key,self.length)



    def construct_soln_field(self, Mysoln, rank):
        soln = defaultdict()
        if self.dataprefix == 'snapshot':
            # For time averaged data output with instant mode
            for key in Mysoln:
                if len(key.split('_')) > 2:
                    _prefix, etype, prank = key.split('_')
                    if etype in self.suffix_etype and prank == f'p{rank}':
                        # Mysoln has structure like: rho,u,v,w,p,du/dx,du/dy,dv/dx,dv/dy,dw/dx,dw/dy
                        soln[f'{etype}_{prank}'] = Mysoln[key]
                        reynold_stress = np.einsum('ijk,ilk->ijlk',Mysoln[key][:,1:4],Mysoln[key][:,1:4]).reshape(Mysoln[key].shape[0],-1,Mysoln[key].shape[-1])
                        soln[f'{etype}_{prank}'] = np.concatenate((soln[f'{etype}_{prank}'],reynold_stress),axis = 1)

                        #print(soln[f'{etype}_{prank}'].shape)
        else:
            # For the normal write out data with conservative format
            for key in Mysoln:
                if len(key.split('_')) > 2:
                    _prefix, etype, prank = key.split('_')
                    if etype in self.suffix_etype and prank == f'p{rank}':
                        # Mysoln has structure like: rho,rhou,rhov,rhow,E
                        soln[f'{etype}_{prank}'] = Mysoln[key]
                        soln[f'{etype}_{prank}'] = np.array(self.con_to_pri(soln[f'{etype}_{prank}'])).swapaxes(0,1)
                        print('checkout routine:', soln[f'{etype}_{prank}'].shape)
        return soln

    def con_to_pri(self, cons):
        cons = cons.swapaxes(0,1)
        rho, E = cons[0], cons[-1]

        # Divide momentum components by rho
        vs = [rhov/rho for rhov in cons[1:-1]]

        # Compute the pressure
        gamma = self.cfg.getfloat('constants', 'gamma')
        p = (gamma - 1)*(E - 0.5*rho*sum(v*v for v in vs))

        return [rho] + vs + [p]







    def exchange_info(self, comm, time):
        self.avgfield_writeout = self.mpi_function(self.avgfield, comm)
        if time == self.time[0]:
            self.avgmesh_writeout = self.mpi_function(self.avgmesh, comm, True)


    def mpi_function(self, avgfield, comm, flag1 = False):
        rank = comm.Get_rank()
        size = comm.Get_size()

        revbuff = defaultdict(list)
        temp = defaultdict(list)
        avgfield_writeout = defaultdict()

        if rank == 0:
            print('communication between ranks')

        # In this case, since it is a dictionary, the best pratice is to boardcast it
        for key in avgfield:
            if len(key.split('_')) > 2 and key.split('_')[-1] != f'p{rank}':
                name = f'{key}_{rank}'
                temp[name] = avgfield[key]
                #self.avgfield.pop(key,None)
            elif len(key.split('_')) == 2:
                if flag1:
                    avgfield_writeout[key] = avgfield[key].swapaxes(1,-1) / self.length / self.order
                else:
                    avgfield_writeout[key] = avgfield[key] / self.length / self.order


        for i in range(size):
            revbuff.update(comm.bcast(temp, root=i))

        #print(revbuff.keys())

        for key in revbuff:
            etype, eid, origin_rank, remote_rank = key.split('_')
            name = f'{etype}_{eid}_{origin_rank}'
            #if name in self.avgfield.keys() and rank == int(origin_rank[-1]):
            if rank == int(origin_rank[-1]):
                avgfield[name] += revbuff[key] #/ self.length / self.order

        # append broken elements to main part
        for key in avgfield:
            if len(key.split('_')) == 3:
                etype, eid, origin_rank= key.split('_')
                if rank == int(origin_rank[-1]):
                    name = f'{etype}_{origin_rank}'
                    #print(avgfield_writeout[name].shape,avgfield[key].shape)
                    avgfield_writeout[name] = np.dstack((avgfield_writeout[name],avgfield[key] / self.length / self.order))


        print(avgfield_writeout.keys())
        return avgfield_writeout




    def write_to_file(self,msh,dir):
        with h5py.File(dir,'a') as f:
            for key in msh.keys():
                if len(msh[key].shape) > 2:
                    msh[key] = msh[key].swapaxes(1,2)
                #f.create_dataset(f'{key}', data=msh[key])
                f.create_dataset(f'{key}', data=msh[key].reshape(-1, msh[key].shape[-1]))

            f.close()
