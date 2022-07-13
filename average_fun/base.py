# -*- coding: utf-8 -*-
import numpy as np

from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader
from pyfr.util import subclass_where
from pyfr.shapes import BaseShape

class BaseAvg(object):
    def __init__(self, args):

        # load mesh and solution files
        self.mesh = NativeReader(args[0])
        soln = NativeReader(args[1])

        # Check solution and mesh are compatible
        if self.mesh['mesh_uuid'] != soln['mesh_uuid']:
            raise RuntimeError('Solution "%s" was not computed on mesh "%s"' %
                               (args.solnf, args.meshf))

        # Load the configuration and stats files
        self.cfg = Inifile(soln['config'])
        self.stats = Inifile(soln['stats'])
        self.dtype = np.dtype(self.cfg.get('backend','precision')).type

        # Data file prefix (defaults to soln for backwards compatibility)
        self.dataprefix = self.stats.get('data', 'prefix', 'tavg')
        #print(self.cfg)

        # Get element types and array shapes
        self.mesh_inf = self.mesh.array_info('spt')
        #self.soln_inf = self.soln.array_info(self.dataprefix)

        # Get the number of elements of each type in each partition
        self.mesh_part = self.mesh.partition_info('spt')

        # Dimensions
        self.ndims = next(iter(self.mesh_inf.values()))[1][2]
        #self.nvars = next(iter(self.soln_inf.values()))[1][1]

        # Decide which types of element to average in span
        self.suffix_etype = ['hex','pri']


        start  = 136.80  #60
        end    = 136.90   #90
        dt     = 0.1     #0.1

        tt = np.arange(start, end, dt)
        self.time = list()
        for i in range(len(tt)):
            self.time.append("{:.2f}".format(tt[i]))


    def soln_loader(self, time):
        name = f'../../../Re5e4/gradient/datafiles_{time}.pyfrs'
        #name = f'../../../Re5e4/pyfrs/naca0012_{time}.pyfrs'
        return NativeReader(name)



    def get_order(self, name, nspts):
        return self.get_shape(name, nspts, self.cfg).order_from_nspts(nspts)

    def get_shape(self, name, nspts, cfg):
        shapecls = subclass_where(BaseShape, name=name)
        return shapecls(nspts, cfg)

    #@memoize
    def get_std_ele(self, name, nspts):
        order = self.get_order(name, nspts) - 1
        #print(order)
        return self.get_shape(name, nspts, self.cfg).std_ele(order)

    #@memoize
    def get_soln_op(self, name, nspts, svpts):
        shape = self.get_shape(name, nspts, self.cfg)
        return shape.ubasis.nodal_basis_at(svpts).astype(self.dtype)
