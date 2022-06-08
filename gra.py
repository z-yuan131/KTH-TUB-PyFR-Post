from pyfr.shapes import BaseShape
from pyfr.util import memoize, subclass_where

import numpy as np

class BaseGrad(object):
    def __init__(self, cfg, Mymesh, Mysoln):
        from pyfr.solvers.base import BaseSystem

        #self.outf = args.outf

        # Load the mesh and solution files
        self.soln = Mysoln
        self.mesh = Mymesh
        self.cfg = cfg
        self.dtype = np.float64

        # Check solution and mesh are compatible
        if self.mesh['mesh_uuid'] != self.soln['mesh_uuid']:
            raise RuntimeError('Solution "%s" was not computed on mesh "%s"' %
                               (args.solnf, args.meshf))

        # Load the configuration and stats files
        #self.cfg = Inifile(self.soln['config'])
        #self.stats = Inifile(self.soln['stats'])

        # Data file prefix (defaults to soln for backwards compatibility)
        #self.dataprefix = self.stats.get('data', 'prefix', 'soln')
        self.dataprefix = 'soln'

        # Get element types and array shapes
        self.mesh_inf = self.mesh.array_info('spt')
        self.soln_inf = self.soln.array_info(self.dataprefix)

        # Dimensions
        self.ndims = next(iter(self.mesh_inf.values()))[1][2]
        self.nvars = next(iter(self.soln_inf.values()))[1][1]

        # System and elements classes
        self.systemscls = subclass_where(
            BaseSystem, name = self.cfg.get('solver', 'system')
        )
        self.elementscls = self.systemscls.elementscls

    def print_out(self):
        for key in self.mesh_inf:
            name = key.split('_')[1]
            slnkey = f'soln_{name}_{key.split("_")[2]}'
            print(self._pre_proc_fields_grad(name, Mymesh[key], Mysoln[slnkey]).shape, Mysoln[slnkey].shape)

    def _pre_proc_fields_grad(self, name, mesh, soln):
        # Call the reference pre-processor
        #soln = self._pre_proc_fields_ref(name, mesh, soln)
        soln = soln.swapaxes(0,1)

        # Dimensions
        nvars, nupts = soln.shape[:2]

        # Get the shape class
        basiscls = subclass_where(BaseShape, name=name)

        # Construct an instance of the relevant elements class
        eles = self.elementscls(basiscls, mesh, self.cfg)

        # Get the smats and |J|^-1 to untransform the gradient
        smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)
        rcpdjac = eles.rcpdjac_at_np('upts')

        # Gradient operator
        gradop = eles.basis.m4.astype(self.dtype)

        # Evaluate the transformed gradient of the solution
        gradsoln = gradop @ soln.swapaxes(0, 1).reshape(nupts, -1)
        gradsoln = gradsoln.reshape(self.ndims, nupts, nvars, -1)

        # Untransform
        gradsoln = np.einsum('ijkl,jkml->mikl', smat*rcpdjac, gradsoln,
                             dtype=self.dtype, casting='same_kind')
        gradsoln = gradsoln.reshape(nvars*self.ndims, nupts, -1)

        return np.vstack([soln, gradsoln]).swapaxes(0,1)


class BaseGrad2(object):
    def __init__(self, cfg):
        from pyfr.solvers.base import BaseSystem

        self.cfg = cfg
        # Dimensions
        self.ndims = 3
        self.nvars = 5
        self.dtype = np.float64

        # System and elements classes
        self.systemscls = subclass_where(
            BaseSystem, name = self.cfg.get('solver', 'system')
        )
        self.elementscls = self.systemscls.elementscls


    def _pre_proc_fields_grad(self, name, mesh, soln):
        # Call the reference pre-processor
        #soln = self._pre_proc_fields_ref(name, mesh, soln)
        soln = soln.swapaxes(0,1)

        # Dimensions
        nvars, nupts = soln.shape[:2]

        # Get the shape class
        basiscls = subclass_where(BaseShape, name=name)

        # Construct an instance of the relevant elements class
        eles = self.elementscls(basiscls, mesh, self.cfg)

        # Get the smats and |J|^-1 to untransform the gradient
        smat = eles.smat_at_np('upts').transpose(2, 0, 1, 3)
        rcpdjac = eles.rcpdjac_at_np('upts')

        # Gradient operator
        gradop = eles.basis.m4.astype(self.dtype)

        # Evaluate the transformed gradient of the solution
        gradsoln = gradop @ soln.swapaxes(0, 1).reshape(nupts, -1)
        gradsoln = gradsoln.reshape(self.ndims, nupts, nvars, -1)

        # Untransform
        gradsoln = np.einsum('ijkl,jkml->mikl', smat*rcpdjac, gradsoln,
                             dtype=self.dtype, casting='same_kind')
        gradsoln = gradsoln.reshape(nvars*self.ndims, nupts, -1)

        return np.vstack([soln, gradsoln]).swapaxes(0,1)

#BaseGrad().print_out()
