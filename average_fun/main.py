# -*- coding: utf-8 -*-
import sys
import numpy as np
from argparse import ArgumentParser, FileType

from localCon import localConMain
from avg import avgcls



def main():

    solnname = '../../Re5e4/gradient/datafiles_136.80.pyfrs'
    meshname = '../../Re5e4/gradient/mesh.pyfrm'



    sys.argv = [meshname,solnname]

    # main solver
    #localConMain(sys.argv).load_connectivity()

    # avgerage in space (span) and in time
    avgcls(sys.argv).load()




if __name__ == "__main__":
    main()
