


import sys

from localCon import localConMain
from avg import Average
from spod import spod_avg_z, spod_time

#from test import test_mpi






def main():
    # main solver
    #localConMain().load_connectivity()

    # avgerage in space (span) and in time
    #Average().load()

    # do spod with averaged data in span
    spod_avg_z().load()
    #spod_time().load(f'./series/')







if __name__ == "__main__":
    main()
