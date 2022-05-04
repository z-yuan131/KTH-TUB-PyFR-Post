from pyfr.inifile import Inifile
from pyfr.readers.native import NativeReader


class load_class(object):
    def __init__(self,names):
        self.cfgname = names[0]
        self.meshname = names[1]
        self.solnname = names[2]



    def load_ini(self):
        from argparse import ArgumentParser, FileType

        ap = ArgumentParser(description='Read interpolation argument.')
        ap.add_argument('--data_dir', type=str, default='./channel_Retau180.ini',
                           help='data directory containing input.txt')
        ap.add_argument('cfg', type=FileType('r'), help='config file')

        args = ap.parse_args([self.cfgname])
        confg = Inifile.load(args.cfg)

        return confg

    def load_soln(self):
        from argparse import ArgumentParser, FileType

        ap = ArgumentParser(description='Read interpolation argument.')
        ap.add_argument('--data_dir', type=str, default='./100.pyfrs',
                           help='data directory containing input.txt')
        ap.add_argument('solnf', type=FileType('r'), help='solution file')

        args = ap.parse_args([self.solnname])
        #print(args)
        soln = NativeReader(args.solnf.name)

        return soln

    def load_mesh(self):
        from argparse import ArgumentParser, FileType

        ap = ArgumentParser(description='Read interpolation argument.')
        ap.add_argument('--data_dir', type=str, default='./100.pyfrs',
                           help='data directory containing input.txt')
        ap.add_argument('meshf', type=FileType('r'), help='solution file')

        args = ap.parse_args([self.meshname])
        #print(args)
        mesh = NativeReader(args.meshf.name)

        return mesh

    def return_fun(self):
        return self.confg,self.soln,self.mesh
