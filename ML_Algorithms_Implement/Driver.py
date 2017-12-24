# -*- coding: utf-8 -*-

import cmd
import sys

class Start(cmd.Cmd):
  
    def do_Knn(self, line):
        from algos import Knn
        sys.exit() 

    def do_KMeans(self, line):
        from algos import KMeans
        sys.exit()
        
    def do_NaiveBayes(self, line):
        from algos import NaiveBayes    
        sys.exit()
    def do_ID3(self, line):
        from algos import ID3    
        sys.exit()
    def do_C45(self, line):
        from algos import C45    
        sys.exit()
   

if __name__ == '__main__':
#    Start().cmdloop()
    from algos import C45    
    sys.exit()
    