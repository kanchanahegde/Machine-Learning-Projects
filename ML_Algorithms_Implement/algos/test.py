# -*- coding: utf-8 -*-
def test1(a,b,c):
    if c == 0:
        d=a
        a=Leaf(4,90)
        print(d)
        print()
    else:
     r=test1(a,b.b,c+1)
     if r:
         b.b=200
         print(a)   
         print("d",b)
    

    
class Leaf:
   def __init__(self,a,b):
        self.a=a
        self.b=b
   def __repr__(self):
        return "<Inode:%s %s>" % (self.a,self.b)
c=Leaf(4,{4:'s'})    
b=Leaf(4,c)
a=Leaf(20,b)

test1(a,a,0)
print(a)

