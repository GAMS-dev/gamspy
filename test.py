import copy

class A:
    
    def __init__(self) -> None:
        print("Calling init")
        
    def __eq__(self, other: object):
        pass
        
a = A()
b = copy.deepcopy(a)