import numpy as np
import numpy.typing as npt

class Component:
    name: str
    mat_index: int
    _params: list

    class Terminal:
        component:"Component"
        sign: int
        def __init__(self, component:"Component", sign:int):
            assert sign==1 or sign==-1
            self.sign=sign
            self.component=component

    def __init__(self, name:str|None=None, **params):
        if name is not None:
            self.name = name
        else:
            self.name = f"{self.__class__.__name__}{id(self)}"
        self.positive_terminal = self.Terminal(self, +1)
        self.negative_terminal = self.Terminal(self, -1)
    
    def _parse_param(self, params:dict[str,float], param_names:list[str]):
        for name in param_names:
            param = params.get(name)
            if param is None:
                continue
            if type(param) is not float:
                raise TypeError(f"Value of '{name}' is not a float")
            setattr(self, name, param)
        self._params = param_names
    
    def __repr__(self):
        l = [f"{name}={getattr(self,name)}" for name in self._params]
        return f"{self.__class__.__name__}({', '.join(l)})"
    
    def __neg__(self):
        return self.negative_terminal
    
    def __pos__(self):
        return self.positive_terminal

class Resistor(Component):
    r: float|None = None
    i: float|None = None
    v: float|None = None
    def __init__(self, name:str|None=None, **params:float):
        super().__init__(name, **params)
        self._parse_param(params, ["r","i","v"])
    
class VoltageSource(Component):
    i: float|None = None
    v: float|None = None
    def __init__(self, name:str|None=None, **params:float):
        super().__init__(name, **params)
        self._parse_param(params, ["i","v"])

class CurrentSource(Component):
    i: float|None = None
    v: float|None = None
    def __init__(self, name:str|None=None, **params:float):
        super().__init__(name, **params)
        self._parse_param(params, ["i","v"])

class LinearCircuitSolver:
    components: list[Component]
    kcl_mat:npt.NDArray[np.int8]
    kvl_mat:npt.NDArray[np.int8]
    net_list: list[list[Component.Terminal]]|None
    _jump_mat:npt.NDArray[np.uint16]
    def __init__(
            self, 
            components:list[Component],
            net_list: list[list[Component.Terminal]]|None = None,
            *,
            kcl_mat:npt.NDArray[np.int8]|None=None,
            ) -> None:
        self.components = components
        for mat_ind, comp in enumerate(self.components):
            comp.mat_index = mat_ind
        if kcl_mat is not None:
            self.net_list = None
            self.kcl_mat = np.array(kcl_mat, dtype=np.int8)
        elif net_list is not None:
            self.net_list = net_list
            self._generate_kcl_mat()
        self._generate_kvl_mat()
    
    def _generate_kcl_mat(self) -> None:
        assert self.net_list is not None
        kcl_mat = np.zeros((len(self.net_list),len(self.components)), dtype=np.int8)
        for line in range(kcl_mat.shape[0]):
            for terminal in self.net_list[line]:
                kcl_mat[line, terminal.component.mat_index] = terminal.sign
        self.kcl_mat =  kcl_mat
    
    def _generate_jump_mat(self) -> None:
        nlines, ncolumns = self.kcl_mat.shape
        NO_CONNECTION = 0xffff
        jump_mat = np.full((nlines, ncolumns), NO_CONNECTION, dtype=np.uint16)
        for y in range(nlines):
            for x in range(ncolumns):
                if self.kcl_mat[y,x] and jump_mat[y,x]==NO_CONNECTION:
                    for j in range(nlines):
                        if j!=y and self.kcl_mat[j,x]:
                            jump_mat[j,x]=y
                            jump_mat[y,x]=j
                            break
        self._jump_mat = jump_mat

    def _generate_kvl_mat(self) -> None:
        self._generate_jump_mat()
        ncolumns = self.kcl_mat.shape[1]

        class StackElement:
            line:int
            candidates:list[int]
            column:int
            def __init__(self,line:int,candidates:list[int]):
                self.line = line
                self.candidates = candidates
        
        stack: list[StackElement] = []
        kvl_mat: list[npt.NDArray[np.int8]] = []
    
        def load_next_candidate(ele: StackElement):
            try:
                ele.column = ele.candidates.pop()
                return True
            except IndexError:
                return False    

        def get_candidates(line:int, excluded_column:int|None=None)->list[int]:
            candidates = []
            for x in range(ncolumns):
                if self.kcl_mat[line, x]:
                    if excluded_column is not None and x==excluded_column:
                        continue
                    candidates.append(x)
            return candidates

        def add_loop(loop_origin: list[StackElement]):
            kvl_mat_line = np.zeros((ncolumns,), dtype=np.int8)
            for ele in loop_origin:
                kvl_mat_line[ele.column] = self.kcl_mat[ele.line, ele.column]
            for line in kvl_mat: # if the loop is in the mat, don't add it to mat
                if np.array_equal(line, kvl_mat_line) or np.array_equal(-line, kvl_mat_line):
                    return
            # print("→".join(f"{e.column}" for e in loop_origin))
            kvl_mat.append(kvl_mat_line)
        
        # search
        stack.append(StackElement(0, get_candidates(0)))
        while stack:
            tos = stack[-1] # tos: top of stack
            if not load_next_candidate(tos): # load the next candidate
                stack.pop()
                continue
            connecting_line = self._jump_mat[tos.line, tos.column]
            if len(stack)>=3:
                continue_ = False
                for ele_index,ele in enumerate(stack[:-2]):
                    if ele.line==connecting_line: # found a loop
                        add_loop(stack[ele_index:])
                        continue_ = True
                        break
                if continue_:
                    continue
            connecting_candidates = get_candidates(connecting_line, excluded_column=tos.column)
            stack.append(StackElement(connecting_line, connecting_candidates))
        
        # store the value
        self.kvl_mat = np.array(kvl_mat, dtype=np.int8)

    def solve(self):
        raise NotImplementedError()



def is_parallel(a,b):
    return not np.any(np.cross(a,b))

