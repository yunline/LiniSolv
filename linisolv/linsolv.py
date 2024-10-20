import numpy as np
import numpy.typing as npt

__all__ = [
    "Component",
    "Resistor",
    "Source",
    "LinearCircuitSolver"
]

PRINT_COLOR_RED = "\033[31m"
PRINT_COLOR_GREEN = "\033[32m"
PRINT_COLOR_DEFAULT = "\033[39m"

class UnknownVarible:
    value: float|None = None
    component:"Component"
    param_name: str
    reciprocal: bool = False
    _rec:"UnknownVarible"
    mat_index: int

    def __init__(self, component:"Component", param_name:str):
        self.component = component
        self.param_name = param_name
    
    def get_reciprocal(self) -> "UnknownVarible":
        if not hasattr(self,"_rec"):
            self._rec = self.__class__.__new__(self.__class__)
            self._rec.component = self.component
            self._rec.param_name = self.param_name
            self._rec.reciprocal = True
            self._rec._rec = self
        return self._rec

    def __repr__(self):
        if self.value is None:
            return f"{PRINT_COLOR_RED}Unknown{PRINT_COLOR_DEFAULT}"
        else:
            return f"{PRINT_COLOR_GREEN}{float(self.value):.2f}{PRINT_COLOR_DEFAULT}"
    
    def __hash__(self) -> int:
        return id(self)
    
    def get_expr(self) -> str:
        if self.reciprocal:
            return f"1/{self.get_reciprocal()}"
        else:
            return f"{self.param_name.upper()}{self.component.name}"

class Equation:
    unknowns: list[UnknownVarible]
    coefficients: list[float]
    const_term: float

    def __init__(self):
        self.unknowns = []
        self.coefficients = []
        self.const_term = 0

    def add_unknown(self, unknown:UnknownVarible, coeff:float):
        self.unknowns.append(unknown)
        self.coefficients.append(coeff)
    
    def add_const(self, const:float):
        self.const_term+=const

    def __repr__(self):
        terms = []
        for unk,coff in zip(self.unknowns,self.coefficients):
            if coff==1.0:
                coff_str = '+'
            elif coff==-1.0:
                coff_str = '-'
            elif coff<0:
                coff_str = f"{coff:.2f}*"
            else:
                coff_str = f"+{coff:.2f}*"
            terms.append(f'{coff_str}{unk.get_expr()}')
        s = " ".join(terms)
        if s[0]=="+":
            s = s[1:]
        return s+f" = {-self.const_term:.2f}"

class Component:
    name: str
    mat_index: int

    _params: tuple[str, ...] = ("i", "v")
    i: float|UnknownVarible
    v: float|UnknownVarible

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
        self._parse_param(params)
        self.positive_terminal = self.Terminal(self, +1)
        self.negative_terminal = self.Terminal(self, -1)
    
    def _parse_param(self, params:dict[str,float]):
        for name in self._params:
            param = params.get(name, UnknownVarible(self, name))
            
            if not isinstance(param, UnknownVarible):
                param = float(param)
            setattr(self, name, param)
    
    def __repr__(self):
        l = [f"{name}={getattr(self,name)}" for name in self._params]
        return f"{self.__class__.__name__}({', '.join(l)})"
    
    def __neg__(self):
        return self.negative_terminal
    
    def __pos__(self):
        return self.positive_terminal

class Resistor(Component):
    _params = ("i", "v", "r")
    r: float|UnknownVarible
    
class Source(Component):
    pass

class LinearCircuitSolver:
    components: list[Component]
    kcl_mat:npt.NDArray[np.int8]
    kvl_mat:npt.NDArray[np.int8]
    net_list: list[list[Component.Terminal]]|None
    equations: list[Equation]
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
            if len(stack)>=2:
                continue_ = False
                for ele_index,ele in enumerate(stack[:-1]):
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

    def _generate_equations(self) -> None:
        equations:list[Equation] = []
        # get KCL equations
        for line in self.kcl_mat:
            eq = Equation()
            equations.append(eq)
            for comp_ind, comp in enumerate(self.components):
                if line[comp_ind]==0:
                    continue
                if not isinstance(comp.i, UnknownVarible):
                    eq.add_const(comp.i*line[comp_ind])
                    continue
                if isinstance(comp, Source):
                    eq.add_unknown(comp.i, line[comp_ind])
                elif isinstance(comp, Resistor):
                    if (not isinstance(comp.r, UnknownVarible) and
                        not isinstance(comp.v, UnknownVarible)):
                        eq.add_const(line[comp_ind]*comp.v/comp.r)
                    elif (not isinstance(comp.r, UnknownVarible) and
                        isinstance(comp.v, UnknownVarible)):
                        eq.add_unknown(comp.v, line[comp_ind]/comp.r)
                    elif (isinstance(comp.r, UnknownVarible) and
                        not isinstance(comp.v, UnknownVarible)):
                        eq.add_unknown(comp.r.get_reciprocal(), line[comp_ind]*comp.v)
                    else:
                        raise RuntimeError("我不会，长大后再学习")
                        eq.add_unknown(comp.i, line[comp_ind])
        # get KVL equations
        for line in self.kvl_mat:
            eq = Equation()
            equations.append(eq)
            for comp_ind, comp in enumerate(self.components):
                if line[comp_ind]==0:
                    continue
                if not isinstance(comp.v, UnknownVarible):
                    eq.add_const(comp.v*line[comp_ind])
                    continue
                if isinstance(comp, Source):
                    eq.add_unknown(comp.v, line[comp_ind])
                elif isinstance(comp, Resistor):
                    if (not isinstance(comp.r, UnknownVarible) and
                        not isinstance(comp.i, UnknownVarible)):
                        eq.add_const(comp.i*comp.r*line[comp_ind])
                    elif (not isinstance(comp.r, UnknownVarible) and
                        isinstance(comp.i, UnknownVarible)):
                        eq.add_unknown(comp.i, comp.r*line[comp_ind])
                    elif (isinstance(comp.r, UnknownVarible) and
                        not isinstance(comp.i, UnknownVarible)):
                        eq.add_unknown(comp.r, comp.i*line[comp_ind])
                    else:
                        raise RuntimeError("我不会，长大后再学习")
                        eq.add_unknown(comp.v, line[comp_ind])
        # get Ohm's Law qeuations
        for comp in self.components:
            if isinstance(comp, Resistor):
                eq = Equation()
                condition = isinstance(comp.r, UnknownVarible)+\
                    isinstance(comp.v, UnknownVarible)+\
                    isinstance(comp.i, UnknownVarible)
                if condition==0:
                    continue
                elif condition==1:
                    if isinstance(comp.r, UnknownVarible):
                        eq.add_unknown(comp.r, 1) 
                        eq.add_const(-comp.v/comp.i) # type: ignore
                    elif isinstance(comp.v, UnknownVarible):
                        eq.add_unknown(comp.v, 1) 
                        eq.add_const(-comp.r*comp.i) # type: ignore
                    elif isinstance(comp.i, UnknownVarible):
                        eq.add_unknown(comp.i, 1)
                        eq.add_const(-comp.v/comp.r) # type: ignore
                elif condition==2:
                    if not isinstance(comp.r, UnknownVarible):
                        eq.add_unknown(comp.v, -1) # type: ignore
                        eq.add_unknown(comp.i, comp.r) # type: ignore
                    elif not isinstance(comp.i, UnknownVarible):
                        eq.add_unknown(comp.v, -1) # type: ignore
                        eq.add_unknown(comp.r, comp.i) # type: ignore
                    elif not isinstance(comp.v, UnknownVarible):
                        raise RuntimeError("我不会，长大后再学习")
                elif condition==3:
                    raise RuntimeError("我不会，长大后再学习")
                if len(eq.unknowns)!=0:
                    equations.append(eq)

        self.equations = equations
    
    def solve(self) -> None:
        self._generate_equations()
        # get all unknowns
        unknowns_set:set[UnknownVarible] = set()
        for eq in self.equations:
            for unk in eq.unknowns:
                unknowns_set.add(unk)
        unknowns = list(unknowns_set)
        # reset the value of unknowns
        for ind,unk in enumerate(unknowns):
            unk.value=None
            unk.mat_index = ind
        solver_mat = np.zeros((len(self.equations),len(unknowns)+1),dtype=np.float64)
        for line,eq in enumerate(self.equations):
            solver_mat[line,-1] = eq.const_term
            for unk,coeff in zip(eq.unknowns, eq.coefficients):
                solver_mat[line, unk.mat_index] = coeff
        solver_mat = remove_linear_dependence(solver_mat)
        result = np.linalg.solve(solver_mat[:,:-1], solver_mat[:,-1])
        for res,unk in zip(result, unknowns):
            unk.value = res

def remove_linear_dependence(mat:npt.NDArray[np.float64])->npt.NDArray[np.float64]:
    cnt = 0
    selected_lines = []
    for i in range(1,mat.shape[0]+1):
        rnk = np.linalg.matrix_rank(mat[:i])
        if rnk==cnt+1:
            selected_lines.append(i-1)
            cnt+=1
        # rnk<cnt+1: the line shouldn't be selected
        # rnk>cnt+1: this case will never happen
    return mat[selected_lines]
