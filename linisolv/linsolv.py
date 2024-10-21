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
            return f"1/{self.get_reciprocal().get_expr()}"
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
                coff_str = ' + '
            elif coff==-1.0:
                coff_str = ' - '
            elif coff<0:
                coff_str = f" - {-coff:.2f}*"
            else:
                coff_str = f" + {coff:.2f}*"
            terms.append(f'{coff_str}{unk.get_expr()}')
        s = "".join(terms)
        if s=="":
            s="0.00"
        if s.startswith(" + "):
            s = s[3:]
        elif s.startswith(" - "):
            s = "-"+s[3:]
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
        def __hash__(self) -> int:
            return id(self)

    def __init__(self, name:str|None=None, **params):
        if name is None:
            name = object.__repr__(self)
        self.name = name
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
        return f"{self.__class__.__name__}({self.name}, {', '.join(l)})"
    
    def __neg__(self) -> Terminal:
        return self.negative_terminal
    
    def __pos__(self) -> Terminal:
        return self.positive_terminal

class Source(Component):
    pass

class Resistor(Component):
    _params = ("i", "v", "r")
    r: float|UnknownVarible
    def autocomplete(self) -> None:
        def check_unknown(param_name:str) -> bool:
            param=getattr(self, param_name)
            if not isinstance(param, UnknownVarible):
                return True
            elif param.value is not None:
                return True
            return False
        def get_value(param_name) -> float:
            param=getattr(self, param_name)
            if isinstance(param, UnknownVarible):
                assert param.value is not None
                return param.value
            return param
        condition = sum(check_unknown(name) for name in self._params)
        if condition<2:
            raise RuntimeError("Too much unknowns, resistor autocompletion failed")
        if condition==3:
            return
        if not check_unknown('v'):
            self.v.value = get_value('i')*get_value('r') # type: ignore
        elif not check_unknown('i'):
            self.i.value = get_value('v')/get_value('r') # type: ignore
        else:
            self.r.value = get_value('v')/get_value('i') # type: ignore

class LinearCircuitSolver:
    components: list[Component]
    kcl_mat:npt.NDArray[np.int8]
    kvl_mat:npt.NDArray[np.int8]
    net_list: list[set[Component.Terminal]]
    equations: list[Equation]
    valid_equation_indices: list[int]
    _jump_mat:npt.NDArray[np.uint16]

    def __init__(
            self, 
            components:list[Component],
            net_list: list[set[Component.Terminal]],
            ) -> None:
        self.components = components
        for mat_ind, comp in enumerate(self.components):
            comp.mat_index = mat_ind
        self.net_list = [set(i) for i in net_list]
        self._generate_kcl_mat()
        self._generate_kvl_mat()
    
    def _generate_kcl_mat(self) -> None:
        kcl_mat = np.zeros((len(self.net_list),len(self.components)), dtype=np.int8)
        excs:list[Exception] = []
        shorted_indices:list[int] = []

        # build kcl mat
        for line_ind in range(kcl_mat.shape[0]):
            for terminal in self.net_list[line_ind]:
                if terminal.component not in self.components:
                    raise RuntimeError(
                        f"Component '{terminal.component.name}' "
                        "is not included in the component list"
                    )
                if kcl_mat[line_ind, terminal.component.mat_index]:
                    shorted_indices.append(terminal.component.mat_index)
                    excs.append(RuntimeError(f"Component '{terminal.component.name}' is shorted"))
                kcl_mat[line_ind, terminal.component.mat_index] = terminal.sign
        # verify kcl mat
        abs_kcl_mat = np.abs(kcl_mat)
        for line_ind,line_sum in enumerate(np.sum(np.abs(kcl_mat),axis=1)):
            if line_sum==0:
                excs.append(RuntimeError(f"Net[{line_ind}]: 0 terminals connected (expected >=2)"))
            elif line_sum==1:
                excs.append(RuntimeError(f"Net[{line_ind}]: Only 1 terminal connected (expected >=2)"))
        mat_abs_sum = np.sum(abs_kcl_mat, axis=0)
        mat_sum = np.sum(kcl_mat, axis=0)
        if not (np.all(mat_abs_sum==2) and np.all(mat_sum==0)):
            for col_ind in range(kcl_mat.shape[1]):
                if col_ind in shorted_indices:
                    continue # component is shorted, skip the check
                comp_name = self.components[col_ind].name
                if mat_abs_sum[col_ind]==0:
                    excs.append(RuntimeError(f"Component '{comp_name}' is not connected to anything"))
                elif mat_abs_sum[col_ind]==1:
                    sign = '+' if mat_sum[col_ind]<0 else '-'
                    excs.append(RuntimeError(f"Terminal '{sign}{comp_name}' is open circuit"))
                elif mat_abs_sum[col_ind]==2:
                    if mat_sum[col_ind]!=0:
                        sign = '+' if mat_sum[col_ind]>0 else '-'
                        excs.append(RuntimeError(
                            f"Terminal '{sign}{comp_name}' is connected twice to different nets"))
                else: # >2
                    excs.append(RuntimeError(f"Component '{comp_name}' is connected to more than 2 nets"))
        # raise if there is any exception
        if excs:
            raise ExceptionGroup("Invalid circuit",excs)        
            
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
            for x in range(self.kcl_mat.shape[1]):
                if self.kcl_mat[line, x]:
                    if excluded_column is not None and x==excluded_column:
                        continue
                    candidates.append(x)
            return candidates

        def add_loop(loop_origin: list[StackElement]):
            kvl_mat_line = np.zeros((self.kcl_mat.shape[1],), dtype=np.int8)
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
        
        # store the value and verify
        self.kvl_mat = np.array(kvl_mat, dtype=np.int8)
        if not np.all(np.sum(np.abs(self.kvl_mat), axis=0)!=0):
            raise RuntimeError("The graph is not connected or have cut-edge")

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
                        pass # in this case Ohm's equation is not needed
                elif condition==3:
                    pass # in this case Ohm's equation is not needed
                if len(eq.unknowns)!=0:
                    equations.append(eq)

        self.equations = []
        for eq in equations:
            if len(eq.unknowns)!=0:
                self.equations.append(eq)
                continue
            if eq.const_term!=0:
                raise RuntimeError("The circuit is over constrained (conflict constraint)")

    def solve(self) -> None:
        self._generate_equations()
        # get all unknowns
        unknowns_set:set[UnknownVarible] = set()
        for eq in self.equations:
            for unk in eq.unknowns:
                unknowns_set.add(unk)
        unknowns = list(unknowns_set)
        # reset the value of unknowns and mark the index
        for ind,unk in enumerate(unknowns):
            unk.value=None
            if unk.reciprocal:
                unk.get_reciprocal().value=None
            unk.mat_index = ind
        # build the solver mat
        solver_mat = np.zeros((len(self.equations),len(unknowns)+1),dtype=np.float64)
        for line_ind, eq in enumerate(self.equations):
            solver_mat[line_ind, -1] = -eq.const_term
            for unk,coeff in zip(eq.unknowns, eq.coefficients):
                solver_mat[line_ind, unk.mat_index] = coeff
        self.valid_equation_indices = select_linear_independent_quations(solver_mat)
        dof = len(unknowns)-len(self.valid_equation_indices)
        if dof>0:
            raise RuntimeError(f"The circuit is not fully constrained (DOF: {dof})")
        elif dof<0:
            raise RuntimeError(f"The circuit is over constrained (DOF: {dof})")
        solver_mat = solver_mat[self.valid_equation_indices]
        # solve
        result = np.linalg.solve(solver_mat[:,:-1], solver_mat[:,-1])
        for res,unk in zip(result, unknowns):
            unk.value = res
            if unk.reciprocal:
                unk.get_reciprocal().value=1/res
        for unk in unknowns:
            if isinstance(unk.component,Resistor):
                unk.component.autocomplete()

def select_linear_independent_quations(mat:npt.NDArray[np.float64]) -> list[int]:
    cnt = 0
    selected_lines = []
    for i in range(0,mat.shape[0]):
        rnk = np.linalg.matrix_rank(mat[:i+1])
        if rnk==cnt+1:
            selected_lines.append(i)
            cnt+=1
        # rnk<cnt+1: the line shouldn't be selected
        # rnk>cnt+1: this case will never happen
    return selected_lines
