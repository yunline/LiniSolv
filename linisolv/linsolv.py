import numpy as np
import numpy.typing as npt
import typing

__all__ = [
    "Component",
    "Resistor",
    "Source",
    "LinearCircuitSolver",
    "LinisolvError",
    "CircuitError",
    "CircuitTopologyError",
    "CircuitSolverError"
]

PRINT_COLOR_RED = "\033[31m"
PRINT_COLOR_GREEN = "\033[32m"
PRINT_COLOR_DEFAULT = "\033[39m"

class LinisolvError(RuntimeError):
    pass

class CircuitError(LinisolvError):
    pass

class CircuitTopologyError(CircuitError):
    disconnected_components:list["Component"]
    def __init__(self,*args,disconnected_comps:list["Component"]|None=None,**kwargs) -> None:
        if disconnected_comps is not None:
            self.disconnected_comps=disconnected_comps
        super().__init__(*args, **kwargs)

class CircuitSolverError(LinisolvError):
    pass


class Varible:
    component:"Component"
    param_name: str
    mat_index: int
    reciprocal: bool = False

    def __init__(self, component:"Component", param_name:str):
        self.component = component
        self.param_name = param_name
    
    def __hash__(self) -> int:
        return id(self)
    
    def get_reciprocal(self) -> typing.Self:
        if not hasattr(self, "_rec"):
            self._rec = self.__class__.__new__(self.__class__)
            self._rec.component = self.component
            self._rec.param_name = self.param_name
            self._rec.reciprocal = True
            self._rec._rec = self
        return self._rec
    
    @property
    def value(self) -> float:
        raise NotImplementedError()


class UnknownVarible(Varible):
    computed_value: float|None = None

    def __repr__(self):
        if self.computed_value is None:
            return f"{PRINT_COLOR_RED}Unknown{PRINT_COLOR_DEFAULT}"
        else:
            return f"{PRINT_COLOR_GREEN}{float(self.computed_value):.2f}{PRINT_COLOR_DEFAULT}"
    
    def reset(self):
        self.computed_value = None
        if not self.reciprocal and hasattr(self,"_rec"):
            self._rec.computed_value = None
    
    @property
    def value(self) -> float:
        if self.computed_value is None:
            raise RuntimeError("The value of unknown varible is not computed")
        return self.computed_value

class ConstrainedVarible(Varible):
    constraint_value: float

    def __repr__(self):
            return f"{float(self.constraint_value):.2f}"

    def __init__(self, component:"Component", param_name:str, constraint_value:float):
        super().__init__(component, param_name)
        self.constraint_value = constraint_value
    
    def get_reciprocal(self):
        rec = super().get_reciprocal()
        rec.constraint_value = 1/self.value
        return rec

    @property
    def value(self) -> float:
        return self.constraint_value

Coefficient = list[float|ConstrainedVarible]

def _get_coefficient_value(coeff: Coefficient):
    value = 1.0
    for v in coeff:
        if isinstance(v, ConstrainedVarible):
            value*=v.value
        else:
            value*=float(v)
    return value

def get_term_expr(term: list[float|Varible])->tuple[float,str]:
    coeff = 1.0
    vars:list[Varible] = []
    for v in term:
        if isinstance(v, Varible):
            vars.append(v)
        else:
            coeff*=float(v)
    sorted_vars = sorted(vars, key=lambda v:v.reciprocal)
    is_first = True
    first_reciprocal = False
    expr_list:list[str] = []
    for v in sorted_vars:
        _expr = f"{v.param_name.upper()}_{v.component.name}"
        if is_first:
            expr_list.append(_expr)
            first_reciprocal = v.reciprocal
            is_first = False
            continue
        if v.reciprocal:
            expr_list.append("/"+_expr)
        else:
            expr_list.append("*"+_expr)
    expr_str = "".join(expr_list)
    if abs(coeff)!=1.0:
        if first_reciprocal:
            expr_str = "/" + expr_str
        expr_str = f"{abs(coeff):.1f}" + expr_str
    elif first_reciprocal:
        expr_str = f"1/" + expr_str
    return (1.0 if coeff>0 else -1.0), expr_str

class Equation:
    unknowns: list[UnknownVarible]
    coefficients: list[Coefficient]
    const_terms: list[Coefficient]

    def __init__(self):
        self.unknowns = []
        self.coefficients = []
        self.const_terms = []

    def add_unknown(self, unknown:UnknownVarible, coeff:Coefficient):
        self.unknowns.append(unknown)
        self.coefficients.append(coeff)
    
    def add_const(self, const:Coefficient):
        self.const_terms.append(const)

    def get_const_term_value(self) -> float:
        return sum(_get_coefficient_value(coeff) for coeff in self.const_terms)
    
    def get_coefficient_values(self) -> list[float]:
        return [_get_coefficient_value(coeff) for coeff in self.coefficients]
    
    def get_expr(self):
        left = []
        is_first_term = True
        for unknwon, coeff in zip(self.unknowns, self.coefficients):
            sign, term_expr = get_term_expr([*coeff, unknwon])
            if is_first_term:
                if sign>0:
                    left.append(term_expr)
                else:
                    left.append("-"+term_expr)
                is_first_term = False
            else:
                left.append((" + " if sign>0 else " - ") + term_expr)
        left_str = "".join(left)
        right = []
        is_first_term = True
        for const_term in self.const_terms:
            sign, term_expr = get_term_expr(const_term)
            if is_first_term:
                if sign>0:
                    right.append(term_expr)
                else:
                    right.append("-"+term_expr)
                is_first_term = False
            else:
                right.append((" + " if sign>0 else " - ") + term_expr)
        if len(right)==0:
            rigth_str = "0.0"
        else:
            rigth_str = "".join(right)
        
        return left_str + " = " + rigth_str

class Component:
    name: str
    mat_index: int

    _params: tuple[str, ...] = ("i", "v")
    i: Varible
    v: Varible

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
            param_raw = params.get(name)
            param: Varible
            if param_raw is None:
                param = UnknownVarible(self, name)
            else:
                param = ConstrainedVarible(self, name, float(param_raw))
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
    r: Varible
    def autocomplete(self) -> None:
        def is_unknown(param_name:str) -> bool:
            param=getattr(self, param_name)
            assert isinstance(param, Varible)
            if isinstance(param, UnknownVarible) and param.computed_value is None:
                return True
            return False
        unknown_state = tuple(is_unknown(name) for name in ['r','v','i'])
        if unknown_state==(False, False, False):
            return
        elif unknown_state==(True, False, False):
            assert isinstance(self.r, UnknownVarible)
            self.r.computed_value = self.v.value/self.i.value
        elif unknown_state==(False, True, False):
            assert isinstance(self.v, UnknownVarible)
            self.v.computed_value = self.i.value*self.r.value
        elif unknown_state==(False, False, True):
            assert isinstance(self.i, UnknownVarible)
            self.i.computed_value = self.v.value/self.r.value
        else:
            raise CircuitSolverError(f"Too much unknowns, resistor '{self.name}' autocompletion failed")


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
                    raise CircuitError(
                        f"Component '{terminal.component.name}' "
                        "is not included in the component list"
                    )
                if kcl_mat[line_ind, terminal.component.mat_index]:
                    shorted_indices.append(terminal.component.mat_index)
                    excs.append(CircuitError(f"Component '{terminal.component.name}' is shorted"))
                kcl_mat[line_ind, terminal.component.mat_index] = terminal.sign
        # verify kcl mat
        abs_kcl_mat = np.abs(kcl_mat)
        for line_ind,line_sum in enumerate(np.sum(np.abs(kcl_mat),axis=1)):
            if line_sum==0:
                excs.append(CircuitError(f"Net[{line_ind}]: 0 terminals connected (expected >=2)"))
            elif line_sum==1:
                excs.append(CircuitError(f"Net[{line_ind}]: Only 1 terminal connected (expected >=2)"))
        mat_abs_sum = np.sum(abs_kcl_mat, axis=0)
        mat_sum = np.sum(kcl_mat, axis=0)
        if not (np.all(mat_abs_sum==2) and np.all(mat_sum==0)):
            for col_ind in range(kcl_mat.shape[1]):
                if col_ind in shorted_indices:
                    continue # component is shorted, skip the check
                comp_name = self.components[col_ind].name
                if mat_abs_sum[col_ind]==0:
                    excs.append(CircuitError(f"Component '{comp_name}' is not connected to anything"))
                elif mat_abs_sum[col_ind]==1:
                    sign = '+' if mat_sum[col_ind]<0 else '-'
                    excs.append(CircuitError(f"Terminal '{sign}{comp_name}' is open circuit"))
                elif mat_abs_sum[col_ind]==2:
                    if mat_sum[col_ind]!=0:
                        sign = '+' if mat_sum[col_ind]>0 else '-'
                        excs.append(CircuitError(
                            f"Terminal '{sign}{comp_name}' is connected twice to different nets"))
                else: # >2
                    excs.append(CircuitError(f"Component '{comp_name}' is connected to more than 2 nets"))
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
        mat_abs_sum = np.sum(np.abs(self.kvl_mat), axis=0)
        if not np.all(mat_abs_sum!=0):
            disconnected = []
            for col_ind in range(self.kvl_mat.shape[1]):
                if mat_abs_sum[col_ind]==0:
                    disconnected.append(self.components[col_ind])
            raise CircuitTopologyError(
                f"Component is disconnected or on cut-edge\n"
                f"Disconnected/cut-edge component(s): {[i.name for i in disconnected]}",
                disconnected_comps=disconnected,
            )

    def generate_equations(self) -> None:
        equations:list[Equation] = []
        line:list[float]
        # get KCL equations
        for line in self.kcl_mat:
            eq = Equation()
            equations.append(eq)
            for comp_ind, comp in enumerate(self.components):
                if line[comp_ind]==0.0:
                    continue
                if isinstance(comp.i, ConstrainedVarible):
                    eq.add_const([comp.i, line[comp_ind]])
                    continue
                assert isinstance(comp.i, UnknownVarible)
                if isinstance(comp, Source):
                    eq.add_unknown(comp.i, [line[comp_ind]])
                elif isinstance(comp, Resistor):
                    if (isinstance(comp.r, ConstrainedVarible) and
                        isinstance(comp.v, ConstrainedVarible)):
                        eq.add_const([line[comp_ind], comp.v, comp.r.get_reciprocal()])
                    elif (isinstance(comp.r, ConstrainedVarible) and
                        isinstance(comp.v, UnknownVarible)):
                        eq.add_unknown(comp.v, [line[comp_ind], comp.r.get_reciprocal()])
                    elif (isinstance(comp.r, UnknownVarible) and
                        isinstance(comp.v, ConstrainedVarible)):
                        eq.add_unknown(comp.r.get_reciprocal(), [line[comp_ind],comp.v])
                    else:
                        eq.add_unknown(comp.i, [line[comp_ind]])
        # get KVL equations
        for line in self.kvl_mat:
            eq = Equation()
            equations.append(eq)
            for comp_ind, comp in enumerate(self.components):
                if line[comp_ind]==0:
                    continue
                if isinstance(comp.v, ConstrainedVarible):
                    eq.add_const([comp.v, line[comp_ind]])
                    continue
                assert isinstance(comp.v, UnknownVarible)
                if isinstance(comp, Source):
                    eq.add_unknown(comp.v, [line[comp_ind]])
                elif isinstance(comp, Resistor):
                    if (isinstance(comp.r, ConstrainedVarible) and
                        isinstance(comp.i, ConstrainedVarible)):
                        eq.add_const([comp.i, comp.r, line[comp_ind]])
                    elif (isinstance(comp.r, ConstrainedVarible) and
                        isinstance(comp.i, UnknownVarible)):
                        eq.add_unknown(comp.i, [comp.r, line[comp_ind]])
                    elif (isinstance(comp.r, UnknownVarible) and
                        isinstance(comp.i, ConstrainedVarible)):
                        eq.add_unknown(comp.r, [comp.i, line[comp_ind]])
                    else:
                        eq.add_unknown(comp.v, [line[comp_ind]])
        # get Ohm's Law qeuations
        for comp in self.components:
            if not isinstance(comp, Resistor):
                continue
            eq = Equation()
            unknown_state = (
                isinstance(comp.r, UnknownVarible),
                isinstance(comp.v, UnknownVarible),
                isinstance(comp.i, UnknownVarible),
            )
            if unknown_state==(True, False, False):
                eq.add_unknown(comp.r, [1.0]) # type: ignore
                eq.add_const([-1, comp.v, comp.i.get_reciprocal()]) # type: ignore
            elif unknown_state==(False, True, False):
                eq.add_unknown(comp.v, [1.0])  # type: ignore
                eq.add_const([-1, comp.r, comp.i]) # type: ignore
            elif unknown_state==(False, False, True):
                eq.add_unknown(comp.i, [1.0]) # type: ignore
                eq.add_const([-1, comp.v, comp.r.get_reciprocal()]) # type: ignore
            elif unknown_state==(False, True, True):
                eq.add_unknown(comp.v, [-1.0]) # type: ignore
                eq.add_unknown(comp.i, [comp.r]) # type: ignore
            elif unknown_state==(True, True, False):
                eq.add_unknown(comp.v, [-1.0]) # type: ignore
                eq.add_unknown(comp.r, [comp.i]) # type: ignore
            else:
                continue # In this case Ohm's equation is not needed
            equations.append(eq)

        self.equations = []
        for eq in equations:
            if len(eq.unknowns)!=0:
                self.equations.append(eq)
            elif len(eq.const_terms)!=0: # 0=C, conflict
                raise CircuitSolverError("The circuit is over constrained (conflict constraint)")

    def solve(self, use_cached_equations = False) -> None:
        if not (use_cached_equations and hasattr(self, "equations")):
            self.generate_equations()
        # get all unknowns
        unknowns_set:set[UnknownVarible] = set()
        for eq in self.equations:
            for unk in eq.unknowns:
                unknowns_set.add(unk)
        unknowns = list(unknowns_set)
        # reset the value of unknowns and mark the index
        for ind,unk in enumerate(unknowns):
            unk.reset()
            unk.mat_index = ind
        # build the solver mat
        solver_mat = np.zeros((len(self.equations),len(unknowns)+1),dtype=np.float64)
        for line_ind, eq in enumerate(self.equations):
            solver_mat[line_ind, -1] = -eq.get_const_term_value()
            for unk,coeff in zip(eq.unknowns, eq.get_coefficient_values()):
                solver_mat[line_ind, unk.mat_index] = coeff
        self.valid_equation_indices = select_linear_independent_quations(solver_mat)
        dof = len(unknowns)-len(self.valid_equation_indices)
        if dof>0:
            raise CircuitSolverError(f"The circuit is not fully constrained (DOF: {dof})")
        elif dof<0:
            raise CircuitSolverError(f"The circuit is over constrained (DOF: {dof})")
        solver_mat = solver_mat[self.valid_equation_indices]
        # solve
        result = np.linalg.solve(solver_mat[:,:-1], solver_mat[:,-1])
        for res,unk in zip(result, unknowns):
            unk.computed_value = res
            if unk.reciprocal:
                unk.get_reciprocal().computed_value=1/res
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
