import io
import itertools
import re
import numpy as np
import numpy.typing as npt

class Component:
    name: str
    mat_index: int
    _params: list

    def __init__(self, name:str, params):
        self.name = name
    
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

class Resistor(Component):
    r: float|None = None
    i: float|None = None
    v: float|None = None
    def __init__(self, name:str, params:dict[str,float]):
        super().__init__(name, params)
        self._parse_param(params, ["r","i","v"])
    
class VoltageSource(Component):
    i: float|None = None
    v: float|None = None
    def __init__(self, name:str, params:dict[str,float]):
        super().__init__(name, params)
        self._parse_param(params, ["i","v"])

class CurrentSource(Component):
    i: float|None = None
    v: float|None = None
    def __init__(self, name:str, params:dict[str,float]):
        super().__init__(name, params)
        self._parse_param(params, ["i","v"])


component_token_to_type:dict[str, type[Component]] = {"R":Resistor, "V":VoltageSource, "I":CurrentSource}
comp_def_pattern = re.compile(r"(?P<comp_type>[RVI]) +(?P<comp_name>[\w\d]+){(?P<comp_params>([\w\d]+=[+-]?\d+(.\d*)?,? ?)*)}")
comp_param_pattern = re.compile(r"(?P<param_name>[\w\d])+=(?P<param_value>[+-]?\d+(.\d*))")
net_def_pattern = re.compile(r"(?P<net_name>[\w\d]+)?{([\w\d]+[+-],? ?)+}")
net_terminal_pattern = re.compile(r"(?P<comp>[\w\d]+)(?P<pol>[+-])")

def parse_circuit(stream:io.TextIOBase) -> tuple[list[Component], npt.NDArray[np.int8]]:
    parsing_net_section = False
    comp_dict:dict[str,Component] = {}
    components = []
    kcl_mat:list[list[int]] = []
    eof = False
    for line_no in itertools.count(1):
        line = stream.readline()
        if line=='': # EOF
            eof = True
        line = line[:line.find("#")].strip()
        if line=='' and not eof: # Empty line
            continue

        if not parsing_net_section:
            if eof:
                raise RuntimeError("No net is defined in this circui file")
            if line=="NET":
                parsing_net_section = True
                for n,comp in enumerate(comp_dict.values()):
                    comp.mat_index=n
                    components.append(comp)
                continue

            comp_def_match = comp_def_pattern.fullmatch(line)
            if comp_def_match is None:
                raise RuntimeError(f"Syntax error at line {line_no}")
            comp_name = comp_def_match["comp_name"]
            if comp_name in comp_dict:
                raise RuntimeError(f"At line {line_no}: {comp_name} has been defined")
            params_dict = {}
            for param_match in comp_param_pattern.finditer(comp_def_match["comp_params"]):
                params_dict[param_match["param_name"]] = float(param_match["param_value"])
            comp_cls = component_token_to_type[comp_def_match["comp_type"]]
            comp_dict[comp_name] = comp_cls(comp_name, params_dict)
            
        else: # parsing net section
            if eof:
                break
            net_def_match = net_def_pattern.fullmatch(line)
            if net_def_match is None:
                raise RuntimeError(f"Syntax error at line {line_no}")
            
            kcl_mat_line = [0]*len(components)
            for terminal_match in net_terminal_pattern.finditer(line):
                try:
                    comp = comp_dict[terminal_match["comp"]]
                except KeyError:
                    raise RuntimeError(f"At line {line_no}: Undefined component '{terminal_match['comp']}'")
                if terminal_match["pol"]=="+":
                    kcl_mat_line[comp.mat_index]=+1
                elif terminal_match["pol"]=="-":
                    kcl_mat_line[comp.mat_index]=-1
                else:
                    raise RuntimeError(f"Unable to resolve +- sign at line {line_no}")
            kcl_mat.append(kcl_mat_line)
    return components, np.array(kcl_mat, dtype=np.int8)

def find_loops(kcl_mat: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    nlines, ncolumns = kcl_mat.shape
    
    # build search mat
    NO_CONNECTION = 0xffff
    search_mat = np.full((nlines, ncolumns), NO_CONNECTION, dtype=np.uint16)
    for y in range(nlines):
        for x in range(ncolumns):
            if kcl_mat[y,x] and search_mat[y,x]==NO_CONNECTION:
                for j in range(nlines):
                    if j!=y and kcl_mat[j,x]:
                        search_mat[j,x]=y
                        search_mat[y,x]=j
                        break
    del x,y,j # clean the namespace

    # var of searching
    class StackElement:
        line:int
        column:int
        candidates:list[int]
        def __init__(self, line, candidates:list[int]):
            self.line = line
            self.candidates = candidates
        def load_candidate(self):
            try:
                self.column = self.candidates.pop()
                return True
            except IndexError:
                return False
        def __repr__(self):
            return f"<ln={self.line}, col={self.column}, {self.candidates}>"

    stack: list[StackElement] = []
    def get_candidate(line:int, excluded_column:int|None=None)->list[int]:
        candidates = []
        for x in range(ncolumns):
            if kcl_mat[line, x]:
                if excluded_column is not None and x==excluded_column:
                    continue
                candidates.append(x)
        return candidates
    
    kvl_mat:list[npt.NDArray[np.int8]] = []
    def add_loop(loop_origin: list[StackElement]):
        kvl_mat_line = np.zeros((ncolumns,), dtype=np.int8)
        for ele in loop_origin:
            kvl_mat_line[ele.column] = kcl_mat[ele.line, ele.column]
        for line in kvl_mat:
            if np.array_equal(line, kvl_mat_line) or np.array_equal(-line, kvl_mat_line):
                return
        # print("→".join(f"{e.column}" for e in loop_origin))
        kvl_mat.append(kvl_mat_line)
    
    # search
    stack.append(StackElement(0, get_candidate(0)))
    while stack:
        tos = stack[-1] # tos: top of stack
        if not tos.load_candidate(): # load the next candidate
            stack.pop()
            continue
        connecting_line = search_mat[tos.line, tos.column]
        if len(stack)>=3:
            continue_ = False
            for ele_index,ele in enumerate(stack[:-2]):
                if ele.line==connecting_line: # found a loop
                    add_loop(stack[ele_index:])
                    continue_ = True
                    break
            if continue_:
                continue
        connecting_candidates = get_candidate(connecting_line, excluded_column=tos.column)
        stack.append(StackElement(connecting_line, connecting_candidates))
    
    # return
    return np.array(kvl_mat, dtype=np.int8)

def is_parallel(a,b):
    return not np.any(np.cross(a,b))

with open("test_circuits/circuit0.txt", "r", encoding="utf-8") as _file:
    components, kcl_mat = parse_circuit(_file)
print(f"KCL mat:\n{kcl_mat}")
kvl_mat = find_loops(kcl_mat)
print(f"KVL mat:\n{kvl_mat}")
