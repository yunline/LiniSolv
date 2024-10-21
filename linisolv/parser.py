import io
import itertools
import re
from .linsolv import *

component_token_to_type:dict[str, type[Component]] = {"R":Resistor, "S":Source}
comp_def_pattern = re.compile(
    r"(?P<comp_type>[RS])\s+"
    r"(?P<comp_name>[\w\d]+){"
        r"(?P<comp_params>"
            r"([\w\d]+\s*=\s*[+-]?\d+(\.\d*)?\s*,\s*)*"
            r"([\w\d]+\s*=\s*[+-]?\d+(\.\d*)?\s*,?\s*)?"
        r")"
    r"}"
)
comp_param_pattern = re.compile(r"(?P<param_name>[\w\d]+)\s*=\s*(?P<param_value>[+-]?\d+(\.\d*)?)")
net_def_pattern = re.compile(
    r"(?P<net_name>[\w\d]+)?{"
        r"([+-][\w\d]+\s*,\s*)+"
        r"([+-][\w\d]+\s*,?\s*)?"
    r"}"
)
net_terminal_pattern = re.compile(r"(?P<pol>[+-])(?P<comp>[\w\d]+)")

def parse_circuit(stream:io.TextIOBase) -> LinearCircuitSolver:
    parsing_net_section = False
    comp_dict:dict[str,Component] = {}
    components = []
    net_list:list[list[Component.Terminal]] = []
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
            comp_dict[comp_name] = comp_cls(comp_name, **params_dict)
            
        else: # parsing net section
            if eof:
                break
            net_def_match = net_def_pattern.fullmatch(line)
            if net_def_match is None:
                raise RuntimeError(f"Syntax error at line {line_no}")
            
            net_list_line = []
            for terminal_match in net_terminal_pattern.finditer(line):
                try:
                    comp = comp_dict[terminal_match["comp"]]
                except KeyError:
                    raise RuntimeError(f"At line {line_no}: Undefined component '{terminal_match['comp']}'")
                if terminal_match["pol"]=="+":
                    net_list_line.append(+comp)
                elif terminal_match["pol"]=="-":
                    net_list_line.append(-comp)
                else:
                    raise RuntimeError(f"Unable to resolve +- sign at line {line_no}")
            net_list.append(net_list_line)
    return LinearCircuitSolver(components, net_list)
