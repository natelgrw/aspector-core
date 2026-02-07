import re
import json
import os

def parse_netlist_to_graph(netlist_path, output_dir, sim_id=1, topology_id=None):
    netlist_name = os.path.basename(netlist_path)
    if topology_id is None:
        topology_id = os.path.splitext(netlist_name)[0]
    
    components = []
    nets = set()
    connections = [] # (component, net, pin)
    
    with open(netlist_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('simulator') or line.startswith('global') or line.startswith('parameters') or line.startswith('include') or line.startswith('{%') or line.startswith('}'):
            continue
            
        parts = line.split()
        if not parts:
            continue
            
        name = parts[0]
        
        # Skip if name contains '=' (likely a parameter assignment like maxnotes=5)
        if '=' in name:
            continue
            
        comp_type = 'unknown'
        
        # Simple heuristic for type
        if name.lower().startswith('m'):
            # Check if it's a simulation command starting with m (like modelParameter, mytran)
            # Mosfets usually have at least 4 nodes then a model name.
            # If parts[1] is a reserved word like 'tran', 'info', it is not a mosfet.
            if len(parts) > 1 and parts[1] in ['tran', 'dc', 'ac', 'noise', 'info', 'check']:
                continue
                
            comp_type = 'mosfet'
            # Format: Name Drain Gate Source Body Model ...
            # MM3 Voutp Vinp net13 gnd! nfet ...
            # Clean parens just in case: MM1 (d g s b) ...
            if len(parts) >= 5:
                d = parts[1].replace('(', '').replace(')', '')
                g = parts[2].replace('(', '').replace(')', '')
                s = parts[3].replace('(', '').replace(')', '')
                b = parts[4].replace('(', '').replace(')', '')
                
                model = parts[5] if len(parts) > 5 else 'mosfet'
                if 'nfet' in model.lower(): comp_type = 'nfet'
                elif 'pfet' in model.lower(): comp_type = 'pfet'
                
                connections.append((name, d, 'D'))
                connections.append((name, g, 'G'))
                connections.append((name, s, 'S'))
                connections.append((name, b, 'B'))
                nets.update([d, g, s, b])

        elif name.lower().startswith('r'):
            # Filter TESTBENCH resistors
            if any(k in name for k in ['Rin', 'Rfeed', 'Rload', 'Rsw', 'Rsrc', 'Rshunt', 'R_unity']):
                continue
                
            comp_type = 'resistor'
            if len(parts) >= 3:
                n1 = parts[1].replace('(', '').replace(')', '')
                n2 = parts[2].replace('(', '').replace(')', '')
                connections.append((name, n1, '1'))
                connections.append((name, n2, '2'))
                nets.update([n1, n2])
                
        elif name.lower().startswith('c'):
             # Check if it's a simulation command (e.g. checklimitdest matches 'c')
            if len(parts) > 1 and parts[1] in ['tran', 'dc', 'ac', 'noise', 'info']:
                continue
            # Filter TESTBENCH capacitors
            if any(k in name for k in ['Ctran', 'Cload']):
                 continue
                 
            comp_type = 'capacitor'
            if len(parts) >= 3:
                n1 = parts[1].replace('(', '').replace(')', '')
                n2 = parts[2].replace('(', '').replace(')', '')
                connections.append((name, n1, '1'))
                connections.append((name, n2, '2'))
                nets.update([n1, n2])

        # Independent Sources REMOVED


        # Dependent Sources REMOVED

        
        if comp_type != 'unknown':
            components.append({'name': name, 'type': comp_type})

    # Build Graph Object
    graph_nodes = {}
    
    # Add Component Nodes
    for comp in components:
        graph_nodes[comp['name']] = {"type": "COMPONENT", "subtype": comp['type']}
        
    # Add Net Nodes
    for net in nets:
        graph_nodes[net] = {"type": "NET"}
        
    # Build Edges
    # Edge is between Component and Net
    edges = []
    for comp_name, net_name, pin in connections:
        edges.append({
            "source": comp_name,
            "target": net_name,
            "pin": pin
        })
        
    graph_obj = {
        "sim_id": sim_id,
        "topology_id": topology_id,
        "netlist": netlist_name,
        "graph": {
            "directed": False,
            "nodes": graph_nodes,
            "edges": edges
        }
    }
    
    output_path = os.path.join(output_dir, "graph.json")
    with open(output_path, 'w') as f:
        json.dump(graph_obj, f, indent=2)
        
    return output_path

def extract_sizing_map(netlist_path):
    """
    Parses a Spectre netlist to map component parameters to optimization variables.
    
    Returns:
    --------
    dict
        { 
          "M1": { "l": "nA1", "nfin": "nB1" },
          "M2": { "l": "nA2", ... } 
        }
    """
    mapping = {}
    
    with open(netlist_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignore comments
            if line.startswith('*') or line.startswith('//'):
                continue
            
            parts = line.split()
            if not parts:
                continue
                
            name = parts[0]
            
            # Filter out obvious non-components
            if '=' in name: 
                continue

            # Check for simulation directives that might mask as components
            # e.g. "mytran tran ...", "modelParameter info ..."
            if len(parts) > 1 and parts[1] in ['options', 'tran', 'dc', 'ac', 'noise', 'info', 'pz', 'sp', 'pss', 'hb', 'envlp']:
                continue

            # Only care about MOS, R, C
            # Using tuple for startswith is cleaner
            if name.upper().startswith(('M', 'R', 'C')):
                # Filter Testbench artifacts
                if any(tb in name for tb in ['Rin', 'Rfeed', 'Rload', 'Ctran', 'Cload', 'Rsw', 'Rsrc', 'Rshunt', 'R_unity']):
                    continue
                    
                # Scan for parameters
                # Format: MM3 V... nfet l=nA1 nfin=nB1 ...
                comp_map = {}
                for part in parts:
                    if '=' in part:
                        key, val = part.split('=', 1)
                        # clean up potential parens or formatting
                        key = key.strip()
                        val = val.strip().replace('(', '').replace(')', '')
                        
                        # We are looking for variables like nA1, nB1, etc.
                        # Heuristic: if val starts with 'n' or matches optimization params?
                        # Or just capture everything that looks like a variable?
                        # Let's capture everything for now, or filter later.
                        # The user specifically mentioned nfet/pfet A2: B2 etc.
                        
                        comp_map[key] = val
                        
                if comp_map:
                    mapping[name] = comp_map
                    
    return mapping
