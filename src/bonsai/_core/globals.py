globalVars = {'nCells', 'nGenes', 'nNodes', 'max_node_ind', 'mem_friendly'}
nCells = None
nGenes = None
nNodes = None
max_node_ind = 0
mem_friendly = False


def reset():
    """Restore module state so repeated in-process runs are independent."""
    global nCells, nGenes, nNodes, max_node_ind, mem_friendly
    nCells = nGenes = nNodes = None
    max_node_ind = 0
    mem_friendly = False
    for name in ("nwk_counter", "nwk_folder", "geneIds", "geneMeans", "getHessStarTreeJax_jit"):
        globals().pop(name, None)
