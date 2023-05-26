import k2
def minimal_topo(tokens) -> 'k2.Fsa':
    """Build the minimal topology.
    See https://arxiv.org/abs/2110.03098
    """
    assert 0 in tokens, "We assume 0 is ID of the blank symbol"

    num_tokens = len(tokens)
    final_state = 1
    arcs = ""
    for i in range(num_tokens):
        arcs += f"0 0 {tokens[i]} {tokens[i]} 0.0\n"
    arcs += f"0 {final_state} -1 -1 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    ans = k2.arc_sort(ans)
    return ans

def build_ctc_topo_compact_sorted(tokens):
    assert 0 in tokens, "We assume 0 is ID of the blank symbol"

    num_blank_states = 1
    num_states = len(tokens) + num_blank_states
    final_state = num_states
    arcs = "0 1 1 0 0.0\n"
    for i in range(num_blank_states + 1, num_states):
        arcs += f"0 {i} {tokens[i - 1] + 1} {tokens[i - 1] + 1} 0.0\n"
    arcs += f"0 {final_state} -1 -1 0.0\n"
    for i in range(num_blank_states, num_states):
        arcs += f"{i} 0 0 0 0.0\n"
        arcs += f"{i} {i} {tokens[i - 1] + 1} 0 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs, num_aux_labels=1)
    return ans

token_shift = minimal_topo(list(range(4)))
token_shift.draw('topo_minimal.pdf')