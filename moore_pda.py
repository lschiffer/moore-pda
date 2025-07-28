from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Set, Tuple, Optional, Dict, List, Any
from enum import Enum
from typeguard import typechecked
import json
from graphviz import Digraph


class StackOp(Enum):
    PUSH = "__push__"
    POP = "__pop__"
    IGNORE = "__ignore__"

BOTTOM = "__bottom_symbol__"
EPS = "__epsilon__"

special_symbols: Set[str] = { BOTTOM, EPS, StackOp.POP.value, StackOp.PUSH.value, StackOp.IGNORE.value }

PDATransition = Tuple[str, str, StackOp, Optional[str], str]
MPDATransition = Tuple[str, StackOp, Optional[str], str]

@typechecked
@dataclass
class PDA():
    """Pushdown automaton.
    In a single transition, only single stack symbols can be pushed or popped
    (or the stack can be ignored).
    No epsilon-transitions are allowed.
    An accepting configuration is reached when a final state is reached while
    emptying the stack. Note that the empty word can therefore not be accepted.
    Transitions have the form (source state, input symbol, stack operation,
    stack symbol, target state). If the stack operation is IGNORE, the stack
    symbol should be None.
    """
    
    states: Set[str]
    input_symbols: Set[str]
    stack_symbols: Set[str]
    transitions: Set[PDATransition]
    initial: Set[str]
    final: Set[str]


    def __post_init__(self):

        self.check_basics()
        self.check_initial_states_no_ingoing()
        self.check_final_states_no_outgoing()
        self.check_transitions_to_final()


    def check_basics(self):
        """Basic check that the PDA definition is not violated: transitions,
        initial, and final may only use valid states, input_symbols,
        and stack_symbols, respectively. Reserved special symbols are not allowed
        as states or stack symbols or input symbols.
        """

        if not self.initial <= self.states:
            raise ValueError("Initial states must be contained in states");
        if not self.final <= self.states:
            raise ValueError("Final states must be contained in states");
        if not BOTTOM in self.stack_symbols:
            raise ValueError(f"{BOTTOM} must be in stack symbols")
        for (src, inp, op, stk, tar) in self.transitions:
            if (not src in self.states) or (not tar in self.states):
                raise ValueError(f"Invalid source or target state in {src, inp, op, stk, tar}")
            if not stk in self.stack_symbols and not (stk is None and op == StackOp.IGNORE):
                raise ValueError(f"Invalid stack symbol in {src, inp, op, stk, tar}")
            if op == StackOp.IGNORE and not (stk is None):
                raise ValueError(f"Invalid stack operation in {src, inp, op, stk, tar}")
        if not self.states or not self.initial or not self.final or not self.input_symbols:
                raise ValueError(f"Sets of states, initial states, final states, input symbols must not be empty")
        for sym in special_symbols:
            if sym in self.states or sym in self.input_symbols or (sym in self.stack_symbols and sym != BOTTOM):
                raise ValueError(f"Special symbol {sym} not allowed as state or stack symbol or input symbol")


    @classmethod
    def from_dict(cls, input_dict) -> PDA:
        """Takes a JSON dict and returns a PDA instance based on it."""

        if input_dict["type"] != "PDA":
                raise ValueError(f"Input automaton type not matching PDA")

        prepared_dict = {}

        for field in fields(PDA):
            if field.name == "transitions":
                prepared_transitions: Set[PDATransition] = set()
                for trans in input_dict["transitions"]:
                    if len(trans) != 5:
                        raise ValueError(f"PDATransition {trans} needs to have 5 elements")
                    prepared_transitions.add((trans[0], trans[1], StackOp(trans[2]), trans[3], trans[4]))

                prepared_dict["transitions"] = prepared_transitions
                
            else:
                if field.name in input_dict:
                    prepared_dict[field.name] = set(input_dict[field.name])

        return PDA(**prepared_dict)


    def check_initial_states_no_ingoing(self):
        """Check that initial states have no ingoing transitions."""

        for (src, inp, op, stk, tar) in self.transitions:
            if tar in self.initial:
                raise ValueError(f"Initial state {tar} must not have ingoing transitions: {src, inp, op, stk, tar}");

    def check_final_states_no_outgoing(self):
        """Check that final states have no outgoing transitions."""

        for (src, inp, op, stk, tar) in self.transitions:
            if src in self.final:
                raise ValueError(f"Final state {src} must not have outgoing transitions: {src, inp, op, stk, tar}");

    def check_transitions_to_final(self):
        """Check that all transitions leading to final states pop the bottom
        symbol and that bottom symbol is never pushed."""

        for (src, inp, op, stk, tar) in self.transitions:
                if tar in self.final and not (op == StackOp.POP and stk == BOTTOM):
                    raise ValueError(f"Transition to final state {tar} must pop bottom symbol: {src, inp, op, stk, tar}");
                if op == StackOp.PUSH and stk == BOTTOM:
                    raise ValueError(f"No transition must push bottom symbol: {src, inp, op, stk, tar}");
        
    def get_length_one_words(self) -> Tuple[Set[str], Set[PDATransition]]:
        """Returns list of all words of length 1. Because of the restrictions
        of the PDA definition, only transitions from an initial to a final
        state need to be considered.
        """

        relevant_transitions: Set[PDATransition] = set()
        length_one_words: Set[str] = set()

        for (src, inp, op, stk, tar) in self.transitions:
            if src in self.initial and tar in self.final and op == StackOp.POP and stk == BOTTOM:
                length_one_words.add(inp)
                relevant_transitions.add((src, inp, op, stk, tar))

        return length_one_words, relevant_transitions
        
    def remove_transitions(self, transitions: set[PDATransition]):
        """Remove the transitions from the PDA. Raises a KeyError if a
        transition does not exist.
        """

        for transition in transitions:
            self.transitions.remove(transition)

    def convert_to_MPDA(self) -> MPDA:
        """Convert PDA to MPDA."""

        new_states: Set[str]
        new_input_symbols: Set[str] = self.input_symbols
        new_stack_symbols: Set[str]
        new_transitions: Set[MPDATransition] = set()
        new_output_function: Dict[str, str]
        new_initial: Set[str] = set()
        new_final: Set[str]

        stack_symbols_temp: Set[str] = self.stack_symbols.copy()
        stack_symbols_temp.add(EPS)

        basic_state_name_list: List[Tuple[Tuple[str, str, str, int], str]] = [((q, i, s, b), (f"{q}_{i}_{s}_{b}"))
                        for q in self.states
                        for i in self.input_symbols 
                        for s in stack_symbols_temp 
                        for b in [0,1]]

        # each state gets an index to prevent name collisions
        state_name_dict: Dict[Tuple[str, str, str, int], str] = dict()
        for (idx, (state_tuple, basic_state_name)) in enumerate(basic_state_name_list):
            state_name_dict[state_tuple] = f"{basic_state_name}-{idx}"

        new_output_function = {state_name_dict[(q, i, s, b)]: i
            for (q, i, s, b) in state_name_dict.keys()}

        new_states = set(state_name_dict.values())

        new_stack_symbols = {f"{s}_{b}"
                for s in self.stack_symbols
                for b in [0,1]}

        new_final = {state_name_dict[(q, i, s, b)]
                for q in self.final
                for i in self.input_symbols 
                for s in [EPS] 
                for b in [1]}

        for (src, inp, op, stk, tar) in self.transitions:
            if src in self.initial and op == StackOp.PUSH:
                new_initial.add(state_name_dict[(tar, inp, stk, 1)])
            elif src in self.initial and op == StackOp.IGNORE:
                new_initial.add(state_name_dict[(tar, inp, EPS, 1)])
            # the case POP does not occur because that would mean that a word
            # of length 1 is accepted, which is not representable in the MPDA

        # transitions: shift transition input symbol to target state
        # input symbol and stack symbol stored in source state can by any symbol
        # simulate stack symbol above stack bottom symbol in state
        for (src, inp, op, stk, tar) in self.transitions:
            for isym in self.input_symbols:
                for ssym in stack_symbols_temp:
                    if op == StackOp.IGNORE:
                        # ignore: stack symbol in state unchanged
                        new_transitions.add((state_name_dict[(src, isym, ssym, 0)], StackOp.IGNORE, None, state_name_dict[(tar, inp, ssym, 0)]))
                        new_transitions.add((state_name_dict[(src, isym, ssym, 1)], StackOp.IGNORE, None, state_name_dict[(tar, inp, ssym, 1)]))
                    elif op == StackOp.PUSH:
                        # push: stack symbol in state not at the top of the stack anymore (0 in target state)
                        new_transitions.add((state_name_dict[(src, isym, ssym, 0)], StackOp.PUSH, f"{stk}_0", state_name_dict[(tar, inp, ssym, 0)]))
                        new_transitions.add((state_name_dict[(src, isym, ssym, 1)], StackOp.PUSH, f"{stk}_1", state_name_dict[(tar, inp, ssym, 0)]))
                    elif op == StackOp.POP:
                        # pop 1: pop symbol from real stack and retrieve top of stack bit
                        new_transitions.add((state_name_dict[(src, isym, ssym, 0)], StackOp.POP, f"{stk}_0", state_name_dict[(tar, inp, ssym, 0)]))
                        new_transitions.add((state_name_dict[(src, isym, ssym, 0)], StackOp.POP, f"{stk}_1", state_name_dict[(tar, inp, ssym, 1)]))
                        # pop 2: stack symbol in state is at the top of the stack, so it is removed instead
                        new_transitions.add((state_name_dict[(src, isym, stk, 1)], StackOp.IGNORE, None, state_name_dict[(tar, inp, EPS, 1)]))
                        # pop 3: bottom of the stack is reached and no stack symbol stored in state
                        new_transitions.add((state_name_dict[(src, isym, EPS, 1)], StackOp.POP, f"{stk}_1", state_name_dict[(tar, inp, EPS, 1)]))
        
        return(MPDA(new_states, new_input_symbols, new_stack_symbols, new_transitions, new_output_function, new_initial, new_final))


@typechecked
@dataclass
class MPDA:
    """Moore Pushdown automaton. Input symbols are produced at the states as
    defined by the output function (and not at the transitions).
    In a single transition, only single stack symbols can be pushed or popped
    (or the stack can be ignored).
    No epsilon-transitions are allowed.
    An accepting configuration is reached when a final state is reached while
    emptying the stack.
    Note that the empty word or words of length 1 can therefore not be accepted.
    Transitions have the form (source state, stack operation, stack symbol,
    target state). If the stack operation is IGNORE, the stack
    symbol should be None.
    Starts nondeterministically with any stack symbols at the bottom of the stack.
    """
    
    states: Set[str]
    input_symbols: Set[str]
    stack_symbols: Set[str]
    transitions: Set[MPDATransition]
    output_function: Dict[str, str]
    initial: Set[str]
    final: Set[str]


    def __post_init__(self):

        self.check_basics()


    def check_basics(self):
        """Basic check that the MPDA definition is not violated: transitions,
        output_function, initial, and final may only use valid states,
        input_symbols, and stack_symbols, respectively.
        """

        if not self.initial <= self.states:
            raise ValueError("Initial states must be contained in states");
        if not self.final <= self.states:
            raise ValueError("Final states must be contained in states");
        for (src, op, stk, tar) in self.transitions:
            if (not src in self.states) or (not tar in self.states):
                raise ValueError(f"Invalid source or target state in ({src, op, stk, tar})")
            if not stk in self.stack_symbols and not (stk is None and op == StackOp.IGNORE):
                raise ValueError(f"Invalid stack symbol in ({src, op, stk, tar})")
            if (not isinstance(op, StackOp)) or (op == StackOp.IGNORE and not (stk is None)):
                raise ValueError(f"Invalid stack operation in ({src, op, stk, tar})")
        if not self.output_function.keys() == self.states:
                raise ValueError(f"Output function does not match with state set")
        if not set(self.output_function.values()) <= self.input_symbols:
                raise ValueError(f"Output function uses invalid input symbols")
        # automaton may become empty as a result of the transformation
        # therefore sets are not checked for emptiness


    def to_dict(self) -> Dict[str, Any]:
        """Returns a dict based on the MPDA instance."""

        output_dict: Dict[str, Any] = dict()

        output_dict["type"] = "MPDA"

        for field in fields(MPDA):
            if field.name == "transitions":
                prepared_transitions = []
                for (src, op, stk, tar) in getattr(self, field.name):
                    prepared_transitions.append((src, op.value, stk, tar))
                output_dict[field.name] = prepared_transitions

            elif field.name == "output_function":
                output_dict[field.name] = getattr(self, field.name)
                
            else:
                output_dict[field.name] = list(getattr(self, field.name))

        return output_dict


    def trim(self):
        """Removes unused states and transitions. Reachability from an initial
        state and to a final state is both required for a state/transition to be
        useful. Also updates the output function by restricting it to remaining
        states.
        """

        forward_adjacency: Dict[str, Set[str]] = dict()
        backward_adjacency: Dict[str, Set[str]] = dict()

        # build dicts for forward/backward adjacency
        for (src, op, stk, tar) in self.transitions:
            forward_adjacency.setdefault(src, set()).add(tar)
            backward_adjacency.setdefault(tar, set()).add(src)

        # states reachable from initial/final states
        forward_reachable: Set[str] = get_reachable_states(self.initial, forward_adjacency)
        backward_reachable: Set[str]  = get_reachable_states(self.final, backward_adjacency)

        useful_states: Set[str] = set.intersection(forward_reachable, backward_reachable)

        # useful transitions
        useful_transitions: Set[MPDATransition] = set()
        for (src, op, stk, tar) in self.transitions:
            if src in useful_states and tar in useful_states:
                useful_transitions.add((src, op, stk, tar))

        useful_output_function: Dict[str, str] = {state: self.output_function[state] for state in useful_states}

        self.states = useful_states
        self.transitions = useful_transitions
        self.output_function = useful_output_function


@typechecked
def get_reachable_states(starting_set: Set[str], adjacency_dict: Dict[str, Set[str]]) -> Set[str]:
    """Given a set of starting nodes and a dict with the successors of
    each node in a graph, the set of nodes reachable from the starting
    nodes is computed. (Traversal in no specific order.)"""

    visited: Set[str] = set()
    reachable: Set[str] = starting_set.copy()
    visit_next: Set[str] = starting_set.copy()

    while len(visit_next) != 0:
        current_node: str = visit_next.pop()
        children: set[str] = adjacency_dict.get(current_node, set())
        for child in children:
            reachable.add(child)
            if child not in visited:
                visit_next.add(child)
        visited.add(current_node)

    return reachable


@typechecked
def show_automaton(automaton: PDA | MPDA, filename: str):
    """Renders the given automaton (PDA or MPDA) using graphviz and stores it as
    a png file under the given file name.
    """

    graph: Digraph = Digraph()

    if isinstance(automaton, PDA):
        for state in automaton.states:
            if state in automaton.initial:
                graph.node(state, state)
                graph.node(f"__start__{state}", shape='point')
                graph.edge(f"__start__{state}", state)
            elif state in automaton.final:
                graph.node(state, state, shape='doublecircle')
            else:
                graph.node(state, state)
        for (src, inp, op, stk, tar) in automaton.transitions:
            pretty_stk: str = '' if stk is None else stk.replace(BOTTOM, '⊥')
            graph.edge(src, tar, f"{inp} {op.name} {pretty_stk}")
    elif isinstance(automaton, MPDA):
        for state in automaton.states:
            output_symbol: str = automaton.output_function[state]
            pretty_state: str = state.replace(BOTTOM, '⊥').replace(EPS, 'ε')
            state_label: str = f"{pretty_state} | {output_symbol}"
            if state in automaton.initial:
                graph.node(state, state_label)
                graph.node(f"__start__{state}", shape='point')
                graph.edge(f"__start__{state}", state)
            elif state in automaton.final:
                graph.node(state, state_label, shape='doublecircle')
            else:
                graph.node(state, state_label)
        for (src, op, stk, tar) in automaton.transitions:
            pretty_stk: str = '' if stk is None else stk.replace(BOTTOM, '⊥')
            graph.edge(src, tar, f"{op.name} {pretty_stk}")

    graph.render(filename, view=True, format='png')




if __name__ == '__main__':

    input_file = "input_pda_3.json"
    output_file = "output_mpda.json"
    
    with open(input_file, "r") as file:
        pda_dict = json.load(file)
    input_pda = PDA.from_dict(pda_dict)

    #show_automaton(input_pda, 'show_input_pda_before_removal')

    length_one_words, transitions = input_pda.get_length_one_words()
    print(length_one_words)
    input_pda.remove_transitions(transitions)

    output_mpda = input_pda.convert_to_MPDA()
    #show_automaton(output_mpda, 'show_output_mpda_before_trim')
    output_mpda.trim()

    show_automaton(input_pda, 'show_input_pda')
    show_automaton(output_mpda, 'show_output_mpda')
    
    mpda_dict = output_mpda.to_dict()
    print(mpda_dict)

    with open(output_file, "w") as file:
        json.dump(mpda_dict, file, indent=4)