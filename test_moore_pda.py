import unittest
import json
from pathlib import Path

from moore_pda import PDA
from moore_pda import MPDA
from moore_pda import get_reachable_states

from moore_pda import StackOp
from moore_pda import BOTTOM
from moore_pda import EPS
from moore_pda import PDATransition
from moore_pda import MPDATransition

##### PDA

class TestInputPDA(unittest.TestCase):

    def setUp(self):
        self.top_dir = Path(__file__).parent
        self.pda_dir = self.top_dir / "test_pdas"

    # helper function to call in tests
    def check_file_name_value_error(self, file_name: Path, error_message: str):
        with self.assertRaises(ValueError) as context:
            with open(file_name, "r") as file:
                pda_dict = json.load(file)
            input_pda: PDA = PDA.from_dict(pda_dict)
        self.assertEqual(str(context.exception), error_message)

    # basic checks
    def test_unknown_initial_state(self):
        self.check_file_name_value_error(self.pda_dir / "pda_unknown_initial_state.json",
            "Initial states must be contained in states")

    def test_unknown_final_state(self):
        self.check_file_name_value_error(self.pda_dir / "pda_unknown_final_state.json",
            "Final states must be contained in states")

    def test_bottom_symbol_missing(self):
        self.check_file_name_value_error(self.pda_dir / "pda_bottom_symbol_missing.json",
            "__bottom_symbol__ must be in stack symbols")

    def test_invalid_transition(self):
        self.check_file_name_value_error(self.pda_dir / "pda_invalid_transition1.json",
            "Invalid source or target state in ('x', 'a', <StackOp.PUSH: '__push__'>, 'z', 'q')")
        self.check_file_name_value_error(self.pda_dir / "pda_invalid_transition2.json",
            "Invalid source or target state in ('q', 'a', <StackOp.PUSH: '__push__'>, 'z', 'x')")

    def test_invalid_stack_symbol(self):
        self.check_file_name_value_error(self.pda_dir / "pda_invalid_stack_symbol1.json",
            "Invalid stack symbol in ('q', 'a', <StackOp.PUSH: '__push__'>, 'x', 'q')")
        self.check_file_name_value_error(self.pda_dir / "pda_invalid_stack_symbol2.json",
            "Invalid stack symbol in ('q', 'a', <StackOp.PUSH: '__push__'>, None, 'q')")

    def test_invalid_stack_operation(self):
        self.check_file_name_value_error(self.pda_dir / "pda_invalid_stack_operation1.json",
            "'__unknown__' is not a valid StackOp")
        self.check_file_name_value_error(self.pda_dir / "pda_invalid_stack_operation2.json",
            "Invalid stack operation in ('p', 'a', <StackOp.IGNORE: '__ignore__'>, 'z', 'q')")

    def test_empty_set(self):
        self.check_file_name_value_error(self.pda_dir / "pda_empty_set1.json",
            "Sets of states, initial states, final states, input symbols must not be empty")
        self.check_file_name_value_error(self.pda_dir / "pda_empty_set2.json",
            "Sets of states, initial states, final states, input symbols must not be empty")
        self.check_file_name_value_error(self.pda_dir / "pda_empty_set3.json",
            "Sets of states, initial states, final states, input symbols must not be empty")
        self.check_file_name_value_error(self.pda_dir / "pda_empty_set4.json",
            "Sets of states, initial states, final states, input symbols must not be empty")

    def test_wrong_automaton_type(self):
        self.check_file_name_value_error(self.pda_dir / "pda_wrong_automaton_type.json",
            "Input automaton type not matching PDA")

    def test_wrong_transition_length(self):
        self.check_file_name_value_error(self.pda_dir / "pda_wrong_transition_length1.json",
            "PDATransition ['p', '__ignore__', None, 'q'] needs to have 5 elements")
        self.check_file_name_value_error(self.pda_dir / "pda_wrong_transition_length2.json",
            "PDATransition ['p', 'a', '__ignore__', None, 'q', 'a'] needs to have 5 elements")

    def test_special_symbols(self):
        self.check_file_name_value_error(self.pda_dir / "pda_special_symbols1.json",
            "Special symbol __epsilon__ not allowed as state or stack symbol or input symbol")
        self.check_file_name_value_error(self.pda_dir / "pda_special_symbols2.json",
            "Special symbol __bottom_symbol__ not allowed as state or stack symbol or input symbol")
        self.check_file_name_value_error(self.pda_dir / "pda_special_symbols3.json",
            "Special symbol __push__ not allowed as state or stack symbol or input symbol")


    # other checks
    def test_initial_states_ingoing(self):
        self.check_file_name_value_error(self.pda_dir / "pda_initial_states_ingoing.json",
            "Initial state p must not have ingoing transitions: ('p', 'a', <StackOp.IGNORE: '__ignore__'>, None, 'p')")

    def test_final_states_outgoing(self):
        self.check_file_name_value_error(self.pda_dir / "pda_final_states_outgoing.json",
            "Final state f must not have outgoing transitions: ('f', 'a', <StackOp.IGNORE: '__ignore__'>, None, 'q')")

    def test_transitions_final_states(self):
        self.check_file_name_value_error(self.pda_dir / "pda_transitions_final_states1.json",
            "Transition to final state f must pop bottom symbol: ('q', 'b', <StackOp.POP: '__pop__'>, 'z', 'f')")
        self.check_file_name_value_error(self.pda_dir / "pda_transitions_final_states2.json",
            "Transition to final state f must pop bottom symbol: ('p', 'a', <StackOp.IGNORE: '__ignore__'>, None, 'f')")
        self.check_file_name_value_error(self.pda_dir / "pda_transitions_final_states3.json",
            "No transition must push bottom symbol: ('q', 'a', <StackOp.PUSH: '__push__'>, '__bottom_symbol__', 'q')")


    # valid input
    def test_valid_pda(self):
        with open(self.pda_dir / "pda_valid.json", "r") as file:
            pda_dict = json.load(file)
        input_pda: PDA = PDA.from_dict(pda_dict)

        self.assertEqual(input_pda.states, {"p", "q", "f"})
        self.assertEqual(input_pda.input_symbols, {"a", "b", "c"})
        self.assertEqual(input_pda.stack_symbols, {"__bottom_symbol__", "z"})
        self.assertEqual(input_pda.initial, {"p"})
        self.assertEqual(input_pda.final, {"f"})

        expected_transitions: List[PDATransition] = {
            ("p", "a", StackOp.IGNORE, None, "q"),
            ("p", "c", StackOp.POP, "__bottom_symbol__", "f"),
            ("q", "a", StackOp.PUSH, "z", "q"),
            ("q", "b", StackOp.POP, "z", "q"),
            ("q", "b", StackOp.POP, "__bottom_symbol__", "f")}

        self.assertEqual(input_pda.transitions, expected_transitions, msg=f"Expected {expected_transitions}, but got {input_pda.transitions}")



class TestPDAGetLengthOne(unittest.TestCase):

    def setUp(self):
        self.top_dir = Path(__file__).parent
        self.pda_dir = self.top_dir / "test_pdas"

    def test_has_one_length_one(self):
        with open(self.pda_dir / "pda_valid.json", "r") as file:
            pda_dict = json.load(file)
        input_pda: PDA = PDA.from_dict(pda_dict)
        words: Set[str]
        transitions: Set[PDATransition]
        words, transitions = input_pda.get_length_one_words()
        self.assertEqual(words, { "c" })
        self.assertEqual(transitions, { ("p", "c", StackOp.POP, "__bottom_symbol__", "f") })

    def test_has_two_length_one(self):
        with open(self.pda_dir / "pda_valid_two_length_one.json", "r") as file:
            pda_dict = json.load(file)
        input_pda: PDA = PDA.from_dict(pda_dict)
        words: Set[str]
        transitions: Set[PDATransition]
        words, transitions = input_pda.get_length_one_words()
        self.assertEqual(words, { "c", "a" })
        self.assertEqual(transitions, { ("p", "c", StackOp.POP, "__bottom_symbol__", "f"), ("x", "a", StackOp.POP, "__bottom_symbol__", "g") })

    def test_has_no_length_one(self):
        with open(self.pda_dir / "pda_valid_no_length_one.json", "r") as file:
            pda_dict = json.load(file)
        input_pda: PDA = PDA.from_dict(pda_dict)
        words: Set[str]
        transitions: Set[PDATransition]
        words, transitions = input_pda.get_length_one_words()
        self.assertEqual(words, set())
        self.assertEqual(transitions, set())



class TestPDARemoveTransitions(unittest.TestCase):

    def setUp(self):
        self.top_dir = Path(__file__).parent
        self.pda_dir = self.top_dir / "test_pdas"

        with open(self.pda_dir / "pda_valid.json", "r") as file:
            pda_dict = json.load(file)
        self.input_pda: PDA = PDA.from_dict(pda_dict)

    def test_remove_no_transitions(self):
        self.input_pda.remove_transitions(set())

        self.assertEqual(self.input_pda.states, {"p", "q", "f"})
        self.assertEqual(self.input_pda.input_symbols, {"a", "b", "c"})
        self.assertEqual(self.input_pda.stack_symbols, {"__bottom_symbol__", "z"})
        self.assertEqual(self.input_pda.initial, {"p"})
        self.assertEqual(self.input_pda.final, {"f"})

        expected_transitions: List[PDATransition] = {
            ("p", "a", StackOp.IGNORE, None, "q"),
            ("p", "c", StackOp.POP, "__bottom_symbol__", "f"),
            ("q", "a", StackOp.PUSH, "z", "q"),
            ("q", "b", StackOp.POP, "z", "q"),
            ("q", "b", StackOp.POP, "__bottom_symbol__", "f")}

        self.assertEqual(self.input_pda.transitions, expected_transitions)

    def test_remove_two_transitions(self):
        self.input_pda.remove_transitions({("q", "b", StackOp.POP, "z", "q"),
            ("q", "b", StackOp.POP, "__bottom_symbol__", "f")})

        self.assertEqual(self.input_pda.states, {"p", "q", "f"})
        self.assertEqual(self.input_pda.input_symbols, {"a", "b", "c"})
        self.assertEqual(self.input_pda.stack_symbols, {"__bottom_symbol__", "z"})
        self.assertEqual(self.input_pda.initial, {"p"})
        self.assertEqual(self.input_pda.final, {"f"})

        expected_transitions: List[PDATransition] = {
            ("p", "a", StackOp.IGNORE, None, "q"),
            ("p", "c", StackOp.POP, "__bottom_symbol__", "f"),
            ("q", "a", StackOp.PUSH, "z", "q")}

        self.assertEqual(self.input_pda.transitions, expected_transitions)

    def test_remove_nonexistent_transition(self):
        with self.assertRaises(KeyError) as context:
            self.input_pda.remove_transitions({("p", "a", StackOp.IGNORE, None, "x")})



class TestPDAConvertMPDA(unittest.TestCase):
    """These tests check that the outputs of convert_to_MPDA are consistent with
    the specification. Better would be to check that the two automata
    have the same behavior, which would require parsing inputs with them."""

    def setUp(self):
        self.top_dir = Path(__file__).parent
        self.pda_dir = self.top_dir / "test_pdas"

    def test_first_transition_ignore(self):

        with open(self.pda_dir / "pda_valid_test_conversion.json", "r") as file:
            pda_dict = json.load(file)
        input_pda: PDA = PDA.from_dict(pda_dict)
        output_mpda: MPDA = input_pda.convert_to_MPDA()

        self.assertEqual(output_mpda.input_symbols, {"b", "a"})
        self.assertEqual(output_mpda.stack_symbols, {"z_1", "z_0", "__bottom_symbol___0", "__bottom_symbol___1"})

        states_no_index: Set[str] = set(map(lambda s: s.split('-')[0], output_mpda.states))
        self.assertEqual(len(states_no_index), len(output_mpda.states))
        expected_states: Set[str] = {f"{q}_{i}_{s}_{b}"
            for q in {"q", "p", "f"}
            for i in {"a", "b"}
            for s in {"z", "__bottom_symbol__", "__epsilon__"}
            for b in {"0", "1"}}

        self.assertEqual(states_no_index, expected_states)

        for state in output_mpda.states:
            self.assertEqual(output_mpda.output_function[state], state[2])

        final_no_index: Set[str] = set(map(lambda s: s.split('-')[0], output_mpda.final))
        initial_no_index: Set[str] = set(map(lambda s: s.split('-')[0], output_mpda.initial))

        self.assertEqual(len(final_no_index), len(output_mpda.final))
        self.assertEqual(len(initial_no_index), len(output_mpda.initial))

        self.assertEqual(final_no_index, {"f_b___epsilon___1", "f_a___epsilon___1"})

        self.assertEqual(initial_no_index, {"q_a___epsilon___1"})

        transitions_no_index: Set[MPDATransition] = set(map(lambda t: 
            (t[0].split('-')[0], t[1], t[2], t[3].split('-')[0]), output_mpda.transitions))

        self.assertEqual(len(transitions_no_index), len(output_mpda.transitions))

        expected_ignore: set[MPDATransition] = {(f"p_{i}_{s}_{b}", StackOp.IGNORE, None, f"q_a_{s}_{b}")
            for i in {"a", "b"} for s in {"z", "__epsilon__", "__bottom_symbol__"} for b in {"0", "1"}}
        self.assertLessEqual(expected_ignore, transitions_no_index)

        expected_push: set[MPDATransition] = {(f"q_{i}_{s}_{b}", StackOp.PUSH, f"z_{b}", f"q_a_{s}_0")
            for i in {"a", "b"} for s in {"z", "__epsilon__", "__bottom_symbol__"} for b in {"0", "1"}}
        self.assertLessEqual(expected_push, transitions_no_index)

        expected_pop1: set[MPDATransition] = set()
        expected_pop1 = expected_pop1.union({(f"q_{i}_{s}_0", StackOp.POP, f"z_{b}", f"q_b_{s}_{b}")
            for i in {"a", "b"} for s in {"z", "__epsilon__", "__bottom_symbol__"} for b in {"0", "1"}})
        expected_pop1 = expected_pop1.union({(f"q_{i}_z_1", StackOp.IGNORE, None, f"q_b___epsilon___1") for i in {"a", "b"}})
        expected_pop1 = expected_pop1.union({(f"q_{i}___epsilon___1", StackOp.POP, "z_1", f"q_b___epsilon___1") for i in {"a", "b"}})
        self.assertLessEqual(expected_pop1, transitions_no_index)

        expected_pop2: set[MPDATransition] = set()
        expected_pop2 = expected_pop2.union({(f"q_{i}_{s}_0", StackOp.POP, f"__bottom_symbol___{b}", f"f_b_{s}_{b}")
            for i in {"a", "b"} for s in {"z", "__epsilon__", "__bottom_symbol__"} for b in {"0", "1"}})
        expected_pop2 = expected_pop2.union({(f"q_{i}___bottom_symbol___1", StackOp.IGNORE, None, f"f_b___epsilon___1") for i in {"a", "b"}})
        expected_pop2 = expected_pop2.union({(f"q_{i}___epsilon___1", StackOp.POP, "__bottom_symbol___1", f"f_b___epsilon___1") for i in {"a", "b"}})
        self.assertLessEqual(expected_pop2, transitions_no_index)

        all_expected_transitions: set[MPDATransition] = set.union(expected_ignore, expected_push, expected_pop1, expected_pop2)
        set_diff = set.difference(transitions_no_index, all_expected_transitions)

        self.assertEqual(len(transitions_no_index),
            len(expected_ignore) + len(expected_push) + len(expected_pop1) + len(expected_pop2), msg=f"Transition Set Difference: {set_diff}")


    def test_first_transition_push(self):

        with open(self.pda_dir / "pda_valid_first_push.json", "r") as file:
            pda_dict = json.load(file)
        input_pda: PDA = PDA.from_dict(pda_dict)
        output_mpda: MPDA = input_pda.convert_to_MPDA()

        initial_no_index: Set[str] = set(map(lambda s: s.split('-')[0], output_mpda.initial))
        expected_initial: set[MPDATransition] = {"q_a_z_1"}
        self.assertEqual(initial_no_index, expected_initial)


    def test_first_transition_pop(self):

        with open(self.pda_dir / "pda_valid_first_pop.json", "r") as file:
            pda_dict = json.load(file)
        input_pda: PDA = PDA.from_dict(pda_dict)
        output_mpda: MPDA = input_pda.convert_to_MPDA()

        self.assertEqual(output_mpda.initial, set())


    def test_name_clash(self):

        with open(self.pda_dir / "pda_valid_name_clash.json", "r") as file:
            pda_dict = json.load(file)
        input_pda: PDA = PDA.from_dict(pda_dict)
        output_mpda: MPDA = input_pda.convert_to_MPDA()
        
        self.assertEqual(len(output_mpda.states), 3*2*4*2)
        self.assertEqual(len(output_mpda.initial), 2)
        initial_list: list[str] = list(output_mpda.initial)
        first_output: str = output_mpda.output_function[initial_list[0]]
        second_output: str = output_mpda.output_function[initial_list[1]]
        self.assertEqual(set([first_output, second_output]), set(["a", "a_x"]))
        self.assertEqual(initial_list[0].split('-')[0], 'q_a_x_b_1')
        self.assertEqual(initial_list[1].split('-')[0], 'q_a_x_b_1')

        target_states: list[str] = list()
        for (src, op, stk, tar) in output_mpda.transitions:
            if src == initial_list[0] or src == initial_list[1]:
                if op == StackOp.IGNORE and stk is None:
                    target_states.append(tar)

        self.assertEqual(len(target_states), 2)
        self.assertEqual(target_states[0], target_states[1])
        self.assertEqual(target_states[0].split('-')[0], "q_a___epsilon___1")

         

##### MPDA

class TestMPDACheckBasics(unittest.TestCase):
    pass

class TestMPDAToDict(unittest.TestCase):
    pass

class TestMPDATrim(unittest.TestCase):
    pass


##### Other

class TestGetReachableStates(unittest.TestCase):
    pass



if __name__ == '__main__':
    unittest.main()