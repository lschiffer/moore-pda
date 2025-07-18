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
    pass

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