import brain
import brain_util as bu
import numpy as np
import pptree
import json
import copy
import generate_sentences

from collections import namedtuple
from collections import defaultdict
from enum import Enum

import argparse

# BrainAreas
LEX = "LEX"
DET = "DET"
SUBJ = "SUBJ"
OBJ = "OBJ"
VERB = "VERB"
PREP = "PREP"
PREP_P = "PREP_P"
ADJ = "ADJ"
ADVERB = "ADVERB"

COMP1 = "COMP1"
COMP2 = "COMP2"

# Unique to Russian
NOM = "NOM"
ACC = "ACC"
DAT = "DAT"

# Fixed area stats for explicit areas
LEX_SIZE = 56
DET_SIZE = 2

# Actions
DISINHIBIT = "DISINHIBIT"
INHIBIT = "INHIBIT"
# Skip firing in this round, just activate the word in LEX/DET/other word areas.
# All other rules for these lexical items should be in PRE_RULES.
ACTIVATE_ONLY = "ACTIVATE_ONLY"
CLEAR_DET = "CLEAR_DET"

AREAS = [LEX, DET, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P, COMP1, COMP2]
EXPLICIT_AREAS = [LEX]
RECURRENT_AREAS = [SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P, COMP1, COMP2]

RUSSIAN_AREAS = [LEX, NOM, VERB, ACC, DAT]
RUSSIAN_EXPLICIT_AREAS = [LEX]
RUSSIAN_LEX_SIZE = 7

AreaRule = namedtuple('AreaRule', ['action', 'area', 'index'])
FiberRule = namedtuple('FiberRule', ['action', 'area1', 'area2', 'index'])
FiringRule = namedtuple('FiringRule', ['action'])
OtherRule = namedtuple('OtherRule', ['action'])


def generic_noun(index):
    return {
        "index": index,
        "type": "noun",
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, SUBJ, 0),
            FiberRule(DISINHIBIT, LEX, OBJ, 0),
            FiberRule(DISINHIBIT, LEX, PREP_P, 0),
            FiberRule(DISINHIBIT, DET, SUBJ, 0),
            FiberRule(DISINHIBIT, DET, OBJ, 0),
            FiberRule(DISINHIBIT, DET, PREP_P, 0),
            FiberRule(DISINHIBIT, ADJ, SUBJ, 0),
            FiberRule(DISINHIBIT, ADJ, OBJ, 0),
            FiberRule(DISINHIBIT, ADJ, PREP_P, 0),
            FiberRule(DISINHIBIT, VERB, OBJ, 0),
            FiberRule(DISINHIBIT, PREP_P, PREP, 0),
            FiberRule(DISINHIBIT, PREP_P, SUBJ, 0),
            FiberRule(DISINHIBIT, PREP_P, OBJ, 0),
        ],
        "POST_RULES": [
            AreaRule(INHIBIT, DET, 0),
            AreaRule(INHIBIT, ADJ, 0),
            AreaRule(INHIBIT, PREP_P, 0),
            AreaRule(INHIBIT, PREP, 0),
            FiberRule(INHIBIT, LEX, SUBJ, 0),
            FiberRule(INHIBIT, LEX, OBJ, 0),
            FiberRule(INHIBIT, LEX, PREP_P, 0),
            FiberRule(INHIBIT, ADJ, SUBJ, 0),
            FiberRule(INHIBIT, ADJ, OBJ, 0),
            FiberRule(INHIBIT, ADJ, PREP_P, 0),
            FiberRule(INHIBIT, DET, SUBJ, 0),
            FiberRule(INHIBIT, DET, OBJ, 0),
            FiberRule(INHIBIT, DET, PREP_P, 0),
            FiberRule(INHIBIT, VERB, OBJ, 0),
            FiberRule(INHIBIT, PREP_P, PREP, 0),
            FiberRule(INHIBIT, PREP_P, VERB, 0),
            FiberRule(DISINHIBIT, LEX, SUBJ, 1),
            FiberRule(DISINHIBIT, LEX, OBJ, 1),
            FiberRule(DISINHIBIT, DET, SUBJ, 1),
            FiberRule(DISINHIBIT, DET, OBJ, 1),
            FiberRule(DISINHIBIT, ADJ, SUBJ, 1),
            FiberRule(DISINHIBIT, ADJ, OBJ, 1),
            FiberRule(INHIBIT, PREP_P, SUBJ, 0),
            FiberRule(INHIBIT, PREP_P, OBJ, 0),
            FiberRule(INHIBIT, VERB, ADJ, 0),
        ]
    }


def generic_trans_verb(index):
    return {
        "index": index,
        "type": "trans_verb",
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, SUBJ, 0),
            FiberRule(DISINHIBIT, VERB, ADVERB, 0),
            AreaRule(DISINHIBIT, ADVERB, 1),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, VERB, 0),
            AreaRule(DISINHIBIT, OBJ, 0),
            AreaRule(INHIBIT, SUBJ, 0),
            AreaRule(INHIBIT, ADVERB, 0),
            FiberRule(DISINHIBIT, PREP_P, VERB, 0),
        ]
    }


def generic_intrans_verb(index):
    return {
        "index": index,
        "type": "intrans_verb",
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, SUBJ, 0),
            FiberRule(DISINHIBIT, VERB, ADVERB, 0),
            AreaRule(DISINHIBIT, ADVERB, 1),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, VERB, 0),
            AreaRule(INHIBIT, SUBJ, 0),
            AreaRule(INHIBIT, ADVERB, 0),
            FiberRule(DISINHIBIT, PREP_P, VERB, 0),
        ]
    }


def generic_copula(index):
    return {
        "index": index,
        "type": "copula",
        "PRE_RULES": [
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, SUBJ, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, VERB, 0),
            AreaRule(DISINHIBIT, OBJ, 0),
            AreaRule(INHIBIT, SUBJ, 0),
            FiberRule(DISINHIBIT, ADJ, VERB, 0)
        ]
    }


def generic_adverb(index):
    return {
        "index": index,
        "type": "adverb",
        "PRE_RULES": [
            AreaRule(DISINHIBIT, ADVERB, 0),
            FiberRule(DISINHIBIT, LEX, ADVERB, 0)
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, ADVERB, 0),
            AreaRule(INHIBIT, ADVERB, 1),
        ]

    }


def generic_determinant(index):
    return {
        "index": index,
        "type": "determinant",
        "PRE_RULES": [
            AreaRule(DISINHIBIT, DET, 0),
            FiberRule(DISINHIBIT, LEX, DET, 0)
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, DET, 0),
            FiberRule(INHIBIT, VERB, ADJ, 0),
        ]
    }


def generic_adjective(index):
    return {
        "index": index,
        "type": "adjective",
        "PRE_RULES": [
            AreaRule(DISINHIBIT, ADJ, 0),
            FiberRule(DISINHIBIT, LEX, ADJ, 0)
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, ADJ, 0),
            FiberRule(INHIBIT, VERB, ADJ, 0),
        ]

    }


def generic_preposition(index):
    return {
        "index": index,
        "type": "generic_preposition",
        "PRE_RULES": [
            AreaRule(DISINHIBIT, PREP, 0),
            FiberRule(DISINHIBIT, LEX, PREP, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, PREP, 0),
            AreaRule(DISINHIBIT, PREP_P, 0),
            FiberRule(INHIBIT, LEX, SUBJ, 1),
            FiberRule(INHIBIT, LEX, OBJ, 1),
            FiberRule(INHIBIT, DET, SUBJ, 1),
            FiberRule(INHIBIT, DET, OBJ, 1),
            FiberRule(INHIBIT, ADJ, SUBJ, 1),
            FiberRule(INHIBIT, ADJ, OBJ, 1),
        ]
    }


LEXEME_DICT = {
    "the": generic_determinant(0),
    "a": generic_determinant(1),
    "dogs": generic_noun(2),
    "cats": generic_noun(3),
    "mice": generic_noun(4),
    "city": generic_noun(5),
    "chase": generic_trans_verb(6),
    "love": generic_trans_verb(7),
    "bite": generic_trans_verb(8),
    "of": generic_preposition(9),
    "big": generic_adjective(10),
    "bad": generic_adjective(11),
    "run": generic_intrans_verb(12),
    "fly": generic_intrans_verb(13),
    "quickly": generic_adverb(14),
    "in": generic_preposition(15),
    "are": generic_copula(16),
    "man": generic_noun(17),
    "woman": generic_noun(18),
    "saw": generic_trans_verb(19),
    "small": generic_adjective(20),
    "good": generic_adjective(21),
    "fat": generic_adjective(22),
    "useful": generic_adjective(23),
    "beautiful": generic_adjective(24),
    "impressive": generic_adjective(25),
    "intense": generic_adjective(26),
    "confident": generic_adjective(27),
    "naive": generic_adjective(28),
    "classic": generic_adjective(29),
    "healthy": generic_adjective(30),
    "kind": generic_adjective(31),
    "warm-hearted": generic_adjective(32),
    "humble": generic_adjective(33),
    "fierce": generic_adjective(34),
    "energetic": generic_adjective(35),
    "friendly": generic_adjective(36),
    "dependable": generic_adjective(37),
    "sheep": generic_noun(38),
    "fish": generic_noun(39),
    "robot": generic_noun(40),
    "child": generic_noun(41),
    "bird": generic_noun(42),
    "eat": generic_trans_verb(43),
    "hit": generic_trans_verb(44),
    "oppress": generic_trans_verb(45),
    "leave": generic_trans_verb(46),
    "touch": generic_trans_verb(47),
    "rapidly": generic_adverb(48),
    "faithfully": generic_adverb(49),
    "frankly": generic_adverb(50),
    "gently": generic_adverb(51),
    "honestly": generic_adverb(52),
    "embrace": generic_trans_verb(53),
    "help": generic_trans_verb(54),
    "scientist": generic_noun(55)
}


def generic_russian_verb(index):
    return {
        "area": LEX,
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, VERB, 0),
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, NOM, 0),
            FiberRule(DISINHIBIT, VERB, ACC, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, VERB, 0)
        ]
    }


def generic_russian_ditransitive_verb(index):
    return {
        "area": LEX,
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, VERB, 0),
            FiberRule(DISINHIBIT, LEX, VERB, 0),
            FiberRule(DISINHIBIT, VERB, NOM, 0),
            FiberRule(DISINHIBIT, VERB, ACC, 0),
            FiberRule(DISINHIBIT, VERB, DAT, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, VERB, 0)
        ]
    }


def generic_russian_nominative_noun(index):
    return {
        "area": LEX,
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, NOM, 0),
            FiberRule(DISINHIBIT, LEX, NOM, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, NOM, 0)
        ]
    }


def generic_russian_accusative_noun(index):
    return {
        "area": LEX,
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, ACC, 0),
            FiberRule(DISINHIBIT, LEX, ACC, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, ACC, 0)
        ]
    }


def generic_russian_dative_noun(index):
    return {
        "area": LEX,
        "index": index,
        "PRE_RULES": [
            AreaRule(DISINHIBIT, DAT, 0),
            FiberRule(DISINHIBIT, LEX, DAT, 0),
        ],
        "POST_RULES": [
            FiberRule(INHIBIT, LEX, DAT, 0)
        ]
    }


RUSSIAN_LEXEME_DICT = {
    "vidit": generic_russian_verb(0),
    "lyubit": generic_russian_verb(1),
    "kot": generic_russian_nominative_noun(2),
    "kota": generic_russian_accusative_noun(2),
    "sobaka": generic_russian_nominative_noun(3),
    "sobaku": generic_russian_accusative_noun(3),
    "sobakie": generic_russian_dative_noun(3),
    "kotu": generic_russian_dative_noun(2),
    "dayet": generic_russian_ditransitive_verb(4)
}

ENGLISH_READOUT_RULES = {
    VERB: [LEX, SUBJ, OBJ, PREP_P, ADVERB, ADJ],
    SUBJ: [LEX, DET, ADJ, PREP_P, COMP1],
    OBJ: [LEX, DET, ADJ, PREP_P, COMP1],
    PREP_P: [LEX, PREP, ADJ, DET],
    PREP: [LEX],
    ADJ: [LEX, COMP1],
    DET: [LEX],
    ADVERB: [LEX, COMP1],
    LEX: [],
    COMP1: [LEX, COMP2],
    COMP2: [LEX, COMP1]
}

RUSSIAN_READOUT_RULES = {
    VERB: [LEX, NOM, ACC, DAT],
    NOM: [LEX],
    ACC: [LEX],
    DAT: [LEX],
    LEX: [],
}


class ParserBrain(brain.Brain):
    def __init__(self, p, lexeme_dict={}, all_areas=[], recurrent_areas=[], initial_areas=[], readout_rules={}):
        brain.Brain.__init__(self, p)
        self.lexeme_dict = lexeme_dict
        self.all_areas = all_areas
        self.recurrent_areas = recurrent_areas
        self.initial_areas = initial_areas

        self.fiber_states = defaultdict()
        self.area_states = defaultdict(set)
        self.activated_fibers = defaultdict(set)
        self.readout_rules = readout_rules
        self.initialize_states()

    def initialize_states(self):
        for from_area in self.all_areas:
            self.fiber_states[from_area] = defaultdict(set)
            for to_area in self.all_areas:
                self.fiber_states[from_area][to_area].add(0)

        for area in self.all_areas:
            self.area_states[area].add(0)

        for area in self.initial_areas:
            self.area_states[area].discard(0)

    def applyFiberRule(self, rule):
        if rule.action == INHIBIT:
            self.fiber_states[rule.area1][rule.area2].add(rule.index)
            self.fiber_states[rule.area2][rule.area1].add(rule.index)
        elif rule.action == DISINHIBIT:
            self.fiber_states[rule.area1][rule.area2].discard(rule.index)
            self.fiber_states[rule.area2][rule.area1].discard(rule.index)

    def applyAreaRule(self, rule):
        if rule.action == INHIBIT:
            self.area_states[rule.area].add(rule.index)
        elif rule.action == DISINHIBIT:
            self.area_states[rule.area].discard(rule.index)

    def applyRule(self, rule):
        if isinstance(rule, FiberRule):
            self.applyFiberRule(rule)
            return True
        if isinstance(rule, AreaRule):
            self.applyAreaRule(rule)
            return True
        return False

    def parse_project(self):
        project_map = self.getProjectMap()
        self.remember_fibers(project_map)
        self.project({}, project_map)

    def parse_project_undirection(self, project_map):
        self.remember_fibers(project_map)
        self.project({}, project_map)

    # For fiber-activation readout, remember all fibers that were ever fired.
    def remember_fibers(self, project_map):
        for from_area, to_areas in project_map.items():
            self.activated_fibers[from_area].update(to_areas)

    def recurrent(self, area):
        return (area in self.recurrent_areas)

    # TODO: Remove brain from ProjectMap somehow
    # perhaps replace Parser state with ParserBrain:Brain, better design
    def getProjectMap(self):
        proj_map = defaultdict(set)
        for area1 in self.all_areas:
            if len(self.area_states[area1]) == 0:
                for area2 in self.all_areas:
                    if area1 == LEX and area2 == LEX:
                        continue
                    if len(self.area_states[area2]) == 0:
                        if len(self.fiber_states[area1][area2]) == 0:
                            if self.areas[area1].winners:
                                proj_map[area1].add(area2)
                            if self.areas[area2].winners:
                                proj_map[area2].add(area2)
        # print(proj_map)
        return proj_map

    def activateWord(self, area_name, word):
        area = self.areas[area_name]
        k = area.k
        assembly_start = self.lexeme_dict[word]["index"] * k
        area.winners = list(range(assembly_start, assembly_start + k))
        area.fix_assembly()

    def activateIndex(self, area_name, index):
        area = self.areas[area_name]
        k = area.k
        assembly_start = index * k
        area.winners = list(range(assembly_start, assembly_start + k))
        area.fix_assembly()

    def interpretAssemblyAsString(self, area_name):
        return self.getWord(area_name, 0.7)

    def getWord(self, area_name, min_overlap=0.7):
        if not self.areas[area_name].winners:
            raise Exception(
                "Cannot get word because no assembly in " + area_name)
        winners = set(self.areas[area_name].winners)
        area_k = self.areas[area_name].k
        threshold = min_overlap * area_k
        for word, lexeme in self.lexeme_dict.items():
            word_index = lexeme["index"]
            word_assembly_start = word_index * area_k
            word_assembly = set(
                range(word_assembly_start, word_assembly_start + area_k))
            if len((winners & word_assembly)) >= threshold:
                return word
        return None

    def getActivatedFibers(self):
        # Prune activated_fibers pased on the readout_rules
        pruned_activated_fibers = defaultdict(set)
        for from_area, to_areas in self.activated_fibers.items():
            for to_area in to_areas:
                if to_area in self.readout_rules[from_area]:
                    pruned_activated_fibers[from_area].add(to_area)

        return pruned_activated_fibers


class RussianParserBrain(ParserBrain):
    def __init__(self, p, non_LEX_n=10000, non_LEX_k=100, LEX_k=10,
                 default_beta=0.2, LEX_beta=1.0, recurrent_beta=0.05, interarea_beta=0.5, verbose=False):

        recurrent_areas = [NOM, VERB, ACC, DAT]
        ParserBrain.__init__(self, p,
                             lexeme_dict=RUSSIAN_LEXEME_DICT,
                             all_areas=RUSSIAN_AREAS,
                             recurrent_areas=recurrent_areas,
                             initial_areas=[LEX],
                             readout_rules=RUSSIAN_READOUT_RULES)
        self.verbose = verbose

        LEX_n = RUSSIAN_LEX_SIZE * LEX_k
        self.add_explicit_area(LEX, LEX_n, LEX_k, default_beta)

        self.add_area(NOM, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(ACC, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(VERB, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(DAT, non_LEX_n, non_LEX_k, default_beta)

        # LEX: all areas -> * strong, * -> * can be strong
        # non LEX: other areas -> * (?), LEX -> * strong, * -> * weak
        # DET? Should it be different?
        custom_plasticities = defaultdict(list)
        for area in recurrent_areas:
            custom_plasticities[LEX].append((area, LEX_beta))
            custom_plasticities[area].append((LEX, LEX_beta))
            custom_plasticities[area].append((area, recurrent_beta))
            for other_area in recurrent_areas:
                if other_area == area:
                    continue
                custom_plasticities[area].append((other_area, interarea_beta))

        self.update_plasticities(area_update_map=custom_plasticities)


class EnglishParserBrain(ParserBrain):
    def __init__(self, p, non_LEX_n=10000, non_LEX_k=100, LEX_k=20,
                 default_beta=0.2, LEX_beta=1.0, recurrent_beta=0.05, interarea_beta=0.5, verbose=False):
        ParserBrain.__init__(self, p,
                             lexeme_dict=LEXEME_DICT,
                             all_areas=AREAS,
                             recurrent_areas=RECURRENT_AREAS,
                             initial_areas=[LEX, SUBJ, VERB],
                             readout_rules=ENGLISH_READOUT_RULES)
        self.verbose = verbose

        LEX_n = LEX_SIZE * LEX_k
        self.add_explicit_area(LEX, LEX_n, LEX_k, default_beta)

        DET_k = LEX_k
        self.add_area(SUBJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(OBJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(VERB, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(ADJ, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(PREP, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(PREP_P, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(DET, non_LEX_n, DET_k, default_beta)
        self.add_area(ADVERB, non_LEX_n, non_LEX_k, default_beta)

        self.add_area(COMP1, non_LEX_n, non_LEX_k, default_beta)
        self.add_area(COMP2, non_LEX_n, non_LEX_k, default_beta)

        # LEX: all areas -> * strong, * -> * can be strong
        # non LEX: other areas -> * (?), LEX -> * strong, * -> * weak
        # DET? Should it be different?
        custom_plasticities = defaultdict(list)
        for area in RECURRENT_AREAS:
            custom_plasticities[LEX].append((area, LEX_beta))
            custom_plasticities[area].append((LEX, LEX_beta))
            custom_plasticities[area].append((area, recurrent_beta))
            for other_area in RECURRENT_AREAS:
                if other_area == area:
                    continue
                custom_plasticities[area].append((other_area, interarea_beta))

        self.update_plasticities(area_update_map=custom_plasticities)

    def getProjectMap(self):
        proj_map = ParserBrain.getProjectMap(self)
        # "War of fibers"
        if LEX in proj_map and len(proj_map[LEX]) > 2:  # because LEX->LEX
            raise Exception(
                "Got that LEX projecting into many areas: " + str(proj_map[LEX]))
        return proj_map

    def getWord(self, area_name, min_overlap=0.7):
        word = ParserBrain.getWord(self, area_name, min_overlap)
        if word:
            return word
        if not word and area_name == DET:
            winners = set(self.areas[area_name].winners)
            area_k = self.areas[area_name].k
            threshold = min_overlap * area_k
            nodet_index = DET_SIZE - 1
            nodet_assembly_start = nodet_index * area_k
            nodet_assembly = set(
                range(nodet_assembly_start, nodet_assembly_start + area_k))
            if len((winners & nodet_assembly)) > threshold:
                return "<null-det>"
        # If nothing matched, at least we can see that in the parse output.
        return "<NON-WORD>"


class ParserDebugger():
    def __init__(self, brain, all_areas, explicit_areas):
        self.b = brain
        self.all_areas = all_areas
        self.explicit_areas = explicit_areas

    def run(self):
        command = input("DEBUGGER: ENTER to continue, 'P' for PEAK \n")
        while command:
            if command == "P":
                self.peak()
                return
            elif command:
                print("DEBUGGER: Command not recognized...")
                command = input("DEBUGGER: ENTER to continue, 'P' for PEAK \n")
            else:
                return

    def peak(self):
        remove_map = defaultdict(int)
        # Temporarily set beta to 0
        self.b.no_plasticity = True
        self.b.save_winners = True

        for area in self.all_areas:
            self.b.areas[area].unfix_assembly()
        while True:
            test_proj_map_string = input(
                "DEBUGGER: enter projection map, eg. {\"VERB\": [\"LEX\"]}, or ENTER to quit\n")
            if not test_proj_map_string:
                break
            test_proj_map = json.loads(test_proj_map_string)
            # Important: save winners to later "remove" this test project round
            to_area_set = set()
            for _, to_area_list in test_proj_map.items():
                for to_area in to_area_list:
                    to_area_set.add(to_area)
                    if not self.b.areas[to_area].saved_winners:
                        self.b.areas[to_area].saved_winners.append(
                            self.b.areas[to_area].winners)

            for to_area in to_area_set:
                remove_map[to_area] += 1

            self.b.project({}, test_proj_map)
            for area in self.explicit_areas:
                if area in to_area_set:
                    area_word = self.b.interpretAssemblyAsString(area)
                    print("DEBUGGER: in explicit area " +
                          area + ", got: " + area_word)

            print_assemblies = input(
                "DEBUGGER: print assemblies in areas? Eg. 'LEX,VERB' or ENTER to cont\n")
            if not print_assemblies:
                continue
            for print_area in print_assemblies.split(","):
                print("DEBUGGER: Printing assembly in area " + print_area)
                print(str(self.b.areas[print_area].winners))
                if print_area in self.explicit_areas:
                    word = self.b.interpretAssemblyAsString(print_area)
                    print("DEBUGGER: in explicit area got assembly = " + word)

        # Restore assemblies (winners) and w values to before test projections
        for area, num_test_projects in remove_map.items():
            self.b.areas[area].winners = self.b.areas[area].saved_winners[0]
            self.b.areas[area].w = self.b.areas[area].saved_w[-num_test_projects - 1]
            self.b.areas[area].saved_w = self.b.areas[area].saved_w[:(
                -num_test_projects)]
        self.b.no_plasticity = False
        self.b.save_winners = False
        for area in self.all_areas:
            self.b.areas[area].saved_winners = []


# strengthen the assembly representing this word in LEX
# possibly useful way to simulate long-term potentiated word assemblies
# so that they are easily completed.
def potentiate_word_in_LEX(b, word, rounds=20):
    b.activateWord(LEX, word)
    for _ in range(20):
        b.project({}, {LEX: [LEX]})


# "dogs chase cats" experiment, what should happen?
# simplifying assumption 1: after every project round, freeze assemblies
# exp version 1: area not fired into until LEX fires into it
# exp version 2: project between all disinhibited fibers/areas, forming some "ghosts"

# "dogs": open fibers LEX<->SUBJ and LEX<->OBJ but only SUBJ disinhibited
# results in "dogs" assembly in LEX<->SUBJ (reciprocal until stable, LEX frozen)
# in version 2 would also have SUBJ<->VERB, so LEX<->SUBJ<->VERB overall

# "chase": opens fibers LEX<->VERB and VERB<->OBJ, inhibit SUBJ, disi
# results in "chase" assembly in LEX<->VERB
# in version 2 would also havee VERB<->OBJ

# "cats":


# Readout types
class ReadoutMethod(Enum):
    FIXED_MAP_READOUT = 1
    FIBER_READOUT = 2
    NATURAL_READOUT = 3


def parse(sentence="cats chase mice", language="English", p=0.1, LEX_k=20,
          project_rounds=10, verbose=True, debug=False, readout_method=ReadoutMethod.FIBER_READOUT):
    if language == "English":
        b = EnglishParserBrain(p, LEX_k=LEX_k, verbose=verbose)
        lexeme_dict = LEXEME_DICT
        all_areas = AREAS
        explicit_areas = EXPLICIT_AREAS
        readout_rules = ENGLISH_READOUT_RULES

    if language == "Russian":
        b = RussianParserBrain(p, LEX_k=LEX_k, verbose=verbose)
        lexeme_dict = RUSSIAN_LEXEME_DICT
        all_areas = RUSSIAN_AREAS
        explicit_areas = RUSSIAN_EXPLICIT_AREAS
        readout_rules = RUSSIAN_READOUT_RULES

    parseHelper(b, sentence, p, LEX_k, project_rounds, verbose, debug,
                lexeme_dict, all_areas, explicit_areas, readout_method, readout_rules)


def parseHelper(b, sentence, p, LEX_k, project_rounds, verbose, debug,
                lexeme_dict, all_areas, explicit_areas, readout_method, readout_rules):
    debugger = ParserDebugger(b, all_areas, explicit_areas)

    sentence = sentence.split(" ")

    extreme_debug = False

    pre_word = None
    comp = 0

    flag2 = False

    for word in sentence:
        lexeme = lexeme_dict[word]
        b.activateWord(LEX, word)
        if verbose:
            print("Activated word: " + word)
            print(b.areas[LEX].winners)

        flag = False

        '''if pre_word!=None:
            print(LEXEME_DICT[pre_word]["type"])
            print(LEXEME_DICT[word]["type"])'''

        if pre_word == None:
            flag = False
        elif (LEXEME_DICT[word]["type"] == LEXEME_DICT[pre_word]["type"] and (LEXEME_DICT[word]["type"] == "adverb"
                                                                              or LEXEME_DICT[word][
                                                                                  "type"] == "adjective" or
                                                                              LEXEME_DICT[word]["type"] == "noun")) or \
                ((LEXEME_DICT[word]["type"] == "adjective" or LEXEME_DICT[word]["type"] == "determinant")
                 and LEXEME_DICT[pre_word]["type"] == "noun"):
            flag = True
        if flag or flag2:

            if LEXEME_DICT[word]["type"] == "adjective" and LEXEME_DICT[pre_word]["type"] == "noun":
                flag2 = True
            elif LEXEME_DICT[word]["type"] == "noun":
                flag2 = False

            PRE_RULES = [AreaRule(DISINHIBIT, COMP1, 0),
                         AreaRule(DISINHIBIT, COMP2, 0)]
            for rule in PRE_RULES:
                b.applyRule(rule)

            last_comp = comp
            comp = comp % 2 + 1  # 选择单项投影区域
            proj_map = defaultdict(set)

            proj_map_first = defaultdict(set)

            if comp == 2:
                b.areas[COMP2].unfix_assembly()
                b.areas[COMP2].winners = []

                proj_map_first[LEX] = {COMP2, LEX}
                proj_map_first[COMP1] = {COMP2, COMP1}

                proj_map[LEX] = {COMP2, LEX}
                proj_map[COMP1] = {COMP2, COMP1}
                proj_map[COMP2] = {COMP2, LEX}

            elif comp == 1:
                b.areas[COMP1].unfix_assembly()
                b.areas[COMP1].winners = []

                proj_map_first[LEX] = {COMP1, LEX}

                proj_map[LEX] = {COMP1, LEX}
                proj_map[COMP1] = {COMP1, LEX}
                if last_comp == 2:
                    proj_map_first[COMP2] = {COMP1, COMP2}

                    proj_map[COMP2] = {COMP1, COMP2}
                elif LEXEME_DICT[pre_word]["type"] == "adverb":
                    proj_map_first[ADVERB] = {ADVERB, COMP1}

                    proj_map[ADVERB] = {ADVERB, COMP1}
                elif LEXEME_DICT[pre_word]["type"] == "adjective":
                    proj_map_first[ADJ] = {ADJ, COMP1}

                    proj_map[ADJ] = {ADJ, COMP1}
                elif LEXEME_DICT[pre_word]["type"] == "noun":
                    if len(b.area_states[SUBJ]) == 0:
                        proj_map_first[SUBJ] = {SUBJ, COMP1}

                        proj_map[SUBJ] = {SUBJ, COMP1}
                    else:
                        proj_map_first[OBJ] = {OBJ, COMP1}

                        proj_map[OBJ] = {OBJ, COMP1}
            # print(proj_map_first)
            for area in proj_map_first:
                # print(area)
                if area not in proj_map_first[LEX]:
                    b.areas[area].fix_assembly()
                    if verbose:
                        print("FIXED assembly bc not LEX->this area in: " + area)
                elif area != LEX:
                    b.areas[area].unfix_assembly()
                    b.areas[area].winners = []
                    if verbose:
                        print("ERASED assembly because LEX->this area in " + area)

            b.parse_project_undirection(proj_map_first)
            if verbose:
                print("Got proj_map = ")
                print(proj_map)

            for i in range(project_rounds):
                b.parse_project_undirection(proj_map)
                if verbose:
                    print("Got proj_map = ")
                    print(proj_map)
                if extreme_debug and word == "a":
                    print("Starting debugger after round " +
                          str(i) + "for word" + word)
                    debugger.run()

            POST_RULES = [AreaRule(INHIBIT, COMP1, 0),
                          AreaRule(INHIBIT, COMP2, 0)]
            for rule in POST_RULES:
                b.applyRule(rule)

        else:
            comp = 0

            for rule in lexeme["PRE_RULES"]:
                b.applyRule(rule)

            proj_map = b.getProjectMap()
            for area in proj_map:
                if area not in proj_map[LEX]:
                    b.areas[area].fix_assembly()
                    if verbose:
                        print("FIXED assembly bc not LEX->this area in: " + area)
                elif area != LEX:
                    b.areas[area].unfix_assembly()
                    b.areas[area].winners = []
                    if verbose:
                        print("ERASED assembly because LEX->this area in " + area)

            proj_map = b.getProjectMap()
            if verbose:
                print("Got proj_map = ")
                print(proj_map)

            for i in range(project_rounds):
                b.parse_project()
                if verbose:
                    proj_map = b.getProjectMap()
                    print("Got proj_map = ")
                    print(proj_map)
                if extreme_debug and word == "a":
                    print("Starting debugger after round " +
                          str(i) + "for word" + word)
                    debugger.run()

            # if verbose:
            #	print("Done projecting for this round")
            #	for area_name in all_areas:
            #		print("Post proj stats for " + area_name)
            #		print("w=" + str(b.areas[area_name].w))
            #		print("num_first_winners=" + str(b.areas[area_name].num_first_winners))

            for rule in lexeme["POST_RULES"]:
                b.applyRule(rule)

            if debug:
                print("Starting debugger after the word " + word)
                debugger.run()
        pre_word = word
    # Readout
    # For all readout methods, unfix assemblies and remove plasticity.
    b.no_plasticity = True
    for area in all_areas:
        b.areas[area].unfix_assembly()

    dependencies = []

    def read_out(area, mapping, lis, father_word=None, father_area=None):
        to_areas = mapping[area]
        for A in AREAS:
            if father_word == None and A == VERB:
                continue
            b.areas[A].unfix_assembly()
            b.areas[A].winners = []
        if father_word != None:
            b.activateWord(LEX, father_word)
            proj_map = defaultdict(set)
            proj_map[LEX] = {father_area}
            b.parse_project_undirection(proj_map)
            b.parse_project_undirection({father_area: mapping[father_area]})
        b.areas[LEX].unfix_assembly()
        b.areas[LEX].winners = []
        b.parse_project_undirection({area: to_areas})
        this_word = b.getWord(LEX, 0.8)
        '''print(area)
        print(this_word)'''
        if this_word in word_set or this_word == "<NON-WORD>":
            return
        word_set.add(this_word)
        lis.append(this_word)
        # print(lis)

        '''for to_area in to_areas:
            if to_area == LEX:
                continue
            b.project({}, {to_area: [LEX]})
            other_word = b.getWord(LEX)
            print(other_word)
            dependencies.append([this_word, other_word, to_area])'''

        for to_area in to_areas:
            if to_area != LEX:
                lis_new = None
                if to_area == ADVERB:
                    lis_new = adv
                elif to_area == OBJ:
                    lis_new = obj
                elif to_area == SUBJ:
                    lis_new = subj
                elif to_area == ADJ:
                    if lis == subj:
                        lis_new = adj_subj
                    else:
                        lis_new = adj_obj
                elif to_area == DET:
                    if lis == subj:
                        lis_new = det_subj
                    else:
                        lis_new = det_obj
                elif to_area == COMP1 or to_area == COMP2:
                    lis_new = lis
                # print(lis_new)
                read_out(to_area, mapping, lis_new, this_word, area)

    def treeify(parsed_dict, parent):
        for key, values in parsed_dict.items():
            key_node = pptree.Node(key, parent)
            if isinstance(values, str):
                _ = pptree.Node(values, key_node)
            else:
                treeify(values, key_node)

    det_subj = []
    adj_subj = []
    subj = []
    det_obj = []
    adj_obj = []
    obj = []
    verb = []
    adv = []

    word_set = set()

    if readout_method == ReadoutMethod.FIXED_MAP_READOUT:
        # Try "reading out" the parse.
        # To do so, start with final assembly in VERB
        # project VERB->SUBJ,OBJ,LEX

        parsed = {VERB: read_out(VERB, readout_rules)}

        print("Final parse dict: ")
        print(parsed)

        root = pptree.Node(VERB)
        treeify(parsed[VERB], root)

    if readout_method == ReadoutMethod.FIBER_READOUT:
        activated_fibers = b.getActivatedFibers()
        if verbose:
            print("Got activated fibers for readout:")
            print(activated_fibers)

        read_out(VERB, activated_fibers, verb)
        '''print("Got dependencies: ")
        print(dependencies)'''
        '''print(det_subj)
        print(adj_subj)
        print(subj)
        print(verb)
        print(adv)
        print(det_obj)
        print(adj_obj)
        print(obj)'''

    # root = pptree.Node(VERB)
    # treeify(parsed[VERB], root)

    def print_result():
        sp = subj[0]
        vp = verb[0]
        op = ""
        for i in range(len(adj_subj)):
            if adj_subj[len(adj_subj) - i - 1] == "<NON-WORD>":
                continue
            print("{NP," + adj_subj[len(adj_subj) - i - 1] + "," + sp + "}")
            sp = adj_subj[len(adj_subj) - i - 1] + " " + sp
        for i in range(1, len(subj)):
            if subj[i] == "<NON-WORD>":
                continue
            print("{NP," + sp + "," + subj[i] + "}")
            sp += " " + subj[i]
        if len(det_subj):
            print("{NP," + det_subj[0] + "," + sp + "}")
            sp = det_subj[0] + " " + sp
        if len(obj):
            op = obj[0]
            for i in range(len(adj_obj)):
                if adj_obj[len(adj_obj) - i - 1] == "<NON-WORD>":
                    continue
                print("{NP," + adj_obj[len(adj_obj) - i - 1] + "," + op + "}")
                op = adj_obj[len(adj_obj) - i - 1] + " " + op
            for i in range(1, len(obj)):
                if obj[i] == "<NON-WORD>":
                    continue
                print("{NP," + op + "," + obj[i] + "}")
                op += " " + obj[i]
            if len(det_obj):
                print("{NP," + det_obj[0] + "," + op + "}")
                op = det_obj[0] + " " + op
        for i in range(len(adv)):
            if adv[i] == "<NON-WORD>":
                continue
            print("{VP," + vp + "," + adv[i] + "}")
            vp += " " + adv[i]
        if len(op):
            print("{VP," + vp + "," + op + "}")
            vp += " " + op
        print("{S," + sp + "," + vp + "}")

    print_result()


# pptree.print_tree(root)


def main(sen):
    # generate_sentences.generate_sentences_samples(LEXEME_DICT)
    print(sen)
    parse(sen)
    # parse(sentence="bad good fat small big man cats chase quickly mice")
    # parse(sentence="big bad good cats chase small fat man")
    # parse(sentence="bad good big man fat small cats chase mice")


def generate_test(size):
    generate_sentences.generate_sentences_samples(LEXEME_DICT, size=size)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument('-s', '--sentence', type=str, default=None, help='Sentence to parse')
    arg.add_argument('-g', '--generator', type=int, default=None, help='Sample Size')
    argp = arg.parse_args()
    if argp.sentence is not None:
        main(argp.sentence)
    elif argp.generator is not None:
        generate_test(argp.generator)
    else:
        with open("tests.txt", "r") as f:
            sentences = f.readlines()
            for sentence in sentences:
                print(sentence)
                parse(sentence)

# TODOs
# BRAIN
# fix brain.py to work when no-assembly areas are projected in

# PARSER IMPLEMENTATION
# Factor out debugger of parse
# Factor out read-out, possibly other aspects of parse
# consider areas where only A->B needed not A<->B, easy to fix
# for example, SUBJ/OBJ->DET, etc?

# PARSER CONCEPTUAL
# 1) NATURAL READ OUT:
# "Fiber-activation read out": Remember fibers that were activated
# "Lexical-item read out": Get word from V, see rules (not sufficient but recovers basic structure)

# 2) PREP area: of, others
# "brand of toys", to merge brand<->of<->toys, look for activated noun areas
# for example if OBJ is the only one, we're done
# if multiple, recency? (first instance of lookahead/memory!)

# 3) Intransitive verbs (in particular wrt read out)

# RESEARCH IDEAS
# 1) Russian experiment (free word order)
# 2) Grammaticality, detect some sort of error for non-grammatical
