# Simple script to attempt to compare lp files.
import argparse
from cStringIO import StringIO
import logging
from pprint import pprint
logging.basicConfig(level=logging.DEBUG)
_log = logging.getLogger(__name__)

section_names = set(['Minimize', 'Subject To', 'Bounds', 'General', 'Binaries'])


def parse_name(name, zero_based=False):
    name_parts = name.split("_")
    for i in xrange(len(name_parts)):
        try:
            number = int(name_parts[i])
            if not zero_based:
                number -= 1
            name_parts[i] = number
        except ValueError:
            continue
    return "_".join(str(x) for x in name_parts)


def process_min(min_section, zero_based=False):
    _log.info(" - Parsing Objective Function")
    results = set()

    parameters = min_section.split(":")[1]
    parameters = parameters.split("+")

    for p in parameters:
        parts = p.strip().split()
        multiplier = round(float(parts[0]), 5)
        name_parts = parse_name(parts[1], zero_based)
        results.add((multiplier, name_parts))

    return results


def compare(set0, set1, name0, name1):
    print "set0 size:", len(set0)
    print "set1 size:", len(set1)
    print "Only in {} objective function:".format(name0)
    set0_only = set0 - set1
    pprint(set0_only)
    print "Diff count:", len(set0_only)

    print "Only in {} objective function:".format(name1)
    set1_only = set1 - set0
    pprint(set1_only)
    print "Diff count:", len(set1_only)

    print "Diff in diff:", abs(len(set0_only) - len(set1_only))

def compare_sources(name_map0, name_map1, name0, name1):
    set0 = set(name_map0.iterkeys())
    set1 = set(name_map1.iterkeys())
    print "Source 0 size:", len(set0)
    print "Source 1 size:", len(set1)
    print "Only in {} objective function:".format(name0)
    set0_only = set0 - set1
    for source in set0_only:
        name = name_map0[source]
        print name, source

    print "Diff count:", len(set0_only)

    print "Only in {} objective function:".format(name1)
    set1_only = set1 - set0
    for source in set1_only:
        name = name_map1[source]
        print name, source

    print "Diff count:", len(set1_only)

    print "Diff in diff:", abs(len(set0_only) - len(set1_only))

    print "Complete set 0"
    print
    for source in set0:
        name = name_map0[source]
        print name, source
    print
    print "Complete set 1"
    print
    for source in set1:
        name = name_map1[source]
        print name, source


def build_atom(atom_parts, **kwargs):
    zero_based = kwargs.get("zero_based", False)
    if len(atom_parts) == 1:
        name = atom_parts[0]
        multiplier = 1.0
        sign = "+"
    elif len(atom_parts) == 2:
        if atom_parts[0] in ("-", "+"):
            sign, name = atom_parts
            multiplier = 1.0
        else:
            multiplier, name = atom_parts
            sign = "+"
    elif len(atom_parts) == 3:
        sign, multiplier, name = atom_parts

    multiplier = float(multiplier)
    if sign == "-":
        multiplier = -multiplier
    return parse_name(name, zero_based), round(multiplier, 5)

def add_or_merge_atom(atoms, atom):
    # I hate jump...

    #Search for matching atom.
    merge_target = None
    for member in atoms:
        if member[0] == atom[0]:
            merge_target = member
            break

    if merge_target is not None:
        atoms.remove(merge_target)
        new_atom = (merge_target[0], merge_target[1] + atom[1])
        _log.debug("Merging: {} and {} -> {}".format(merge_target, atom, new_atom))
        atoms.add(new_atom)
    else:
        atoms.add(atom)

def process_subject_expression(expression, zero_based=False):
    if not expression:
        return None

    atoms = set()

    equality_operators = ("=", "<=", ">=")

    operators = ("+", "-") + equality_operators

    atom_parts = []

    value = round(float(expression[-1]), 5)

    equality_operator = expression[-2]

    assert equality_operator in equality_operators

    # Include the equality operator so that we will process the last atom.
    for token in expression[:-1]:
        if token in operators:
            if atom_parts:
                atom = build_atom(atom_parts, zero_based=zero_based)
                add_or_merge_atom(atoms, atom)
            atom_parts = []
        if token in equality_operators:
            break
        atom_parts.append(token)

    return frozenset(atoms), equality_operator, value


def process_subject(subject_section, zero_based=False):
    _log.info(" - Parsing Subject To")
    name_map = {}

    tokens = subject_section.split()

    current_expression = []

    current_name = next_name = ""

    for token in tokens:
        if token.endswith(":"):
            current_name = next_name
            next_name = token
            parsed_expression = process_subject_expression(current_expression, zero_based)
            if parsed_expression is not None:
                name_map[parsed_expression] = current_name
            current_expression = []
        else:
            current_expression.append(token)

    # Grab the last expression.
    parsed_expression = process_subject_expression(current_expression, zero_based)
    if parsed_expression is not None:
        name_map[parsed_expression] = current_name

    return name_map


def process_bounds(bounds_section, zero_based):
    _log.info(" - Parsing Bounds")
    results = set()

    lines = bounds_section.splitlines()

    inf = float("+inf")

    for line in lines:
        parts = line.strip().split()

        if len(parts) == 5:
            parts[0] = float(parts[0])
            parts[-1] = float(parts[-1])
            parts[2] = parse_name(parts[2], zero_based)
            parts = tuple(parts)
        elif len(parts) == 3:
            parts[-1] = float(parts[-1])
            parts[0] = parse_name(parts[0], zero_based)
            parts = tuple([0.0, "<="] + parts)

        if parts[0] == 0.0 and (parts[-1] == inf or parts[-1] == 1.0):
            continue

        results.add(parts)

    return results


def process_binary_general(binaries, generals, zero_based):
    _log.info(" - Parsing binaries and general")
    results = set()

    names = binaries.split() + generals.split()

    for name in names:
        name = parse_name(name, zero_based)
        results.add(name)

    return results


def process_lp(lp, zero_based=False):
    """Process an individual lp file."""
    _log.info("Processing file {}".format(lp.name))
    sections = {}
    results = {}
    # Get sections
    current_section = None
    current_text = StringIO()
    current_line = lp.readline().strip()

    while current_line != "End":
        if current_line in section_names:
            if current_section is not None:
                sections[current_section] = current_text.getvalue()
            current_text = StringIO()
            current_section = current_line
            _log.info(" - Ingesting section {}".format(current_section))
        else:
            current_text.write(current_line+'\n')
        current_line = lp.readline().strip()

    sections[current_section] = current_text.getvalue()

    results["Minimize"] = process_min(sections["Minimize"], zero_based)
    results["Subject To"] = process_subject(sections["Subject To"], zero_based)
    results["Bounds"] = process_bounds(sections["Bounds"], zero_based)
    results["Binaries"] = process_binary_general(sections.get("Binaries", ""), sections.get("General", ""), zero_based)

    return results


parser = argparse.ArgumentParser()

parser.add_argument("lp", type=argparse.FileType('r'), nargs=2)
args = parser.parse_args()

lps = args.lp
print [x.name for x in lps]

lp_results = []
lp_results.append(process_lp(lps[0], True))
lp_results.append(process_lp(lps[1], False))

# Compare results.
_log.info("Comparing objective functions")
min0 = lp_results[0]["Minimize"]
min1 = lp_results[1]["Minimize"]

compare(min0, min1, lps[0].name, lps[1].name)

_log.info("Comparing Subject To")
subject0 = lp_results[0]["Subject To"]
subject1 = lp_results[1]["Subject To"]

compare_sources(subject0, subject1, lps[0].name, lps[1].name)

_log.info("Comparing Bounds")
bounds0 = lp_results[0]["Bounds"]
bounds1 = lp_results[1]["Bounds"]

compare(bounds0, bounds1, lps[0].name, lps[1].name)

_log.info("Comparing Binaries")
binaries0 = lp_results[0]["Binaries"]
binaries1 = lp_results[1]["Binaries"]

compare(binaries0, binaries1, lps[0].name, lps[1].name)

