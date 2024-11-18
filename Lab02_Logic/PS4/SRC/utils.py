import os
from literals import Literal
from typing import List
from knowledge_base import Clause, KnowledgeBase

def read_input(path):
    if not os.path.exists(path):
            return None
    with open(path) as f:
        try:
            return [line.strip() for line in f]
        finally:
            f.close()


def write_output(path: str, status: bool, steps: List[List[Clause]], kb: KnowledgeBase):
    with open(path, 'w') as fs:
        for step_clauses in steps:
            fs.write(f"{len(step_clauses)}\n")
            for clause in step_clauses:
                fs.write(f"{str(clause)}\n")
        fs.write("YES" if status else "NO")


def process_line(line):
    list_elem = line.split('OR')
    list_elem = [item.strip() for item in list_elem]
    return list_elem


def data_structuring(lines):
    clauses = []
    alpha = process_line(lines[0])  # First line is the query (alpha)
    n = int(lines[1])              # Second line is number of KB clauses
    for i in range(2, 2 + n):      # Read n clauses starting from line 3
        clauses.append(process_line(lines[i]))
    return alpha, clauses


def remove(literal, clause):
    return [item for item in clause if item != literal]


def check_complement(clause: Clause) -> bool:
    """Check if clause contains complementary literals"""
    pairs = [(lit1, lit2) 
            for lit1 in clause.literals 
            for lit2 in clause.literals 
            if lit1 != lit2]
    return not any(lit1 == lit2.negative() for lit1, lit2 in pairs)


def issubset(this, src):
    for item in this:
        if not include(src, item):
            return False
    return True


def same_clause(clause_i, clause_j):
    return set(clause_i.literals) == set(clause_j.literals)


def include(clauses, current_clause):
    if current_clause == '{}':
        return False
    for clause in clauses:
        if same_clause(current_clause, clause):
            return True
    return False


def check_duplicate_and_remove(this, src):
    return [clause for clause in this 
            if not any(same_clause(clause, src_clause) for src_clause in src)]


def pl_resolve(clause_i: Clause, clause_j: Clause) -> List[Clause]:
    clauses = []
    for literal_i in clause_i.literals:
        for literal_j in clause_j.literals:
            if literal_i == literal_j.negative():
                # Create new clause without the complementary literals
                new_literals = set(clause_i.literals) | set(clause_j.literals)
                new_literals.remove(literal_i)
                new_literals.remove(literal_j)
                
                if not new_literals:
                    clauses.append(Clause())  # Empty clause
                else:
                    new_clause = Clause(new_literals)
                    if check_complement(new_clause):
                        clauses.append(new_clause)
    return clauses


def pl_resolution(kb: KnowledgeBase, alpha: List[str]) -> tuple[bool, List[List[Clause]]]:
    # Add negation of alpha to KB
    for literal in alpha:
        kb.add_clause(Clause([Literal(literal).negative()]))

    clauses = kb.clauses[:]
    result_arr = []
    new = []

    while True:
        temp = []
        n = len(clauses)
        pairs = [(clauses[i], clauses[j])
                for i in range(n) for j in range(i+1, n)]
        
        for (clause_i, clause_j) in pairs:
            resolvents = pl_resolve(clause_i, clause_j)
            new.extend(resolvents)

            if any(clause.is_empty() for clause in resolvents):
                # Found empty clause - contradiction
                result_arr.append([clause for clause in new 
                                if not any(clause == existing for existing in clauses)])
                return True, result_arr
        
        if all(any(new_clause == existing for existing in clauses) 
              for new_clause in new):
            result_arr.append([])
            return False, result_arr
        
        # Add new unique clauses
        for new_clause in new:
            if not any(new_clause == existing for existing in clauses):
                clauses.append(new_clause)
                temp.append(new_clause)
        
        result_arr.append(temp)