from typing import Set, List
from literals import Literal

class Clause:
    def __init__(self, literals=None):
        """Initialize a clause from a list of literals or empty set"""
        self.literals = set(literals) if literals else set()
    
    @classmethod
    def from_string(cls, clause_str: str) -> 'Clause':
        """Create a clause from a string (e.g., 'A OR -B OR C')"""
        if clause_str == "{}":
            return cls()
        return cls([Literal(lit.strip()) for lit in clause_str.split('OR')])
    
    def is_empty(self) -> bool:
        """Check if clause is empty (represents {})"""
        return len(self.literals) == 0
    
    def __str__(self) -> str:
        if self.is_empty():
            return "{}"
        return " OR ".join(sorted(lit.to_string() for lit in self.literals))
    
    def __eq__(self, other) -> bool:
        if isinstance(other, str) and other == "{}":
            return self.is_empty()
        return isinstance(other, Clause) and self.literals == other.literals
    
    def __hash__(self) -> int:
        return hash(frozenset(self.literals))

class KnowledgeBase:
    def __init__(self):
        self.clauses: List[Clause] = []
        
    def add_clause(self, clause: Clause):
        self.clauses.append(clause)