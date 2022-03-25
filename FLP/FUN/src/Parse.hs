module Parse (parseGrammar) where 

import Data.List.Split ( splitOn )
import GrammarData

parseGrammar :: [String] -> Grammar
parseGrammar [] = error "Wrong input grammar, missing something."
parseGrammar [_] = error "Wrong input grammar, missing something."
parseGrammar [_,_] = error "Wrong input grammar, missing something."
parseGrammar [_,_,_] = error "Wrong input grammar, missing something."
parseGrammar (x:y:z:xz) = 
        Grammar {
            nonterminals = splitOn "," x, 
            terminals = splitOn "," y, 
            rules = map parseRules xz,
            startingNonterminal = z
        }

parseRules :: String -> Rule
parseRules x =
    if length subParts == 2
        then Rule {left = head subParts, right = tail subParts}
        else error "Wrong input rules."
    where subParts = splitOn "->" x

