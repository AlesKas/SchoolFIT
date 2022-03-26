module Parse (parseGrammar) where 

import Data.List.Split ( splitOn )
import GrammarData
import Debug.Trace

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

-- Parse input intu Rule
-- Right side of every rule is devided into characters for easier work later
parseRules :: String -> Rule
parseRules inputRule =
    if length subParts == 2
        then Rule {left = head subParts, right = filter (/="") (splitOn "" (head $ tail subParts))}
        else error "Wrong input rules."
    where subParts = splitOn "->" inputRule

