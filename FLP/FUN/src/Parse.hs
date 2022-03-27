module Parse (parseGrammar) where 

import Data.List.Split ( splitOn )
import GrammarData
import Debug.Trace

parseGrammar :: [String] -> Grammar
parseGrammar [] = error "Wrong input grammar, missing something."
parseGrammar [_] = error "Wrong input grammar, missing something."
parseGrammar [_,_] = error "Wrong input grammar, missing something."
parseGrammar [_,_,_] = error "Wrong input grammar, missing something."
parseGrammar (x:y:z:xz) =  Grammar (splitOn "," x) (splitOn "," y) (map parseRules xz) z

-- Parse input intu Rule
-- Right side of every rule is devided into characters for easier work later
parseRules :: String -> Rule
parseRules inputRule = Rule {left = head subParts, right = splitIntoChars (head (tail subParts))}
    where subParts = splitOn "->" inputRule

-- splitOn "" returns "" at 0 index, I dont want that
splitIntoChars :: String -> [[Char]]
splitIntoChars str = tail (splitOn "" str)