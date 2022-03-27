module Simple (removeSimpleRules) where

import GrammarData
import Debug.Trace
import Data.List

removeSimpleRules :: Grammar -> Grammar
removeSimpleRules (Grammar nonTerms terms rules start) = Grammar nonTerms terms (nub (processRules nonTerms rules)) start

-- For each nonterminal create Na set and create new nonsimple rules
processRules :: [String] -> [Rule] -> [Rule]
processRules [] _ = error "Missing some nonterminals."
processRules [nonTerm] rules = createNonSimpleRules nonTerm (nub (createNaSet nonTerm [nonTerm] rules)) rules
processRules (nonTerm:nonTerms) rules = createNonSimpleRules nonTerm (nub (createNaSet nonTerm [nonTerm] rules)) rules ++ processRules nonTerms rules

-- Create Na set
-- For each nonterminal check if there is simple rule that is in the non list, if yes, add right side of that rule to non list and continue
createNaSet :: String -> [String] -> [Rule] -> [String]
createNaSet nonTerm non [] = []
createNaSet nonTerm non [Rule left right] = non ++ [ head right | left `elem` non && isSimpleRule (Rule left right)]
createNaSet nonTerm non (Rule left right:rest) = createNaSet nonTerm (non ++ [ head right | isSimpleRule (Rule left right) && left `elem` non]) rest

-- Create Nonsimple rule based on Na set
createNonSimpleRules :: String -> [String] -> [Rule] -> [Rule]
createNonSimpleRules _ _ [] = error "Missong something to create nonsimple rules."
createNonSimpleRules nonTerm naSet [Rule left right] = [Rule left right | elem left naSet && not (isSimpleRule (Rule left right))]
createNonSimpleRules nonTerm naSet (Rule left right : rest) =
    if elem left naSet && not (isSimpleRule (Rule left right))
        then Rule nonTerm right : createNonSimpleRules nonTerm naSet rest 
        else createNonSimpleRules nonTerm naSet rest

-- Checks if input rule is simple rule
isSimpleRule :: Rule -> Bool
isSimpleRule (Rule l [r])
    | head r `elem` ['A'..'Z'] && length r == 1 = True
    | otherwise = False
isSimpleRule (Rule l r) = False