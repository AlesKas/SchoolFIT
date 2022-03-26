module Simple (removeSimpleRules) where

import GrammarData
import Debug.Trace
import Data.List

removeSimpleRules :: Grammar -> Grammar
removeSimpleRules (Grammar nonTerms terms rules start) = Grammar nonTerms terms (nub (processRules nonTerms rules)) start

-- For each nonterminal create Na set and create new nonsimple rules
processRules :: [String] -> [Rule] -> [Rule]
processRules [] _ = error "Missing some nonterminals."
processRules [nonTerm] rules = makeNonSimpleRules nonTerm (nub (createNaSet nonTerm [nonTerm] rules)) rules
processRules (nonTerm:nonTerms) rules = makeNonSimpleRules nonTerm (nub (createNaSet nonTerm [nonTerm] rules)) rules ++ processRules nonTerms rules

-- Create Na set
createNaSet :: String -> [String] -> [Rule] -> [String]
createNaSet nonTerm non [] = []
createNaSet nonTerm non [Rule left right] = non ++ [ head right | left `elem` non && isSimpleRule (Rule left right)]
createNaSet nonTerm non (Rule left right:rest) = createNaSet nonTerm (non ++ [ head right | isSimpleRule (Rule left right) && left `elem` non]) rest

-- Create Nonsimple rule based on Na set
makeNonSimpleRules :: String -> [String] -> [Rule] -> [Rule]
makeNonSimpleRules _ _ [] = error "Missong something to create nonsimple rules."
makeNonSimpleRules nonTerm naSet [Rule left right] = [Rule left right | elem left naSet && not (isSimpleRule (Rule left right))]
makeNonSimpleRules nonTerm naSet (Rule left right : rest) = 
    if elem left naSet && not (isSimpleRule (Rule left right))
        then Rule nonTerm right : makeNonSimpleRules nonTerm naSet rest 
        else makeNonSimpleRules nonTerm naSet rest

-- Checks if input rule is simple rule
isSimpleRule :: Rule -> Bool
isSimpleRule (Rule l [r])
    | head r `elem` ['A'..'Z'] = True
    | otherwise = False
isSimpleRule (Rule l r) = False