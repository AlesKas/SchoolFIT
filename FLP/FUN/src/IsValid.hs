module IsValid (isValidBKG, validTerminals) where

import GrammarData

-- Check, if input grammar is valid CFG
isValidBKG :: Grammar -> Bool
isValidBKG (Grammar nonterminals terminals rules startingNonterminal) =
    validNonterminals nonterminals && validTerminals terminals  && validNonTerm nonterminals startingNonterminal  && validRules rules nonterminals terminals

-- Check if all nonterminals are length 1 and are from 'A' to 'Z'
validNonterminals :: [String] -> Bool
validNonterminals [] = False
validNonterminals [nonTerm] = maxLength1 nonTerm && elem (head nonTerm) ['A'..'Z']
validNonterminals (nonTerm:nonTerms) = if maxLength1 nonTerm && elem (head nonTerm) ['A'..'Z']
    then validNonterminals nonTerms else error "Invalid input nonterminals."

-- Check if all terminals are length 1 and are from 'a' to 'z'
validTerminals :: [String] -> Bool
validTerminals [] = False
validTerminals [term] = maxLength1 term && elem (head term) ['a'..'z']
validTerminals (term:terms) = if maxLength1 term && elem (head term) ['a'..'z']
    then validTerminals terms else error "Invalid input terminals."

-- Check if tested nonterminal is in the set of nonterminals
validNonTerm :: [String] -> String -> Bool
validNonTerm non start = elem start non || error "Nonterminal is not in set of nonterminals."

-- Check if there are valid terminals and nonterminals in rules
validRules :: [Rule] -> [String] -> [String] -> Bool
validRules [] nonTerm term = False 
validRules [r] nonTerm term = validRule r nonTerm term
validRules (r:rs) nonTerm term = (validRule r nonTerm term && validRules rs nonTerm term) || error "Wrong input rules."

-- Check if ther is valid nonterminal on the left side of the rule
-- And then check if on the right side there are valid non/terminals as well
validRule :: Rule -> [String] -> [String] -> Bool 
validRule (Rule left right) nonTerm term = validNonTerm nonTerm left && validRightSide (head right) nonTerm term

-- Well, this is kinda complicated
-- take first char of right side of rule and check if there is proper non/terminal and continue with tail of right side
validRightSide :: String -> [String] -> [String] -> Bool 
validRightSide [] nonTerm term = False 
validRightSide x nonTerm term = if length x > 1 then [head x] `elem` (nonTerm++term) && validRightSide (tail x) nonTerm term else x `elem` (nonTerm++term)

maxLength1 :: String -> Bool
maxLength1 x = length x == 1