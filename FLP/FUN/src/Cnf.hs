module Cnf (bkg2cnf) where

import GrammarData
import Data.List
import Debug.Trace
import IsValid(validTerminals)

-- Transform bkg to cnf
bkg2cnf :: Grammar -> Grammar
bkg2cnf (Grammar nonterminals terminals rules start) = Grammar newNonterms terminals newRules start
    where newRules = nub (transformTerms (bkgRulesToCnfRules rules))
          newNonterms = nub (collectNewNonterminals newRules)

-- Lastly, go through all the newly created rules and add their left side to the set of nonterminals
collectNewNonterminals :: [Rule] -> [String]
collectNewNonterminals [] = []
collectNewNonterminals [Rule left right] = [left]
collectNewNonterminals (Rule left right : rest) = left : collectNewNonterminals rest

-- Rule is valid cnf rule if on the right side ther is only one terminal
-- or two nonterminals
-- It's kinda ugly but it works
isValidCnfRule :: [String] -> Bool
isValidCnfRule rule
    | length rule == 1 && elem (head (head rule)) ['a'..'z'] = True
    | length rule == 2 && elem (head (head rule)) ['A'..'Z'] && elem (head (head (tail rule))) ['A'..'Z'] = True
    | length rule == 2 && head (head rule) == '<' = True
    | otherwise = False

-- If rule is already valid CNF rule carry on,
-- else transform it into CNF rule
bkgRulesToCnfRules :: [Rule] -> [Rule]
bkgRulesToCnfRules [] = []
bkgRulesToCnfRules [Rule left right] = 
    if isValidCnfRule right 
        then [Rule left right] 
        else transformRulesToCnf (Rule left right) []
bkgRulesToCnfRules (Rule left right : rest) = 
    if isValidCnfRule right 
        then Rule left right : bkgRulesToCnfRules rest
        else bkgRulesToCnfRules rest ++ transformRulesToCnf (Rule left right) []

-- Splits BKG rule into two
-- first rule is currently processed rule
-- [Rules] are list of rules that originated from spliting that rule, so I need to check also that, if they are valid CNF rules
transformRulesToCnf :: Rule -> [Rule] -> [Rule]
transformRulesToCnf (Rule left right) rule =
    if isValidCnfRule right
        then rule ++ [Rule left right]
        else transformRulesToCnf (newToOld (createComplexNonterm right) (tail right)) (rule ++ [oldToNew left (head right : [createComplexNonterm right])])

-- Alias to create rule from original nonterminal to newly created nonterminal
oldToNew :: String -> [String] -> Rule
oldToNew = Rule

-- Alias to create rule from newly created nonterminal to original values
newToOld :: String -> [String] -> Rule
newToOld = Rule

-- Create new complex nonterminal of simple nonterminals
createComplexNonterm :: [String] -> String
createComplexNonterm xs = concat (["<"] ++ tail xs ++ [">"])

-- Transform terminals from left side of the rule to nonterminals
-- Right side of the rule can now take shape:
-- Terminal Nonterminal
-- Nonterminal Terminal
-- Nonterminal Nonterminal
-- this function creates new Nonterminal Nonterminal and Nonterminal => terminal rules
transformTerms :: [Rule] -> [Rule]
transformTerms [] = []
transformTerms (Rule _ []:_) = []
transformTerms (Rule _ (_:_:_:_):_) = []
transformTerms [Rule left [rightLeft, rightRight]] = termToNonterm left [rightLeft, rightRight]
transformTerms [Rule left [right]] = [Rule left [right]]
transformTerms (Rule left [rightLeft, rightRight] : rest) = termToNonterm left [rightLeft, rightRight] ++ transformTerms rest
transformTerms (Rule left [right] : rest) = Rule left [right] : transformTerms rest

-- Creates new nonterminal that derives to terminal
termToNonterm :: String -> [String] -> [Rule]
termToNonterm _ [] = []
termToNonterm _ [_] = []
termToNonterm _ (_:_:_:_) = []
termToNonterm left [rightLeft, rightRight]
    -- If there are two terminals on the right side, create nonterminal for both of them
    | rightLeft /= rightLeftNew && rightRight /= rightRightNew = [Rule left [rightLeftNew,rightRightNew], Rule rightLeftNew [rightLeft], Rule rightRightNew [rightRight]]
    -- If there is at least one terminal on the right side, create new nonterminal for it
    | rightLeft /= rightLeftNew = [Rule left [rightLeftNew,rightRightNew], Rule rightLeftNew [rightLeft]]
    | rightRight /= rightRightNew = [Rule left [rightLeftNew,rightRightNew], Rule rightRightNew [rightRight]]
    -- Otherwise return original rule
    | otherwise = [Rule left [rightLeft,rightRight]]
    -- Create new nonterminal for in case that on the right side of rule there are some terminals
    where rightLeftNew = if validTerminals [rightLeft] then "<" ++ rightLeft ++ ">" else rightLeft
          rightRightNew = if validTerminals [rightRight] then "<" ++ rightRight ++ ">" else rightRight