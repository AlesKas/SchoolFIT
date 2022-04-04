module Cnf (bkg2cnf) where

import GrammarData
import Data.List
import Debug.Trace
import IsValid(validTerminals)

-- Transform bkg to cnf
bkg2cnf :: Grammar -> Grammar
bkg2cnf (Grammar nonterminals terminals rules start) = Grammar newNonterms terminals newRules start
    where newRules = nub (transformTerminalsToNonterminals (transformBkgRulesToCnfRules rules))
          newNonterms = nub (collectNewNonterminals newRules)

-- Lastly, go through all the newly created rules and add their left side to the set of nonterminals
collectNewNonterminals :: [Rule] -> [String]
collectNewNonterminals [] = []
collectNewNonterminals [Rule left right] = [left]
collectNewNonterminals (Rule left right : rest) = left : collectNewNonterminals rest

-- If rule is already valid CNF rule carry on,
-- else transform it into CNF rule
transformBkgRulesToCnfRules :: [Rule] -> [Rule]
transformBkgRulesToCnfRules [] = []
transformBkgRulesToCnfRules [Rule left right] = 
    if isValidCnfRule right 
        then [Rule left right] 
        else transformBkgRuleToCnfRule (Rule left right) []
transformBkgRulesToCnfRules (Rule left right : rest) = 
    if isValidCnfRule right 
        then Rule left right : transformBkgRulesToCnfRules rest
        else transformBkgRulesToCnfRules rest ++ transformBkgRuleToCnfRule (Rule left right) []

-- Splits BKG rule into two
-- first rule is currently processed rule
-- [Rules] are list of rules that originated from spliting original that rule, so I need to check also that, if they are valid CNF rules
transformBkgRuleToCnfRule :: Rule -> [Rule] -> [Rule]
transformBkgRuleToCnfRule (Rule left right) rule =
    if length right == 2
        then rule ++ [Rule left right]
        else transformBkgRuleToCnfRule 
            (newToOld (createComplexNonterm right) (tail right)) 
            (rule ++ [oldToNew left (head right : [createComplexNonterm right])])

-- Alias to create rule from original nonterminal to newly created nonterminal
oldToNew :: String -> [String] -> Rule
oldToNew = Rule

-- Alias to create rule from newly created nonterminal to original values
newToOld :: String -> [String] -> Rule
newToOld = Rule

-- Create new complex nonterminal of simple nonterminals
createComplexNonterm :: [String] -> String
createComplexNonterm xs = concat (["<"] ++ tail xs ++ [">"])

-- Transform terminals from right side of the rule to nonterminals
-- Right side of the rule can now take shape:
-- Terminal Terminal
-- Terminal Nonterminal
-- Nonterminal Terminal
-- Nonterminal Nonterminal
transformTerminalsToNonterminals :: [Rule] -> [Rule]
transformTerminalsToNonterminals [] = []
transformTerminalsToNonterminals (Rule _ []:_) = []
transformTerminalsToNonterminals (Rule _ (_:_:_:_):_) = []
transformTerminalsToNonterminals [Rule left [rightLeft, rightRight]] = createNonterminalFromTerminal left [rightLeft, rightRight]
transformTerminalsToNonterminals [Rule left [right]] = [Rule left [right]]
transformTerminalsToNonterminals (Rule left [rightLeft, rightRight] : rest) = createNonterminalFromTerminal left [rightLeft, rightRight] ++ transformTerminalsToNonterminals rest
transformTerminalsToNonterminals (Rule left [right] : rest) = Rule left [right] : transformTerminalsToNonterminals rest

-- Creates new nonterminal that derives to terminal
-- this function creates new Nonterminal => terminal rules
createNonterminalFromTerminal :: String -> [String] -> [Rule]
createNonterminalFromTerminal _ [] = []
createNonterminalFromTerminal _ [_] = []
createNonterminalFromTerminal _ (_:_:_:_) = []
createNonterminalFromTerminal left [rightLeft, rightRight]
    -- If there are two terminals on the right side, create nonterminal for both of them
    | rightLeft /= rightLeftNew && rightRight /= rightRightNew = [Rule left [rightLeftNew,rightRightNew], Rule rightLeftNew [rightLeft], Rule rightRightNew [rightRight]]
    -- If there is at least one terminal on the right side, create new nonterminal for it
    | rightLeft /= rightLeftNew = [Rule left [rightLeftNew,rightRightNew], Rule rightLeftNew [rightLeft]]
    | rightRight /= rightRightNew = [Rule left [rightLeftNew,rightRightNew], Rule rightRightNew [rightRight]]
    -- Otherwise return original rule
    | otherwise = [Rule left [rightLeft,rightRight]]
    where rightLeftNew = createNewNonterminal rightLeft
          rightRightNew = createNewNonterminal rightRight

-- Create new nonterminal for in case that on the right side of rule there are some terminals
createNewNonterminal :: String -> String
createNewNonterminal terminal =
    if validTerminals [terminal]
        then "<" ++ terminal ++ ">"
        else terminal

-- Rule is valid cnf rule if on the right side ther is only one terminal
-- or two nonterminals
-- It's kinda ugly but it works
isValidCnfRule :: [String] -> Bool
isValidCnfRule rule
    | length rule == 1 && elem (head (head rule)) ['a'..'z'] = True
    | length rule == 2 && elem (head (head rule)) ['A'..'Z'] && elem (head (head (tail rule))) ['A'..'Z'] = True
    | otherwise = False