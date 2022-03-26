module GrammarData where

import Data.List

data Rule = Rule {
    left :: String,
    right :: [String]
} deriving (Eq)

instance Show Rule where
    show (Rule l r) = l ++ "->" ++ join "" r

data Grammar = Grammar {
    nonterminals :: [String],
    terminals :: [String],
    rules :: [Rule],
    startingNonterminal :: String
} deriving (Eq)

instance Show Grammar where
    show (Grammar nonterm term rules start) =
        join "," nonterm ++ "\n"
        ++ join "," term ++ "\n"
        ++ start ++ "\n"
        ++ join "\n" (map show rules)

-- Helper function to join list of strings
join :: String -> [String] -> String
join _ []    = ""
join sep (x:xs) = foldl' (\a b -> a ++ sep ++ b) x xs