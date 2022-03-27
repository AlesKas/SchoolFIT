module Main (main) where

-- Library imports
import System.Environment
import Parse
import IsValid
import Simple
import Cnf
import System.IO

-- Process input arguments, expected aruments are (-i, -1, -2) and input file
-- Input file can also be empty, in that case read input grammar fron STDIN
processArguments :: [String] -> (String, String)
processArguments [] = error "Expected some arguments"
processArguments [option] = (option, "")
processArguments [option,filePath]
    | option == "-i" = ("-i", filePath)
    | option == "-1" = ("-1", filePath)
    | option == "-2" = ("-2", filePath)
    | otherwise = error "Unknown argument"
processArguments _ = error "Expects max 2 arguments"

readGrammar :: String -> String -> IO String
readGrammar option filePath = if filePath == "" then getContents else readFile filePath

main :: IO()
main = do
    args <- getArgs
    let (setting, inputFile) = processArguments args
    inputGrammar <- readGrammar setting inputFile
    let bkg = parseGrammar (Prelude.lines inputGrammar)
    if isValidBKG bkg
        then
            case setting of "-i" -> print bkg
                            "-1" -> print (removeSimpleRules bkg)
                            "-2" -> print (bkg2cnf (removeSimpleRules bkg))
                            _ -> error "Already tested aguments, adding this to become exhaustive"
        else error "Wrong input grammar."
    return()