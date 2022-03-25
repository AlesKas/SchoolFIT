isAsc' :: Ord a => [a] -> Bool
isAsc' [] = True
isAsc' [x] = True
isAsc' (x:y:xs) = (x<=y) && isAsc' (y : xs)

-- Sort a list using quicksort algorithm.
-- quicksort :: Ord a => [a] -> [a]
-- quicksort [3,1,2] = [1,2,3]
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort [x] = [x]
quicksort (x:xs) = quicksort [a | a <- xs, a <= x] ++ [x] ++ quicksort[a | a <- xs, a > x]

main :: IO()
main = do
    print(isAsc' [1..10])
    print(isAsc' [3,2,1])
    print(quicksort [5,10,1,2,3,8,4,6,7,9])