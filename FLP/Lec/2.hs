-- Find out whether element is in list.
elem' :: Eq a => a -> [a] -> Bool
elem' _ [] = False
elem' e (x : xs) = (e == x) || elem' e xs

-- Remove duplicates from list.
rem' :: Eq a => [a] -> [a]
rem' [] = []
rem' (x:xs)
    | x `elem'` xs = rem' xs
    | otherwise = x : rem' xs

-- Remove consecutive duplicates from list.
remd' :: Eq a => [a] -> [a]
remd' [] = []
remd' (x:xs) = x : remd' (dropWhile (== x) xs)

remd2' :: Eq a => [a] -> [a]
remd2' [] = []
remd2' [x] = [x]
remd2' (x:y:xs)
    | x == y = remd2' (x : xs)
    | otherwise = x : remd2' (y : xs)

-- Find all right triangles with integer length sides, with all sides <= 10,
-- and a perimeter of 24.
-- triangles :: [(Integer, Integer, Integer)]
-- triangles = [(6,8,10)]
triangles :: [(Integer, Integer, Integer)]
triangles = [(a,b,c) | c <- [1..10], b <- [1..c], a <- [1..b], a ^ 2 + b ^ 2 == c ^ 2, a + b + c == 24]

main :: IO ()
main = do
    print(elem' 2 [1..10])
    print(rem' "aaaabbbbbcccccaaaaa")
    print(remd' "aaaabbbbbcccccaaaaa")
    print(remd2' "aaaabbbbbcccccaaaaa")
    print(triangles)