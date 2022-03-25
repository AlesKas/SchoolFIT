data Expr
    = Val Integer 
        | Add Expr Expr
        | Sub Expr Expr
        deriving (Show, Eq)

eval :: Expr -> Integer
eval (Val v) = v
eval (Add e1 e2) = eval e1 + eval e2
eval (Sub e1 e2) = eval e1 - eval e2

sum3 :: (Num a) => [a] -> a 
sum3 [] = 0
sum3 (x : xs) = x + sum3 xs

min' :: (Ord a) => a -> a -> a
min' a b
    | a > b = b
    | a < b = a
    | a == b = a

minim :: [Int] -> Int
minim [] = 0
minim [x] = x
minim (x:xs) = min' x (minim xs)

max' :: (Ord a) => a -> a -> a
max' a b
    | a > b = a
    | b > a = b
    | a == b = a

maxim :: [Int] -> Int
maxim [] = 0
maxim [x] = x
maxim (x:xs) = max x (maxim xs)

sum' :: (Num a) => [a] -> a
sum' [] = 0
sum' [x] = x
sum' (x:xs) = x + sum' xs

prod' :: (Num a) => [a] -> a
prod' [] = 0
prod' [x] = x
prod' (x:xs) = x * sum' xs

elem' :: (Ord a) => a -> [a] -> Bool
elem' _ [] = False
elem' a (x:xs)
    | a == x = True
    | otherwise = elem' a xs

zip' :: [a] -> [b] -> [(a,b)]
zip' [] [] = []
zip' [x] [] = []
zip' [] [y] = []
zip' [x] [y] = [(x,y)]
zip' (x:xs) (y:ys) = [(x,y)] ++ zip' xs ys

main :: IO()
main = do
    let exp1 = Val 5
    let expr2 = Val 3
    print(eval $ Add (Val 5) expr2)
    print(eval $ Sub exp1 expr2)
    print(sum3 [1..10])
    print(foldr (+) 0 [1..10])
    print(minim [1..10])
    print(maxim [1..10])
    print(sum' [1..10])
    print(prod' [0..10])
    print(elem' 5 [1..10])
    print(5 `elem'` [1..10])
    print([x^2 | x <- [1..10], x `mod` 2 == 0])
    print(zip' [1..5] [5..10])