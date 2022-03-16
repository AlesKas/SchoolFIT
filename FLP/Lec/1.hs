import Data.Ord
import Distribution.Simple (UserHooks(instHook))

elem' :: Eq a => a -> [a] -> Bool
elem' _ [] = False
elem' e (x : xs) = (x == e) || elem e xs

elem'' :: Eq t => t -> [t] -> Bool
elem'' _ [] = False
elem'' e (x : xs) = if x == e then True else elem'' e xs

factorial :: (Integral a) => a -> a
factorial 0 = 1
factorial n = n * factorial (n - 1)

addVectors :: (Num a) => (a, a) -> (a, a) -> (a, a)
addVectors a b = (fst a + fst b, snd a + snd b)

addVectors2 :: (Num a) => (a, a) -> (a, a) -> (a, a)
addVectors2 (x1, x2) (y1, y2) = (x1 + y1, x2 + y2)

sum' :: (Num a) => [a] -> a
sum' [] = 0
sum' (x:xs) = x + sum' xs

fun :: (Num a, Ord a) => a -> a
fun x = (if x > 100 then x else x * 2) + 42

max' :: (Ord a) => a -> a -> a
max' a b
    | a > b     = a
    | otherwise = b

inRange :: Ord a => a -> a -> a -> Bool
inRange low high x = il && ih
    where
        il = x >= low
        ih = x <= high

inRange2 :: Ord a => a -> a -> a -> Bool 
inRange2 low hight x =
    let il = x >= low
        ih = x <= hight
    in il && ih

head' :: [a] -> a
head' [] = error "Empty"
head' (x:_) = x

main :: IO ()
main = do
    print (elem' 2 [1..10])
    print (elem'' 20 [1..10])
    print(factorial 5)

    putStr "\nSeznamy:\n\n"

    -- Seznamy
    print (head [1..10])
    print (tail [1..10])
    print (last [1..10])
    print(1 : 2 : 3 : [])
    print(null [1..10])
    print(null [])
    print(take 5 [1..10])
    print(takeWhile (< 5) ([1..10]))
    print(takeWhile (> 5) ([1..10]))

    putStr "\nList komprehenze:\n\n"
    print([x | x <- [1..100], x `mod` 7 == 0])
    print([x + y | x <- [1..10], even x, y <- [1..10], odd y])
    print([[x | x <- xs, x `elem` ['a'..'z']] | xs <- ["abXcd", "efghX", "Xijkl"]])

    putStr "\nTuply:\n\n"
    print(fst (1,2))
    print(snd (1,2))
    print(zip [1,2,3] [4,5,6])
    print(zipWith (+) [1,2,3] [4,5,6])

    putStr "\nPatter Matching:\n\n"
    print(addVectors (1, 2) (3, 4))
    print(addVectors2 (1, 2) (3, 4))
    print(sum' [1..10])
    print(fun 42)
    print(max' 5 3)
    print(max' 3 5)
    print(inRange 1 5 3)
    print(inRange2 1 5 10)
    print(4 * (let a = 9 in a + 1) + 2)
    print(head' [1..10])