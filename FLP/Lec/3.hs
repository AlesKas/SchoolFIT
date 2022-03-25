elemC :: Char -> [Char] -> Bool
elemC _ [] = False 
elemC c (x:xs) = if c == x then True else elemC c xs

map' :: (a -> b) -> [a] -> [b]
map' _ [] = []
map' f (x:xs) = f x : map' f xs

add1 x = x + 1

filter2 :: (a -> Bool) -> [a] -> [a]
filter2 _ [] = []
-- filter2 p (x:xs) = if p x then x : filter2 p xs else filter2 p xs
filter2 p (x:xs)
    | p x = x : filter2 p xs
    | otherwise = filter2 p xs

foldl2 :: (a -> b -> a) -> a -> [b] -> a
foldl2 f z []  = z
foldl2 f z (x:xs) = foldl2 f (f z x) xs

foldr2 :: (a -> b -> b) -> b -> [a] -> b
foldr2 f z [] = z
foldr2 f z (x:xs) = f x (foldr2 f z xs)

map2 :: (a -> b) -> [a] -> [b]
map2 f = foldr2 (\x acc -> f x : acc) []

elem3 :: (Eq a) => a -> [a] -> Bool
elem3 y = foldr f False 
    where
        f x acc = if x == y then True else acc

findKey :: (Eq k) => k -> [(k,v)] -> Maybe v
findKey key [] = Nothing
findKey key ((k,v):xs)
    | key == k = Just v
    | otherwise = findKey key xs

data Shape
    = Circle Float Float Float
    | Rectangle Float Float Float Float
    deriving (Show)

area :: Shape -> Float
area (Circle _ _ r) = pi * r ^ 2
area (Rectangle x1 y1 x2 y2) = abs (x2 - x1) * abs (y2 - y1)

data Point = Point Float Float deriving (Show)

data Shape2
    = Circle2 Point Float
    | Rectangle2 Point Point
    deriving (Show)

area2 :: Shape2 -> Float
area2 (Circle2 _ r) = pi * r ^ 2
area2 (Rectangle2 (Point x1 x2) (Point y1 y2)) = abs (x2 - x1) * (y2 - y1)

data Student = Student {
    login :: String,
    firstName :: String,
    lastName :: String,
    points :: Int
} deriving (Show)

data Day
    = Monday
    | Tuesday
    | Wadnesday
    | Thursday
    | Friday
    | Saturday
    | Sunday
    deriving (Eq, Ord, Show, Read, Bounded, Enum)

type AssocList k v = [(k,v)]

infixr 5 :<

data List a = Empty | a :< (List a)

instance (Show a) => Show (List a) where
    show l = "[" ++ helper l ++ "]"
        where
            helper Empty = ""
            helper (x :< Empty) = show x
            helper (x :< xs) = show x ++ "," ++ helper xs

data Tree a = EmptyTree | Node a (Tree a) (Tree a)

main :: IO()
main = do
    print([d | d <- [1..100], 100 `mod` d == 0])
    print([d | d <- filter(\d -> 100 `mod` d == 0)[1..100]])
    print(filter(\d -> 100 `mod` d == 0)[1..100])
    print(elemC 'b' "abbbbbsss")
    print(map' add1 [1..10])
    print(filter2 (\d -> d > 5)[1..10])
    print(foldl2 (+) 0 [1..10])
    print(foldr2 (+) 0 [1..10])
    print(map2 (\d -> d + 1)[1..10])
    print(elem3 'b' "abbbbbsss")
    print(findKey 3 [(3,'3'),(1,'1')])
    print(area $ Circle 10 20 30)
    print(Point 1 2)
    print(Student "xkaspa48" "Ales" "Kasparek" 100)
    print(Monday > Friday)
    print([1..10])
    print(Rectangle2 (Point 1 2) (Point 1 2))