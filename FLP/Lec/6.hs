len [] = 0
len (x:xs) = 1 + len xs

foldr' f a [] = a
foldr' f a (x:xs) = f x (foldr' f a xs)

main :: IO()
main = do
    print(len [1..10])
    print(foldr' (\_ d -> d + 1) 0 [1..10]) 
    print (elem 'X' (['a'..'z']++['A'..'Z']))