
#1.a
def prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

#print(prime(3))
#print(prime(4))
#print(prime(49))

#1.b
def select_primes(array):
    return [x for x in array if prime(x)]

print(select_primes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

