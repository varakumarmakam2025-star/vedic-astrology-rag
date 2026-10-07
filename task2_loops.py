# Task 2: Loops and Lists

# Part 1: Basic for loop
print("Counting 1 to 5:")
for i in range(1, 6):
    print(i)

# Part 2: Looping through a list
fruits = ["apple", "banana", "mango", "orange"]
print("\nFruits list:")
for fruit in fruits:
    print(fruit)

# Part 3: While loop
print("\nWhile loop countdown:")
count = 5
while count > 0:
    print(count)
    count -= 1
print("Done!")

# Part 4: List operations
numbers = [10, 25, 3, 47, 8]
print("\nOriginal list:", numbers)
print("Sum of numbers:", sum(numbers))
print("Max number:", max(numbers))
print("Min number:", min(numbers))

numbers.append(100)
print("After adding 100:", numbers)

numbers.remove(3)
print("After removing 3:", numbers)

# Part 5: List comprehension (shortcut way to build a list)
squares = [n * n for n in range(1, 6)]
print("\nSquares of 1-5:", squares)

# Part 6: Looping with index using enumerate
print("\nFruits with index:")
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")